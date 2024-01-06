import io
import os
import copy
import pickle
import random
import time
import json
import torch
import numpy as np
import multiprocessing
import torch.nn.functional as F
from collections import OrderedDict
from itertools import chain
from Heron.pretrain.dataset import *


import torch.nn as nn


from .metric import (
    metric_rmse,
    metric_r_squared,
    metric_pairwise_comp_accuracy,
    metric_top_k_recall,
    metric_peak_score,
    random_mix,
)

def integer_to_one_hot(integer_vector, num_classes):
    assert integer_vector.max() < num_classes
    one_hot_matrix = torch.zeros(integer_vector.size(0), num_classes)
    one_hot_matrix.scatter_(1, integer_vector.unsqueeze(1), 1)
    return one_hot_matrix

class SegmentDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            device,
            fea_norm_vec=None,
            shuffle=False,
    ):
        self.device = device
        self.shuffle = shuffle
        self.number = len(dataset)
        self.batch_size = batch_size

        self.labels = torch.empty((self.number,), dtype=torch.float32)

        # Flatten features
        flatten_features = []
        ct = 0
        for task in dataset.features:
            perfs = dataset.perfs[task]
            self.labels[ct: ct + len(perfs)] = torch.tensor(perfs)
            for idx, row in enumerate(dataset.features[task]):
                pad_num = 70 - row.shape[0]
                assert pad_num >= 0
                if pad_num > 0:
                    row = np.concatenate([row, np.zeros((pad_num, row.shape[1]))], axis = 0)
                flatten_features.extend(row)
                ct += 1

        self.features = torch.tensor(np.array(flatten_features, dtype=np.float32))

        st_ids = self.features[:, 0]
        st_feas = integer_to_one_hot(st_ids.view(-1).to(torch.long), 15)

        ax_ids = self.features[:, 1]
        ax_feas = integer_to_one_hot(ax_ids.view(-1).to(torch.long), 10)

        extent_feas = self.features[:, 2:4]

        pa_ids = self.features[:, 4]
        pa_feas = integer_to_one_hot(pa_ids.view(-1).to(torch.long), 20)

        v_feas = self.features[:, 5].unsqueeze(1)
        self.features = torch.cat([st_feas, ax_feas, extent_feas, pa_feas, v_feas], dim = 1)
        self.features = self.features.reshape(-1, 70, self.features.shape[-1])

        if fea_norm_vec is not None:
            self.normalize(fea_norm_vec)

        self.iter_order = self.pointer = None

    def normalize(self, norm_vector=None):
        if norm_vector is None:
            norm_vector = torch.ones((self.features.shape[-1],))
            for i in range(self.features.shape[-1]):
                max_val = self.features[:, :, i].max().item()
                if max_val > 0:
                    norm_vector[i] = max_val
        self.features /= norm_vector.reshape(1, 1, -1)
        return norm_vector

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.number)
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0

        return self

    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration

        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):
        features = self.features[indices]
        labels = self.labels[indices]
        return (x.to(self.device) for x in (features, labels))

    def __len__(self):
        return self.number


class AttentionModule(nn.Module):  
    def __init__(self, in_dim, hidden_dim, out_dim, attention_head=8, res_block_cnt = 2):
        super().__init__()
        hidden_dim_1 = hidden_dim[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim_1, attention_head)

        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
        )
        self.l_list = []
        for i in range(res_block_cnt):
            self.l_list.append(nn.Sequential(
                nn.Linear(hidden_dim_1, hidden_dim_1), 
                nn.ReLU()
            ))
        self.l_list = nn.Sequential(*self.l_list)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_1, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )
        

    def forward(self, batch_datas_steps):
        encoder_output = self.encoder(batch_datas_steps)

        encoder_output = encoder_output.transpose(0, 1)
        output = self.attention(encoder_output, encoder_output, encoder_output)[0] + encoder_output

        for l in self.l_list:
            output = l(output) + output

        output = torch.sigmoid(self.decoder(output).sum(0))

        return output.squeeze()

def moving_average(average, update):
    if average is None:
        return update
    else:
        return average * 0.95 + update * 0.05


class TransformerModel:
    def __init__(self, device="cuda:0",  loss_type='rmse'):
        self.in_dim = 48
        self.hidden_dim = [64, 128, 256, 256]
        self.out_dim = [256, 128, 64, 1]
        self.loss_type = loss_type
        self.n_epoch = 100
        self.lr = 7e-4
        

        self.loss_func = torch.nn.MSELoss()

        self.grad_clip = 0.5
        self.fea_norm_vec = None

        # Hyperparameters for self.fit_base
        self.batch_size = 512
        self.infer_batch_size = 4096
        self.wd = 1e-6
        self.device = device
        self.print_per_epoches = 5

        # models
        self.base_model = None

        self.vocab = None

        # Others
        self.save_name = None
        self.writer = None

    def predict(self, inp):
        if not isinstance(inp, list):
            return self._predict_a_dataset(self.base_model, inp)
        else:
            assert self.vocab != None
        #   assert isinstance(inp, list)
            dset = Dataset()
            dset.stage_names, dset.ax_names, dset.var_names = self.vocab
            dset.fromSamples(inp)
            return self._predict_a_dataset(self.base_model, dset)

    def fit_base(self, train_set, valid_set, n_epoch=None):
        print("=" * 60 + "\nFit a net. Train size: %d" % len(train_set))
        train_loader = SegmentDataLoader(
            train_set, self.batch_size, self.device,
            shuffle=True
        )
        # Normalize features
        if self.fea_norm_vec is None:
            self.fea_norm_vec = train_loader.normalize()
        else:
            train_loader.normalize(self.fea_norm_vec)
        valid_loader = SegmentDataLoader(valid_set, self.infer_batch_size, self.device,
                                         fea_norm_vec=self.fea_norm_vec)
        n_epoch = n_epoch or self.n_epoch
        net = AttentionModule(self.in_dim, self.hidden_dim, self.out_dim).to(self.device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.wd
        )
        train_loss = None
        best_train_loss = 1e10
        total_steps = 0
        all_top1s = []
        for epoch in range(n_epoch):
            tic = time.time()
            # train
            net.train()
            for batch, (features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outs = net(features)
                loss = self.loss_func(outs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                optimizer.step()

                train_loss = moving_average(train_loss, loss.item())
                total_steps += 1
                if total_steps % 10 == 0 and self.writer != None:
                    self.writer.add_scalar("%s/train_loss"%self.writer.platform,
                                             np.sqrt(train_loss), total_steps)
            train_time = time.time() - tic
            tic = time.time()
            valid_loss = self._validate(net, valid_loader)
            top1s = self.eval_model(net, valid_set)
            all_top1s.append(top1s)
            valid_time = time.time() - tic

            t_loss = np.sqrt(train_loss)
            valid_loss = np.sqrt(valid_loss)
            loss_msg = "Train Loss: %.4f\tValid Loss: %.4f"%(t_loss, valid_loss)
            if self.writer != None:
                self.writer.add_scalar("%s/batch_val_loss"%self.writer.platform,
                                         valid_loss, epoch)
                self.writer.add_scalar("%s/prec"%self.writer.platform,
                                         np.average(top1s), epoch)

            print("Epoch: %d\tBatch: %d\t%s\tTrain Speed: %.0f" % (
                epoch, batch, loss_msg, len(train_loader) / train_time,))
            print("Valid time ", valid_time)

            if train_loss < best_train_loss:
                best_train_loss = train_loss
            if self.save_name != None:
                print("Saving")
                self.base_model = net
                self.save(self.save_name + "/epoch_%d.pkl"%epoch)
        with open(self.save_name + "/top1s.pkl", 'wb') as f:
            pickle.dump(all_top1s, f)
        return net

    def finetune(self, train_set, n_epoch):
        print("=" * 60 + "\nFinetune. Train size: %d" % len(train_set))
        train_loader = SegmentDataLoader(train_set, self.batch_size, self.device, shuffle=True)
        assert self.fea_norm_vec!= None
        train_loader.normalize(self.fea_norm_vec)
        net = self.base_model.to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        train_loss = None
        for epoch in range(n_epoch):
            net.train()
            for batch, (features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outs = net(features)
                loss = self.loss_func(outs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                optimizer.step()
                train_loss = moving_average(train_loss, loss.item())
            print("Train Loss: %.4f"%(np.sqrt(train_loss)))
        self.base_model = net

    def eval_model(self, model, valid_set):
        prediction = self._predict_a_dataset(model, valid_set)
        top1s = []
        for task in valid_set.tasks():
            preds = prediction[task]
            labels = valid_set.perfs[task]
            top1s.append(metric_peak_score(preds, labels, 1))

      # top1 = np.average(top1s)
        return top1s


    def _validate(self, model, valid_loader):
        model.eval()
        valid_losses = []

        for features, labels in valid_loader:
            out = model(features)
            loss = self.loss_func(out, labels)
            valid_losses.append(loss.item())

        return np.mean(valid_losses)

    def _predict_a_dataset(self, model, dataset):
        ret = {}
        for task, features in dataset.features.items():
            ret[task] = self._predict_a_task(model, task, features)
        return ret

    def _predict_a_task(self, model, task, features):
        if model is None:
            return np.zeros(len(features), dtype=np.float32)

        model.eval()
        tmp_set = create_one_task(task, features, np.zeros((len(features),)))

        preds = []
        for features, labels in SegmentDataLoader(
                tmp_set, self.infer_batch_size, self.device,
                fea_norm_vec=self.fea_norm_vec,
        ):
            preds.append(model(features))
        return torch.cat(preds).detach().cpu().numpy()


    def load(self, filename):
        if self.device == 'cpu':
            self.base_model, self.fea_norm_vec = \
                CPU_Unpickler(open(filename, 'rb')).load()
        else:
            self.base_model, self.fea_norm_vec = \
                pickle.load(open(filename, 'rb'))
            self.base_model = self.base_model.cuda() if self.base_model else None 

    def save(self, filename):
        base_model = self.base_model.cpu() if self.base_model else None 
        pickle.dump((base_model, self.fea_norm_vec),
                    open(filename, 'wb'))
        self.base_model = self.base_model.to(self.device) if self.base_model else None 

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


