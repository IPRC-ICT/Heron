import pickle
from dataset import *
import time
import argparse
import os
import shutil
from tensorboardX import SummaryWriter 
from cost_model.metric import (
    metric_rmse,
    metric_r_squared,
    metric_pairwise_comp_accuracy,
    metric_top_k_recall,
    metric_peak_score,
    random_mix,
)

from utils import make_model

def evaluate_model(models, test_set):
    # make prediction
    predictions = [model.predict(test_set) for  model in models]
    for i in range(len(predictions)):
        all_preds = predictions[:i+1]
        rmse_list = []
        r_sqaured_list = []
        pair_acc_list = []
        peak_score1_list = []
        peak_score5_list = []
        workload_maps = {}

        for i, task in enumerate(test_set.tasks()):
            apreds = [p[task].reshape(1, -1) for p in all_preds]
            if len(all_preds) > 1:
                m_preds = np.concatenate(apreds, axis = 0)
                preds = np.mean(m_preds, axis = 0).reshape(-1)
            else:
                preds = apreds[0].reshape(-1)
            labels = test_set.perfs[task]

            rmse_list.append(np.square(metric_rmse(preds, labels)))
            r_sqaured_list.append(metric_r_squared(preds, labels))
            pair_acc_list.append(metric_pairwise_comp_accuracy(preds, labels))
            peak_score1_list.append(metric_peak_score(preds, test_set.perfs[task], 1))
            peak_score5_list.append(metric_peak_score(preds, test_set.perfs[task], 5))


        rmse = np.sqrt(np.average(rmse_list))
        r_sqaured = np.average(r_sqaured_list)
        pair_acc = np.average(pair_acc_list)
        peak_score1 = np.average(peak_score1_list)
        peak_score5 = np.average(peak_score5_list)

        eval_res = {
            "RMSE": rmse,
            "R^2": r_sqaured,
            "pairwise comparision accuracy": pair_acc,
            "average peak score@1": peak_score1,
            "average peak score@5": peak_score5,
        }
        print(eval_res)
    return eval_res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-name", type=str, default='trans')
    parser.add_argument("--model", type=str, default='trans')
    args = parser.parse_args()
    print("Arguments: %s" % str(args))


    cur = time.time()
    with open("./data.pkl", "rb") as f:
        dset = pickle.load(f)
    print("Loaded Time : ", time.time() - cur)
    cur = time.time()
    train_set, val_set = dset.split_within_task()
    print("Split Time : ", time.time() - cur)

    models = []
    for idx in [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]:
        model = make_model(args.model)
        model.save_name = "save/"+args.save_name 
        model.load(model.save_name + "/epoch_%d.pkl"%idx)
        model.vocab = [dset.stage_names, dset.ax_names, dset.var_names]
        models.append(model)
    evaluate_model(models, val_set)
    
