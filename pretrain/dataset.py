import os
import json
import math
import pickle
import numpy as np
from tqdm import tqdm

class Dataset:
    def __init__(self):
        self.perfs = {}
        self.features = {}
        self.stage_names = []
        self.ax_names = []
        self.var_names = []

    def _encode_stage(self, info):
        assert "ST:" in info
        name = info.split("ST:")[-1]
        if name == "None":
            return [0]
        name_id = None
        if name not in self.stage_names:
            self.stage_names.append(name)
            name_id = len(self.stage_names)
        else:
            name_id = self.stage_names.index(name)
        return [name_id]

    def _encode_ax(self, info):
        assert "AX:" in info
        name = info.split("AX:")[-1]
        if name == "None":
            return [0] + [0, 0]
        assert "fused" not in name
        ax_id = None
        axname = name.split(".")[0] 
        if axname not in self.ax_names:
            self.ax_names.append(axname)
            ax_id = len(self.ax_names)
        else:
            ax_id = self.ax_names.index(axname)
        names = name.split(".")
        innernames = [x for x in names if x == "inner"]
        outernames = [x for x in names if x == "outer"]
        inner_num = min(len(innernames), 5)
        outer_num = min(len(outernames), 5)
        return [ax_id] + [inner_num, outer_num + 5]


    def _encode_param(self, info):
        assert "PA:" in info
        name = info.split("PA:")[-1]
        if name == "None":
            return [0]
        name_id = None
        if name not in self.var_names:
            self.var_names.append(name)
            name_id = len(self.var_names)
        else:
            name_id = self.var_names.index(name)
        return [name_id]

    def _encode_variable(self, info):
        assert "VA:" in info
        name = info.split("VA:")[-1]
        if name == "None":
            return [0]
        return self._encode_usr_variable(name)

    def _encode_usr_variable(self, name):
        assert name != "None"
        name_id = None
        if name not in self.var_names:
            self.var_names.append(name)
            name_id = len(self.var_names)
        else:
            name_id = self.var_names.index(name)
        return [name_id]

    def fromSamples(self, samples):
        self.features[samples[0].task.name] = [self.feasFromParam(sample.knob_manager.solved_knob_vals_phenotype) for sample in samples]
    
    def fromRecord(self, record, task_name, max_predict = 1):
        features = []; perfs = []
        for rec in record:
            json_dict = json.loads(rec)
            perf = json_dict['perf']
            feas = self.feasFromParam(json_dict['param'])
            features.append(feas)
            perfs.append(math.exp(-perf))
        self.features[task_name] = np.array(features)
        perfs = np.array(perfs)
        self.perfs[task_name] = max_predict * perfs.min() / perfs


    def initilize_per_param(self):
        return

    def omitkeys(self):
        # omit loop lengths and other variables for shorter seq lenth
        return ["L#", "O#"]

    def feasFromParam(self, param):
        self.initilize_per_param()
        feas = []
        for pkey in param.keys():
            omit = False
            for key in self.omitkeys():
                if key in pkey:
                    omit = True
            if omit:
                continue

            if "P#" in pkey:
                info = pkey.split("#")[-1]
                st, ax, pa = info.split(",")
                st_vec  = self._encode_stage(st)
                ax_vec  = self._encode_ax(ax)
                res_vec = self._encode_param(pa)
            elif "V#" in pkey:
                info = pkey.split("#")[-1]
                st, ax, va = info.split(",")
                st_vec = self._encode_stage(st)
                ax_vec = self._encode_ax(ax)
                res_vec = self._encode_variable(va)
            else:
                st_vec = self._encode_stage("ST:None")
                ax_vec = self._encode_ax("AX:None")
                res_vec = self._encode_usr_variable(pkey)
            val_vec = [self._encode_number(param[pkey])]
            feas.append(st_vec + ax_vec + res_vec + val_vec)
        return feas

    def _encode_number(self, val):
        return math.log(1 + val) 

    def _read_one_record(self, task_name, path):
        self.features[task_name] = []
        self.perfs[task_name] = []
        for row in open(path):
            json_dict = json.loads(row)
            perf = json_dict['perf']
            feas = self.feasFromParam(json_dict['param'])
            self.features[task_name].append(feas)
            self.perfs[task_name].append(math.exp(-perf))
        min_perf = min(self.perfs[task_name])
        self.perfs[task_name] = [min_perf / x for x in self.perfs[task_name]]

    def read_records(self, path):
        op_names = os.listdir(path) 
        for opname in tqdm(op_names):
            op_path = os.path.join(path, opname)
            task_names = os.listdir(op_path)
            for task_name in tqdm(task_names):
                task_path = os.path.join(op_path, task_name)
                rec_path = os.path.join(task_path, "records.txt")
                try:
                    self._read_one_record(task_name, rec_path)
                except:
                    continue
        self.save("./data.pkl")

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    def split_within_task(self, ratio = 0.9):
        train_set = Dataset()
        test_set = Dataset()

        for task in self.features:
            features, perfs = self.features[task], self.perfs[task]
            perfs = np.exp(-np.array(perfs))
            features = np.array(features)
            split = int(ratio * len(features))

            arange = np.arange(len(features))
            arange = np.flip(arange)
            train_indices, test_indices = arange[:split], arange[split:]

            if len(train_indices):
                train_perfs = perfs[train_indices]
                train_perfs = min(train_perfs) / train_perfs
                train_features = features[train_indices]
                train_set.features[task] = train_features
                train_set.perfs[task] = train_perfs

            if len(test_indices):
                test_perfs = perfs[test_indices]
                test_perfs = min(test_perfs) / test_perfs
                test_features = features[test_indices]
                test_set.features[task] = test_features
                test_set.perfs[task] = test_perfs
        return train_set, test_set
    
    def __len__(self):
        lens = 0
        for key in self.perfs.keys():
            lens += len(self.perfs[key])
        return lens
    
    def tasks(self):
        return self.perfs.keys()

def create_one_task(task, features, perfs):
    ret = Dataset()
    ret.perfs[task] = np.array(perfs)
    ret.features[task] = np.array(features)
    return ret

if __name__ == "__main__":
    dataset = Dataset()
    dataset.read_records("./records")
