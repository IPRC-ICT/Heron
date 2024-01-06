import pickle
from dataset import Dataset
import time
import argparse
import os
import shutil
from tensorboardX import SummaryWriter 

from utils import make_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-name", type=str, default='trans')
    parser.add_argument("--path", type=str, default='./data.pkl')
    parser.add_argument("--model", type=str, default='trans')
    args = parser.parse_args()
    print("Arguments: %s" % str(args))
    cur = time.time()
    with open(args.path, "rb") as f:
        dset = pickle.load(f)
    print("Loaded Time : ", time.time() - cur)
    cur = time.time()
    train_set, val_set = dset.split_within_task()
    print("Split Time : ", time.time() - cur)

    if not os.path.exists("save"):
        os.makedirs("save")
    summary_path = "summary/%s"%args.save_name
    if os.path.exists(summary_path):
        shutil.rmtree(summary_path)
    os.makedirs(summary_path)
    writer = SummaryWriter(summary_path)

    model = make_model(args.model)
    filename = "save/" + args.save_name + ".pkl"
    model.save_name = "save/"+args.save_name 
    model.vocab = [dset.stage_names, dset.ax_names, dset.var_names]
    if not os.path.exists(model.save_name):
        os.makedirs(model.save_name)
    writer.platform = "V100"
    model.writer = writer
    
    model.fit_base(train_set, val_set)
