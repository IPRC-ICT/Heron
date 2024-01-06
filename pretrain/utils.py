import os
import json
from Heron.pretrain.cost_model import MLPModel, TransformerModel

def list_files_in_folder(folder_path):
    files_list = []
    fpaths = []
    for file_name in os.listdir(folder_path):
        fpath = os.path.join(folder_path, file_name)
        if os.path.isfile(fpath):
            files_list.append(file_name)
            fpaths.append(fpath)
    return files_list, fpaths

def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def make_model(name, path = None):
    if name == "mlp":
        model = MLPModel()
    elif name == "trans":
        model = TransformerModel()
    if path != None:
        model.load(path)
    return model
