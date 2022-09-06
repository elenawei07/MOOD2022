#  Copyright (C) 2020 Canon Medical Systems Corporation. All rights reserved

from pathlib import Path
import random
from functools import partial
from PIL import Image
import bcolz
import torch
import numpy as np
from torch.nn import functional as F


DATAPATH = Path(__file__).parent.parent / "data"


class ArrayDataset(torch.utils.data.Dataset):
    """
    Returns processed bcolz items.
    x_array: bcolz array
    y_array: bcolz array
    process: function (x_item, y_item -> dataset return type)
    index_map: dictionary mapping the torch dataset indices to the bcolz array indices.
    """

    def __init__(self, x_array, y_array, process):
        self.x_array = x_array
        self.y_array = y_array
        self.process = process

    def __getitem__(self, idx):
        return self.process(self.x_array[idx], self.y_array[idx])

    def __len__(self):
        return len(self.x_array)

class TestDataset(torch.utils.data.Dataset):
    """
    Returns processed bcolz items.
    x_array: bcolz array
    y_array: bcolz array
    process: function (x_item, y_item -> dataset return type)
    index_map: dictionary mapping the torch dataset indices to the bcolz array indices.
    """

    def __init__(self, x_array, y_array, process):
        self.x_array = x_array
        self.y_array = y_array
        
        self.process = process

    def __getitem__(self, idx):
        return self.process(self.x_array[idx], self.y_array[idx])

    def __len__(self):
        return len(self.x_array)


def resize_process(x, y, scale_factor=0.5):
    x = np.concatenate([x, x, x])
    x = torch.from_numpy(x).view(1, 3, x.shape[-2], x.shape[-1]).float()
    x = F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)
    y = np.concatenate([y, y, y])
    y = torch.from_numpy(y).view(1, 3, y.shape[-2], y.shape[-1]).float()
    y = F.interpolate(y, scale_factor=scale_factor, mode="bilinear", align_corners=False)
    return x[0], y[0]

def get_resize_test_data(datapath: Path, gtpath: Path, scale_factor=0.5):

    arr_data = bcolz.open(datapath)
    arr_gt = bcolz.open(gtpath)
    return TestDataset(arr_data, arr_gt, process=partial(resize_process, scale_factor=scale_factor))

def get_resize_data(datapath: Path, scale_factor=0.5):

    arr = bcolz.open(datapath)
    return ArrayDataset(arr, arr, process=partial(resize_process, scale_factor=scale_factor))


def get_train_val_datasets(datapath: Path, data="brain"):
    path = datapath

    val_dataset = get_resize_data(scale_factor=0.5, datapath=path/f"{data}_val")
    train_dataset = get_resize_data(scale_factor=0.5, datapath=path/f"{data}_train")


    return train_dataset, val_dataset

def get_test_dataset(datapath: Path, gtpath: Path, data='brain'):
    test_data_path = datapath
    ground_truth_path = gtpath

    test_dataset = get_resize_test_data(scale_factor=0.5, datapath=test_data_path, gtpath = ground_truth_path)

    return test_dataset


