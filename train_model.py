"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
from datetime import datetime
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import cv2
from PIL import Image
import streamlit as st

sys.path.append("detr")
sys.path.append("src")

#from engine import evaluate, train_one_epoch
from models import build_model
import util.misc as utils
#import datasets.transforms as R

from config import Args
from table_datasets import PDFTablesDataset,RandomMaxResize

from grits import grits
from train_core import get_class_map
from train_core import get_transform
from train_core import get_model
from train_core import train

def get_data2(args):
    """
    Based on the args, retrieves the necessary data to perform training,
    evaluation or GriTS metric evaluation
    """
    # Datasets
    print("loading data for training")
    class_map = get_class_map(args.data_type)

    dataset_train = PDFTablesDataset(
        os.path.join(args.data_root_dir, "val"),
        get_transform(args.data_type, "val"),
        do_crop=False,
        max_neg=0,
        make_coco=False,
        image_extension=".jpg",
        xml_fileset="val_filelist.txt",
        class_map=class_map)
    dataset_val = PDFTablesDataset(os.path.join(args.data_root_dir, "val"),
                                   get_transform(args.data_type, "val"),
                                   do_crop=False,
                                   make_coco=True,
                                   image_extension=".jpg",
                                   xml_fileset="val_filelist.txt",
                                   class_map=class_map)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                        args.batch_size,
                                                        drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val,
                                 2 * args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)
    return data_loader_train, data_loader_val, dataset_val, len(
        dataset_train)


# def main():
class TrainModel:
    def __init__(self,  data_root_dir):
        args = Args

        print(args.__dict__)
        print("-" * 100)

        args.data_root_dir = data_root_dir
        args.config_file="structure_config.json"
        args.data_type='structure'
        args.mode='train'
        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("loading model")
        self.device = torch.device(args.device)
        self.model, self.criterion, self.postprocessors = get_model(args, self.device)
        self.args=args
        self.model.eval()

        #assert args.data_type == "structure", "GriTS is only applicable to structure recognition"
        #self.dataset_test = get_data2(args)

    def run_training(self, debug=True):
        self.args.debug = debug
        train(self.args, self.model, self.criterion, self.postprocessors, self.device)


def main():
    t = TrainModel( "data/PubTables1M-Structure-PASCAL-VOC")
    t.run_training()


if __name__ == "__main__":
    main()
