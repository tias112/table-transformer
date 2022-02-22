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


def get_data(args):
    """
    Based on the args, retrieves the necessary data to perform training,
    evaluation or GriTS metric evaluation
    """
    # Datasets
    print("loading data")
    class_map = get_class_map(args.data_type)


    dataset_test = PDFTablesDataset(os.path.join(args.data_root_dir,
                                                 "val"),
                                    RandomMaxResize(1000, 1000),
                                    include_original=True,
                                    make_coco=False,
                                    image_extension=".jpg",
                                    xml_fileset="val_filelist.txt",
                                    class_map=class_map)
    return dataset_test

def get_model(args, device):
    """
    Loads DETR model on to the device specified.
    If a load path is specified, the state dict is updated accordingly.
    """
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    if args.model_load_path:
        print("loading model from checkpoint")
        loaded_state_dict = torch.load(args.model_load_path, map_location=device)
        model_state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in loaded_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=True)
    return model, criterion, postprocessors



# def main():
class GritsReport:
    def __init__(self, checkpoint_path, data_root_dir,  table_words_dir, metrics_save_filepath):
        args = Args

        assert os.path.exists(checkpoint_path), checkpoint_path
        print(args.__dict__)
        print("-" * 100)

        args.model_load_path = checkpoint_path
        args.data_root_dir = data_root_dir
        args.metrics_save_filepath = metrics_save_filepath
        args.table_words_dir=table_words_dir
        args.data_type='structure'
        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("loading model")
        self.device = torch.device(args.device)
        self.model, _, self.postprocessors = get_model(args, self.device)
        self.model.eval()

        assert args.data_type == "structure", "GriTS is only applicable to structure recognition"
        self.dataset_test = get_data(args)

    def run_grits(self, debug=True):
        self.args.debug = debug
        grits(self.args, self.dataset_test, self.device)


def main():
    g = GritsReport("model_11.pth", "data/PubTables1M-Structure-PASCAL-VOC", "data/PubTables1M-Table-Words-JSON", "stats.txt")
    g.run_grits()


if __name__ == "__main__":
    main()
