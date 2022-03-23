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
from train_core import eval

# def main():
class EvalDetectionModel:
    def __init__(self,  data_root_dir):
        args = Args

        print(args.__dict__)
        print("-" * 100)

        args.data_root_dir = data_root_dir
        args.config_file="detection_config.json"
        args.data_type='detection'
        args.mode='eval'
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


    def run_eval(self, debug=True):
        self.args.debug = debug
        eval(self.args, self.model, self.criterion, self.postprocessors, self.device)


def main():
    t = EvalDetectionModel("data/PubTables1M-Structure-PASCAL-VOC")
    t.run_eval()


if __name__ == "__main__":
    main()
