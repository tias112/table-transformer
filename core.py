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

# fuck this line :)
# reason 
# File "/home/research/table-transformer/detr/engine.py", line 12, in <module>
#     import util.misc as utils
#     ModuleNotFoundError: No module named 'util'
sys.path.append("detr")
sys.path.append("src")

from engine import evaluate, train_one_epoch
from models import build_model
import util.misc as utils
import datasets.transforms as R

from config import Args
from table_datasets import (
    PDFTablesDataset,
    TightAnnotationCrop,
    RandomPercentageCrop,
    RandomErasingWithTarget,
    ToPILImageWithTarget,
    RandomMaxResize,
    RandomCrop,
)


def get_class_map():
    class_map = {
        "table": 0,
        "table column": 1,
        "table row": 2,
        "table column header": 3,
        "table projected row header": 4,
        "table spanning cell": 5,
        "no object": 6,
    }
    return class_map


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
class TableRecognizer:
    def __init__(self, checkpoint_path):
        args = Args

        assert os.path.exists(checkpoint_path), checkpoint_path
        print(args.__dict__)
        print("-" * 100)

        args.model_load_path = checkpoint_path

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("loading model")
        self.device = torch.device(args.device)
        self.model, _, self.postprocessors = get_model(args, self.device)
        print(self.model)
        self.model.eval()

        class_map = get_class_map()

        self.normalize = R.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def predict(self, image_path = None, debug=True, thresh=0.9):
        if image_path is None:
            image_path = "/data/pubtables1m/PubTables1M-Structure-PASCAL-VOC/images/PMC514496_table_0.jpg"  

        image = image_path
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")

        w, h = image.size

        img_tensor = self.normalize(F.to_tensor(image))[0]
        img_tensor = torch.unsqueeze(img_tensor, 0).to(self.device)
   
        outputs = None
        with torch.no_grad():
            outputs = self.model(img_tensor)
        print(outputs['pred_boxes'])
        #TODO: visualize/print outputs
        image_size = torch.unsqueeze(torch.as_tensor([int(h), int(w)]), 0).to(self.device)
        print(image_size)
        #TODO: print image_size
        results = self.postprocessors['bbox'](outputs, image_size)[0]
        print(results)

        if debug is True:
            image = np.array(image)
            for idx, score in enumerate(results["scores"].tolist()):
                if score < thresh: continue

                xmin, ymin, xmax, ymax = list(map(int, results["boxes"][idx]))

                print("hee")

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            results["debug_image"] = image

        return results


def main():
    m = TableRecognizer("model_11.pth")
    import glob
    from tqdm import tqdm

    for image_path in tqdm(glob.glob("/data/pubtables1m/PubTables1M-Structure-PASCAL-VOC/images/*.jpg")[:100], total=100): 
        output = m.predict(image_path)
        cv2.imwrite(f"debug/{os.path.basename(image_path)}.jpg", output["debug_image"])



if __name__ == "__main__":
    main()
