"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
from datetime import datetime
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import cv2
from PIL import Image
import streamlit as st
import glob
from tqdm import tqdm
from bd_dataset import BDTablesDataset

sys.path.append("detr")
sys.path.append("src")

from engine import evaluate, train_one_epoch
from models import build_model
from grits import objects_to_cells
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
        "table column": 1,  # red
        "table row": 2,  # blue
        "table column header": 3,  # magenta
        "table projected row header": 4,  # cyan
        "table spanning cell": 5,
        "no object": 6,
    }
    return class_map


def get_colors_map():
    colors_map = [
        {'name': 'brown', 'label': 0, 'category_name': 'table', 'r': 128, 'g': 64, 'b': 64, 'dx': 3, 'dy': 3},
        {'name': 'red', 'label': 1, 'category_name': 'table column', 'r': 255, 'g': 0, 'b': 0, 'dx': 4, 'dy': 4},
        {'name': 'blue', 'label': 2, 'category_name': 'table row', 'r': 0, 'g': 0, 'b': 255, 'dx': 3, 'dy': 3},
        {'name': 'magenta', 'label': 3, 'category_name': 'table column header', 'r': 255, 'g': 0, 'b': 255, 'dx': 1,
         'dy': 1},
        {'name': 'cyan', 'label': 4, 'category_name': 'table projected row header', 'r': 0, 'g': 255, 'b': 255, 'dx': 2,
         'dy': 2},
        {'name': 'green', 'label': 5, 'category_name': 'table spanning cell', 'r': 0, 'g': 255, 'b': 0, 'dx': 3,
         'dy': 3},
        {'name': 'orange', 'label': 6, 'category_name': 'other', 'r': 255, 'g': 127, 'b': 39, 'dx': 3, 'dy': 3},
        {'name': 'green', 'label': 7, 'category_name': 'header cell', 'r': 0, 'g': 255, 'b': 0, 'dx': 3,
         'dy': 3},
        {'name': 'green', 'label': 8, 'category_name': 'subheader cell', 'r': 0, 'g': 255, 'b': 0, 'dx': 3,
         'dy': 3},
        {'name': 'green', 'label': 9, 'category_name': 'sub cell', 'r': 0, 'g': 255, 'b': 0, 'dx': 3,
         'dy': 3},
        {'name': 'green', 'label': 10, 'category_name': 'cell', 'r': 0, 'g': 255, 'b': 0, 'dx': 3,
         'dy': 3}
    ]
    return colors_map


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
    def __init__(self, checkpoint_path,
                 make_coco=False,
                 export_objects=True,
                 root=None,
                 images_dir="processed",
                 image_extension=".png",
                 save_debug_images=False,
                 original_xy_offset=True,
                 ds=None):
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
        self.model.eval()
        self.make_coco = make_coco
        self.export_objects = export_objects
        # class_map = get_class_map()
        self.images_dir = os.path.join(root, images_dir)
        self.normalize = R.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.save_debug_images = save_debug_images
        self.original_xy_offset = original_xy_offset
        self.ds = ds
        self.image_extension = image_extension
        self.root = root
        self.class_list = [{'label': c['label'], 'name': c['category_name']} for c in get_colors_map()]

        self.debug_images_dir = f"{os.path.split(os.path.abspath(self.images_dir))[0]}/debug"
        if self.save_debug_images and not os.path.exists(self.debug_images_dir):
            os.makedirs(self.debug_images_dir)

        try:
            with open(os.path.join(self.root, "filelist.txt"), 'r') as file:
                lines = [line.rstrip('\n') for line in file.readlines()]
        except:
            lines = []
        png_page_ids = set([f for f in lines if f.strip().endswith(self.image_extension)])
        file_list = set([f for f in os.listdir(self.images_dir) if f.strip().endswith(self.image_extension)])
        self.page_ids = list(file_list.union(png_page_ids))

    def process(self, max_count):
        result_objs = []
        if self.export_objects:
            for image_id, image_path in enumerate(tqdm(glob.glob(f"{self.images_dir}/*.png"), total=max_count)):
                img_filename = os.path.basename(image_path)
                crop_box = self.ds._get_cropped_bbox(img_filename)
                padding_box = self.ds._get_padding_bbox(img_filename)

                rows, cols, cells, debug_image = self.get_objects(image_path, crop_box, padding_box)
                if self.export_objects:
                    obj_details = {
                        "original_file": img_filename,
                        "cropped_bbox": crop_box,
                        "padding": padding_box,
                        "cells_structure": cells,
                        "table_structure": {
                            "rows": rows,
                            "columns": cols
                        }

                    }
                    if self.save_debug_images:
                        cv2.imwrite(f"{self.debug_images_dir}/{img_filename}", debug_image)

                    result_objs.append(obj_details)
            result = {"objs": result_objs,
                      "categories": self.class_list[:-3]
                      }
            jsonString = json.dumps(result)
            jsonFile = open("output.json", "w")
            jsonFile.write(jsonString)
            jsonFile.close()

        if self.make_coco:
            self.process_coco(max_count)

    def get_objects(self, image_path, crop_box, padding_box):
        img_filename = os.path.basename(image_path)
        table_words_dir = f"{self.root}/words/lines/"
        img_words_filepath = os.path.join(table_words_dir, img_filename.replace(".png", "_words.json"))
        page_tokens = []
        if os.path.exists(img_words_filepath):
            with open(img_words_filepath, 'r') as f:
                page_tokens = json.load(f)
        output = self.predict(image_path, page_tokens)
        print(output)
        rows = output["pred_table_structures"]["rows"]
        cols = output["pred_table_structures"]["columns"]
        cells = output["pred_cells"]
        if cells is None:
            cells = []
        if rows is None:
            rows = []
        if cols is None:
            rows = []
        print(rows,cols,cells)
        if self.original_xy_offset:
            rows = self.origin_img_table_xy(rows, crop_box, padding_box)
            cols = self.origin_img_table_xy(cols, crop_box, padding_box)
            cells = self.origin_img_cell_xy(cells, crop_box, padding_box)
        print(rows, cols, cells)
        return rows, cols, cells, output["debug_image"]

    def process_coco(self, max_count):
        # for coco
        self.dataset = {}
        self.dataset['images'] = [{'id': idx, 'file_name': page_id} for idx, page_id in enumerate(self.page_ids[:max_count])]
        self.dataset['annotations'] = []
        ann_id = 0
        for image_id, page_id in enumerate(self.page_ids[:max_count]):
            img_filename = page_id
            crop_box = self.ds._get_cropped_bbox(img_filename)
            padding_box = self.ds._get_padding_bbox(img_filename)
            image_path = os.path.join(self.images_dir, img_filename)

            rows, cols, cells, _ = self.get_objects(image_path, crop_box, padding_box)

            # Reduce class set
            # keep_indices = [idx for idx, label in enumerate(labels) if label in self.class_set]
            bboxes = [cell['bbox'] for cell in cells]
            bboxes.extend([row['bbox'] for row in rows])
            bboxes.extend([col['bbox'] for col in cols])

            labels = [self.cell_label(cell) for cell in cells]
            labels.extend([row['label'] for row in rows])
            labels.extend([col['label'] for col in cols])

            for bbox, label in zip(bboxes, labels):
                ann = {'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                       'iscrowd': 0,
                       'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                       'category_id': label,
                       'image_id': image_id,
                       'id': ann_id,
                       'ignore': 0,
                       'segmentation': []}
                self.dataset['annotations'].append(ann)
                ann_id += 1
            self.dataset['categories'] = [{'id': idx} for idx in self.class_list]

        jsonString = json.dumps(self.dataset)
        jsonFile = open("coco_output.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    def cell_label(self, cell):
        if 'header' in cell.keys() and cell['header'] :
            return 7
        if 'subheader' in cell.keys() and cell['subheader']:
            return 8
        if 'subcell' in cell.keys() and cell['subcell']:
            return 9
        return 10

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def origin_img_cell_xy(self, pred_cells, crop_box, padding_box):
        for cell in pred_cells:
            cell['bbox'] = list(np.array(cell['bbox']) + np.array(crop_box) - np.array(padding_box))
            for span in cell['spans']:
                span['bbox'] = list(np.array(span['bbox']) + np.array(crop_box) - np.array(padding_box))
        return pred_cells

    def origin_img_table_xy(self, items, crop_box, padding_box):
        for item in items:
            item['bbox'] = list(np.array(item['bbox']) + np.array(crop_box) - np.array(padding_box))
        return items

    def objects_to_grid_cells(self, outputs, page_tokens, image):
        """     from grits
                 This function runs the GriTS proposed in the paper. We also have a debug
                 mode which let's you see the outputs of a model on the pdf pages.
                 """
        structure_class_names = [
            'table', 'table column', 'table row', 'table column header',
            'table projected row header', 'table spanning cell', 'no object'
        ]
        structure_class_map = {k: v for v, k in enumerate(structure_class_names)}
        structure_class_thresholds = {
            "table": 0.5,
            "table column": 0.5,
            "table row": 0.5,
            "table column header": 0.5,
            "table projected row header": 0.5,
            "table spanning cell": 0.5,
            "no object": 10
        }
        boxes = outputs['pred_boxes']
        m = outputs['pred_logits'].softmax(-1).max(-1)
        scores = m.values
        labels = m.indices
        # rescaled_bboxes = rescale_bboxes(torch.tensor(boxes[0], dtype=torch.float32), img_test.size)
        rescaled_bboxes = self.rescale_bboxes(boxes[0].cpu(), image.size)
        pred_bboxes = [bbox.tolist() for bbox in rescaled_bboxes]
        pred_labels = labels[0].tolist()
        pred_scores = scores[0].tolist()
        pred_table_structures, pred_cells, pred_confidence_score = objects_to_cells(pred_bboxes, pred_labels,
                                                                                    pred_scores,
                                                                                    page_tokens, structure_class_names,
                                                                                    structure_class_thresholds,
                                                                                    structure_class_map)

        return pred_table_structures, pred_cells, pred_confidence_score

    def predict(self, image_path=None, page_tokens=None, debug=True, thresh=0.9):
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

        pred_table_structures, pred_cells, pred_confidence_score = self.objects_to_grid_cells(outputs, page_tokens, image)

        image_size = torch.unsqueeze(torch.as_tensor([int(h), int(w)]), 0).to(self.device)
        results = self.postprocessors['bbox'](outputs, image_size)[0]
        print(results)

        if debug is True:
            image = np.array(image)
            result_objects = []
            for idx, score in enumerate(results["scores"].tolist()):
                if score < thresh: continue

                xmin, ymin, xmax, ymax = list(map(int, results["boxes"][idx]))
                category_type = results["labels"][idx].item()
                colors = get_colors_map()

                r, g, b = colors[category_type]['r'], colors[category_type]['g'], colors[category_type]['b']
                dx, dy = colors[category_type]['dx'], colors[category_type]['dy']

                cv2.rectangle(image, (xmin, ymin), (xmax + dx, ymax + dy), (r, g, b), 2)
                result_objects.append({'category_type': category_type, 'bbox': [xmin, ymin, xmax, ymax]})
            results["debug_image"] = image
            results["debug_objects"] = result_objects
            results["pred_table_structures"] = pred_table_structures
            results["pred_cells"] = pred_cells
        return results


def main():
    m = TableRecognizer()
    import glob
    from tqdm import tqdm

    for image_path in tqdm(glob.glob("/data/pubtables1m/PubTables1M-Structure-PASCAL-VOC/images/*.jpg")[:100],
                           total=100):
        output = m.predict(image_path)
        cv2.imwrite(f"debug/{os.path.basename(image_path)}.jpg", output["debug_image"])


if __name__ == "__main__":
    main()
