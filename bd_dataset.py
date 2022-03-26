from PIL import Image, ImageOps
import json
import os
import cv2
import numpy as np
import torch
import easyocr
import src.eval_utils as eval_utils
import src.schema as m
from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.text_flow import arrange_text
from src.schema import TextToken

def get_category_map():
    category_map = {
        "table": 0,
        "Cell": 1,
        "header": 2
    }
    return category_map


# def createIndex():

class BDTablesDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root='./data/nc_ttv/val',
            do_crop=True,
            do_padding=True,
            make_coco=False,
            bbox_data="val.json",
            image_extension=".png",
            category_map=None
    ):
        self.root = root
        self.do_crop = do_crop
        self.make_coco = make_coco
        self.image_extension = image_extension
        self.do_padding = do_padding
        self.category_map = category_map
        self.bbox_data = bbox_data
        self.table_objs = {}
        self.pad = 3
        self.root = root
        self.image_directory = os.path.join(root, "images")
        self.image_processed = os.path.join(root, "processed") #"C:/temp/processed/"#
        self.words_directory = os.path.join(root, "words")
        self.category_map = get_category_map()
        lines = os.listdir(self.image_directory)
        png_pages = set(
            [f for f in lines if f.strip().endswith(self.image_extension)])

        with open(os.path.join(root, "val.json"), "r") as file:
            data = json.load(file)
            imageid_dict = {k['id']: k['file_name'] for k in data['images']}
            self.table_objs = {imageid_dict[obj['image_id']]: obj for obj in data['annotations'] if
                               'category_id' in obj
                               and obj['category_id'] == category_map['table']
                               and imageid_dict[obj['image_id']] in png_pages}


    #return bounding box of cropped table from original image
    def _get_cropped_bbox(self, filename):
        if filename in self.table_objs.keys():
            table_obj = self.table_objs[filename]
            if 'cropped_bbox' in table_obj.keys():
                return table_obj['cropped_bbox']
        return [0, 0, 0, 0]

    def _get_padding_bbox(self, filename):
        if filename in self.table_objs.keys():
            table_obj = self.table_objs[filename]
            if 'padding' in table_obj.keys():
                return table_obj['padding']
        return [0, 0, 0, 0]

    # Cropped image of above dimension with small padding
    # (It will not change original image)
    def _do_extract_table_img(self, filename, bbox, padding_sizes):
        img = os.path.join(self.image_directory, filename)
        if isinstance(filename, str):
            img = Image.open(img).convert("RGB")
        table_bbox = bbox

        if self.do_crop:
            table_bbox = self._get_bbox_with_borders(bbox)
            img = img.crop(table_bbox)

        self.table_objs[filename].update({"cropped_bbox": table_bbox})

        self.table_objs[filename].update({"padding": padding_sizes})
        # extend with white paddings
        if self.do_padding:
            img = ImageOps.expand(img, padding_sizes, (255, 255, 255))

        return img

    # keeping borders within cropped image
    def _get_bbox_with_borders(self, bbox):
        return bbox[0] - self.pad, bbox[1] - self.pad, bbox[0] + bbox[2] + 2 * self.pad, bbox[1] + bbox[
            3] + 2 * self.pad

    def _process_images(self, max_samples):
        # Iterating through the tables and cut
        # list
        f = open(os.path.join(self.root, "filelist.txt"), "w")
        first_samples = {k: self.table_objs[k] for k in list(self.table_objs)[:max_samples]}

        for image_file in first_samples.keys():
            table_obj = first_samples[image_file]
            img = self._do_extract_table_img(image_file, table_obj['bbox'], (35, 30, 30, 30))
            f.write(f"{image_file}\n")

            img = np.array(img)
            if not os.path.exists(self.image_processed):
                os.makedirs(self.image_processed)
            cv2.imwrite(f"{self.image_processed}/{image_file}", img)
            # returns JSON object as
            # a dictionary

            # Closing file
        f.close()
        print("processed:", first_samples.keys())

    def _process_words(
        self, words, image_file
    ) -> List[m.TextToken]:
        image_path = os.path.join(self.image_processed, image_file)

        tokens = self.words_to_tokens(words)
        raster_text = arrange_text(
            tokens, image_path, self.words_directory
        )
        return raster_text

    def words_to_tokens(self, words):
        return [TextToken(bbox=tuple([float(word[0][0][0]), float(word[0][0][1]), float(word[0][2][0]), float(word[0][3][1])]), text=word[1]) for word in words]


    def _process_text(self, max_samples):

        reader = easyocr.Reader(["en"], gpu=False)
        first_samples = {k: self.table_objs[k] for k in list(self.table_objs)[:max_samples]}

        for image_file in first_samples.keys():
            img = os.path.join(self.image_processed, image_file)
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")

            epsilon = 0.001
            result = reader.readtext(
                image=np.asarray(img)
             #   slope_ths=epsilon,
            #    ycenter_ths=epsilon,
            #    height_ths=epsilon,
            #    width_ths=epsilon,
#                decoder="wordbeamsearch"
            #   add_margin=0.15
            )
            tokens = self._process_words(result, image_file)

            print(result)


if __name__ == "__main__":
    ds = BDTablesDataset(
        root="C:/temp/data/nc_ttv/val",
         do_crop=True,
         do_padding=True,
        category_map=get_category_map())
    ds._process_images(226)
    ds._process_text(226)
