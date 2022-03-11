from PIL import Image, ImageOps
import json
import os
import cv2
import numpy as np
import torch
import easyocr

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
            category_map=None,
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
        self.category_map = get_category_map()
        with open(os.path.join(root, "val.json"), "r") as file:
            data = json.load(file)
            imageid_dict = {k['id']: k['file_name'] for k in data['images']}
            self.table_objs = {imageid_dict[obj['image_id']]: obj for obj in data['annotations'] if
                               'category_id' in obj and obj['category_id'] == category_map['table']}


    #return bounding box of cropped table from original image
    def get_cropped_bbox(self, filename):
        table_obj = self.table_objs[filename]
        return table_obj['cropped_bbox']

    # Cropped image of above dimension with small padding
    # (It will not change original image)
    def padding(self, filename, bbox, padding_sizes):
        img = os.path.join(self.image_directory, filename)
        if isinstance(filename, str):
            img = Image.open(img).convert("RGB")
        table_bbox = bbox
        if self.do_padding:
            table_bbox = self.padding_bbox(bbox)
        if self.do_crop:
            img = img.crop(table_bbox)
        self.table_objs[filename].update({"cropped_bbox": table_bbox})
        # extend with white paddings
        return ImageOps.expand(img, padding_sizes, (255, 255, 255))

    # keeping borders within cropped image
    def padding_bbox(self, bbox):
        return bbox[0] - self.pad, bbox[1] - self.pad, bbox[0] + bbox[2] + 2 * self.pad, bbox[1] + bbox[
            3] + 2 * self.pad

    def process_images(self, max_samples):
        # Iterating through the tables and cut
        # list
        f = open(os.path.join(self.root, "filelist.txt"), "w")
        first_samples = {k: self.table_objs[k] for k in list(self.table_objs)[:max_samples]}

        for image_file in first_samples.keys():
            table_obj = first_samples[image_file]
            img = self.padding(image_file, table_obj['bbox'], (35, 30, 30, 30))
            f.write(f"{image_file}\n")

            # cv2.imwrite(f"processed/{os.path.basename(image_path)}.jpg", output["debug_image"])
            img = np.array(img)
            self.image_directory
            # cv2.imwrite(f"processed/{image_file}.jpg", img)
            print(cv2.imwrite(f"{self.image_processed}/{image_file}", img))   #TODO: /data/nc_ttv/processed/
            # returns JSON object as
            # a dictionary

            # Closing file
        f.close()

    def process_text(self, max_samples):

        reader = easyocr.Reader(["en"], gpu=False)
        first_samples = {k: self.table_objs[k] for k in list(self.table_objs)[:max_samples]}

        for image_file in first_samples.keys():
            img = os.path.join(self.image_processed, image_file)
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")

            epsilon = 0.001
            result = reader.readtext(
                image=np.asarray(img),
                slope_ths=epsilon,
                ycenter_ths=epsilon,
                height_ths=epsilon,
                width_ths=epsilon,
                decoder="wordbeamsearch",
                add_margin=0.15
            )
            print(result)

if __name__ == "__main__":
    ds = BDTablesDataset(category_map=get_category_map(),
                         do_crop=True,
                         do_padding=True)
    ds.process_images(3)
    ds.process_text(3)
