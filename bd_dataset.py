from PIL import Image, ImageOps
import json
import os
import cv2
import numpy as np
import torch


def get_category_map():
    category_map = {
        "table": 0,
        "Cell": 1,
        "header": 2
    }
    return category_map


class BDTablesDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root='./data/nc_ttv',
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
            img = self.padding(image_file, table_obj['bbox'], (35, 20, 20, 20))
            f.write(f"{image_file}\n")

            # cv2.imwrite(f"processed/{os.path.basename(image_path)}.jpg", output["debug_image"])
            img = np.array(img)
            cv2.imwrite(f"data/nc_ttv/processed/{image_file}", img)
            #print(cv2.imwrite(f"C:/temp/processed/{image_file}", img))
            # returns JSON object as
            # a dictionary

            # Closing file
        f.close()


if __name__ == "__main__":
    ds = BDTablesDataset(category_map=get_category_map(),
                         do_crop=True,
                         do_padding=True)
    ds.process_images(3)
