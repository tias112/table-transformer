from __future__ import annotations
import os
from PIL import Image
import numpy as np
import cv2
import json

from copy import deepcopy
from typing import Any, List, Optional, Tuple

#from pdfplumber.display import COLORS
#from pdfplumber.page import Page as PlumberPage
#from pdfplumber.utils import decimalize
from pydantic import BaseModel

#from src.config import settings
from src.schema import TextToken
from src.utils.box_util import (  # type: ignore
    convert_to_bd,
    convert_to_box,
    stitch_boxes_into_lines,
    stitch_boxes_into_lines_as_boxes,
)
from src.utils.logger import get_logger
from src.utils.table import BorderBox  # type: ignore

logger = get_logger(__file__)

NOT_TRANSPARENT = (255,)


class Token(BaseModel):
    bbox: List[float]
    text: str = ""
    type: str = "text"
    children: Optional[List[Any]] = None


def export_from_text_tokens(text_tokens: List[TextToken]) -> List[Token]:
    return [
        Token(bbox=list(tt.bbox), text=tt.text, type=tt.type)
        for tt in text_tokens
    ]


def import_to_text_tokens(tokens: List[Token]) -> List[TextToken]:
    return [
        TextToken(bbox=tuple(t.bbox), text=t.text, type=t.type) for t in tokens if t.text != ""  # type: ignore
    ]


def stitch_text_tokens_into_lines(
    words: List[Token],
    max_x_dist: float = 10,
    min_y_overlap_ratio: float = 0.8,
) -> List[Token]:
    """
    Stitch tokens to lines.
        `words` - list of words/chars/etc. that will be merged to line.
        `max_x_dist` - If the distance between two sub-lines is greater than
                `max_x_dist` then there are two lines.
        `min_y_overlap_ratio` - If two tokens have y_overlap_ratio is greater than
                `min_y_overlap_ratio` then probably they are on the same line
                 (check `max_x_dist`).
    """
    words = deepcopy(words)
    merged_boxes = stitch_boxes_into_lines(
        convert_to_box(words),
        max_x_dist=max_x_dist,
        min_y_overlap_ratio=min_y_overlap_ratio,
    )
    return convert_to_bd(merged_boxes)  # type: ignore






def stitch_bd_bboxes_into_lines_as_boxes(
    text_tokens: List[Token],
    min_y_overlap_ratio: float = 0.8,
) -> List[Token]:
    merged_boxes = stitch_boxes_into_lines_as_boxes(
        convert_to_box(text_tokens), min_y_overlap_ratio=min_y_overlap_ratio
    )
    return convert_to_bd(merged_boxes)  # type: ignore



def save_to_file(
    tokens: List[Token],
    image_path=None,
    path_to_save=None,
    stage: str = ""
) -> Tuple[int, int]:
    if image_path is None:
        image_path = "data/nc_ttv/val/images/cigorig_3_tbls8.png"
    if path_to_save is None:
        path_to_save = "data/nc_ttv/val/words"

    image = image_path
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    (w,h) = (image.width, image.height)
    image = np.array(image)
    words_to_save = []
    for token in tokens:
        xmin, ymin, xmax, ymax = list(map(int, token.bbox))
        cv2.rectangle(image,
                    (xmin, ymin),
                    (xmax, ymax), (128,128,128), 1)
        word_json = {
                "bbox": [
                    token.bbox[0],
                    token.bbox[1],
                    token.bbox[2],
                    token.bbox[3]
                ],
                "text": token.text,
                "flags": 0,
                "span_num": 98,
                "line_num": 0,
                "block_num": 0
            }
        words_to_save.append(word_json)
    image_id = os.path.basename(image_path).strip().replace(".png", "")
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    if not os.path.exists(f"{path_to_save}/{stage}"):
        os.makedirs(f"{path_to_save}/{stage}")
    cv2.imwrite(f"{path_to_save}/{stage}/{os.path.basename(image_path)}", image)
    with open(f"{path_to_save}/{stage}/{image_id}_words.json", "w") as f:
        f.write(json.dumps(words_to_save))
    return (w,h)

# def save_to_file(tokens: List[Token],  stage_name: str, page_num: int = 0) -> None:
#     save_path = settings.verbose_save_path / f"{stage_name}" / "jsons"
#     save_path.mkdir(exist_ok=True, parents=True)
#     path_to_file = save_path / f"{page_num}.json"
#     path_to_file.write_text(
#         f"[{', '.join((token.json() for token in tokens))}]"
#     )


def arrange_text(
    input_tokens: List[TextToken], image_path, save_path, verbose: bool = True
) -> List[TextToken]:
    tokens = export_from_text_tokens(input_tokens)

    lines = stitch_text_tokens_into_lines(tokens)
   # if verbose:
    (w,h) = save_to_file(lines, image_path, save_path, "lines")
#        save_to_file(lines, "lines")

    return import_to_text_tokens(lines)
