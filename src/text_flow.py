from __future__ import annotations
import os
from PIL import Image
import numpy as np
import cv2

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


def find_gap(
    lines: List[Token],
) -> List[Token]:
    """
    Detect gaps between two columns of text
    """
    # sort lines by Y
    lines.sort(key=lambda x: x.bbox[1])

    gaps: List[Token] = []
    while len(lines) > 10:
        merged_intervals = merge_bbox_intervals(deepcopy(lines))
        gaps = define_gaps_between_intervals(merged_intervals, lines, gaps)
        lines.pop(0)
    return gaps


def merge_bbox_intervals(intervals: List[Token]) -> List[Token]:
    intervals.sort(key=lambda x: x.bbox[0])
    i = 0
    j = 1
    while j < len(intervals):
        if intervals[i].bbox[2] > intervals[j].bbox[2]:
            intervals.pop(j)
        elif intervals[i].bbox[2] < intervals[j].bbox[0]:
            i += 1
            j += 1
        else:
            tmp = intervals.pop(j)
            intervals[i].bbox = [
                intervals[i].bbox[0],
                intervals[i].bbox[1],
                tmp.bbox[2],
                intervals[i].bbox[3],
            ]
    return intervals


def insert_gap(gap: Token, gaps: List[Token]) -> List[Token]:
    i = 0
    while i < len(gaps):
        if bbox_inside_another(gap, gaps[i]):
            return gaps
        elif bbox_inside_another(gaps[i], gap):
            gaps.pop(i)
        i += 1
    gaps.append(gap)
    return gaps


def define_gaps_between_intervals(
    intervals: List[Token],
    lines_for_x: List[Token],
    gaps: List[Token],
) -> List[Token]:
    l_old = intervals.pop(0)
    while intervals:
        l = intervals.pop(0)
        gap = Token(
            type="gap",
            bbox=[
                l_old.bbox[2],
                lines_for_x[0].bbox[1],
                l.bbox[0],
                lines_for_x[-1].bbox[3],
            ],
            text="",
        )
        insert_gap(gap, gaps)
        l_old = l
    return gaps


def find_paragraphs(
    gaps: List[Token], page_height: float, page_width: float
) -> List[Token]:
    # Add the general paragraphs. It will contain all other paragraphs.
    paragraphs = [
        Token(
            bbox=[0, 0, page_width, page_height],
            type="paragraph",
            children=[],
            text="",
        )
    ]

    if not gaps:
        return paragraphs

    # Sort gaps. Paragraphs (and gaps between paragraphs) can be nested.
    # The shortest gaps always is the deepest one.
    gaps.sort(key=lambda x: x.bbox[3] - x.bbox[1])

    # Add left and right borders. They are longer then any real gaps
    gaps.append(Token(bbox=[0, 0, 0, page_height + 1], type="gap", text=""))
    gaps.append(
        Token(
            bbox=[page_width, 0, page_width, page_height + 1],
            type="gap",
            text="",
        )
    )

    while len(gaps) > 2:
        # Take the shortest gap
        gap = gaps.pop(0)

        # Find paragraphs to the left and to the right about gap.
        # Left (for left paragraph) and right(for right) border will define later.
        paragraph_left = Token(
            bbox=[0, gap.bbox[1], gap.bbox[0], gap.bbox[3]],
            type="paragraph",
            children=[],
            text="",
        )
        paragraph_right = Token(
            bbox=[gap.bbox[2], gap.bbox[1], page_width, gap.bbox[3]],
            type="paragraph",
            children=[],
            text="",
        )

        # find borders
        for g in gaps:
            paragraph_left.bbox[0] = (
                max(paragraph_left.bbox[0], g.bbox[2])
                if max(paragraph_left.bbox[0], g.bbox[2])
                < paragraph_left.bbox[2]
                else paragraph_left.bbox[0]
            )
            paragraph_right.bbox[2] = (
                min(paragraph_right.bbox[2], g.bbox[0])
                if min(paragraph_right.bbox[2], g.bbox[0])
                > paragraph_right.bbox[0]
                else paragraph_right.bbox[2]
            )

        paragraphs.append(paragraph_left)
        paragraphs.append(paragraph_right)

    return paragraphs


def get_nested_paragraphs(
    tokens: List[Token], paragraphs: List[Token]
) -> List[Token]:
    # Fill the smallest paragraph first.
    paragraphs.sort(key=lambda x: x.bbox[3] - x.bbox[1])

    for parag in paragraphs[:]:
        for token in tokens[:]:
            if bbox_inside_another(token, parag):
                tokens.remove(token)
                parag.children.append(token)

        paragraphs.remove(parag)
        if not parag.children:
            logger.error("Paragraph has no children")

        tokens.append(parag)

    return tokens


def stitch_bd_bboxes_into_lines_as_boxes(
    text_tokens: List[Token],
    min_y_overlap_ratio: float = 0.8,
) -> List[Token]:
    merged_boxes = stitch_boxes_into_lines_as_boxes(
        convert_to_box(text_tokens), min_y_overlap_ratio=min_y_overlap_ratio
    )
    return convert_to_bd(merged_boxes)  # type: ignore


def bbox_inside_another(box1: Token, box2: Token) -> bool:
    """Box1 inside box2"""
    bbox1 = BorderBox(
        top_left_x=round(box1.bbox[0]),
        top_left_y=round(box1.bbox[1]),
        bottom_right_x=round(box1.bbox[2]),
        bottom_right_y=round(box1.bbox[3]),
    )
    bbox2 = BorderBox(
        top_left_x=round(box2.bbox[0]),
        top_left_y=round(box2.bbox[1]),
        bottom_right_x=round(box2.bbox[2]),
        bottom_right_y=round(box2.bbox[3]),
    )
    return bbox1.box_is_inside_box(bbox2, threshold=0.5)  # type: ignore


def get_ordered_text(objects: List[Token]) -> List[Token]:
    result = []

    def sort_text(tokens: List[Token]) -> None:
        tokens = stitch_bd_bboxes_into_lines_as_boxes(
            tokens, min_y_overlap_ratio=0.6
        )

        for token in tokens:
            if not token.children:
                result.append(token)
            else:
                sort_text(token.children)

    sort_text(objects)
    return result

#TODO: move to my code
import json
def draw_bboxes2(
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
    (w,h) = draw_bboxes2(lines, image_path,save_path, "lines")
#        save_to_file(lines, "lines")

    gaps = find_gap(lines)
    if verbose:
        draw_bboxes2(gaps, image_path, save_path,"gaps")
    #    save_to_file(gaps, "gaps")

    paragraphs = find_paragraphs(gaps, h, w)
    if verbose:
        draw_bboxes2(paragraphs, image_path,save_path, "paragraphs")
     #   save_to_file(paragraphs,  "paragraphs")

    nested_paragraphs = get_nested_paragraphs(tokens, paragraphs)
    if verbose:
        draw_bboxes2(nested_paragraphs, image_path,save_path, "nested_paragraphs")
    #    save_to_file(nested_paragraphs, "nested_paragraphs")

    result = get_ordered_text(nested_paragraphs)
    if verbose:
        draw_bboxes2(result, image_path,save_path, "result")
     #   save_to_file(result,  "result")

    return import_to_text_tokens(lines)
