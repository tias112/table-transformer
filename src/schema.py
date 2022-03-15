"""Contain model for saving page data as JSON"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pdfplumber.page import Page as PlumberPage
from PIL.Image import Image as PImage
from pydantic import BaseModel, Field, validator

from src.utils.logger import get_logger

logger = get_logger(__name__)

GEOM_OBJECT = TypeVar("GEOM_OBJECT", bound="GeomObject")
BBOX = Tuple[float, float, float, float]
POINTS = Tuple[Tuple[float, ...], Tuple[float, ...]]


class GeomObject(BaseModel):
    type: str = "bbox"
    bbox: BBOX
    text: Optional[str] = None
    segmentation: Optional[POINTS] = None

    @classmethod
    def parse_plumber_obj(
        cls: Type[GEOM_OBJECT], plumber_obj: Dict[str, Any], **kwargs: Any
    ) -> GEOM_OBJECT:
        return cls(
            **kwargs,
            bbox=(
                float(plumber_obj["x0"]),
                float(plumber_obj["top"]),
                float(plumber_obj["x1"]),
                float(plumber_obj["bottom"]),
            )
        )


class Line(GeomObject):
    """Describe line"""

    def __init__(self, *, bbox: BBOX, type: str = "line") -> None:
        super().__init__(type=type, bbox=bbox)


class Rectangle(GeomObject):
    """Describe rectangle"""

    def __init__(
        self, *, bbox: BBOX, type: Optional[str] = "rectangle"
    ) -> None:
        super().__init__(type=type, bbox=bbox)


class Image(GeomObject):
    """Describe image bbox"""

    def __init__(self, *, bbox: BBOX, type: Optional[str] = "image") -> None:
        super().__init__(type=type, bbox=bbox)


# pylint: disable=no-self-argument
# pylint: disable=bad-super-call
class ImageWithData(GeomObject):
    """Describe image for internal purposes"""

    image: PImage

    def __init__(
        self, *, bbox: BBOX, image: PImage, type: str = "image_with_data"
    ):
        super().__init__(type=type, bbox=bbox, image=image)

    @validator("image")
    def check_image_size(cls, image: PImage) -> PImage:
        if not image.size[0] or not image.size[1]:
            raise ValueError("Zero image dimension")
        return image

    class Config:
        arbitrary_types_allowed = True


class Curve(GeomObject):
    """Describe curve"""

    def __init__(
        self, *, bbox: BBOX, segmentation: POINTS, type: str = "curve"
    ) -> None:
        super().__init__(type=type, bbox=bbox, segmentation=segmentation)

    @staticmethod
    def _get_points(plumber_obj: Dict[str, Any]) -> POINTS:
        return (
            tuple(float(x) for x, _ in plumber_obj["points"]),
            tuple(float(y) for _, y in plumber_obj["points"]),
        )

    @classmethod
    def parse_plumber(cls, plumber_obj: Dict[str, Any]) -> Curve:
        return super().parse_plumber_obj(
            plumber_obj, segmentation=cls._get_points(plumber_obj)
        )


class TextToken(GeomObject):
    """Describe text field"""

    def __init__(
        self,
        *,
        bbox: BBOX,
        text: str,
        type: str = "text",
        children: Optional[List[TextToken]] = None
    ) -> None:
        super().__init__(type=type, bbox=bbox, text=text, children=children)

    @classmethod
    def parse_plumber(cls, plumber_obj: Dict[str, Any]) -> TextToken:
        return super().parse_plumber_obj(plumber_obj, text=plumber_obj["text"])


TextToken.update_forward_refs()


class PageSize(BaseModel):
    width: float
    height: float


class Page(BaseModel):
    """Describe page"""

    size: PageSize
    page_num: int
    objs: List[Union[Line, Rectangle, TextToken, Image, Curve]]

    @classmethod
    def parse_plumber(cls, plumber_page: PlumberPage) -> Page:
        return cls(
            size=PageSize(
                width=plumber_page.width, height=plumber_page.height
            ),
            page_num=plumber_page.page_number,
            objs=[],
        )


class RequestObjectTypes(str, Enum):
    """Possible objects to extract."""

    TEXT = "text"
    LINE = "line"
    RECTANGLE = "rectangle"
    CURVE = "curve"
    IMAGE = "image"


class PlumberObjectTypes(str, Enum):
    CHAR = "char"
    LINE = "line"
    RECTANGLE = "rect"
    CURVE = "curve"
    IMAGE = "image"


class TextTokenType(str, Enum):
    CHAR = "char"
    WORD = "word"


class PreprocessResponse(BaseModel):
    file: Path = Field(
        title="Path to file, that was preprocessed", example="path/to/file.pdf"
    )


class PreprocessArgs(BaseModel):
    object_types: Tuple[RequestObjectTypes, ...] = Field(
        default=(RequestObjectTypes.TEXT,),
        title="Processing object types",
        description="Texts are all words from page, both vector and raster. "
        "Rectangle and image contain bounding boxes only. "
        "Line are horizontal and vertical lines. "
        "Curve are 2 points non vertical and horizontal line or with more points"
        "Curve contains bounding box and all points. "
        "Point are stored in two arrays, X in the first and Y in the other.",
        example=("text", "curve", "line", "rectangle", "image"),
    )

    language: Tuple[str, ...] = Field(
        default=("eng",),
        title="Languages for OCR",
        description="See shortcut for languages here: "
        "https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html",
        example=("eng", "rus"),
    )

    text_token_type: TextTokenType = Field(
        default=TextTokenType.WORD,
        title="Which tokens you want to receive",
        example=TextTokenType.WORD,
    )


# pylint: disable=E0213
class PreprocessRequest(BaseModel):
    """Request to extract pdf objects.
    https://kb.epam.com/display/EPMUII/Inference+format"""

    file: Path = Field(
        title="Path to input PDF file without bucket", example="path/to/file"
    )
    bucket: str = Field(title="Bucket in the MinIO", example="test")
    pages: Optional[Tuple[int, ...]] = Field(
        title="Pages to process. If null then process all pages",
        description="Pages begins with 1.",
        example=(1, 2, 3),
    )
    output_bucket: Optional[str] = Field(
        title="Output bucket",
        description="'output_bucket' is optional. If it is not specified, "
        "then 'bucket' should be used for both 'input_path', 'file' and "
        "'output_path'. ",
        example="tenant_name",
    )
    output_path: Optional[Path] = Field(
        title="Path to save JSON",
        description='Default path is a folder "ocr" nearby PDF.',
        example="path/to/output",
    )

    args: PreprocessArgs = Field(
        default=PreprocessArgs(),
        title="Object_types, languages and token type ",
    )

    @validator("file")
    def path_extension_must_be_pdf(cls, file: Path) -> Path:
        if file.suffix != ".pdf":
            raise ValueError("File extension must be pdf")
        return file

    @validator("output_path", always=True)
    def output_path_must_exist_for_output_path(
        cls,
        output_path: str,
        values: Dict[str, Any],
    ) -> str:
        if output_path is None and values["output_bucket"]:
            raise ValueError(
                "Output bucket needed for output path only! "
                "Set both `output_path` and `output_bucket` or `output_path` only."
            )
        return output_path

    @validator("output_path", always=True)
    def set_default_output(
        cls,
        output_path: Optional[Path],
        values: Dict[str, Any],
    ) -> Optional[Path]:
        # As validator's flag "always" is True it can suppress for
        # `path_extension_must_be_pdf` validator error, so check
        # that `file` validator passed successfully
        file: Path = values.get("file")
        if file is None:
            return None
        return output_path if output_path else file.parent / "ocr/"


class LocalPreprocessRequest(BaseModel):
    """Request to work with volume"""

    file: Path = Field(example="path/to/file")
    pages: Optional[Tuple[int, ...]] = Field(
        title="Pages to process",
        example=[1, 2, 3],
    )
    output_path: Optional[Path] = Field(
        title="Path to save JSON",
        example="path/to/output",
    )
    args: PreprocessArgs = Field(default=PreprocessArgs())
