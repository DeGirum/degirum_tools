#
# image_tools.py: image processing functions
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements miscellaneous functions for image processing
#

import cv2, numpy as np
import PIL.Image
from typing import Optional, Union


ImageType = Union[np.ndarray, PIL.Image.Image]
PILImage = PIL.Image.Image


def luminance(color: tuple) -> float:
    """Calculate luminance from RGB color

    Args:
        color (tuple): RGB color
    """
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def image_size(img: ImageType) -> tuple:
    """Get image size

    Args:
        img: OpenCV or PIL image
    """
    if isinstance(img, PILImage):
        return img.size
    else:
        return (img.shape[1], img.shape[0])


def crop_image(img: ImageType, bbox: list):
    """Crop and return PIL/OpenCV image to given bbox

    Args:
        img: image
        bbox: bounding box in format [x0, y0, x1, y1]
    """
    if isinstance(img, PILImage):
        return img.crop(tuple(bbox))
    else:
        return img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]


def resize_image(img: ImageType, w: int, h: int):
    """Resize and return PIL/OpenCV image to given size

    Args:
        img: image
        w, h: new width and height
    """
    if isinstance(img, PILImage):
        return img.resize((w, h))
    else:
        return cv2.resize(img, (w, h))


def paste_image(img: ImageType, crop: ImageType, bbox: list):
    """Paste given crop image into bigger image into specified bbox

    Args:
        img: image to paste into
        crop: crop image to use for pasting
        bbox: bounding box to paste into in format [x0, y0, x1, y1]
    """
    if isinstance(img, PILImage) and isinstance(crop, PILImage):
        img.paste(crop, (bbox[0], bbox[1]))
    elif isinstance(img, np.ndarray) and isinstance(crop, np.ndarray):
        img[round(bbox[1]) : round(bbox[3]), round(bbox[0]) : round(bbox[2])] = crop
    else:
        raise ValueError(
            f"paste_image: image types {type(img)} and {type(crop)} do not match"
        )


def to_opencv(img: ImageType) -> np.ndarray:
    """Convert any image to OpenCV image"""

    if isinstance(img, PILImage):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def detect_motion(base_img: Optional[np.ndarray], img: ImageType) -> tuple:
    """
    Detect areas with motion on given image in respect to base image.

    Args:
        base_img: base image; pass None to use `img` as base image
        img: image to detect motion on

    Returns:
        A tuple of motion image and updated base image.
        Motion image is black image with white pixels where motion is detected.

    """

    if isinstance(img, PILImage):
        cur_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    else:
        cur_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cur_img = cv2.GaussianBlur(src=cur_img, ksize=(5, 5), sigmaX=0)

    if base_img is None:
        base_img = cur_img
        return None, base_img

    diff = cv2.absdiff(base_img, cur_img)
    base_img = cur_img

    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.array([]))

    return thresh, base_img
