#
# image_tools.py: image processing functions
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements miscellaneous functions for image processing
#

import cv2, numpy as np
from typing import Optional


def luminance(color: tuple) -> float:
    """Calculate luminance from RGB color

    Args:
        color (tuple): RGB color
    """
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def crop(img: np.ndarray, bbox: list):
    """Crop and return OpenCV image to given bbox

    Args:
        img: OpenCV image
        bbox (list): bounding box in format [x0, y0, x1, y1]
    """
    return img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]


def detect_motion(base_img: Optional[np.ndarray], img: np.ndarray) -> tuple:
    """
    Detect areas with motion on given image in respect to base image.

    Args:
        base_img: base image; pass None to use `img` as base image
        img: image to detect motion on

    Returns:
        A tuple of motion image and updated base image.
        Motion image is black image with white pixels where motion is detected.

    """
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
