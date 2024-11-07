#
# crop_extent.py: bbox crop extent support
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements enum and function for bbox crop extension
#

from enum import Enum
from typing import Sequence
import numpy as np


class CropExtentOptions(Enum):
    """
    Options for applying extending crop to the input image.
    """

    ASPECT_RATIO_NO_ADJUSTMENT = 1
    """
    Each dimension of the bounding box is extended by 'crop_extent' parameter
    """

    ASPECT_RATIO_ADJUSTMENT_BY_LONG_SIDE = 2
    """
    The longer dimension of the bounding box is extended by 'crop_extent' parameter, and the other side is
    calculated as new_bbox_h * aspect_ratio (if the longer dimension is the height of the bounding box)
    or new_bbox_w / aspect_ratio (if the longer dimension is the width of the bounding box),
    aspect_ratio is defined as the ratio of model's input width to model's input height
    """

    ASPECT_RATIO_ADJUSTMENT_BY_AREA = 3
    """
    The bounding box is extended by an area factor of ('crop_extent' * 0.01 + 1) ^ 2; the new height
    is determined from the desired area and aspect ratio, and is adjusted to be at least as long as
    the original bounding box height; the new width is calculated as new_bbox_h * aspect_ratio, where
    aspect_ratio is defined as the ratio of model's input width to model's input height,
    and then adjusted to be at least as long as the original bounding box width; the new
    height is then adjusted as new_bbox_w / aspect_ratio
    """


def extend_bbox(
    bbox: Sequence,
    crop_extent_option: CropExtentOptions,
    crop_extent: float,
    aspect_ratio: float,
    image_size: Sequence,
) -> list:
    """
    Inflate bbox coordinates to crop extent according to chosen crop extent approach
    and adjust to image size

    Args:
        bbox: bbox coordinates in [x1, y1, x2, y2] format
        crop_extent_option: method of applying extending crop
        crop_extent: extent of cropping in percent of bbox size
        aspect_ratio: aspect ratio of the model
        image_size: image size in (width, height) format

    Returns:
        bbox coordinates adjusted to crop extent and image size
    """
    bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if crop_extent_option == CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT:
        if crop_extent == 0.0:
            dx = dy = 0
        else:
            scale = crop_extent * 0.01 * 0.5
            dx = bbox_w * scale
            dy = bbox_h * scale

    elif crop_extent_option == CropExtentOptions.ASPECT_RATIO_ADJUSTMENT_BY_LONG_SIDE:
        scale = crop_extent * 0.01 + 1.0
        if bbox_h > bbox_w:
            new_bbox_h = bbox_h * scale
            new_bbox_w = new_bbox_h * aspect_ratio
        else:
            new_bbox_w = bbox_w * scale
            new_bbox_h = new_bbox_w / aspect_ratio
        dx = (new_bbox_w - bbox_w) / 2
        dy = (new_bbox_h - bbox_h) / 2

    elif crop_extent_option == CropExtentOptions.ASPECT_RATIO_ADJUSTMENT_BY_AREA:
        expansion_factor = np.power(crop_extent * 0.01 + 1, 2)
        new_bbox_h = np.sqrt(bbox_w * bbox_h * expansion_factor / aspect_ratio)
        new_bbox_h = max(new_bbox_h, bbox_h)
        new_bbox_w = new_bbox_h * aspect_ratio
        new_bbox_w = max(new_bbox_w, bbox_w)
        new_bbox_h = new_bbox_w / aspect_ratio
        dx = (new_bbox_w - bbox_w) / 2
        dy = (new_bbox_h - bbox_h) / 2

    maxval = [image_size[0], image_size[1], image_size[0], image_size[1]]
    adjust = [-dx, -dy, dx, dy]
    return [min(maxval[i], max(0, round(bbox[i] + adjust[i]))) for i in range(4)]
