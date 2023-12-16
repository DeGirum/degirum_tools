#
# math_support.py: miscellaneous math/geometry functions
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#

# MIT License
#
# Copyright (c) 2022 Roboflow
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from enum import Enum
from typing import Union


def area(box: np.ndarray) -> np.ndarray:
    """
    Computes bbox(es) area: is vectorized.

    Parameters
    ----------
    box : np.array
        Box(es) in format (x0, y0, x1, y1)

    Returns
    -------
    np.array
        area(s)
    """
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])


def intersection(boxA: np.ndarray, boxB: np.ndarray) -> Union[float, np.ndarray]:
    """
    Compute area of intersection of two boxes

    Parameters
    ----------
    boxA : np.array
        First box
    boxB : np.array
        Second box

    Returns
    -------
    float64
        Area of intersection
    """
    xA = np.fmax(boxA[..., 0], boxB[..., 0])
    xB = np.fmin(boxA[..., 2], boxB[..., 2])
    dx = np.fmax(xB - xA, 0)

    yA = np.fmax(boxA[..., 1], boxB[..., 1])
    yB = np.fmin(boxA[..., 3], boxB[..., 3])
    dy = np.fmax(yB - yA, 0)

    # compute the area of intersection rectangle
    return dx * dy


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray): The bounding box coordinates in (x, y, width, height) format.
    """
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes -
        `boxes_true` and `boxes_detection`. Both sets
        of boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.

    Args:
        boxes_true (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        boxes_detection (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `boxes_true` and `boxes_detection`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_true[:, None] + area_detection - area_inter)


class AnchorPoint(Enum):
    """Position of a point of interest within the bounding box"""

    CENTER = 1
    CENTER_LEFT = 2
    CENTER_RIGHT = 3
    TOP_CENTER = 4
    TOP_LEFT = 5
    TOP_RIGHT = 6
    BOTTOM_LEFT = 7
    BOTTOM_CENTER = 8
    BOTTOM_RIGHT = 9


def get_anchor_coordinates(xyxy: np.ndarray, anchor: AnchorPoint) -> np.ndarray:
    """
    Calculates and returns the coordinates of a specific anchor point
    within the bounding boxes defined by the `xyxy` attribute. The anchor
    point can be any of the predefined positions,
    such as `AnchorPoint.CENTER`, `AnchorPoint.CENTER_LEFT`, `AnchorPoint.BOTTOM_RIGHT`, etc.

    Args:
        xyxy (nd.array): An array of shape `(n, 4)` of bounding box coordinates,
            where `n` is the number of bounding boxes.
        anchor (str): An string specifying the position of the anchor point
            within the bounding box.

    Returns:
        np.ndarray: An array of shape `(n, 2)`, where `n` is the number of bounding
            boxes. Each row contains the `[x, y]` coordinates of the specified
            anchor point for the corresponding bounding box.

    Raises:
        ValueError: If the provided `anchor` is not supported.
    """
    if anchor == AnchorPoint.CENTER:
        return np.array(
            [
                (xyxy[:, 0] + xyxy[:, 2]) / 2,
                (xyxy[:, 1] + xyxy[:, 3]) / 2,
            ]
        ).transpose()
    elif anchor == AnchorPoint.CENTER_LEFT:
        return np.array(
            [
                xyxy[:, 0],
                (xyxy[:, 1] + xyxy[:, 3]) / 2,
            ]
        ).transpose()
    elif anchor == AnchorPoint.CENTER_RIGHT:
        return np.array(
            [
                xyxy[:, 2],
                (xyxy[:, 1] + xyxy[:, 3]) / 2,
            ]
        ).transpose()
    elif anchor == AnchorPoint.BOTTOM_CENTER:
        return np.array([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]]).transpose()
    elif anchor == AnchorPoint.BOTTOM_LEFT:
        return np.array([xyxy[:, 0], xyxy[:, 3]]).transpose()
    elif anchor == AnchorPoint.BOTTOM_RIGHT:
        return np.array([xyxy[:, 2], xyxy[:, 3]]).transpose()
    elif anchor == AnchorPoint.TOP_CENTER:
        return np.array([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 1]]).transpose()
    elif anchor == AnchorPoint.TOP_LEFT:
        return np.array([xyxy[:, 0], xyxy[:, 1]]).transpose()
    elif anchor == AnchorPoint.TOP_RIGHT:
        return np.array([xyxy[:, 2], xyxy[:, 1]]).transpose()

    raise ValueError(f"{anchor} is not supported.")


def intersect(a, b, c, d) -> bool:
    """Check intersection of two lines

    Args:
        a (tuple): starting point of the first line
        b (tuple): ending point of the first line
        c (tuple): starting point of the second line
        d (tuple): ending point of the second line

    Returns:
        bool: True if lines intersect, False otherwise
    """

    s = (a[0] - b[0]) * (c[1] - a[1]) - (a[1] - b[1]) * (c[0] - a[0])
    t = (a[0] - b[0]) * (d[1] - a[1]) - (a[1] - b[1]) * (d[0] - a[0])
    if np.sign(s) == np.sign(t):
        return False
    s = (c[0] - d[0]) * (a[1] - c[1]) - (c[1] - d[1]) * (a[0] - c[0])
    t = (c[0] - d[0]) * (b[1] - c[1]) - (c[1] - d[1]) * (b[0] - c[0])
    if np.sign(s) == np.sign(t):
        return False
    return True
