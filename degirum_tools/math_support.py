#
# math_support.py: miscellaneous math/geometry functions
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#

import numpy as np
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
