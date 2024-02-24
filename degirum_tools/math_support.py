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
from typing import Union, Sequence


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


class NmsBoxSelectionPolicy(Enum):
    """Bounding box selection policy for non-maximum suppression"""

    MOST_PROBABLE = 1  # traditional NMS: select box with highest probability
    LARGEST_AREA = 2  # select box with largest area
    AVERAGE = 3  # average all boxes in a cluster
    MERGE = 4  # merge all boxes in a cluster


def _nms_custom(
    bboxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    use_iou: bool,
    box_select: NmsBoxSelectionPolicy,
    max_wh: int,
):
    """
    Perform non-maximum suppression on a set of detections.

    Args:
        bboxes (np.ndarray): 2D `np.ndarray` representing bboxes in (x1, y1, x2, y2) format
        scores (np.ndarray): 1D `np.ndarray` representing confidence scores of detections
        classes (np.ndarray): 1D `np.ndarray` representing class IDs
        iou_threshold (float): IoU/IoS threshold used for detecting overlapping boxes.
        use_iou (bool): If True, IoU is used for detecting overlapping boxes. Otherwise, IoS is used.
        box_select (NmsBoxSelectionPolicy): bounding box selection policy
        max_wh (int): maximum width/height of an image. Used for category separation.

    Returns:
        np.ndarray: An array of indices of detections that have survived non-maximum suppression.
        If `box_select` option dictates box coordinates update, the survived object box coordinates are modified in-place

    """

    # adjust bboxes by class offset so bboxes of different classes would never overlap
    class_offsets = classes * float(max_wh)
    bboxes[:, 0:4] += class_offsets[:, None]

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()  # sort boxes by IoUs in ascending order

    keep = []
    while order.size > 0:
        i = order[-1]  # pick maximum IoU box
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maximum height

        inter = w * h
        if use_iou:
            overlap = inter / (areas[i] + areas[order] - inter)
        else:  # use IoS
            overlap = inter / np.minimum(areas[i], areas[order])

        if box_select == NmsBoxSelectionPolicy.MOST_PROBABLE:
            pass
        elif box_select == NmsBoxSelectionPolicy.LARGEST_AREA:
            sel = order[overlap > iou_threshold]
            bboxes[i] = bboxes[sel[np.argmax(areas[sel])]] - class_offsets[i]
        elif box_select == NmsBoxSelectionPolicy.AVERAGE:
            sel = order[overlap > iou_threshold]
            bboxes[i] = (
                np.average(bboxes[sel], axis=0, weights=scores[sel]) - class_offsets[i]
            )
        elif box_select == NmsBoxSelectionPolicy.MERGE:
            sel = order[overlap > iou_threshold]
            enclosing_rect = np.array(
                [np.min(x1[sel]), np.min(y1[sel]), np.max(x2[sel]), np.max(y2[sel])]
            )
            bboxes[i] = enclosing_rect - class_offsets[i]

        order = order[overlap <= iou_threshold]

    return keep


def nms(
    detections,
    iou_threshold: float = 0.3,
    use_iou: bool = True,
    box_select: NmsBoxSelectionPolicy = NmsBoxSelectionPolicy.MOST_PROBABLE,
    max_wh: int = 10000,
):
    """
    Perform non-maximum suppression on a set of detections.

    Args:
        detections (InferenceResults): PySDK inference result object
        iou_threshold (float): IoU/IoS threshold used for detecting overlapping boxes.
        use_iou (bool): If True, IoU is used for detecting overlapping boxes. Otherwise, IoS is used.
        box_select (NmsBoxSelectionPolicy): bounding box selection policy
        max_wh (int): Maximum width/height of an image. Used for category separation.

    `detections` will be updated with objects, which survived non-maximum suppression.
    If `box_select` option dictates box coordinates update, the survived object box coordinates are modified in-place

    """

    result_list = detections._inference_results
    n_results = len(result_list)
    bboxes = np.empty((n_results, 4), dtype=np.float64)
    scores = np.empty(n_results, dtype=np.float64)
    classes = np.empty(n_results, dtype=np.int32)
    unique_ids: dict = {}

    try:
        for i, r in enumerate(result_list):
            bboxes[i] = r["bbox"]
            scores[i] = r["score"]
            classes[i] = unique_ids.setdefault(r["label"], len(unique_ids))
    except KeyError as e:
        raise Exception(
            "Detections must be a list of dicts with 'bbox', 'score' and 'category_id' keys"
        ) from e

    if use_iou and box_select == NmsBoxSelectionPolicy.MOST_PROBABLE:
        import cv2

        # use fast OpenCV implementation when possible
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  # TLBR to TLWH
        keep = cv2.dnn.NMSBoxesBatched(bboxes, scores, classes, 0, iou_threshold)  # type: ignore[arg-type]
    else:
        keep = _nms_custom(
            bboxes, scores, classes, iou_threshold, use_iou, box_select, max_wh
        )

        if box_select != NmsBoxSelectionPolicy.MOST_PROBABLE:
            for i in keep:
                result_list[i]["bbox"] = bboxes[i].tolist()

    detections._inference_results = [result_list[i] for i in keep]


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


def _generate_tiles_core(
    tile_size: np.ndarray, tiles_cnt: np.ndarray, image_size: np.ndarray
):
    overlaps = 1.0 - (image_size - tile_size) / (
        np.maximum(1, (tiles_cnt - 1)) * tile_size
    )

    tile_inds = np.array(
        np.meshgrid(np.arange(tiles_cnt[0]), np.arange(tiles_cnt[1]), indexing="ij")
    ).T.reshape(-1, 2)
    top_left = np.floor(tile_inds * tile_size * (1 - overlaps))
    flat = np.column_stack((top_left, top_left + tile_size)).astype(np.int32)
    return np.reshape(flat, (-1, tiles_cnt[0], 4))


def _tiles_count(
    tile_size: np.ndarray, image_size: np.ndarray, min_overlap_percent: np.ndarray
):
    return 1 + np.ceil(
        (image_size - tile_size) / (tile_size * (1.0 - 0.01 * min_overlap_percent))
    ).astype(np.int32)


def generate_tiles_fixed_size(
    tile_size: Union[np.ndarray, Sequence],
    image_size: Union[np.ndarray, Sequence],
    min_overlap_percent: Union[np.ndarray, Sequence],
):
    """
    Generate a set of rectangular boxes (tiles) of given fixed size
    covering given rectangular area (image) with given overlap.

    Args:
        tile_size: desired tile size (width,height)
        image_size: image size (width,height)
        min_overlap_percent: minimum overlaps between tiles in percent (x_overlap,y_overlap)

    Returns:
        np.ndarray: array of tile coordinates in format (x0, y0, x1, y1)
    """

    if not isinstance(tile_size, np.ndarray):
        tile_size = np.array(tile_size)
    if not isinstance(image_size, np.ndarray):
        image_size = np.array(image_size)
    if not isinstance(min_overlap_percent, np.ndarray):
        min_overlap_percent = np.array(min_overlap_percent)
    if min_overlap_percent.size < 2:
        min_overlap_percent = np.repeat(min_overlap_percent, 2)

    return _generate_tiles_core(
        tile_size, _tiles_count(tile_size, image_size, min_overlap_percent), image_size
    )


def generate_tiles_fixed_ratio(
    tile_aspect_ratio: Union[float, np.ndarray, Sequence],
    grid_size: Union[np.ndarray, Sequence],
    image_size: Union[np.ndarray, Sequence],
    min_overlap_percent: Union[np.ndarray, Sequence, float],
):
    """
    Generate a set of rectangular boxes (tiles) of given fixed size
    covering given rectangular area (image) with given overlap.

    Args:
        tile_aspect_ratio: desired tile aspect ratio,
            it can be number width/height or 2-element sequence [width,height]
        grid_size: desired number of tiles in each direction (column,rows);
            only one of the values (which is non-zero) is used to compute the other one
        image_size: image size (width,height)
        min_overlap_percent: minimum overlaps between tiles in percent (x_overlap,y_overlap)

    Returns:
        np.ndarray: array of tile coordinates in format (x0, y0, x1, y1)
    """

    if not isinstance(grid_size, np.ndarray):
        grid_size = np.array(grid_size)
    if not isinstance(image_size, np.ndarray):
        image_size = np.array(image_size)
    if not isinstance(min_overlap_percent, np.ndarray):
        min_overlap_percent = np.array(min_overlap_percent)
    if min_overlap_percent.size < 2:
        min_overlap_percent = np.repeat(min_overlap_percent, 2)

    if not isinstance(tile_aspect_ratio, float):
        tile_aspect_ratio = tile_aspect_ratio[0] / tile_aspect_ratio[1]

    def get_tile_dimension(dim):
        return np.round(
            image_size[dim]
            / (1 + (grid_size[dim] - 1) * (1.0 - 0.01 * min_overlap_percent[dim]))
        )

    tile_size = np.zeros(2)
    if grid_size[0] != 0:
        tile_size[0] = get_tile_dimension(0)
        tile_size[1] = np.round(tile_size[0] / tile_aspect_ratio)
        grid_size[1:] = _tiles_count(
            tile_size[1:], image_size[1:], min_overlap_percent[1:]
        )

    elif grid_size[1] != 0:
        tile_size[1] = get_tile_dimension(1)
        tile_size[0] = np.round(tile_size[1] * tile_aspect_ratio)
        grid_size[:1] = _tiles_count(
            tile_size[:1], image_size[:1], min_overlap_percent[:1]
        )
    else:
        raise ValueError("At least one of grid_size values must be non-zero")

    return _generate_tiles_core(tile_size, grid_size, image_size)


class FIRFilterLP:
    """
    FIR low-pass filter class
    """

    def __init__(self, normalized_cutoff: float, taps_cnt: int, dimension: int = 1):
        """
        Constructor

        Args:
            normalized_cutoff: normalized cutoff frequency
            taps_cnt: number of taps
            dimension: dimension of the input signal
        """
        import scipy.signal

        self._fir_coeffs = scipy.signal.firwin(
            taps_cnt, normalized_cutoff, window="blackmanharris"
        )
        self._buffer = np.array([])
        self._taps_cnt = taps_cnt
        self._initialized = False
        self._result = np.zeros(dimension)

    def update(self, sample: Union[float, np.ndarray]) -> np.ndarray:
        """
        Apply filter to the input signal

        Args:
            sample: input signal sample if `dimension` is 1
                or array of samples if `dimension` is greater than 1

        Returns:
            filtered signal
        """
        if not self._initialized:
            self._buffer = np.column_stack((sample,) * self._taps_cnt)
            self._initialized = True
        self._buffer = np.column_stack((self._buffer[:, 1:], sample))
        self._result = np.dot(self._buffer, self._fir_coeffs)
        return self._result

    def get(self):
        """
        Get last filtered result
        """
        return self._result

    def __call__(self, sample: Union[float, np.ndarray]) -> np.ndarray:
        """Synonym for update() method"""
        return self.update(sample)
