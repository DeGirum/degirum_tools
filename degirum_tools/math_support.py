#
# math_support.py: mathematical utilities for array operations
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements mathematical functions for array operations
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

"""
Math Support Module Overview
===========================

This module provides mathematical utilities for geometric operations and signal processing.
It includes functions for bounding box manipulation, non-maximum suppression, image tiling,
and a lightweight FIR low-pass filter implementation.

Key Features:
    - **Bounding Box Operations**: Area calculation, IoU computation, coordinate conversions
    - **Non-maximum Suppression**: Multiple selection policies for detection filtering
    - **Edge Box Fusion**: Handling overlapping detections with configurable thresholds
    - **Image Tiling**: Utilities for fixed size or aspect ratio tiling
    - **FIR Filter**: Lightweight low-pass filter implementation for signal smoothing

Typical Usage:
    1. Import required functions from the module
    2. Use bounding box operations for detection processing
    3. Apply NMS or box fusion for post-processing detections
    4. Generate image tiles for processing large images
    5. Use FIRFilterLP for smoothing numeric sequences

Integration Notes:
    - Built on NumPy for efficient array operations
    - Compatible with various bounding box formats (xyxy, xywh)
    - Provides consistent results across platforms
    - Includes comprehensive error handling

Key Functions:
    - `area()`: Calculate bounding box areas
    - `intersection()`: Compute intersection area of bounding boxes
    - `nms()`: Apply non-maximum suppression
    - `edge_box_fusion()`: Fuse overlapping edge detections
    - `generate_tiles_fixed_size()`: Generate overlapping image tiles

Configuration Options:
    - NMS selection policies
    - Box fusion thresholds
    - Tile overlap parameters
    - Filter coefficients
"""

import numpy as np
from enum import Enum
from typing import Any, Dict, List, Union, Sequence, Tuple


def area(box: np.ndarray) -> np.ndarray:
    """Compute bounding box areas.

    Args:
        box (np.ndarray): Single box ``(x1, y1, x2, y2)`` or an array of such
            boxes with shape ``(N, 4)``.

    Returns:
        Area of each input box.
    """
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])


def intersection(boxA: np.ndarray, boxB: np.ndarray) -> Union[float, np.ndarray]:
    """Compute intersection area of bounding boxes.

    Args:
        boxA (np.ndarray): First set of boxes ``(x1, y1, x2, y2)``.
        boxB (np.ndarray): Second set of boxes ``(x1, y1, x2, y2)``.

    Returns:
        Intersection area for each pair of boxes.
    """
    xA = np.fmax(boxA[..., 0], boxB[..., 0])
    xB = np.fmin(boxA[..., 2], boxB[..., 2])
    dx = np.fmax(xB - xA, 0)

    yA = np.fmax(boxA[..., 1], boxB[..., 1])
    yB = np.fmin(boxA[..., 3], boxB[..., 3])
    dy = np.fmax(yB - yA, 0)

    return dx * dy


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    """Convert ``(x1, y1, x2, y2)`` boxes to ``(x, y, w, h)`` format."""
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def tlbr2allcorners(x: np.ndarray) -> np.ndarray:
    """Convert ``(x1, y1, x2, y2)`` boxes to all four corner points."""
    y = np.empty((x.shape[0], 8))
    y[..., 0:2] = x[..., 0:2]  # top left (x, y)
    y[:, [2, 3]] = x[:, [2, 1]]  # top right (x, y)
    y[..., 4:6] = x[..., 2:4]  # bottom right (x, y)
    y[:, [6, 7]] = x[:, [0, 3]]  # bottom right (x, y)
    return y


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert ``(x, y, w, h)`` boxes to ``(x1, y1, x2, y2)`` format."""
    y = np.copy(x)
    y[..., 0:2] = x[..., 0:2] - x[..., 2:4] / 2  # top left (x, y)
    y[..., 2:4] = x[..., 0:2] + x[..., 2:4] / 2  # bottom right (x, y)
    return y


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes.

    Args:
        boxes_true (np.ndarray): Ground-truth boxes ``(N, 4)``.
        boxes_detection (np.ndarray): Detection boxes ``(M, 4)``.

    Returns:
        IoU matrix of shape ``(N, M)``.
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
    """Bounding box selection policy for non-maximum suppression.

    This enum defines different strategies for selecting which bounding box to keep
    when multiple overlapping boxes are detected. Each policy has different use cases
    and trade-offs.

    Attributes:
        MOST_PROBABLE (int): Traditional NMS approach that keeps the box with highest confidence score.
            Best for high-confidence detections where false positives are costly.
        LARGEST_AREA (int): Keeps the box with the largest area. Useful when larger detections
            are more likely to be correct or when object size is important.
        AVERAGE (int): Averages the coordinates of all overlapping boxes. Good for reducing
            jitter in tracking applications.
        MERGE (int): Merges all overlapping boxes into a single box. Useful when multiple
            detections of the same object are expected.
    """

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
    agnostic: bool,
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
    if not agnostic:
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
            if not agnostic:
                bboxes[i] = bboxes[sel[np.argmax(areas[sel])]] - class_offsets[i]
            else:
                bboxes[i] = bboxes[sel[np.argmax(areas[sel])]]
        elif box_select == NmsBoxSelectionPolicy.AVERAGE:
            sel = order[overlap > iou_threshold]
            if not agnostic:
                bboxes[i] = (
                    np.average(bboxes[sel], axis=0, weights=scores[sel])
                    - class_offsets[i]
                )
            else:
                bboxes[i] = np.average(bboxes[sel], axis=0, weights=scores[sel])
        elif box_select == NmsBoxSelectionPolicy.MERGE:
            sel = order[overlap > iou_threshold]
            enclosing_rect = np.array(
                [np.min(x1[sel]), np.min(y1[sel]), np.max(x2[sel]), np.max(y2[sel])]
            )
            if not agnostic:
                bboxes[i] = enclosing_rect - class_offsets[i]
            else:
                bboxes[i] = enclosing_rect

        order = order[overlap <= iou_threshold]

    return keep


def nms(
    detections,
    iou_threshold: float = 0.3,
    use_iou: bool = True,
    box_select: NmsBoxSelectionPolicy = NmsBoxSelectionPolicy.MOST_PROBABLE,
    max_wh: int = 10000,
    agnostic: bool = False,
):
    """Apply non-maximum suppression to detection results.

    Args:
        detections (InferenceResults): Iterable of detection dictionaries
            containing ``bbox``, ``score`` and ``class`` fields.
        iou_threshold (float, optional): IoU/IoS threshold. Defaults to ``0.3``.
        use_iou (bool, optional): If ``True`` use IoU, otherwise IoS.
        box_select (NmsBoxSelectionPolicy, optional): Box selection strategy.
        max_wh (int, optional): Maximum image dimension for class separation.
        agnostic (bool, optional): If ``True`` perform class-agnostic NMS.

    Returns:
        (None): ``detections`` is modified in place with the filtered results.
    """

    result_list = detections._inference_results
    n_results = len(result_list)
    bboxes = np.empty((n_results, 4), dtype=np.float64)
    scores = np.empty(n_results, dtype=np.float64)
    classes = np.empty(n_results, dtype=np.int32)
    unique_ids: dict = {}

    try:
        for i, r in enumerate(result_list):
            # Need this to benchmark blank image. Not sure if its necessary for a real image.
            if "bbox" not in r:
                continue
            if "score" not in r:
                continue
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
        if not agnostic:
            keep = cv2.dnn.NMSBoxesBatched(bboxes, scores, classes, 0, iou_threshold)  # type: ignore[arg-type]
        else:
            keep = cv2.dnn.NMSBoxes(bboxes, scores, 0, iou_threshold)  # type: ignore[arg-type]
    else:
        keep = _nms_custom(
            bboxes,
            scores,
            classes,
            iou_threshold,
            use_iou,
            box_select,
            max_wh,
            agnostic,
        )

        if box_select != NmsBoxSelectionPolicy.MOST_PROBABLE:
            for i in keep:
                result_list[i]["bbox"] = bboxes[i].tolist()

    detections._inference_results = [result_list[i] for i in keep]


def _prefilter_boxes(
    boxes: Sequence[Sequence],
    scores: Sequence[float],
    labels: Sequence[int],
    thr: float,
) -> Tuple[Dict[int, np.ndarray], List]:
    # dict with boxes stored by category id
    new_boxes: Dict[int, Any] = dict()
    skipped_boxes = []

    for j in range(len(boxes)):
        score = scores[j]

        label = int(labels[j])
        box_part = boxes[j]
        x1 = float(box_part[0])
        y1 = float(box_part[1])
        x2 = float(box_part[2])
        y2 = float(box_part[3])

        # [label, score, x1, y1, x2, y2]
        b = [label, float(score), x1, y1, x2, y2]

        if score < thr:
            skipped_boxes.append(b)
            continue

        if label not in new_boxes:
            new_boxes[label] = []

        new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return (new_boxes, skipped_boxes)


def _find_matching_box(
    boxes: Sequence[np.ndarray], new_box: np.ndarray, match_iou: float
) -> Tuple[int, Union[float, int]]:

    def _bb_1d_iou(boxes: np.ndarray, new_box: np.ndarray) -> np.ndarray:
        # Returns the larger of the x or y 1D-IoU if the boxes overlap.
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        # Mask out boxes with no overlap in one of the dimensions.
        inter_x, inter_y = np.maximum(xB - xA, 0), np.maximum(yB - yA, 0)
        mask = np.minimum(inter_x, inter_y) == 0
        inter_x[mask] = 0
        inter_y[mask] = 0

        w_a, h_a = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        w_b, h_b = new_box[2] - new_box[0], new_box[3] - new_box[1]

        iou_x, iou_y = inter_x / (w_a + w_b - inter_x), inter_y / (h_a + h_b - inter_y)

        return np.maximum(iou_x, iou_y)

    if len(boxes) == 0:
        return -1, match_iou

    # Create a mask for already matched boxes.
    matched_mask = np.full(len(boxes), False)
    masked_boxes = np.empty((0, 6))

    for i, box_set in enumerate(boxes):
        if len(box_set) >= 2:
            matched_mask[i] = True

        masked_boxes = np.vstack((masked_boxes, box_set[0]))

    ious = _bb_1d_iou(masked_boxes[:, 2:], new_box[2:])

    ious[masked_boxes[:, 0] != new_box[0]] = -1
    ious[matched_mask] = -1

    best_idx = int(np.argmax(ious))  # for type checker.
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def _fuse_boxes(box_sets: Sequence[np.ndarray]) -> np.ndarray:
    fused_boxes = []

    for box_set in box_sets:
        if len(box_set) == 1:
            fused_boxes.append(box_set[0])
        else:
            score = (box_set[0][1] + box_set[1][1]) / 2
            coord_max = np.amax(box_set, axis=0)
            coord_min = np.amin(box_set, axis=0)
            bbox = np.concatenate((coord_min[2:4], coord_max[4:6]))

            fused_box = np.zeros(6, dtype=np.float32)
            fused_box[0] = box_set[0][0]
            fused_box[1] = score
            fused_box[2:] = bbox
            fused_boxes.append(fused_box)

    return np.vstack(fused_boxes)


def edge_box_fusion(
    detections: Sequence[dict],
    iou_threshold: float = 0.55,
    skip_threshold: float = 0.0,
    destructive=True,
) -> List[dict]:
    """
    Perform box fusion on a set of edge detections. Edge detections are detections within a certain threshold of an edge.

    Args:
        detections (DetectionResults.results): A list of dictionaries that contain detection results as defined in PySDK.
        iou_threshold (float): 1D-IoU threshold used for selecting boxes for fusion.
        skip_threshold (float): Score threshold for which boxes to fuse.
        destructive (bool): Keep skipped boxes (underneath the `score_threshold`) in the results.

    Returns:
        (DetectionResult.results): A list of dictionaries that contain detection results as defined in PySDK.
            Boxes that are not fused are kept if destructive is False, otherwise they are discarded.
    """

    boxes_list = []
    scores_list = []
    labels_list = []

    if len(detections) == 0:
        return []

    for det in detections:
        boxes_list.append(det["wbf_info"])
        scores_list.append(det["score"])
        labels_list.append(det["category_id"])

    filtered_boxes, skipped_boxes = _prefilter_boxes(
        boxes_list, scores_list, labels_list, skip_threshold
    )

    if len(filtered_boxes) == 0:
        if not destructive:
            if len(skipped_boxes) > 0:
                results = []
                for box in skipped_boxes:
                    det = dict()
                    det["score"] = float(box[1])
                    det["bbox"] = box[2:6]
                    det["category_id"] = int(box[0])
                    results.append(det)

                return results

        return []

    overall_boxes = []

    # Only fuse same category id.
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes: List[np.ndarray] = []

        # Initial fusion, max two boxes in one fused box.
        for j in range(len(boxes)):
            index, _ = _find_matching_box(new_boxes, boxes[j], iou_threshold)

            if index != -1:
                new_boxes[index] = np.vstack((new_boxes[index], boxes[j]))
            else:
                new_boxes.append(np.expand_dims(boxes[j].copy(), axis=0))

        # Second fusion, max two boxes, fuses corner detections if all four were detected.
        boxes = _fuse_boxes(new_boxes)
        new_boxes = []
        for j in range(len(boxes)):
            index, _ = _find_matching_box(new_boxes, boxes[j], iou_threshold)

            if index != -1:
                new_boxes[index] = np.vstack((new_boxes[index], boxes[j]))
            else:
                new_boxes.append(np.expand_dims(boxes[j].copy(), axis=0))

        overall_boxes.append(_fuse_boxes(new_boxes))

    fused_boxes = np.vstack(overall_boxes)
    fused_boxes = fused_boxes[fused_boxes[:, 1].argsort()[::-1]]
    boxes = fused_boxes[:, 2:]
    scores = fused_boxes[:, 1]
    labels = fused_boxes[:, 0]

    results = []
    for i in range(boxes.shape[0]):
        det = dict()
        det["score"] = float(scores[i])
        det["bbox"] = boxes[i].tolist()
        det["category_id"] = int(labels[i])
        results.append(det)

    if not destructive:
        for box in skipped_boxes:
            det = dict()
            det["score"] = float(box[1])
            det["bbox"] = box[2:6]
            det["category_id"] = int(box[0])
            results.append(det)

    return results


class AnchorPoint(Enum):
    """Position of a point of interest within the bounding box."""

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
    """Get coordinates of an anchor point inside bounding boxes.

    Args:
        xyxy (np.ndarray): Array of boxes ``(N, 4)``.
        anchor (AnchorPoint): Desired anchor location.

    Returns:
        Array ``(N, 2)`` with ``[x, y]`` coordinates.

    Raises:
        ValueError: If ``anchor`` is unsupported.
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


def get_image_anchor_point(w: int, h: int, anchor: AnchorPoint) -> tuple:
    """Return coordinates of an anchor point inside an image."""
    return tuple(
        get_anchor_coordinates(np.array([[0, 0, w, h]]), anchor).astype(int)[0]
    )


def intersect(a, b, c, d) -> bool:
    """Return ``True`` if two line segments intersect."""

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
    """Generate overlapping tiles with fixed size.

    Args:
        tile_size (Union[np.ndarray, Sequence]): Tile size ``(w, h)``.
        image_size (Union[np.ndarray, Sequence]): Image size ``(w, h)``.
        min_overlap_percent (Union[np.ndarray, Sequence]): Minimum overlap in percent.

    Returns:
        (np.ndarray): Array of shape ``(N, M, 4)`` containing tile coordinates, where
            N is the number of rows in the tile grid,
            M is the number of columns in the tile grid,
            and 4 represents the coordinates ``(x1, y1, x2, y2)`` for each tile.
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
    """Generate overlapping tiles with fixed aspect ratio.

    Args:
        tile_aspect_ratio (Union[float, np.ndarray, Sequence]): Desired aspect ratio.
        grid_size (Union[np.ndarray, Sequence]): Grid size ``(x, y)``.
        image_size (Union[np.ndarray, Sequence]): Image size ``(w, h)``.
        min_overlap_percent (Union[np.ndarray, Sequence, float]): Minimum overlap percent.

    Returns:
        (np.ndarray): Array of shape ``(N, M, 4)`` containing tile coordinates, where
            N is the number of rows in the tile grid,
            M is the number of columns in the tile grid,
            and 4 represents the coordinates ``(x1, y1, x2, y2)`` for each tile.
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
    """Low-pass Finite Impulse Response (FIR) filter implementation.

    This class implements a low-pass FIR filter for smoothing signals. It uses convolution
    with designed filter coefficients (FIR kernel) with configurable cutoff frequency and number of taps.

    Attributes:
        normalized_cutoff (float): Normalized cutoff frequency (0 to 1).
        taps_cnt (int): Number of filter taps (order of the filter).
        dimension (int): Number of dimensions in the input signal.

    """

    def __init__(self, normalized_cutoff: float, taps_cnt: int, dimension: int = 1):
        """Initialize the FIR filter with specified parameters.

        Args:
            normalized_cutoff (float): Normalized cutoff frequency between 0 and 1.
                A value of 1 corresponds to the Nyquist frequency.
            taps_cnt (int): Number of filter taps (order of the filter). Higher values
                provide better frequency response but increase computational cost and delay.
            dimension (int, optional): Number of dimensions in the input signal.
                Defaults to 1 for scalar signals.

        Raises:
            ValueError: If normalized_cutoff is not between 0 and 1.
            ValueError: If taps_cnt is less than 1.
            ValueError: If dimension is less than 1.
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
        """Update the filter with a new sample and return the filtered value.

        This method adds a new sample to the filter's buffer and computes the
        filtered output by convolving the input samples with the FIR filter coefficients.

        Args:
            sample (Union[float, np.ndarray]): New input sample. Can be a scalar
                or an array of length equal to the filter's dimension.

        Returns:
            Filtered output value with the same shape as the input sample.

        Raises:
            ValueError: If the input sample's shape doesn't match the filter's dimension.
        """
        if not self._initialized:
            self._buffer = np.column_stack((sample,) * self._taps_cnt)
            self._initialized = True
        self._buffer = np.column_stack((self._buffer[:, 1:], sample))
        self._result = np.dot(self._buffer, self._fir_coeffs)
        return self._result

    def get(self):
        """Get the current filtered value without updating the filter.

        Returns:
            Current filtered value based on the samples in the buffer.
        """
        return self._result

    def __call__(self, sample: Union[float, np.ndarray]) -> np.ndarray:
        """Update the filter with a new sample and return the filtered value.

        This is a convenience method that calls update().

        Args:
            sample (Union[float, np.ndarray]): New input sample. Can be a scalar
                or an array of length equal to the filter's dimension.

        Returns:
            Filtered output value with the same shape as the input sample.
        """
        return self.update(sample)


def compute_kmeans(embeddings: List[np.ndarray], k: int = 0):
    """
    Compute k-means clustering on the given embeddings.
    Args:
        embeddings (List[np.ndarray]): List of embedding vectors.
        k (int): Number of clusters. If 0, it will be deduced based on the number of embeddings.
    Returns:
        List[nd.ndarray]: List of embeddings closest to cluster centroid.
    """
    from scipy.cluster.vq import kmeans2

    if k == 0:
        k = int(max(3, len(embeddings) / 25))

    embeddings_arr = np.array(embeddings)
    centroids, labels = kmeans2(embeddings_arr, k, minit="++")

    ret = []
    for i in range(k):
        cluster_embeddings = embeddings_arr[labels == i]
        if len(cluster_embeddings) == 0:
            continue
        distances = np.linalg.norm(cluster_embeddings - centroids[i], axis=1)
        ret.append(cluster_embeddings[np.argmin(distances)])

    return ret
