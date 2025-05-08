# object_selector.py: object selection analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
# Implements analyzer class to select top K objects based on selection strategy.
#

"""
Object Selector Analyzer Module Overview
========================================

This module provides an analyzer (`ObjectSelector`) for selecting the top-K detections from
object detection results based on various strategies and optional tracking.

Key Concepts:
    - **Selection Strategies**: Supports selecting by highest confidence score or largest bounding-box area.
    - **Tracking Integration**: Optionally uses `track_id` fields to persist selections across frames with a timeout.
    - **Annotation Overlay**: Optionally draws bounding boxes for selected objects on images.

"""

import numpy as np, copy, cv2
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
from .result_analyzer_base import ResultAnalyzerBase
from .math_support import area
from .ui_support import color_complement, rgb_to_bgr


class ObjectSelectionStrategies(Enum):
    """
    Enumeration of object selection strategies.

    Attributes:
        HIGHEST_SCORE (int): Select objects with the highest confidence scores.
        LARGEST_AREA (int): Select objects with the largest bounding-box area.
    """

    HIGHEST_SCORE = 1  # select objects with highest score
    LARGEST_AREA = 2  # select objects with largest area


class ObjectSelector(ResultAnalyzerBase):
    """
    Selects and optionally tracks top-K objects per frame.

    Inspects detection results and retains top-K based on strategy.

    When tracking is enabled, maintains selections using track IDs,
    expiring after tracking timeout.

    Args:
        top_k (int): Number of objects to select.
        selection_strategy (ObjectSelectionStrategies): Ranking strategy.
        use_tracking (bool): Enable tracking-based selection.
        tracking_timeout (int): Frames to wait before dropping unseen object.
        show_overlay (bool): Whether to draw rectangles around selected objects..
        annotation_color (tuple, optional): RGB color for boxes.
    """

    @dataclass
    class _SelectedObject:
        """
        Selected object data structure.

        Attributes:
            detection (dict): The detection result dictionary.
            counter (int): Frames since last seen before removal.
        """

        detection: dict  # detection result
        counter: int = (
            0  # counter to keep track of how long the object has not been found in new results
        )

    def __init__(
        self,
        *,
        top_k: int = 1,
        selection_strategy: ObjectSelectionStrategies = ObjectSelectionStrategies.HIGHEST_SCORE,
        use_tracking: bool = True,
        tracking_timeout: int = 30,
        show_overlay: bool = True,
        annotation_color: Optional[tuple] = None,
    ):
        """
        Constructor.

        Args:
            top_k (int): Number of objects to select.
            selection_strategy (ObjectSelectionStrategies): Strategy for ranking objects.
            use_tracking (bool): If True, enables tracking-based selection: only objects with "track_id" are selected
                (object tracker must precede this analyzer in the pipeline).
            tracking_timeout (int): Number of frames to wait before removing an object from selection.
            show_overlay (bool): If True, annotate image; if False, pass image through unchanged.
            annotation_color (Optional[tuple]): Color to use for annotations; None to use complement of result overlay color.
        """
        self._top_k = top_k
        self._selection_strategy = selection_strategy
        self._use_tracking = use_tracking
        self._selected_objects: List[ObjectSelector._SelectedObject] = []
        self._tracking_timeout = tracking_timeout
        self._show_overlay = show_overlay
        self._annotation_color = annotation_color

    def analyze(self, result):
        """
        Select the top-K objects based on the configured strategy, updating the result.

        Uses tracking IDs to update selected objects when tracking is enabled.

        All other objects not selected are removed from results.

        Args:
            result (InferenceResults): Model result with detection information.
        """

        all_detections = result._inference_results

        if self._selection_strategy == ObjectSelectionStrategies.LARGEST_AREA:

            def metric(det):
                return area(np.array(det["bbox"]))

        elif self._selection_strategy == ObjectSelectionStrategies.HIGHEST_SCORE:

            def metric(det):
                return det["score"]

        else:
            raise ValueError(f"Invalid selection strategy {self._selection_strategy}")

        if self._use_tracking:
            tracked_detections = {
                det["track_id"]: det for det in all_detections if "track_id" in det
            }

            # update existing selected objects
            for obj in self._selected_objects:
                matching_detection = tracked_detections.get(
                    obj.detection["track_id"], None
                )
                if matching_detection is not None:
                    # update bbox from new result
                    obj.detection = copy.deepcopy(matching_detection)
                else:
                    # delete object if not found in new result for a while
                    obj.counter += 1
                    if obj.counter > self._tracking_timeout:
                        self._selected_objects.remove(obj)

            # add new objects
            if len(self._selected_objects) < self._top_k:
                sorted_detections = sorted(
                    tracked_detections.values(), key=metric, reverse=True
                )
                self._selected_objects += [
                    ObjectSelector._SelectedObject(copy.deepcopy(det))
                    for det in sorted_detections[
                        : self._top_k - len(self._selected_objects)
                    ]
                ]

            result._inference_results = [
                copy.deepcopy(obj.detection) for obj in self._selected_objects
            ]
        else:
            sorted_detections = sorted(all_detections, key=metric, reverse=True)
            result._inference_results = sorted_detections[: self._top_k]

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes for the selected objects on the image.

        Args:
            result (InferenceResults): The result containing selected objects.
            image (np.ndarray): Image to annotate.

        Returns:
            np.ndarray: Annotated image.
        """

        if not self._show_overlay:
            return image

        color = (
            color_complement(result.overlay_color)
            if self._annotation_color is None
            else self._annotation_color
        )

        for obj in result._inference_results:
            bbox = obj.get("bbox")
            if bbox is not None:
                int_bbox = tuple(map(int, bbox))
                cv2.rectangle(
                    image,
                    int_bbox[:2],
                    int_bbox[2:],
                    rgb_to_bgr(color),
                    2,
                )
        return image
