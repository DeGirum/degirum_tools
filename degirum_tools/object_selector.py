#
# object_selector.py: object selection analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements analyzer class to select top K objects based on selection strategy.
#

import numpy as np, copy, cv2
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
from .result_analyzer_base import ResultAnalyzerBase
from .math_support import area
from .ui_support import color_complement


class ObjectSelectionStrategies(Enum):
    """
    Object selection strategies
    """

    HIGHEST_SCORE = 1  # select objects with highest score
    LARGEST_AREA = 2  # select objects with largest area


class ObjectSelector(ResultAnalyzerBase):
    """
    Class to select top K objects based on selection strategy.

    Analyzes the object detection `result` object passed to `analyze` method and selects top K detections based on
    selection strategy (see ObjectSelectionStrategies enum). Removes all other detections.

    If `use_tracking` is True, it uses tracking information to select objects. Object tracker must precede this
    analyzer in the pipeline. It also uses `tracking_timeout` to remove an object from selection if it is not found in
    new results for a while.

    """

    @dataclass
    class _SelectedObject:
        """
        Selected object data structure
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
        Constructor

        Args:
            top_k: Number of objects to select
            selection_strategy: Selection strategy
            use_tracking: If True, use tracking information to select objects
                (object tracker must precede this analyzer in the pipeline)
            tracking_timeout: Number of frames to wait before removing an object from selection
            show_overlay: if True, annotate image; if False, send through original image
            annotation_color: Color to use for annotations, None to use complement to result overlay color
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
        Find top-K objects based on selection strategy. Remove all other objects.

        Args:
            result: PySDK model result object
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
        Display object bboxes on a given image

        Args:
            result: PySDK result object to display (should be the same as used in analyze() method)
            image (np.ndarray): image to display on

        Returns:
            np.ndarray: annotated image
        """

        if not self._show_overlay:
            return image

        color = (
            color_complement(result.overlay_color)
            if self._annotation_color is None
            else self._annotation_color
        )

        for obj in result._inference_results:
            bbox = tuple(map(int, obj["bbox"]))

            cv2.rectangle(
                image,
                bbox[:2],
                bbox[2:],
                color,
                2,
            )
        return image
