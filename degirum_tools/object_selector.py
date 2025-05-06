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

Key Concepts
------------

- **Selection Strategies**: Supports selecting by highest confidence score or largest bounding-box area.
- **Tracking Integration**: Optionally uses `track_id` fields to persist selections across frames with a timeout.
- **Annotation Overlay**: Optionally draws bounding boxes for selected objects on images.

Basic Usage
-----------
```python
from degirum_tools.object_selector import ObjectSelector, ObjectSelectionStrategies

# Create selector to choose top 3 by score
selector = ObjectSelector(top_k=3, selection_strategy=ObjectSelectionStrategies.HIGHEST_SCORE)

# Attach to model and run inference
model.attach_analyzers(selector)
for res in model.predict_batch(images):
    img = res.image_overlay
    # process annotated image
```
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
    Analyzer to select and optionally track the top-K objects per frame.

    This analyzer inspects a detection `result` and retains only the top-K entries based on the
    specified `ObjectSelectionStrategies`. When `use_tracking` is enabled, it maintains selected
    objects across frames using `track_id`, expiring them after `tracking_timeout` frames of absence.

    Args:
        top_k (int): Number of objects to select.
        selection_strategy (ObjectSelectionStrategies): Strategy for ranking objects.
        use_tracking (bool): Enable tracking across frames.
        tracking_timeout (int): Frames to wait before dropping an unseen object.
        show_overlay (bool): Draw bounding boxes on the output image.
        annotation_color (Optional[tuple]): RGB color for annotation; defaults to complement of model overlay color.
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
            use_tracking (bool): If True, use tracking information to select objects
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
        Finds top-K objects based on the selection strategy and updates the result.

        Args:
            result (InferenceResults): PySDK model result object to analyze.
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
        Displays object bounding boxes on a given image.

        Args:
            result (InferenceResults): Model result object to display (same as used in analyze()).
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
