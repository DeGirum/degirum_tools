# object_selector.py: object selection analyzer
#
# Copyright DeGirum Corporation 2025
# All rights reserved
# Implements analyzer class to select top K objects based on selection strategy.
#

"""
Object Selector Analyzer Module Overview
====================================

This module provides an analyzer (`ObjectSelector`) for selecting the top-K detections from
object detection results based on various strategies and optional tracking. It enables
intelligent filtering of detection results to focus on the most relevant objects.

Key Features:
    - **Selection Strategies**: Supports selecting by highest confidence score or largest bounding-box area
    - **Tracking Integration**: Uses `track_id` fields to persist selections across frames with configurable timeout
    - **Top-K Selection**: Configurable number of objects to select per frame
    - **Visual Overlay**: Draws bounding boxes for selected objects on images
    - **Selection Persistence**: Maintains selection state across frames when tracking is enabled
    - **Timeout Control**: Configurable frame count before removing lost objects from selection

Typical Usage:
    1. Create an `ObjectSelector` instance with desired selection parameters
    2. Process each frame's detection results through the selector
    3. Access selected objects from the augmented results
    4. Optionally visualize selected objects using the annotate method
    5. Use selected objects in downstream analyzers for focused processing

Integration Notes:
    - Works with any detection results containing bounding boxes and confidence scores
    - Optional integration with `ObjectTracker` for persistent selection across frames
    - Selected objects are marked in the result object for downstream processing
    - Supports both frame-based and tracking-based selection modes

Key Classes:
    - `ObjectSelector`: Main analyzer class that processes detections and maintains selections
    - `ObjectSelectionStrategies`: Enumeration of available selection strategies

Configuration Options:
    - `top_k`: Number of objects to select per frame
    - `selection_strategy`: Strategy for ranking objects (by highest confidence score or by largest bounding box area)
    - `use_tracking`: Enable/disable tracking-based selection persistence
    - `tracking_timeout`: Frames to wait before removing lost objects from selection
    - `show_overlay`: Enable/disable visual annotations
    - `annotation_color`: Customize overlay appearance
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

    Members:
        HIGHEST_SCORE (int): Selects objects with the highest confidence scores.
        LARGEST_AREA (int): Selects objects with the largest bounding-box area.
    """

    HIGHEST_SCORE = 1  # select objects with highest score
    LARGEST_AREA = 2  # select objects with largest area


class ObjectSelector(ResultAnalyzerBase):
    """
    Selects the top-K detected objects per frame based on a specified strategy.

    This analyzer examines the detection results for each frame and retains only the
    top-K detections according to the chosen `ObjectSelectionStrategies` (e.g.,
    highest confidence score or largest bounding-box area).

    When tracking is enabled, it uses object `track_id` information to continue
    selecting the same objects across successive frames, removing an object from the
    selection if it has not appeared for a certain number of frames (the tracking timeout).
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
            top_k (int, optional): Number of objects to select. Default 1.
            selection_strategy (ObjectSelectionStrategies, optional): Strategy for ranking objects. Default ObjectSelectionStrategies.HIGHEST_SCORE.
            use_tracking (bool, optional): Whether to enable tracking-based selection. If True, only objects with a `track_id` field are selected (requires an ObjectTracker to precede this analyzer in the pipeline). Default True.
            tracking_timeout (int, optional): Number of frames to wait before removing an object from selection if it is not detected. Default 30.
            show_overlay (bool, optional): Whether to draw bounding boxes around selected objects on the output image. If False, the image is passed through unchanged. Default True.
            annotation_color (tuple, optional): RGB color for annotation boxes. Default None (uses the complement of the result overlay color).

        Raises:
            ValueError: If an unsupported selection strategy is provided.
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

        Returns:
            None: The result object is modified in-place.
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
            image (np.ndarray): Image to annotate, shape (H, W, 3) in RGB format.

        Returns:
            np.ndarray: Annotated image, shape (H, W, 3) in RGB format.
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
