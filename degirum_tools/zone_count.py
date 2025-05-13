#
# zone_count.py: polygon zone object counting support
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes for polygon zone object counting
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
Zone Count Module Overview
==========================

This module provides support for counting objects within polygonal zones in video frames or images.
It offers classes that integrate with AI inference results to determine whether detected or tracked objects lie within user-defined polygon zones.

Core Features:
    - **Polygonal zone definition:** Allows specifying arbitrary polygon shapes as counting zones.
    - **Anchor-point and IoPA triggering:** Supports two approaches to determine if an object is inside a zone:
        - Using object bounding box anchor points (e.g., bottom center, center).
        - Using Intersection over Polygon Area (IoPA) threshold based on bounding boxes intersecting polygon.
    - **Per-class counting support:** Optionally counts objects separately by class labels.
    - **Object tracking integration:** Supports smoothing of counts over time using frame and time-based track presence.
    - **Timeout support:** Configurable grace period for objects temporarily missing from zones before being considered outside.
    - **Comprehensive overlay annotation:** Draws zones and optionally per-zone object counts or in-zone presence duration on images.
    - **Interactive adjustment (optional):** OpenCV mouse callback support to interactively edit polygon zones in a GUI.

Typical Use Case:
    1. Define one or several polygon zones over the target video/image frame.
    2. Instantiate `ZoneCounter`, specifying zones, classes to count, triggering method, and tracking options.
    3. In the inference pipeline, feed detection or tracking results to `ZoneCounter.analyze()`.
    4. Retrieve per-zone counts and per-object in-zone flags for further processing or event detection.
    5. Optionally, call `ZoneCounter.annotate()` to draw zones and counts on images for visualization.

Integration Notes:
    - Requires detections to include bounding boxes and optionally track IDs for tracking-related functionality.
    - Works effectively when used downstream of an object tracker analyzer to leverage track IDs.
    - The module expects standard DeGirum PySDK result formats (e.g., `result.results` with dict entries for detected objects).
    - Optional parameters enforce consistency and handling of partial/missing detections over time.

"""

import numpy as np, cv2, time
from typing import Tuple, Optional, Dict, List, Union, Any
from dataclasses import dataclass
from .ui_support import (
    put_text,
    color_complement,
    deduce_text_color,
    rgb_to_bgr,
    CornerPosition,
)
from .result_analyzer_base import ResultAnalyzerBase
from .math_support import (
    AnchorPoint,
    get_anchor_coordinates,
    xyxy2xywh,
    xywh2xyxy,
    tlbr2allcorners,
)


@dataclass
class _ObjectState:
    """
    Internal class to track the state of an object within a zone.

    Attributes
    ----------
    timeout_count : int
        Number of frames the object is allowed to be missing before it is considered exited.
    object_label : str
        The class label of the object being tracked.
    presence_count : int
        Number of consecutive frames the object has been present in the zone.
    entering_time : float
        Timestamp when the object was first detected in the zone.
    """

    timeout_count: int  # number of frames to buffer when an object disappears from zone
    object_label: str  # label of the object
    presence_count: int  # number of frames the object is present in the zone
    entering_time: float  # timestamp when the object entered the zone


class _PolygonZone:
    """
    Represents a polygonal zone within a video frame or image, used for testing whether
    a detection's bounding box or anchor points lie inside the zone.

    This class encapsulates:
    - The polygon shape and frame resolution.
    - Triggering logic based on anchor points or IoPA (Intersection over Polygon Area).
    - State tracking of objects within the zone for filtering transient entries/exits.

    Parameters such as bounding box scale and IoPA threshold allow fine control over triggering.

    Attributes:
        polygon (np.ndarray): Polygon vertices as (N,2) array.
        frame_resolution_wh (Tuple[int, int]): Frame width and height.
        triggering_position (list[AnchorPoint] or None): Anchor points to trigger inside zone.
        bounding_box_scale (float): Scale to shrink bounding boxes before zone trigger.
        iopa_threshold (float): IoPA threshold to trigger zone.
        timeout_frames (int): Frames to tolerate temporary absence of object.


    """

    def __init__(
        self,
        polygon: np.ndarray,
        frame_resolution_wh: Tuple[int, int],
        triggering_position: Optional[Union[List[AnchorPoint], AnchorPoint]],
        bounding_box_scale: float = 1.0,
        iopa_threshold: float = 0.0,
        timeout_frames: int = 0,
    ):
        """
        Constructor.

        Args:
            polygon (np.ndarray): Polygon vertices.
            frame_resolution_wh (Tuple[int, int]): Frame resolution (width, height).
            triggering_position (list or AnchorPoint or None): Anchor positions to check.
            bounding_box_scale (float, optional): Box scaling factor. Default 1.0.
            iopa_threshold (float, optional): IoPA threshold. Default 0.0.
            timeout_frames (int, optional): Timeout frames for tracking absence. Default 0.
        """
        self.frame_resolution_wh = frame_resolution_wh
        self.triggering_position = (
            triggering_position
            if triggering_position is None or isinstance(triggering_position, list)
            else [triggering_position]
        )
        self.bounding_box_scale = bounding_box_scale
        self.iopa_threshold = iopa_threshold
        self._timeout_count_initial = timeout_frames
        self._object_states: Dict[int, _ObjectState] = {}
        self.width, self.height = frame_resolution_wh
        self.mask = np.zeros((self.height + 1, self.width + 1))
        cv2.fillPoly(self.mask, [cv2.Mat(polygon.astype(int))], color=[1])
        self.polygon = polygon.astype(np.float32)
        self.polygon_area = cv2.contourArea(polygon.reshape(1, -1, 2))

    def trigger(self, bboxes: np.ndarray) -> np.ndarray:
        """
        Determine which bounding boxes are considered inside the polygon zone.

        This method applies the configured triggering logic:
        - If `triggering_position` is set, tests if any specified anchor point
          of each bounding box lies inside the polygon.
        - Otherwise, uses IoPA (Intersection over Polygon Area) with the polygon
          and thresholds it against `iopa_threshold`.

        Parameters
        ----------
        bboxes : np.ndarray
            Array of shape `(N, 4)` with bounding boxes in `(x0, y0, x1, y1)` format.

        Returns
        -------
        np.ndarray
            Boolean array of length `N`, indicating if each bounding box is inside the zone.
        """
        # clip to frame
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, self.width)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, self.height)
        # scale down bounding box by scale factor
        if self.bounding_box_scale < 1.0:
            bboxes_xywh = xyxy2xywh(bboxes).astype(float)
            bboxes_xywh[:, [2, 3]] *= self.bounding_box_scale
            bboxes = xywh2xyxy(bboxes_xywh)
        if self.triggering_position is None:
            # trigger zones based on IoPA
            iopa = np.zeros((bboxes.shape[0]), dtype=float)
            bboxes_corners = tlbr2allcorners(bboxes).astype(np.float32)
            for i in range(len(bboxes_corners)):
                iopa[i] = cv2.intersectConvexConvex(
                    bboxes_corners[i].reshape(-1, 2), self.polygon
                )[0]
            iopa /= self.polygon_area
            is_in_zone = iopa > self.iopa_threshold
        else:
            # trigger zones based on trigger points (anchor points)
            clipped_anchors = np.array(
                [
                    np.ceil(get_anchor_coordinates(xyxy=bboxes, anchor=anchor)).astype(
                        int
                    )
                    for anchor in self.triggering_position
                ]
            )
            is_in_zone = np.logical_or.reduce(
                self.mask[clipped_anchors[:, :, 1], clipped_anchors[:, :, 0]]
            )
        return is_in_zone


class ZoneCounter(ResultAnalyzerBase):
    """
    Analyzer class that counts objects detected within polygonal zones.

    This class is designed to integrate with PySDK inference results,
    augmenting them with per-zone in/out flags and counts, with support
    for per-class counting and optional object tracking integration.

    Functionality:
    - Determines if objects are inside polygonal zones according to configuration.
    - Supports per-class counting or total counts.
    - Can use object tracking to provide frame-based and time-based presence metrics.
    - Tracks objects with timeout to tolerate short absences.
    - Overlay polygon zones and object counts on output images.

    This class can be used as a step in an inference result processing pipeline
    to provide high-level analytics on object presence per spatial zone.

    Attributes:
        zone_counts (List[dict]): List of per-zone count dictionaries updated
            after each call to `analyze()`.
        key_in_zone (str): Name of the key inserted into result objects that
            stores zone-presence flags.
        key_frames_in_zone (str): Name of the key inserted into result objects
            that stores the number of frames spent in each zone.
        key_time_in_zone (str): Name of the key inserted into result objects
            that stores the total time (in seconds) spent in each zone.

    Methods:
        analyze(result): Determine per-zone presence for *result* and update
            cumulative counts.
        annotate(result, image): Draw zones and their counts on *image*.
        window_attach(win_name): Attach an OpenCV window for interactive
            editing, if supported.

    Examples:
        See the module-level docstring for a complete usage example.
    """

    key_in_zone = "in_zone"
    key_frames_in_zone = "frames_in_zone"
    key_time_in_zone = "time_in_zone"

    def __init__(
        self,
        count_polygons: Union[np.ndarray, list],
        *,
        class_list: Optional[List] = None,
        per_class_display: Optional[bool] = False,
        triggering_position: Optional[
            Union[List[AnchorPoint], AnchorPoint]
        ] = AnchorPoint.BOTTOM_CENTER,
        bounding_box_scale: float = 1.0,
        iopa_threshold: float = 0.0,
        use_tracking: Optional[bool] = False,
        timeout_frames: int = 0,
        window_name: Optional[str] = None,
        show_overlay: bool = True,
        show_inzone_counters: Optional[str] = None,
        annotation_color: Optional[tuple] = None,
        annotation_line_width: Optional[int] = None,
    ):
        """
        Constructor.

        Args:
            count_polygons (List[np.ndarray]): List of polygon zones.
            class_list (List[str], optional): Classes to count. None counts all.
            per_class_display (bool, optional): Display counts per class if True.
            triggering_position (list or AnchorPoint or None, optional): Anchor points to trigger. If None, IoPA used.
            bounding_box_scale (float, optional): Scale bounding boxes before trigger.
            iopa_threshold (float, optional): IoPA threshold if triggering_position is None.
            use_tracking (bool, optional): Use tracking info if True.
            timeout_frames (int, optional): Frames to tolerate absence of object in zone.
            window_name (str, optional): Window name for interactive adjustment.
            show_overlay (bool, optional): Draw annotations if True.
            show_inzone_counters (str or None, optional): 'time', 'frames', 'all' or None.
            annotation_color (tuple, optional): RGB color for annotations.
            annotation_line_width (int, optional): Annotation line thickness.

        """
        self._wh: Optional[Tuple] = None
        self._zones: Optional[List] = None
        self._win_name = window_name
        self._mouse_callback_installed = False
        self._class_list = [] if class_list is None else class_list
        self._per_class_display = per_class_display
        if class_list is None and per_class_display:
            raise ValueError(
                "class_list must be specified when per_class_display is True"
            )
        self._triggering_position = triggering_position
        self._bounding_box_scale = bounding_box_scale
        if self._bounding_box_scale <= 0.0 or self._bounding_box_scale > 1.0:
            raise ValueError("bounding_box_scale must be from 0 to 1")
        self._iopa_threshold = iopa_threshold
        if self._iopa_threshold <= 0 and self._triggering_position is None:
            raise ValueError(
                "iopa_threshold must be specified when triggering_position is None"
            )
        self._use_tracking = use_tracking
        self._timeout_frames = timeout_frames
        if self._timeout_frames > 0 and not self._use_tracking:
            raise ValueError(
                "timeout_frames can be used only when use_tracking is True"
            )
        self._polygons = [
            np.array(polygon, dtype=np.int32) for polygon in count_polygons
        ]
        self._show_overlay = show_overlay
        self._show_inzone_counters = show_inzone_counters
        if self._show_inzone_counters not in [None, "time", "frames", "all"]:
            raise ValueError(
                "show_inzone_counters must be one of 'time', 'frames', 'all', or None"
            )
        if self._show_inzone_counters is not None and not self._show_overlay:
            raise ValueError(
                "show_inzone_counters can be used only when show_overlay is True"
            )
        if self._show_inzone_counters is not None and not self._use_tracking:
            raise ValueError(
                "show_inzone_counters can be used only when use_tracking is True"
            )
        self._annotation_color = annotation_color
        self._annotation_line_width = annotation_line_width

    def analyze(self, result):
        """
        Analyze the given PySDK inference result to count objects in polygon zones.

        Modifies `result` by adding:
        - `result.zone_counts`: list of dictionaries with counts per polygon zone.
        - Adds `"in_zone"` list of booleans per detected object.
        - If tracking is enabled, adds `"frames_in_zone"` and `"time_in_zone"` counters per object.

        Handles:
        - Filtering detections by class_list and tracking presence.
        - Updating internal object states to tolerate temporary absences.
        - Counting total and per-class object presence in each zone.

        Args:
            result (InferenceResults): Inference result to analyze.
        """
        result.zone_counts = [
            dict.fromkeys(
                self._class_list + ["total"] if self._per_class_display else ["total"],
                0,
            )
            for _ in self._polygons
        ]
        self._lazy_init(result)
        if self._zones is None:
            return

        def in_class_list(label):
            return (
                True
                if not self._class_list
                else False if label is None else label in self._class_list
            )

        filtered_results = [
            obj
            for obj in result.results
            if "bbox" in obj
            and in_class_list(obj.get("label"))
            and (not self._use_tracking or "track_id" in obj)
        ]
        # here `filtered_results` contains only objects with labels from the class list, having track IDs if tracking is used
        if self._use_tracking and hasattr(result, "trails"):
            active_tids = [obj["track_id"] for obj in filtered_results]
            lost_results = [
                {
                    "bbox": result.trails[tid][-1],
                    "label": label,
                    "track_id": tid,
                }
                for tid, label in result.trail_classes.items()
                if tid not in active_tids and in_class_list(label)
            ]
            filtered_results.extend(lost_results)
            # here `filtered_results` is extended with "lost" objects: objects that were not detected but still tracked
        # initialize per-object results
        for obj in filtered_results:
            nzones = len(self._zones)
            obj[self.key_in_zone] = [False] * nzones
            if self._use_tracking:
                obj[self.key_frames_in_zone] = [0] * nzones
                obj[self.key_time_in_zone] = [0.0] * nzones
        bboxes = np.array([obj["bbox"] for obj in filtered_results])
        time_now = time.time()

        def set_in_zone_and_increment_counts(obj, zi, zone_counts):
            obj[self.key_in_zone][zi] = True
            zone_counts["total"] += 1
            if self._per_class_display:
                label = obj.get("label")
                if label:
                    zone_counts[label] += 1

        def set_time_in_zone(obj, zi, obj_state):
            obj[self.key_frames_in_zone][zi] = obj_state.presence_count
            obj[self.key_time_in_zone][zi] = time_now - obj_state.entering_time

        for zi, zone in enumerate(self._zones):
            # compute object-in-zone flags
            triggers = zone.trigger(bboxes) if bboxes.size > 0 else np.array([], bool)
            zone_counts = result.zone_counts[zi]
            all_tids_in_zone = []
            # iterate over all filtered objects
            for obj, flag in zip(filtered_results, triggers):
                if flag:
                    # object is in zone
                    set_in_zone_and_increment_counts(obj, zi, zone_counts)
                    if self._use_tracking:
                        tid = obj["track_id"]
                        obj_state = zone._object_states.get(tid)
                        if obj_state:
                            # object was already in zone before
                            obj_state.timeout_count = zone._timeout_count_initial
                            obj_state.presence_count += 1
                        else:
                            # fresh object, never seen in this zone
                            zone._object_states[tid] = obj_state = _ObjectState(
                                timeout_count=zone._timeout_count_initial,
                                object_label=obj.get("label"),
                                presence_count=1,
                                entering_time=time_now,
                            )
                        set_time_in_zone(obj, zi, obj_state)
                        all_tids_in_zone.append(tid)
                else:
                    # object is NOT in zone
                    if self._use_tracking and self._timeout_frames > 0:
                        tid = obj["track_id"]
                        obj_state = zone._object_states.get(tid)
                        if obj_state is not None:
                            # object is not in zone, but still tracked
                            obj_state.presence_count += 1
                            set_in_zone_and_increment_counts(obj, zi, zone_counts)
                            set_time_in_zone(obj, zi, obj_state)
                            obj_state.timeout_count -= 1
                            if obj_state.timeout_count > 0:
                                all_tids_in_zone.append(tid)
                            else:
                                del zone._object_states[tid]
            if self._use_tracking:
                # process inactive objects: still tracked, but not detected on the current frame
                inactive_set = set(zone._object_states.keys())
                if len(all_tids_in_zone) > 0:
                    inactive_set -= set(all_tids_in_zone)
                for tid in inactive_set:
                    obj_state = zone._object_states[tid]
                    if self._timeout_frames > 0:
                        obj_state.presence_count += 1
                        obj_state.timeout_count -= 1
                        zone_counts["total"] += 1
                        if self._per_class_display and obj_state.object_label:
                            zone_counts[obj_state.object_label] += 1
                    # here we delete both timed-out and lost objects (in case of zero timeout)
                    if obj_state.timeout_count == 0:
                        del zone._object_states[tid]

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Annotate the given image with polygon zones and counts.

        Draws polygon outlines and optionally displays per-zone total or per-class counts.
        If tracking is used together with presence counters, optionally draws frames-in-zone
        or time-in-zone indicators near detected objects.

        Args:
            result (InferenceResults): Result with zone counts and per-object flags.
            image (np.ndarray): Image to annotate.

        Returns:
            np.ndarray: Annotated image
        """
        if not self._show_overlay:
            return image
        line_color = (
            color_complement(result.overlay_color)
            if self._annotation_color is None
            else self._annotation_color
        )
        text_color = deduce_text_color(line_color)
        line_width = (
            result.overlay_line_width
            if self._annotation_line_width is None
            else self._annotation_line_width
        )
        # draw object annotations (presence counters)
        if self._show_inzone_counters:
            for r in result.results:
                if "in_zone" in r:
                    text = ""
                    for zi, in_zone in enumerate(r["in_zone"]):
                        if in_zone:
                            if text:
                                text += "\n"
                            if len(self._polygons) > 1:
                                text += f"Z{zi}: "
                            text += (
                                f"{r['frames_in_zone'][zi]}#"
                                if self._show_inzone_counters == "frames"
                                else (
                                    f"{r['time_in_zone'][zi]:.1f}s"
                                    if self._show_inzone_counters == "time"
                                    else f"{r['frames_in_zone'][zi]}#/{r['time_in_zone'][zi]:.1f}s"
                                )
                            )
                    if text:
                        xy = (
                            np.array(r["bbox"][:2]).astype(int)
                            + result.overlay_line_width
                        )
                        put_text(
                            image,
                            text,
                            tuple(xy),
                            corner_position=CornerPosition.TOP_LEFT,
                            font_color=text_color,
                            bg_color=line_color,
                            font_scale=result.overlay_font_scale,
                        )
        # draw polygon zone outlines and display per-zone counts
        for zi in range(len(self._polygons)):
            cv2.polylines(
                image,
                [cv2.Mat(self._polygons[zi])],
                True,
                rgb_to_bgr(line_color),
                line_width,
            )
            if self._per_class_display and self._class_list is not None:
                text = f"Zone {zi}:"
                for class_name in self._class_list:
                    text += (
                        f"\n {class_name}: {result.zone_counts[zi].get(class_name, 0)}"
                    )
            else:
                text = f"Zone {zi}: {result.zone_counts[zi].get('total', 0)}"
            put_text(
                image,
                text,
                tuple(x + line_width for x in self._polygons[zi][0]),
                font_color=text_color,
                bg_color=line_color,
                font_scale=result.overlay_font_scale,
            )
        return image

    def window_attach(self, win_name: str):
        """
        Attach an OpenCV window for optional interactive zone adjustment.

        This installs a mouse callback on the window whose name is provided,
        allowing the user to drag polygons or their vertices to adjust zones.

        Args:
            win_name (str): OpenCV window name to attach the callback to.
        """
        self._win_name = win_name
        self._mouse_callback_installed = False

    def _lazy_init(self, result):
        """
        Performs deferred initialization tasks.

        - Initializes polygon zones from the model result, computing necessary masks and data.
        - Installs mouse callback if interactive adjustment is enabled.

        Args:
            result: PySDK inference result object containing an image for dimension reference.
        """
        if self._zones is None:
            self._wh = (result.image.shape[1], result.image.shape[0])
            self._zones = [
                _PolygonZone(
                    polygon,
                    self._wh,
                    self._triggering_position,
                    self._bounding_box_scale,
                    self._iopa_threshold,
                    self._timeout_frames,
                )
                for polygon in self._polygons
            ]
        if not self._mouse_callback_installed and self._win_name is not None:
            self._install_mouse_callback()

    @staticmethod
    def _mouse_callback(event: int, x: int, y: int, flags: int, self: Any):
        """
        Mouse event callback for interactive polygon zone editing.

        Supports:
        - Left-click and drag to move entire polygon.
        - Right-click and drag to move individual vertices.
        - Updates internal polygon state on changes.

        Args:
            event (int): OpenCV mouse event code.
            x (int): X coordinate of the mouse event.
            y (int): Y coordinate of the mouse event.
            flags (int): Additional event flags.
            self: Instance of ZoneCounter.
        """
        click_point = np.array((x, y))

        def zone_update():
            idx = self._gui_state["update"]
            if idx >= 0 and self._wh is not None:
                if not np.array_equal(self._zones[idx].polygon, self._polygons[idx]):
                    self._zones[idx] = _PolygonZone(
                        self._polygons[idx],
                        self._wh,
                        self._triggering_position,
                        self._bounding_box_scale,
                        self._iopa_threshold,
                        self._timeout_frames,
                    )

        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, polygon in enumerate(self._polygons):
                if cv2.pointPolygonTest(polygon, (x, y), False) > 0:
                    self._gui_state["dragging"] = polygon
                    self._gui_state["offset"] = click_point
                    self._gui_state["update"] = idx
                    break
        if event == cv2.EVENT_RBUTTONDOWN:
            for idx, polygon in enumerate(self._polygons):
                for pt in polygon:
                    if np.linalg.norm(pt - click_point) < 10:
                        self._gui_state["dragging"] = pt
                        self._gui_state["offset"] = click_point
                        self._gui_state["update"] = idx
                        break
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._gui_state["dragging"] is not None:
                delta = click_point - self._gui_state["offset"]
                self._gui_state["dragging"] += delta
                self._gui_state["offset"] = click_point
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self._gui_state["dragging"] = None
            zone_update()
            self._gui_state["update"] = -1

    def _install_mouse_callback(self):
        """
        Internal method to install the OpenCV mouse callback on the attached window.
        """
        if self._win_name is not None:
            try:
                cv2.setMouseCallback(self._win_name, ZoneCounter._mouse_callback, self)  # type: ignore[attr-defined]
                self._gui_state = {"dragging": None, "update": -1}
                self._mouse_callback_installed = True
            except Exception:
                pass  # ignore errors
