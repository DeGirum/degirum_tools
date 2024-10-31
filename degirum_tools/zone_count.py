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
    A class for tracking object state within a zone.
    """

    timeout_count: int  # number of frames to buffer when an object disappears from zone
    object_label: str  # label of the object
    presence_count: int  # number of frames the object is present in the zone
    entering_time: float  # timestamp when the object entered the zone


class _PolygonZone:
    """
    A class for defining a polygon-shaped zone within a frame for detecting objects.

    Attributes:
        polygon (np.ndarray): A polygon represented by a numpy array of shape
            `(N, 2)`, containing the `x`, `y` coordinates of the points.
        frame_resolution_wh (Tuple[int, int]): The frame resolution (width, height)
        triggering_position (Union[List[AnchorPoint], AnchorPoint], optional): the position(s) within the bounding box
            that trigger(s) the zone; if None, iopa_threshold is used and must be specified
        bounding_box_scale (float, optional): scale factor used to downsize detection result bounding boxes before zone
            triggering is performed, no matter whether triggering positions or IoPA is used; useful when only a portion
            of a detected object (a "critical mass") inside a bounding box should trigger the zone
        iopa_threshold (float, optional): intersection over polygon area (IoPA) threshold; if triggering_position is None,
            IoPA of bounding boxes greater than this threshold triggers the zone, otherwise this method is not used
        timeout_frames (int, optional): number of frames to buffer when an object disappears from zone
        current_count (int): The current count of detected objects within the zone
        mask (np.ndarray): The 2D bool mask for the polygon zone
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
        Determines if the detections are within the polygon zone.

        Parameters:
            bboxes (np.ndarray): the numpy array of shape `(N, 4)` of bounding boxes to be checked against the polygon zone

        Returns:
            np.ndarray: A boolean numpy array indicating
                if each detection is within the polygon zone
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
            # trigger zones based on trigger points
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
    Class to count objects in polygon zones.

    Analyzes the object detection `result` object passed to `analyze` method and for each detected
    object checks, does its anchor point belongs to any of the polygon zones specified by the `count_polygons`
    constructor parameter. Only objects belonging to the class list specified by the `class_list`
    constructor parameter are counted.

    Updates detected objects in `result.results[]` list by adding the "in_zone" key containing
    the list of boolean flags, one flag per zone, indicating if the object is detected in the corresponding zone.
    If `use_tracking` option is enabled, only objects with track IDs are updated.
    Additionally, when `use_tracking` option is enabled, "frames_in_zone" and "time_in_zone" keys are added to
    objects with track IDs. The "frames_in_zone" value is the list containing the number of frames the object is
    present in each zone. The "time_in_zone" value is the list containing the total time in seconds the object is
    present in each zone.

    Adds `zone_counts` list of dictionaries to the `result` object - one element per polygon zone.
    Each dictionary contains the count of objects detected in the corresponding zone. The key is the
    class name and the value is the count of objects of this class detected in the zone.
    If the `per_class_display` constructor parameter is False, the dictionary contains only one key "total".

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
        """Constructor

        Args:
            count_polygons (nd.array): list of polygons to count objects in; each polygon is a list of points (x,y)
            class_list (List, optional): list of classes to count; if None, all classes are counted
            per_class_display (bool, optional): when True, display zone counts per class, otherwise display total zone counts
            triggering_position (Union[List[AnchorPoint], AnchorPoint], optional): the position(s) within the bounding box
                that trigger(s) the zone; if None, iopa_threshold is used and must be specified
            bounding_box_scale (float, optional): scale factor used to downsize detection result bounding boxes before zone
                triggering is performed, no matter whether triggering positions or IoPA is used; useful when only a portion
                of a detected object (a "critical mass") inside a bounding box should trigger the zone
            iopa_threshold (float, optional): intersection over polygon area (IoPA) threshold; if triggering_position is None,
                IoPA of bounding boxes greater than this threshold triggers the zone, otherwise this method is not used
            use_tracking (bool, optional): If True, use tracking information to select objects
                (object tracker must precede this analyzer in the pipeline)
            timeout_frames (int, optional): number of frames to tolerate temporary absence of the object when it disappears from zone
            window_name (str, optional): optional OpenCV window name to configure for interactive zone adjustment
            show_overlay: if True, annotate image; if False, send through original image
            show_inzone_counters: "time" to show time-in-zone, "frames" to show frames-in-zone, "all" to show both, None to show nothing
            annotation_color (tuple, optional): Color to use for annotations, None to use complement to result overlay color
            annotation_line_width (int, optional): Line width to use for annotations, None to use result overlay line width
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
        Detect object bounding boxes in polygon zones.

        Updates detected objects in `result.results[]` list by adding the "in_zone" key containing
        the list of boolean flags, one flag per zone, indicating if the object is detected in the corresponding zone.
        If `use_tracking` option is enabled, only objects with track IDs are updated.
        Additionally, when `use_tracking` option is enabled, "frames_in_zone" and "time_in_zone" keys are added to
        objects with track IDs. The "frames_in_zone" value is the list containing the number of frames the object is
        present in each zone. The "time_in_zone" value is the list containing the total time in seconds the object is
        present in each zone.

        Adds `zone_counts` list of dictionaries to the `result` object - one element per polygon zone.
        Each dictionary contains the count of objects detected in the corresponding zone. The key is the
        class name and the value is the count of objects of this class detected in the zone.
        If the `per_class_display` constructor parameter is False, the dictionary contains only one key "total".

        Args:
            result: PySDK model result object
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
        Display polygon zones and zone counts on a given image

        Args:
            result: PySDK result object to display (should be the same as used in analyze() method)
            image (np.ndarray): image to display on

        Returns:
            np.ndarray: annotated image
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

        # draw object annotations
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

        # draw zone annotations
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
        """Attach OpenCV window for interactive zone adjustment by installing mouse callback

        Args:
            win_name (str): OpenCV window name to attach to
        """

        self._win_name = win_name
        self._mouse_callback_installed = False

    def _lazy_init(self, result):
        """Complete deferred initialization steps
            - initialize polygon zones from model result object
            - install mouse callback

        Args:
            result: PySDK model result object
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
        """Mouse callback for OpenCV window for interactive zone operations"""

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
        if self._win_name is not None:
            try:
                cv2.setMouseCallback(self._win_name, ZoneCounter._mouse_callback, self)  # type: ignore[attr-defined]
                self._gui_state = {"dragging": None, "update": -1}
                self._mouse_callback_installed = True
            except Exception:
                pass  # ignore errors
