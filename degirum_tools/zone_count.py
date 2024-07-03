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


import numpy as np, cv2
from typing import Tuple, Optional, Any
from .ui_support import put_text, color_complement, deduce_text_color
from .result_analyzer_base import ResultAnalyzerBase
from .math_support import (
    AnchorPoint,
    get_anchor_coordinates,
    xyxy2xywh,
    xywh2xyxy,
    xyxy2xyxyxyxy,
)


class _PolygonZone:
    """
    A class for defining a polygon-shaped zone within a frame for detecting objects.

    Attributes:
        polygon (np.ndarray): A polygon represented by a numpy array of shape
            `(N, 2)`, containing the `x`, `y` coordinates of the points.
        frame_resolution_wh (Tuple[int, int]): The frame resolution (width, height)
        triggering_positions: The position(s) within the bounding box that trigger(s) the zone
        bounding_box_scale: scale factor used to resize bounding boxes
        iopa_threshold: intersection over polygon area (IoPA) threshold
        current_count (int): The current count of detected objects within the zone
        mask (np.ndarray): The 2D bool mask for the polygon zone
    """

    def __init__(
        self,
        polygon: np.ndarray,
        frame_resolution_wh: Tuple[int, int],
        triggering_positions: list[AnchorPoint],
        bounding_box_scale: float,
        iopa_threshold: float,
    ):
        self.frame_resolution_wh = frame_resolution_wh
        self.triggering_positions = triggering_positions
        self.bounding_box_scale = bounding_box_scale
        self.iopa_threshold = iopa_threshold

        self.width, self.height = frame_resolution_wh
        self.mask = np.zeros((self.height + 1, self.width + 1))
        cv2.fillPoly(self.mask, [polygon.astype(int)], color=[1])
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

        if self.triggering_positions is None:
            # trigger zones based on IoPA
            iopa = np.zeros((bboxes.shape[0]), dtype=float)
            bboxes_corners = xyxy2xyxyxyxy(bboxes).astype(np.float32)
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
                    for anchor in self.triggering_positions
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

    Updates each element of `result.results[]` list by adding the "in_zone" key containing
    the index of the polygon zone where the corresponding is detected.

    Adds `zone_counts` list of dictionaries to the `result` object - one element per polygon zone.
    Each dictionary contains the count of objects detected in the corresponding zone. The key is the
    class name and the value is the count of objects of this class detected in the zone.
    If the `per_class_display` constructor parameter is False, the dictionary contains only one key "total".

    """

    def __init__(
        self,
        count_polygons: np.ndarray,
        *,
        class_list: Optional[list] = None,
        per_class_display: bool = False,
        triggering_positions: list[AnchorPoint] = [AnchorPoint.BOTTOM_CENTER],
        bounding_box_scale: float = 1.0,
        iopa_threshold: float = 0.0,
        use_tracking: bool = False,
        window_name: Optional[str] = None,
    ):
        """Constructor

        Args:
            count_polygons (nd.array): list of polygons to count objects in; each polygon is a list of points (x,y)
            class_list (list, optional): list of classes to count; if None, all classes are counted
            per_class_display (bool, optional): when True, display zone counts per class, otherwise display total zone counts
            triggering_positions (AnchorPoint, optional): the position(s) within the bounding box that trigger(s) the zone
            bounding_box_scale: scale factor used to resize bounding boxes
            iopa_threshold: intersection over polygon area (IoPA) threshold
            use_tracking: If True, use tracking information to select objects
                (object tracker must precede this analyzer in the pipeline)
            window_name (str, optional): optional OpenCV window name to configure for interactive zone adjustment
        """

        self._wh: Optional[Tuple] = None
        self._zones: Optional[list] = None
        self._win_name = window_name
        self._mouse_callback_installed = False
        self._class_list = class_list
        self._per_class_display = per_class_display
        if class_list is None and per_class_display:
            raise ValueError(
                "class_list must be specified when per_class_display is True"
            )

        self._triggering_positions = triggering_positions
        self._bounding_box_scale = bounding_box_scale
        self._iopa_threshold = iopa_threshold
        self._use_tracking = use_tracking
        self._polygons = [
            np.array(polygon, dtype=np.int32) for polygon in count_polygons
        ]

    def analyze(self, result):
        """
        Detect object bounding boxes in polygon zones.

        Updates each result object `result.results[i]` by adding "in_zone" key to it,
        when this object is in a zone and its class belongs to a class list specified
        in a constructor. "in_zone" key value is the index of the zone where this object
        is detected.

        Adds `zone_counts` list of dictionaries to the `result` object - one element per polygon zone.
        Each dictionary contains the count of objects detected in the corresponding zone. The key is the
        class name and the value is the count of objects of this class detected in the zone.
        If the `per_class_display` constructor parameter is False, the dictionary contains only one key "total".


        Args:
            result: PySDK model result object
        """

        result.zone_counts = [{} for _ in self._polygons]

        self._lazy_init(result)

        if self._zones is None:
            return

        def in_class_list(label):
            return (
                True
                if self._class_list is None
                else False if label is None else label in self._class_list
            )

        if self._use_tracking:
            filtered_results = [
                {"bbox": result.trails[tid][-1], "label": result.trail_classes[tid]}
                for tid in set(result.trails.keys())
                if in_class_list(result.trail_classes[tid])
            ]
        else:
            filtered_results = [
                obj
                for obj in result.results
                if "bbox" in obj and in_class_list(obj.get("label", None))
            ]

        if len(filtered_results) == 0:
            return

        bboxes = np.array([obj["bbox"] for obj in filtered_results])

        for zi, zone in enumerate(self._zones):
            triggers = zone.trigger(bboxes)
            zone_counts = result.zone_counts[zi]
            for obj, flag in zip(filtered_results, triggers):
                if flag:
                    obj["in_zone"] = zi
                    label = (
                        obj["label"]
                        if self._per_class_display and "label" in obj
                        else "total"
                    )
                    zone_counts[label] = zone_counts.get(label, 0) + 1

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Display polygon zones and zone counts on a given image

        Args:
            result: PySDK result object to display (should be the same as used in analyze() method)
            image (np.ndarray): image to display on

        Returns:
            np.ndarray: annotated image
        """

        line_color = color_complement(result.overlay_color)
        text_color = deduce_text_color(line_color)

        # draw annotations
        for zi in range(len(self._polygons)):
            cv2.polylines(
                image, [self._polygons[zi]], True, line_color, result.overlay_line_width
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
                tuple(x + result.overlay_line_width for x in self._polygons[zi][0]),
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
        """
        Complete deferred initialization steps
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
                    self._triggering_positions,
                    self._bounding_box_scale,
                    self._iopa_threshold,
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
                self._zones[idx] = _PolygonZone(
                    self._polygons[idx],
                    self._wh,
                    self._triggering_positions,
                    self._bounding_box_scale,
                    self._iopa_threshold,
                )

        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, polygon in enumerate(self._polygons):
                if cv2.pointPolygonTest(polygon, (x, y), False) > 0:
                    zone_update()
                    self._gui_state["dragging"] = polygon
                    self._gui_state["offset"] = click_point
                    self._gui_state["update"] = idx
                    break

        if event == cv2.EVENT_RBUTTONDOWN:
            for idx, polygon in enumerate(self._polygons):
                for pt in polygon:
                    if np.linalg.norm(pt - click_point) < 10:
                        zone_update()
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
