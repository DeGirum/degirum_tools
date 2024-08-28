#
# line_count.py: line crossing object counting support
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes for line crossing object counting
#

import numpy as np, cv2
from typing import List, Dict, Optional, Any
from copy import deepcopy
from .ui_support import put_text, color_complement, deduce_text_color, CornerPosition
from .result_analyzer_base import ResultAnalyzerBase
from .math_support import intersect, get_anchor_coordinates, AnchorPoint


class SingleLineCounts:
    """Class to hold line crossing counts"""

    def __init__(self):
        self.left: int = 0
        self.right: int = 0
        self.top: int = 0
        self.bottom: int = 0

    def __iadd__(self, other):
        if not isinstance(other, SingleLineCounts):
            return NotImplemented
        self.left += other.left
        self.right += other.right
        self.top += other.top
        self.bottom += other.bottom
        return self


class LineCounts(SingleLineCounts):
    """Class to hold total line crossing counts and counts for multiple classes"""

    def __init__(self):
        super().__init__()
        self.for_class: Dict[str, SingleLineCounts] = {}

    def __iadd__(self, other):
        if not isinstance(other, SingleLineCounts):
            return NotImplemented
        self.left += other.left
        self.right += other.right
        self.top += other.top
        self.bottom += other.bottom
        return self


class LineCounter(ResultAnalyzerBase):
    """
    Class to count object tracking trails crossing lines.

    Analyzes the object detection `result` object passed to `analyze` method and, for each object trail
    in the `result.trails` dictionary, checks if this trail crosses any lines specified by the `lines`
    constructor parameter. If the trail crosses the line, the corresponding object is counted in
    two out of four directions: left-to-right vs right-to-left, and top-to-bottom vs bottom-to-top.

    Adds `line_counts` list of dataclasses to the `result` object - one element per crossing line.
    Each dataclass contains four attributes: `left`, `right`, `top`, and `bottom`. Each attribute
    value is the number of occurrences of a trail crossing the corresponding line from the
    corresponding direction. For each trail crossing, two directions are updated:
    `left` vs `right`, and `top` vs `bottom`.

    This class works in conjunction with `ObjectTracker` class that should be used to track object trails.

    """

    def __init__(
        self,
        lines: List[tuple],
        anchor_point: AnchorPoint = AnchorPoint.BOTTOM_CENTER,
        *,
        whole_trail: bool = True,
        count_first_crossing: bool = True,
        absolute_directions: bool = True,
        accumulate: bool = True,
        per_class_display: bool = False,
        show_overlay: bool = True,
        window_name: Optional[str] = None,
    ):
        """Constructor

        Args:
            lines (list[tuple]): list of line coordinates;
                each list element is 4-element tuple of (x1,y1,x2,y2) line coordinates
            anchor_point (AnchorPoint, optional): bbox anchor point to be used for tracing object trails
            whole_trail (bool, optional): when True, last and first points of trail are used to determine if
                trail intersects a line; when False, last and second-to-last points of trail are used
            count_first_crossing (bool, optional): when True, count only first time a trail intersects a line;
                when False, count all times when trail interstects a line
            absolute_directions (bool, optional): when True, direction of trail is calculated relative to coordinate
                system of image; when False, direction of trail is calculated relative to coordinate system defined
                by line that it intersects
            accumulate (bool, optional): when True, accumulate line counts; when False, store line counts only for current
                frame
            per_class_display (bool, optional): when True, display counts per class,
                otherwise display total counts
            show_overlay (bool, optional): when True, apply annotations; when False, annotate() returns original image
            window_name (str, optional): optional OpenCV window name to configure for interactive line adjustment

        """

        self._lines = lines
        self._line_vectors = [self._line_to_vector(line) for line in lines]
        self._anchor_point = anchor_point
        self._whole_trail = whole_trail
        self._count_first_crossing = count_first_crossing
        self._absolute_directions = absolute_directions
        self._accumulate = accumulate
        self._win_name = window_name
        self._mouse_callback_installed = False
        self._per_class_display = per_class_display
        self._show_overlay = show_overlay
        self.reset()

    def reset(self):
        """
        Reset line counts
        """
        self._counted_trails = set()
        self._line_counts = [LineCounts() for _ in self._lines]

    def analyze(self, result):
        """
        Detect trails crossing the line.

        Adds `line_counts` list of dataclasses to the `result` object - one element per crossing line.
        Each dataclass contains four attributes: `left`, `right`, `top`, and `bottom`. Each attribute
        value is the number of occurrences of a trail crossing the corresponding line to the
        corresponding direction. For each trail crossing, two directions are updated:
        `left` vs `right`, and `top` vs `bottom`.

        Args:
            result: PySDK model result object, containing `trails` dictionary from ObjectTracker
        """

        self._lazy_init()

        if not hasattr(result, "trails") or len(result.trails) == 0:
            return

        active_trails = set(result.trails.keys())

        # remove old trails, which are not active anymore (if self._count_first_crossing = True)
        if self._count_first_crossing:
            self._counted_trails = self._counted_trails & active_trails

        # obtain a set of new trails, which were not counted yet (if self._count_first_crossing = True)
        new_trails = active_trails
        if self._count_first_crossing:
            new_trails = new_trails - self._counted_trails

        def count_increment(trail_vector, line_vector):
            counts = SingleLineCounts()
            if self._absolute_directions:
                if trail_vector[0] < 0:
                    counts.left += 1
                else:
                    counts.right += 1
                if trail_vector[1] < 0:
                    counts.top += 1
                else:
                    counts.bottom += 1
            else:
                cross_product = np.cross(trail_vector, line_vector)
                if cross_product > 0:
                    counts.left += 1
                elif cross_product < 0:
                    counts.right += 1
                else:
                    if np.sign(trail_vector) == np.sign(line_vector):
                        counts.left += 1
                    else:
                        counts.right += 1

                trail_onto_line_projection = self._projection(line_vector, trail_vector)
                trail_onto_line_projection_sign = np.sign(trail_onto_line_projection)
                if np.all(trail_onto_line_projection_sign == np.sign(line_vector)):
                    counts.top += 1
                else:
                    if not np.any(trail_onto_line_projection_sign):
                        if cross_product > 0:
                            counts.bottom += 1
                        else:
                            counts.top += 1
                    else:
                        counts.bottom += 1
            return counts

        if not self._accumulate:
            self._line_counts = [LineCounts() for _ in self._lines]

        for tid in new_trails:
            trail = get_anchor_coordinates(
                np.array(result.trails[tid]), self._anchor_point
            )
            if len(trail) > 1:
                trail_start = trail[0] if self._whole_trail else trail[-2]
                trail_end = trail[-1]
                trail_vector = self._line_to_vector(
                    trail_start.tolist() + trail_end.tolist()
                )

                for total_count, line, line_vector in zip(
                    self._line_counts, self._lines, self._line_vectors
                ):
                    if intersect(line[:2], line[2:], trail_start, trail_end):
                        if self._count_first_crossing:
                            self._counted_trails.add(tid)
                        increment = count_increment(trail_vector, line_vector)
                        total_count += increment
                        if self._per_class_display:
                            class_count = total_count.for_class.setdefault(
                                result.trail_classes[tid], SingleLineCounts()
                            )
                            class_count += increment

        result.line_counts = deepcopy(self._line_counts)

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Display crossing lines and line crossing counters on a given image

        Args:
            result: PySDK result object to display (should be the same as used in analyze() method)
            image (np.ndarray): image to display on

        Returns:
            np.ndarray: annotated image
        """

        if hasattr(result, "line_counts") and self._show_overlay:
            line_color = color_complement(result.overlay_color)
            text_color = deduce_text_color(line_color)
            margin = 3
            img_center = (image.shape[1] // 2, image.shape[0] // 2)

            for line_count, line in zip(result.line_counts, self._lines):
                line_start = line[:2]
                line_end = line[2:]

                cv2.line(
                    image,
                    line_start,
                    line_end,
                    line_color,
                    result.overlay_line_width,
                )

                mostly_horizontal = abs(line_start[0] - line_end[0]) > abs(
                    line_start[1] - line_end[1]
                )

                # compute coordinate where to put text
                if mostly_horizontal:
                    cx = min(line_start[0], line_end[0]) + margin
                    if max(line_start[1], line_end[1]) < img_center[1]:
                        cy = max(line_start[1], line_end[1]) + margin
                        corner = CornerPosition.TOP_LEFT
                    elif min(line_start[1], line_end[1]) > img_center[1]:
                        cy = min(line_start[1], line_end[1]) - margin
                        corner = CornerPosition.BOTTOM_LEFT
                    else:
                        cy = (line_start[1] + line_end[1]) // 2
                        corner = CornerPosition.TOP_LEFT
                else:
                    cy = min(line_start[1], line_end[1]) + margin
                    if max(line_start[0], line_end[0]) < img_center[0]:
                        cx = max(line_start[0], line_end[0]) + margin
                        corner = CornerPosition.TOP_LEFT
                    elif min(line_start[0], line_end[0]) > img_center[1]:
                        cx = min(line_start[0], line_end[0]) - margin
                        corner = CornerPosition.TOP_RIGHT
                    else:
                        cx = (line_start[0] + line_end[0]) // 2
                        corner = CornerPosition.TOP_LEFT

                def line_count_str(lc: SingleLineCounts, prefix: str = "") -> str:
                    return (
                        f"{prefix}^({lc.top}) v({lc.bottom}) <({lc.left}) >({lc.right})"
                    )

                if self._per_class_display:
                    capt = "\n".join(
                        [
                            line_count_str(class_count, f"{class_name}: ")
                            for class_name, class_count in line_count.for_class.items()
                        ]
                        + [line_count_str(line_count, "Total: ")]
                    )
                else:
                    capt = line_count_str(line_count)

                put_text(
                    image,
                    capt,
                    (cx, cy),
                    corner_position=corner,
                    font_color=text_color,
                    bg_color=line_color,
                    font_scale=result.overlay_font_scale,
                )

        return image

    def window_attach(self, win_name: str):
        """Attach OpenCV window for interactive line adjustment by installing mouse callback

        Args:
            win_name (str): OpenCV window name to attach to
        """

        self._win_name = win_name
        self._mouse_callback_installed = False

    def _lazy_init(self):
        """
        Complete deferred initialization steps
            - install mouse callback
        """
        if not self._mouse_callback_installed and self._win_name is not None:
            self._install_mouse_callback()

    def _line_to_vector(self, line):
        """
        Return vector defined by line segment.
        """
        return np.array([line[2] - line[0], line[3] - line[1]])

    def _projection(self, a: np.ndarray, b: np.ndarray):
        """
        Return projection of vector b onto vector a.
        """
        return np.dot(a, b) * a / np.dot(a, a)

    @staticmethod
    def _mouse_callback(event: int, x: int, y: int, flags: int, self: Any):
        """Mouse callback for OpenCV window for interactive line operations"""

        click_point = np.array((x, y))

        def line_update():
            idx = self._gui_state["update"]
            if idx >= 0:
                self._line_vectors[idx] = self._line_to_vector(self._lines[idx])

        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, line in enumerate(self._lines):
                line_start_to_point_vector = click_point - line[:2]
                line_vector = self._line_vectors[idx]
                if (
                    np.linalg.norm(
                        line_start_to_point_vector
                        - self._projection(line_vector, line_start_to_point_vector)
                    )
                    < 10
                ):
                    line_update()
                    self._gui_state["dragging"] = line
                    self._gui_state["offset"] = click_point
                    self._gui_state["update"] = idx
                    break

        if event == cv2.EVENT_RBUTTONDOWN:
            for idx, line in enumerate(self._lines):
                for pt in [line[:2], line[2:]]:
                    if np.linalg.norm(pt - click_point) < 10:
                        line_update()
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
            line_update()
            self._gui_state["update"] = -1

    def _install_mouse_callback(self):
        if self._win_name is not None:
            try:
                cv2.setMouseCallback(self._win_name, LineCounter._mouse_callback, self)  # type: ignore[attr-defined]
                self._gui_state = {"dragging": None, "update": -1}
                self._mouse_callback_installed = True
            except Exception:
                pass  # ignore errors
