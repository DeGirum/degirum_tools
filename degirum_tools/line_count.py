#
# line_count.py: line crossing object counting support
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes for line crossing object counting
#

import numpy as np, cv2
from typing import List, Dict, Optional, Union, Any, Type
from copy import deepcopy
from .ui_support import (
    put_text,
    color_complement,
    deduce_text_color,
    rgb_to_bgr,
    CornerPosition,
)
from .result_analyzer_base import ResultAnalyzerBase
from .math_support import intersect, get_anchor_coordinates, AnchorPoint


class SingleLineCounts:
    """Class to hold line crossing counts"""

    def __init__(self):
        self.left: int = 0
        self.right: int = 0
        self.top: int = 0
        self.bottom: int = 0

    def __eq__(self, other):
        if not isinstance(other, SingleLineCounts):
            return NotImplemented
        return (
            self.left == other.left
            and self.right == other.right
            and self.top == other.top
            and self.bottom == other.bottom
        )

    def __iadd__(self, other):
        if not isinstance(other, SingleLineCounts):
            return NotImplemented
        self.left += other.left
        self.right += other.right
        self.top += other.top
        self.bottom += other.bottom
        return self

    def to_dict(self):
        return {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
        }


class LineCounts(SingleLineCounts):
    """Class to hold total line crossing counts and counts for multiple classes"""

    def __init__(self):
        super().__init__()
        self.for_class: Dict[str, SingleLineCounts] = {}

    def to_dict(self):
        return {
            **super().to_dict(),
            "for_class": {
                class_name: class_count.to_dict()
                for class_name, class_count in self.for_class.items()
            },
        }


class SingleVectorCounts:
    """
    Class to hold vector crossing counts.

    In relation to a vector, "right" specifies crossing of a trail from the left side
    to the right side of the vector, and "left" specifies crossing of a trail from the
    right side to the left side of the vector.
    """

    def __init__(self):
        self.right: int = 0
        self.left: int = 0

    def __eq__(self, other):
        if not isinstance(other, SingleVectorCounts):
            return NotImplemented
        return self.right == other.right and self.left == other.left

    def __iadd__(self, other):
        if not isinstance(other, SingleVectorCounts):
            return NotImplemented
        self.right += other.right
        self.left += other.left
        return self

    def to_dict(self):
        return {
            "right": self.right,
            "left": self.left,
        }


class VectorCounts(SingleVectorCounts):
    """Class to hold total vector crossing counts and counts for multiple classes"""

    def __init__(self):
        super().__init__()
        self.for_class: Dict[str, SingleVectorCounts] = {}

    def to_dict(self):
        return {
            **super().to_dict(),
            "for_class": {
                class_name: class_count.to_dict()
                for class_name, class_count in self.for_class.items()
            },
        }


class LineCounter(ResultAnalyzerBase):
    """
    Class to count object tracking trails crossing lines.

    Analyzes the object detection `result` object passed to `analyze` method and, for each object trail
    in the `result.trails` dictionary, checks if this trail crosses any lines specified by the `lines`
    constructor parameter. If `absolute_directions` is set to True, an object's trail crossing a line
    is counted in two out of four directions relative to the frame (left-to-right vs right-to-left,
    and top-to-bottom vs bottom-to-top). If `absolute_directions` is set to False, an object's trail
    crossing a line is counted in one out of two directions relative to the line (left-to-right vs
    right-to-left).

    Adds `line_counts` list of either `LineCounts` or `VectorCounts` objects to the `result` object -
    one object per crossing line. The object contains the following attributes: `left`, `right`, `top`,
    and `bottom` in a `LineCounts` object, `left` and `right` in a `VectorCounts` object.
    Each attribute value is the number of occurrences of a trail crossing the corresponding line to the
    corresponding direction. For each trail crossing, the following directions are updated:
        `left` vs `right`, and `top` vs `bottom` - for `LineCounts`
        `left` (left side of vector) vs `right` (right side of vector) - for `VectorCounts`
    Additionally, if `per_class_display` constructor parameter is set to True, the per-class counts are
    stored in the `for_class` dictionary of the `LineCounts` or `VectorCounts` object.

    Also updates each element of `result.results[]` list by adding the "cross_line" key containing
    the list of boolean flags, one per line, which are True for lines which are crossed and False otherwise.

    This class works in conjunction with `ObjectTracker` class that should be used to track object trails.

    """

    def __init__(
        self,
        lines: List[tuple],
        anchor_point: AnchorPoint = AnchorPoint.BOTTOM_CENTER,
        *,
        whole_trail: bool = True,
        count_first_crossing: bool = True,
        absolute_directions: bool = False,
        accumulate: bool = True,
        per_class_display: bool = False,
        show_overlay: bool = True,
        annotation_color: Optional[tuple] = None,
        annotation_line_width: Optional[int] = None,
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
                when False, count all times when trail intersects a line
            absolute_directions (bool, optional): when True, direction of trail is calculated relative to coordinate
                system of image, and four directions are updated; when False, direction of trail is calculated
                relative to coordinate system defined by line that it intersects, and two directions are updated
            accumulate (bool, optional): when True, accumulate line counts; when False, store line counts only for current
                frame
            per_class_display (bool, optional): when True, display counts per class,
                otherwise display total counts
            show_overlay (bool, optional): if True, annotate image; if False, send through original image
            annotation_color (tuple, optional): Color to use for annotations, None to use complement to result overlay color
            annotation_line_width (int, optional): Line width to use for annotations, None to use result overlay line width
            window_name (str, optional): optional OpenCV window name to configure for interactive line adjustment
        """

        self._lines = [np.array(line).astype(int) for line in lines]
        self._line_vectors = [self._line_to_vector(line) for line in lines]
        self._anchor_point = anchor_point
        self._whole_trail = whole_trail
        self._count_first_crossing = count_first_crossing
        self._absolute_directions = absolute_directions
        self._count_type: Union[Type[LineCounts], Type[VectorCounts]] = (
            LineCounts if absolute_directions else VectorCounts
        )
        self._accumulate = accumulate
        self._mouse_callback_installed = False
        self._per_class_display = per_class_display
        self._show_overlay = show_overlay
        self._annotation_color = annotation_color
        self._annotation_line_width = annotation_line_width
        self._win_name = window_name
        self.reset()

    def reset(self):
        """
        Reset line counts
        """
        self._counted_trails_list: List[set] = [set() for _ in self._lines]
        self._line_counts: List[Union[LineCounts, VectorCounts]] = [
            self._count_type() for _ in self._lines
        ]

    def analyze(self, result):
        """
        Detect trails crossing the line.

        Adds `line_counts` list of dataclasses to the `result` object - one element per crossing line.
        The dataclass contains the following attributes:
            `left`, `right`, `top`, and `bottom` in a `LineCounts` object
            `left` and `right` in a `VectorCounts` object

        Each attribute value is the number of occurrences of a trail crossing the corresponding line to the
        corresponding direction. For each trail crossing, the following directions are updated:
            `left` vs `right`, and `top` vs `bottom` - for `LineCounts`
            `left` (left side of vector) vs `right` (right side of vector) - for `VectorCounts`

        Args:
            result: PySDK model result object, containing `trails` dictionary from ObjectTracker
        """

        self._lazy_init()

        if not hasattr(result, "trails") or len(result.trails) == 0:
            result.line_counts = deepcopy(self._line_counts)
            return

        lines_cnt = len(self._lines)

        new_trails = set(result.trails.keys())
        new_trails_list = [new_trails for _ in self._counted_trails_list]
        if self._count_first_crossing:
            for i in range(len(self._counted_trails_list)):
                # remove old trails, which are not active anymore
                self._counted_trails_list[i] = (
                    self._counted_trails_list[i] & new_trails_list[i]
                )
                # obtain a set of new trails, which were not counted yet
                new_trails_list[i] = new_trails_list[i] - self._counted_trails_list[i]

        def count_increment(trail_vector, line_vector):
            increment_counts: Optional[Union[SingleLineCounts, SingleVectorCounts]] = (
                None
            )
            if self._absolute_directions:
                increment_counts = SingleLineCounts()
                if trail_vector[0] < 0:
                    increment_counts.left += 1
                else:
                    increment_counts.right += 1
                if trail_vector[1] < 0:
                    increment_counts.top += 1
                else:
                    increment_counts.bottom += 1
            else:
                increment_counts = SingleVectorCounts()
                cross_product = np.cross(trail_vector, line_vector)
                if cross_product > 0:
                    increment_counts.left += 1
                elif cross_product < 0:
                    increment_counts.right += 1
                else:
                    if np.sign(trail_vector) == np.sign(line_vector):
                        increment_counts.left += 1
                    else:
                        increment_counts.right += 1
            return increment_counts

        if not self._accumulate:
            self._line_counts = [self._count_type() for _ in self._lines]

        crossed_tids: Dict[int, list] = (
            {}
        )  # map of track IDs to crossed line boolean flags

        for li, new_trails, counted_trails, total_count, line, line_vector in zip(
            range(lines_cnt),
            new_trails_list,
            self._counted_trails_list,
            self._line_counts,
            self._lines,
            self._line_vectors,
        ):
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
                    if intersect(line[:2], line[2:], trail_start, trail_end):
                        crossed_tids.setdefault(tid, [False] * lines_cnt)[li] = True
                        if self._count_first_crossing:
                            counted_trails.add(tid)
                        increment = count_increment(trail_vector, line_vector)
                        total_count += increment
                        if self._per_class_display:
                            class_count: Optional[
                                Union[SingleLineCounts, SingleVectorCounts]
                            ] = None
                            if isinstance(total_count, LineCounts):
                                class_count = total_count.for_class.setdefault(
                                    result.trail_classes[tid], SingleLineCounts()
                                )
                            elif isinstance(total_count, VectorCounts):
                                class_count = total_count.for_class.setdefault(
                                    result.trail_classes[tid], SingleVectorCounts()
                                )
                            class_count += increment

        result.line_counts = deepcopy(self._line_counts)

        for obj in result.results:
            tid = obj.get("track_id")
            if tid is not None:
                flags = crossed_tids.get(tid)
                if flags is not None:
                    obj["cross_line"] = flags
                    continue

            obj["cross_line"] = [False] * lines_cnt

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Display crossing lines and line crossing counters on a given image

        Args:
            result: PySDK result object to display (should be the same as used in analyze() method)
            image (np.ndarray): image to display on

        Returns:
            np.ndarray: annotated image
        """

        if not self._show_overlay or not hasattr(result, "line_counts"):
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

        margin = 3
        img_center = (image.shape[1] // 2, image.shape[0] // 2)

        for line_count, line in zip(result.line_counts, self._lines):
            line_start = tuple(line[:2])
            line_end = tuple(line[2:])

            if self._absolute_directions:
                cv2.line(
                    image,
                    line_start,
                    line_end,
                    rgb_to_bgr(line_color),
                    line_width,
                )
            else:
                cv2.arrowedLine(
                    image,
                    line_start,
                    line_end,
                    rgb_to_bgr(line_color),
                    line_width,
                    tipLength=0.05,
                )

            mostly_horizontal = abs(line_start[0] - line_end[0]) > abs(
                line_start[1] - line_end[1]
            )

            # compute coordinate where to put text
            if self._absolute_directions:
                if mostly_horizontal:
                    cx = line_start[0] + margin
                    if line_start[1] <= img_center[1]:
                        cy = line_start[1] + margin
                        corner = CornerPosition.TOP_LEFT
                    elif line_start[1] > img_center[1]:
                        cy = line_start[1] - margin
                        corner = CornerPosition.BOTTOM_LEFT
                else:
                    cy = line_start[1] + margin
                    if line_start[0] <= img_center[0]:
                        cx = line_start[0] + margin
                        corner = CornerPosition.TOP_LEFT
                    elif line_start[0] > img_center[1]:
                        cx = line_start[0] - margin
                        corner = CornerPosition.TOP_RIGHT

                def line_count_str(
                    lc: SingleLineCounts,
                    prefix: str = "",
                ) -> str:
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
            else:
                if mostly_horizontal:
                    cx_left = cx_right = line_start[0] + margin
                    if line_start[0] <= line_end[0]:
                        cy_right = line_start[1] + margin
                        cy_left = line_start[1] - margin
                        if line_start[1] < line_end[1]:
                            corner_right = CornerPosition.TOP_RIGHT
                            corner_left = CornerPosition.BOTTOM_LEFT
                        elif line_start[1] > line_end[1]:
                            corner_right = CornerPosition.TOP_LEFT
                            corner_left = CornerPosition.BOTTOM_RIGHT
                        else:
                            corner_right = CornerPosition.TOP_LEFT
                            corner_left = CornerPosition.BOTTOM_LEFT
                    elif line_start[0] > line_end[0]:
                        cy_right = line_start[1] - margin
                        cy_left = line_start[1] + margin
                        if line_start[1] < line_end[1]:
                            corner_right = CornerPosition.BOTTOM_RIGHT
                            corner_left = CornerPosition.TOP_LEFT
                        elif line_start[1] > line_end[1]:
                            corner_right = CornerPosition.BOTTOM_LEFT
                            corner_left = CornerPosition.TOP_RIGHT
                        else:
                            corner_right = CornerPosition.BOTTOM_LEFT
                            corner_left = CornerPosition.TOP_LEFT
                else:
                    cy_left = cy_right = line_start[1] + margin
                    if line_start[1] <= line_end[1]:
                        cx_right = line_start[0] - margin
                        cx_left = line_start[0] + margin
                        if line_start[0] < line_end[0]:
                            corner_right = CornerPosition.TOP_RIGHT
                            corner_left = CornerPosition.BOTTOM_LEFT
                        elif line_start[0] > line_end[0]:
                            corner_right = CornerPosition.BOTTOM_RIGHT
                            corner_left = CornerPosition.TOP_LEFT
                        else:
                            corner_right = CornerPosition.TOP_RIGHT
                            corner_left = CornerPosition.TOP_LEFT
                    elif line_start[1] > line_end[1]:
                        cx_right = line_start[0] + margin
                        cx_left = line_start[0] - margin
                        if line_start[0] < line_end[0]:
                            corner_right = CornerPosition.TOP_LEFT
                            corner_left = CornerPosition.BOTTOM_RIGHT
                        elif line_start[0] > line_end[0]:
                            corner_right = CornerPosition.BOTTOM_LEFT
                            corner_left = CornerPosition.TOP_RIGHT
                        else:
                            corner_right = CornerPosition.BOTTOM_LEFT
                            corner_left = CornerPosition.BOTTOM_RIGHT

                def vector_count_str(
                    lc: SingleVectorCounts, prefix: str = "", right: bool = True
                ) -> str:
                    return f"{prefix}{lc.right if right else lc.left}"

                capt_right = "right\n"
                capt_left = "left\n"
                if self._per_class_display:
                    capt_right += "\n".join(
                        [
                            vector_count_str(class_count, f"{class_name}: ", True)
                            for class_name, class_count in line_count.for_class.items()
                        ]
                        + [vector_count_str(line_count, "Total: ", True)]
                    )
                    capt_left += "\n".join(
                        [
                            vector_count_str(class_count, f"{class_name}: ", False)
                            for class_name, class_count in line_count.for_class.items()
                        ]
                        + [vector_count_str(line_count, "Total: ", False)]
                    )
                else:
                    capt_right += vector_count_str(line_count, right=True)
                    capt_left += vector_count_str(line_count, right=False)

                put_text(
                    image,
                    capt_right,
                    (cx_right, cy_right),
                    corner_position=corner_right,
                    font_color=text_color,
                    bg_color=line_color,
                    font_scale=result.overlay_font_scale,
                )
                put_text(
                    image,
                    capt_left,
                    (cx_left, cy_left),
                    corner_position=corner_left,
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
                new_vector = self._line_to_vector(self._lines[idx])
                if not np.array_equal(new_vector, self._line_vectors[idx]):
                    self._line_vectors[idx] = new_vector
                    self.reset()

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
                    self._gui_state["dragging"] = line
                    self._gui_state["offset"] = click_point
                    self._gui_state["update"] = idx
                    break

        if event == cv2.EVENT_RBUTTONDOWN:
            for idx, line in enumerate(self._lines):
                for i in range(0, len(line), 2):
                    if np.linalg.norm(line[i : i + 2] - click_point) < 10:
                        self._gui_state["dragging"] = line[i : i + 2]
                        self._gui_state["offset"] = click_point
                        self._gui_state["update"] = idx
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._gui_state["dragging"] is not None:
                delta = click_point - self._gui_state["offset"]
                reshaped_view = self._gui_state["dragging"].reshape(-1, len(delta))
                reshaped_view += delta
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
