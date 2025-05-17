#
# line_count.py: line crossing object counting support
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements classes for line crossing object counting
#
"""
Line Count Analyzer Module Overview
====================================

This module provides an analyzer (`LineCounter`) for detecting and counting objects as they cross
user-defined lines within video frames. It enables precise tracking of object movements across
virtual boundaries for applications like traffic monitoring and crowd management.

Key Features:
    - **Flexible Line Definitions**: Support for multiple lines defined by endpoints
    - **Directional Counting**: Track crossings in absolute (up/down/left/right) or relative directions
    - **Object Trail Tracking**: Use tracked object paths for accurate crossing detection
    - **Per-Class Counting**: Maintain separate counts for different object classes
    - **Visual Overlay**: Display crossing lines and count statistics on frames
    - **Interactive Editing**: Optional OpenCV mouse callback for line adjustment
    - **First Crossing Mode**: Option to count each object only once per line
    - **Trail Analysis**: Support for analyzing entire object trails or just latest segments

Typical Usage:
    1. Define lines to monitor within video frames
    2. Create a LineCounter instance with desired settings
    3. Process inference results through the analyzer chain
    4. Access crossing counts from result.line_counts
    5. Optionally visualize lines and counts using annotate method

Integration Notes:
    - Requires ObjectTracker analyzer upstream for trail data
    - Works with any detection results containing bounding boxes
    - Supports standard DeGirum PySDK result formats
    - Handles partial/missing detections gracefully

Key Classes:
    - `LineCounter`: Main analyzer class for counting line crossings
    - `SingleLineCounts`: Tracks directional counts for absolute frame directions
    - `LineCounts`: Extends SingleLineCounts with per-class counting
    - `SingleVectorCounts`: Tracks directional counts relative to line orientation
    - `VectorCounts`: Extends SingleVectorCounts with per-class counting

Configuration Options:
    - `lines`: List of line coordinates (x1, y1, x2, y2) to monitor
    - `anchor_point`: Bounding box point used for crossing detection
    - `whole_trail`: Use entire trail or just latest segment
    - `count_first_crossing`: Count each object once per line
    - `absolute_directions`: Use absolute or relative directions
    - `per_class_display`: Enable per-class counting
    - `show_overlay`: Enable visual annotations
    - `annotation_color`: Customize overlay appearance
    - `window_name`: Enable interactive line adjustment
"""
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
    """Holds counts of line crossings in four directions.

    This class records the number of objects that crossed a line in each cardinal direction
    relative to the frame: leftward, rightward, upward, and downward. It is typically used
    within a `LineCounter` result to represent the counts for one monitored line when counting
    with absolute directions.

    Attributes:
        left (int): Number of objects crossing the line moving leftward (e.g., from right to left).
        right (int): Number of objects crossing the line moving rightward (left to right).
        top (int): Number of objects crossing the line moving upward (from bottom toward top).
        bottom (int): Number of objects crossing the line moving downward (from top toward bottom).

    """

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
    """Extends SingleLineCounts to include per-class crossing counts.

    This class tracks line crossing counts with a breakdown by object class. In addition to the
    total counts for all objects (inherited attributes left, right, top, bottom), it maintains
    a dictionary of counts for each object class label. It is typically used by `LineCounter`
    when `per_class_display=True` to provide class-specific crossing statistics for each line.

    Attributes:
        left (int): Total number of objects crossing leftward (all classes combined).
        right (int): Total number of objects crossing rightward (all classes combined).
        top (int): Total number of objects crossing upward (all classes combined).
        bottom (int): Total number of objects crossing downward (all classes combined).
        for_class (Dict[str, SingleLineCounts]): Mapping from class label to a `SingleLineCounts`
            object for that class. Each entry holds the counts of crossings for that specific object class.

    """

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
    """Holds counts of line crossings relative to a line's orientation.

    This class is used for counting crossing events in the two opposite directions defined by a line
    (as opposed to absolute frame directions). It measures how many objects crossed from one side of
    the line to the other. Specifically, `right` represents crossings from the left side to the right
    side of the line (following the line's direction vector), and `left` represents crossings from the
    right side to the left side of the line. This is used by `LineCounter` when
    `absolute_directions=False` (relative direction mode).

    Attributes:
        right (int): Count of objects crossing from the line's left side to its right side.
        left (int): Count of objects crossing from the line's right side to its left side.
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
    """Extends SingleVectorCounts to include per-class crossing counts.

    This class maintains overall crossing counts for a line (relative to its orientation) and also
    tracks counts per object class. It inherits the total `left` and `right` counts (for all objects)
    from `SingleVectorCounts`, and adds a dictionary of per-class counts. It is used by `LineCounter` when
    `per_class_display=True` and `absolute_directions=False`.

    Attributes:
        left (int): Total number of objects crossing from the right side to the left side of the line (all classes).
        right (int): Total number of objects crossing from the left side to the right side of the line (all classes).
        for_class (Dict[str, SingleVectorCounts]): Mapping from class label to a `SingleVectorCounts`
            object for that class. Each entry contains the left/right counts for objects of that specific class.
    """

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
    """Counts objects crossing specified lines in a video stream.

    This analyzer processes tracked object trajectories to detect and tally crossing events for each
    predefined line. It monitors a list of user-defined lines and increments the appropriate count
    whenever an object's trail crosses a line, determining the direction of each crossing event.

    Key features:
        - Supports absolute (frame-axis) mode counting in four directions, or relative (line-oriented)
          mode counting in two directions.
        - Options to use an object's entire trail versus just the latest segment for crossing detection
          (`whole_trail`), and to count only the first crossing per object (`count_first_crossing`).
        - Can accumulate counts over multiple frames or reset counts each frame (`accumulate` flag).
        - Maintains counts for each line (as `LineCounts` in absolute mode or `VectorCounts` in relative mode)
          and can breakdown counts by object class (if `per_class_display=True`).
        - Provides an `annotate(image, result)` method to overlay the lines and current counts on video frames.
        - Supports interactive line adjustment via an OpenCV window (see the `window_attach()` method).

    After calling `analyze(result)` on a detection/tracking result, the `result` object is augmented with
    a new attribute `line_counts`. This attribute is a list of count objects (one per line) representing
    the crossing totals. Each element is either a `LineCounts` (for absolute directions) or `VectorCounts`
    (for relative directions) instance. If `per_class_display` is enabled, each count object also contains
    a `for_class` dictionary for per-class counts. Additionally, each detection entry in `result.results`
    receives a boolean list `cross_line` indicating which lines that object's trail has crossed (True/False
    for each monitored line).

    Note:
        This analyzer requires that object trajectories (trails) are available in the `result` (e.g.,
        provided by an `ObjectTracker`), since counting is based on each object's movement across frames.

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
        """
        Initialize a LineCounter with specified lines and counting options.

        Creates a new line counter instance that will track object crossings over the specified lines.
        The counter can operate in either absolute (frame-axis) or relative (line-oriented) counting modes,
        and supports various options for trail analysis and visualization.

        Args:
            lines (List[tuple]): List of line coordinates, each as (x1, y1, x2, y2).
            anchor_point (AnchorPoint, optional): Anchor point on bbox for trails. Default is BOTTOM_CENTER.
            whole_trail (bool, optional): Use entire trail or last segment only for intersection. Default True.
            count_first_crossing (bool, optional): Count only first crossing per trail if True. Default True.
            absolute_directions (bool, optional): Directions relative to image axes if True. Default False.
            accumulate (bool, optional): Accumulate counts over frames if True. Default True.
            per_class_display (bool, optional): Display counts per object class if True. Default False.
            show_overlay (bool, optional): Draw annotations if True. Default True.
            annotation_color (tuple, optional): RGB color for annotations. Default is complement of overlay color.
            annotation_line_width (int, optional): Thickness of annotation lines.
            window_name (str, optional): OpenCV window name to attach for interactive adjustment.
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
        Reset line crossing counts and crossing history.
        Clears all previously counted trails and counters.
        """
        self._counted_trails_list: List[set] = [set() for _ in self._lines]
        self._line_counts: List[Union[LineCounts, VectorCounts]] = [
            self._count_type() for _ in self._lines
        ]

    def analyze(self, result):
        """
        Analyzes object trails for line crossings and updates crossing counts.

        Checks every tracked trail in `result.trails` for intersections with all lines,
        computes crossing direction, updates counts, and adds `line_counts` attribute
        to the result.

        Adds a `cross_line` boolean list to each detected object's dictionary indicating
        which lines they crossed on this frame.

        Args:
            result (InferenceResults): Model result object containing trails and detection info.

        Returns:
            (None): This method modifies the input result object in-place.

        Raises:
            AttributeError: If result object is missing required attributes.
            TypeError: If result object is not of the expected type.
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
                # remove trails that are no longer active
                self._counted_trails_list[i] = (
                    self._counted_trails_list[i] & new_trails_list[i]
                )
                # new trails not previously counted
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
        Draws the defined lines and crossing counts on the image.

        Args:
            result (InferenceResults): Model result that contains updated `line_counts`.
            image (np.ndarray): BGR image to annotate.

        Returns:
            image (np.ndarray): Annotated image.
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
        """
        Attaches OpenCV window for interactive line adjustment.

        Installs mouse callbacks enabling line dragging.

        Args:
            win_name (str): Name of the OpenCV window.
        """
        self._win_name = win_name
        self._mouse_callback_installed = False

    def _lazy_init(self):
        """
        Perform deferred initialization such as installing the mouse callback
        if a window name has been set and the callback hasn't been installed yet.
        """
        if not self._mouse_callback_installed and self._win_name is not None:
            self._install_mouse_callback()

    def _line_to_vector(self, line):
        """
        Return vector defined by line segment.

        Args:
            line (list or np.ndarray): Two endpoints of a line [x1, y1, x2, y2].

        Returns:
            np.ndarray: Vector representing the direction of the line (x2 - x1, y2 - y1).
        """
        return np.array([line[2] - line[0], line[3] - line[1]])

    def _projection(self, a: np.ndarray, b: np.ndarray):
        """
        Return projection of vector b onto vector a.

        Args:
            a (np.ndarray): Base vector.
            b (np.ndarray): Vector to project.

        Returns:
            np.ndarray: Projection of b onto a.
        """
        return np.dot(a, b) * a / np.dot(a, a)

    @staticmethod
    def _mouse_callback(event: int, x: int, y: int, flags: int, self: Any):
        """
        Mouse event callback for interactive line editing.

        Supports:
        - Left-click and drag to move entire line.
        - Right-click and drag to move individual line endpoints.
        - Updates internal line state on changes.

        Args:
            event (int): OpenCV mouse event code.
            x (int): X coordinate of the mouse event.
            y (int): Y coordinate of the mouse event.
            flags (int): Additional event flags.
            self: Instance of LineCounter.
        """
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
        """
        Internal method to install the OpenCV mouse callback on the attached window.
        """
        if self._win_name is not None:
            try:
                cv2.setMouseCallback(self._win_name, LineCounter._mouse_callback, self)  # type: ignore[attr-defined]
                self._gui_state = {"dragging": None, "update": -1}
                self._mouse_callback_installed = True
            except Exception:
                pass  # ignore errors
