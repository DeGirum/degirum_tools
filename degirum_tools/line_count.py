#
# line_count.py: line crossing object counting support
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes for line crossing object counting
#

import numpy as np, cv2
from dataclasses import dataclass
from typing import List
from copy import deepcopy
from .ui_support import put_text, color_complement, deduce_text_color, CornerPosition
from .result_analyzer_base import ResultAnalyzerBase
from .math_support import intersect


@dataclass
class LineCounts:
    """Dataclass to hold line crossing counts"""

    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0


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

    def __init__(self, lines: List[tuple]):
        """Constructor

        Args:
            lines (list[tuple]): list of line coordinates;
            each list element is 4-element tuple of (x1,y1,x2,y2) line coordinates
        """

        self._lines = lines
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
        value is the number of occurrences of a trail crossing the corresponding line from the
        corresponding direction. For each trail crossing, two directions are updated:
        `left` vs `right`, and `top` vs `bottom`.

        Args:
            result: PySDK model result object, containing `trails` dictionary from ObjectTracker
        """

        if not hasattr(result, "trails") or len(result.trails) == 0:
            return

        active_trails = set(result.trails.keys())

        # remove old trails, which are not active anymore
        self._counted_trails = self._counted_trails & active_trails

        # obtain a set of new trails, which were not counted yet
        new_trails = active_trails - self._counted_trails

        for tid in new_trails:
            trail = result.trails[tid]
            if len(trail) > 1:
                trail_start = trail[0]
                trail_end = trail[-1]

                for line_count, line in zip(self._line_counts, self._lines):
                    if intersect(line[:2], line[2:], trail_start, trail_end):
                        if trail_start[0] > trail_end[0]:
                            line_count.left += 1
                        if trail_start[0] < trail_end[0]:
                            line_count.right += 1
                        if trail_start[1] < trail_end[1]:
                            line_count.top += 1
                        if trail_start[1] > trail_end[1]:
                            line_count.bottom += 1
                        self._counted_trails.add(tid)

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

        if hasattr(result, "line_counts"):
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
                    sep = " "
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
                    sep = "\n"
                    if max(line_start[0], line_end[0]) < img_center[0]:
                        cx = max(line_start[0], line_end[0]) + margin
                        corner = CornerPosition.TOP_LEFT
                    elif min(line_start[0], line_end[0]) > img_center[1]:
                        cx = min(line_start[0], line_end[0]) - margin
                        corner = CornerPosition.TOP_RIGHT
                    else:
                        cx = (line_start[0] + line_end[0]) // 2
                        corner = CornerPosition.TOP_LEFT

                put_text(
                    image,
                    f"Top={line_count.top}{sep}Bottom={line_count.bottom}{sep}Left={line_count.left}{sep}Right={line_count.right}",
                    (cx, cy),
                    corner_position=corner,
                    font_color=text_color,
                    bg_color=line_color,
                    font_scale=result.overlay_font_scale,
                )

        return image
