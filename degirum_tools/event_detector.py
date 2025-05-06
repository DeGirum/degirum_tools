#
# event_detector.py: event detector analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
# Implements analyzer class to detect various events by analyzing inference results
#

"""
Event Detector Analyzer
=======================

This module provides [`EventDetector`](#eventdetector), a **result-post-processing analyzer**
that turns results of analyzers ([zone counters](zone_count.md#zonecounter), [line counters](line_count.md#linecounter), etc.)
into high-level, **human-readable "events."**
Typical uses include:

* Raising a *"PersonInZone"* event when a person stays inside an ROI for *N* seconds.
* Firing a *"VehicleCountExceeded"* event whenever more than *K* cars are in view.
* Detecting complex temporal patterns—e.g. *"Person crosses entry line then exit
  line within 3s"*—by combining multiple analyzers upstream.

The analyzer is **data-driven**: behaviour is described by a YAML (or
`dict`) that follows the schema defined in `event_definition_schema`.
A single analyzer instance can therefore be re-configured without code changes.

**Integration pattern**

1. Attach auxiliary analyzers (e.g. [`ZoneCounter`](zone_count.md#zonecounter),
   [`LineCounter`](line_count.md#linecounter)) upstream so that the required metrics are present in
   each InferenceResults.
2. Construct [`EventDetector`](#eventdetector) with a YAML description (see *Example* below).
3. Attach it to a model or compound model via
   [`attach_analyzers`](compound_models.md#compoundmodelbase).
4. Downstream components can inspect `result.events_detected` or rely on
   [`EventNotifier`](notifier.md#eventnotifier) to convert events into user notifications.

Example
-------

```python
from degirum_tools.event_detector import EventDetector

event_yaml = \"\"\"
Trigger: PersonInZone
when:   ZoneCount
with:
    classes: [person]
    index:   0          # zone id
is greater than or equal to: 1
during: [5, seconds]    # look at a 5-second window
for at least: [80, percent]
\"\"\"

detector = EventDetector(event_yaml)
model.attach_analyzers(detector)
```

Key Concepts
------------

* **Metric** - scalar number extracted from `InferenceResults`
  (zone count, line count, object count, or a user-supplied custom metric).
* **Comparator** - how the metric is compared to a threshold
  (>, >=, <, <=, ==, !=).
* **Temporal window** - *"during"* clause defines sliding window length
  (in frames or seconds).
* **Quantifier** - *"for at least / at most"* clause defines how long within
  the window the condition must hold.

All timing logic is handled internally via a ring-buffer; no external state is
required.

"""

import numpy as np
import yaml, time, jsonschema
from .result_analyzer_base import ResultAnalyzerBase
from .ui_support import put_text, color_complement, deduce_text_color, CornerPosition
from .math_support import AnchorPoint, get_image_anchor_point
from collections import deque
from typing import Union, Optional

#
# Schema for event definition
#

# keys
Key_Trigger = "Trigger"
Key_When = "when"
Key_With = "with"
Key_During = "during"
Key_IsEqualTo = "is equal to"
Key_IsNotEqualTo = "is not equal to"
Key_IsGreaterThan = "is greater than"
Key_IsGreaterThanOrEqualTo = "is greater than or equal to"
Key_IsLessThan = "is less than"
Key_IsLessThanOrEqualTo = "is less than or equal to"
Key_ForAtLeast = "for at least"
Key_ForAtMost = "for at most"
Key_Classes = "classes"
Key_Index = "index"
Key_Directions = "directions"
Key_MinScore = "min score"
Key_Aggregation = "aggregation"

# units
Unit_Second = "second"
Unit_Seconds = "seconds"
Unit_Frame = "frame"
Unit_Frames = "frames"
Unit_Percent = "percent"


# directions
Direction_Left = "left"
Direction_Right = "right"
Direction_Top = "top"
Direction_Bottom = "bottom"

# metrics
Metric_ZoneCount = "ZoneCount"
Metric_LineCount = "LineCount"
Metric_ObjectCount = "ObjectCount"
Metric_Custom = "Custom"

# aggregate functions
Aggregate_Sum = "sum"
Aggregate_Max = "max"
Aggregate_Min = "min"
Aggregate_Mean = "mean"
Aggregate_Std = "std"

# schema YAML
event_definition_schema_text = f"""
type: object
additionalProperties: false
properties:
    {Key_Trigger}:
        type: string
        description: The name of event to raise
    {Key_When}:
        type: string
        enum: [{Metric_ZoneCount}, {Metric_LineCount}, {Metric_ObjectCount}, {Metric_Custom}]
        description: The name of the metric to evaluate
    {Key_With}:
        type: object
        additionalProperties: false
        properties:
            {Key_Classes}:
                type: array
                items:
                    type: string
                description: The class labels to count; if not specified, all classes are counted
            {Key_Index}:
                type: integer
                description: The location number (zone or line index) to count; if not specified, all locations are counted
            {Key_Directions}:
                type: array
                items:
                    type: string
                    enum: [{Direction_Left}, {Direction_Right}, {Direction_Top}, {Direction_Bottom}]
                description: The line intersection directions to count; if not specified, all directions are counted
            {Key_MinScore}:
                type: number
                description: The minimum score of the object to count
                minimum: 0
                maximum: 1
            {Key_Aggregation}:
                type: string
                enum: [{Aggregate_Sum}, {Aggregate_Max}, {Aggregate_Min}, {Aggregate_Mean}, {Aggregate_Std}]
    {Key_IsEqualTo}:
        type: number
        description: The value to compare against
    {Key_IsNotEqualTo}:
        type: number
        description: The value to compare against
    {Key_IsGreaterThan}:
        type: number
        description: The value to compare against
    {Key_IsGreaterThanOrEqualTo}:
        type: number
        description: The value to compare against
    {Key_IsLessThan}:
        type: number
        description: The value to compare against
    {Key_IsLessThanOrEqualTo}:
        type: number
        description: The value to compare against
    {Key_During}:
        type: array
        prefixItems:
            - type: number
            - enum: [{Unit_Seconds}, {Unit_Frames}, {Unit_Second}, {Unit_Frame}]
        items: false
        description: Duration to evaluate the metric
    {Key_ForAtLeast}:
        type: array
        prefixItems:
            - type: number
            - enum: [{Unit_Percent}, {Unit_Frames}, {Unit_Frame}]
        items: false
        description: Minimum duration the metric needs to be true in order to trigger event
    {Key_ForAtMost}:
        type: array
        prefixItems:
            - type: number
            - enum: [{Unit_Percent}, {Unit_Frames}, {Unit_Frame}]
        items: false
        description: Maximum duration the metric needs to be true in order to trigger event
required: [{Key_Trigger}, {Key_When}, {Key_During}]
oneOf:
    - required: [{Key_IsEqualTo}]
      type: object
    - required: [{Key_IsNotEqualTo}]
      type: object
    - required: [{Key_IsGreaterThan}]
      type: object
    - required: [{Key_IsGreaterThanOrEqualTo}]
      type: object
    - required: [{Key_IsLessThan}]
      type: object
    - required: [{Key_IsLessThanOrEqualTo}]
      type: object
"""

event_definition_schema = yaml.safe_load(event_definition_schema_text)

#
# Metric functions
# Function name should match the metric name defined in the schema
#


def ZoneCount(result, params):
    """
    Return object count inside a zone.

    Requires `result.zone_counts` supplied by [`ZoneCounter`](zone_count.md#zonecounter).

    Args:
        result (InferenceResults): The inference results object.
        params (dict): May contain:
            - `classes` (List[str]): Classes to include.
            - `index` (int): Zone index to count (None → all zones).
            - `aggregation` (str): Aggregation function name (e.g., 'sum', 'max').

    Returns:
        count (int or float): Aggregated count (sum, max, …) across the selected zone(s).
    """

    if not hasattr(result, "zone_counts"):
        raise AttributeError(
            "Zone counts are not available in the result: insert ZoneCounter analyzer in a chain"
        )

    index = None
    classes = None
    aggregate = Aggregate_Sum

    if params is not None:
        index = params.get(Key_Index)
        classes = params.get(Key_Classes)
        aggregate = params.get(Key_Aggregation, Aggregate_Sum)

    if index is None:
        zone_counts = result.zone_counts
    else:
        if index >= 0 and index < len(result.zone_counts):
            # select particular zone
            zone_counts = result.zone_counts[index : index + 1]
        else:
            raise ValueError(
                f"Zone index {index} is out of range [0, {len(result.zone_counts)})"
            )

    # count objects in the zone(s) belonging to the specified classes
    counts = [
        (
            sum(zone.values())
            if classes is None
            else sum(zone[key] for key in classes if key in zone)
        )
        for zone in zone_counts
    ]
    # return the aggregated value
    return getattr(np, aggregate)(counts)


def LineCount(result, params):
    """
    Return intersection count on a virtual line.

    Relies on `result.line_counts` provided by [`LineCounter`](line_count.md#linecounter).

    Args:
        result (InferenceResults): The inference results object.
        params (dict): Dict from the *with* clause; keys include:
            - `index` (int): Line index to count (None → all lines).
            - `classes` (List[str]): Classes to include (None → all classes).
            - `directions` (List[str]): Directions to include (None → all directions).

    Returns:
        count (int): Number of line intersections matching the filter.
    """
    if not hasattr(result, "line_counts"):
        raise AttributeError(
            "Line counts are not available in the result: insert LineCounter analyzer upstream."
        )

    index = None
    classes = None
    directions = None

    if params is not None:
        index = params.get(Key_Index)
        classes = params.get(Key_Classes)
        directions = params.get(Key_Directions)

    if not hasattr(result, "line_counts"):
        raise AttributeError(
            "Line counts are not available in the result: insert LineCounter analyzer in a chain"
        )

    if index is None:
        line_counts = result.line_counts
    else:
        if index >= 0 and index < len(result.line_counts):
            # select particular line
            line_counts = result.line_counts[index : index + 1]
        else:
            raise ValueError(
                f"Line index {index} is out of range [0, {len(result.line_counts)})"
            )

    # count line intersections belonging to the specified classes
    count = 0
    all_dirs = [Direction_Left, Direction_Right, Direction_Top, Direction_Bottom]
    for line in line_counts:
        if classes is not None:
            count += sum(
                sum(
                    (
                        getattr(line.for_class[key], dir)
                        if hasattr(line.for_class[key], dir)
                        else 0
                    )
                    for dir in (all_dirs if directions is None else directions)
                )
                for key in classes
                if key in line.for_class
            )
        else:
            count += sum(
                getattr(line, dir) if hasattr(line, dir) else 0
                for dir in (all_dirs if directions is None else directions)
            )

    return count


def ObjectCount(result, params):
    """
    Return plain object count filtered by class and score.

    Args:
        result (InferenceResults): The inference results object.
        params (dict): May contain:
            - `classes` (List[str]): Classes to include.
            - `min_score` (float): Minimum confidence score.

    Returns:
        count (int): Number of detections satisfying the filter.
    """

    classes = None
    min_score = None

    if params is not None:
        classes = params.get(Key_Classes)
        min_score = params.get(Key_MinScore)

    if classes is not None:
        count = sum(
            ("label" in obj and obj["label"] in classes)
            and (min_score is None or ("score" in obj and obj["score"] >= min_score))
            for obj in result.results
        )
    else:
        count = sum(
            min_score is None or ("score" in obj and obj["score"] >= min_score)
            for obj in result.results
        )
    return count


class EventDetector(ResultAnalyzerBase):
    """
    Detect high-level events from inference results.

    The detector evaluates a metric over a sliding window and checks whether it
    satisfies a comparison with a threshold for a required proportion of that window.
    When satisfied, the event name is added to `result.events_detected` (a set managed
    by this class).

    Attributes:
        key_events_detected (str): Name of the attribute where fired events are stored ("events_detected").
        _event_history (collections.deque): Ring buffer storing (timestamp, condition_met_bool) tuples.

    Notes:
        - The class is purely stateless from the outside; all state lives inside
          the analyzer instance.
        - Thread-safe as long as each instance is confined to a single inference
          thread.
    """

    key_events_detected = "events_detected"

    def __init__(
        self,
        event_description: Union[str, dict],
        *,
        show_overlay: bool = True,
        annotation_color: Optional[tuple] = None,
        annotation_font_scale: Optional[float] = None,
        annotation_pos: Union[AnchorPoint, tuple] = AnchorPoint.BOTTOM_LEFT,
    ):
        """
        Create a new detector from a YAML/dict definition.

        Args:
            event_description (str | dict): YAML string or parsed dict matching *event_definition_schema*.
            show_overlay (bool, optional): If True, annotate frames whenever the event fires. Defaults to True.
            annotation_color (tuple, optional): RGB background for the text label. Defaults to the complement of model overlay colour.
            annotation_font_scale (float, optional): OpenCV font scale. None uses the model's default.
            annotation_pos (AnchorPoint | tuple): Where to place the text label; either an `AnchorPoint` enum value or an `(x, y)` pixel tuple.
        """

        self._show_overlay = show_overlay
        self._annotation_color = annotation_color
        self._annotation_font_scale = annotation_font_scale
        self._annotation_pos = annotation_pos

        if isinstance(event_description, str):
            event_desc = yaml.safe_load(event_description)
        else:
            event_desc = event_description

        jsonschema.validate(instance=event_desc, schema=event_definition_schema)
        # from here we guarantee that all keys are present in the schema since it passed validation

        self._event_desc = event_desc
        self._event_name = event_desc[Key_Trigger]
        self._metric_name = event_desc[Key_When]
        self._metric_params = event_desc.get(Key_With)
        self._duration = event_desc[Key_During][0]
        duration_unit = event_desc[Key_During][1]

        self._op = ""
        self._threshold = 0.0
        for op in EventDetector.comparators:
            if op in event_desc:
                self._op = op
                self._threshold = event_desc[op]
                break

        if not self._op:
            raise ValueError("Comparison operator should be specified")

        if Key_ForAtLeast in event_desc:
            self._quantifier = Key_ForAtLeast
            self._quota = event_desc[Key_ForAtLeast][0]
            self._quota_unit = event_desc[Key_ForAtLeast][1]
        elif Key_ForAtMost in event_desc:
            self._quantifier = Key_ForAtMost
            self._quota = event_desc[Key_ForAtMost][0]
            self._quota_unit = event_desc[Key_ForAtMost][1]
        else:
            self._quantifier = Key_ForAtLeast
            self._quota = 100.0
            self._quota_unit = Unit_Percent

        if self._duration <= 0:
            raise ValueError("Duration should be greater than zero")

        self._event_history: deque = deque(
            maxlen=int(self._duration) if duration_unit in Unit_Frames else None
        )

    comparators = {
        Key_IsEqualTo: lambda a, b: a == b,
        Key_IsNotEqualTo: lambda a, b: a != b,
        Key_IsGreaterThan: lambda a, b: a > b,
        Key_IsGreaterThanOrEqualTo: lambda a, b: a >= b,
        Key_IsLessThan: lambda a, b: a < b,
        Key_IsLessThanOrEqualTo: lambda a, b: a <= b,
    }

    def analyze(self, result):
        """
        Evaluate metric and update `result.events_detected`.

        Steps
        -----
        1. Compute metric value via a global helper (e.g. [`ZoneCount`](zone_count.md)).
        2. Check comparator against threshold → **condition_met**.
        3. Append `(timestamp, condition_met)` to history deque.
        4. Drop stale entries if time-based window.
        5. Test quantifier (at least / at most) against quota.
        6. Add event name to `result.events_detected` when satisfied.

        Args:
            result (InferenceResults): Inference result to analyze (modified in-place).
        """

        # evaluate metric and compare with threshold
        metric_value = globals()[self._metric_name](result, self._metric_params)
        condition_met = EventDetector.comparators[self._op](
            metric_value, self._threshold
        )

        # add metric value to the history
        self._event_history.append((time.time(), condition_met))

        if self._event_history.maxlen is None:
            # remove old results
            time_now = time.time()
            while (
                self._event_history
                and self._event_history[0][0] < time_now - self._duration
            ):
                self._event_history.popleft()

        # check if the condition is met for the required duration
        true_cnt = sum(x[1] for x in self._event_history)
        quantity_thr = (
            self._quota * len(self._event_history) / 100.0
            if self._quota_unit == Unit_Percent
            else self._quota
        )
        event_is_detected = False
        if self._quantifier == Key_ForAtLeast:
            event_is_detected = true_cnt >= quantity_thr
        elif self._quantifier == Key_ForAtMost:
            event_is_detected = true_cnt <= quantity_thr

        # add detected event to the result object
        if not hasattr(result, self.key_events_detected):
            result.events_detected = set()
        if event_is_detected:
            result.events_detected.add(self._event_name)

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Overlay the event label onto the frame.

        The overlay is rendered only when:
        - `show_overlay` is True, and
        - this frame has the current event in `result.events_detected`.

        The label background colour defaults to the complement of the model's
        overlay colour, and text colour is auto-chosen for contrast.

        Args:
            result (InferenceResults): Same result instance passed to `analyze()`.
            image (np.ndarray): BGR image to annotate.

        Returns:
            np.ndarray: Annotated image (same instance for efficiency).
        """

        if (
            not self._show_overlay
            or not hasattr(result, self.key_events_detected)
            or not result.events_detected
            or self._event_name not in result.events_detected
        ):
            return image

        bg_color = (
            color_complement(result.overlay_color)
            if self._annotation_color is None
            else self._annotation_color
        )
        text_color = deduce_text_color(bg_color)

        if isinstance(self._annotation_pos, AnchorPoint):
            pos = get_image_anchor_point(
                image.shape[1], image.shape[0], self._annotation_pos
            )
        else:
            pos = self._annotation_pos

        return put_text(
            image,
            self._event_name,
            pos,
            font_color=text_color,
            bg_color=bg_color,
            font_scale=(
                result.overlay_font_scale
                if self._annotation_font_scale is None
                else self._annotation_font_scale
            ),
            corner_position=CornerPosition.AUTO,
        )
