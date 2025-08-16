#
# event_detector.py: event detector analyzer
#
# Copyright DeGirum Corporation 2025
# All rights reserved
# Implements analyzer class to detect various events by analyzing inference results
#

"""
Event Detector Analyzer Module Overview
====================================

This module provides an analyzer (`EventDetector`) for converting analyzer outputs into high-level,
human-readable events. It enables detection of complex temporal patterns and conditions based on
metrics from other analyzers like zone counters and line counters.

Key Features:
    - **Metric-Based Events**: Convert analyzer metrics into meaningful events
    - **Temporal Patterns**: Detect conditions that must hold for specific durations
    - **Complex Conditions**: Combine multiple metrics with logical operators
    - **Data-Driven**: Configure events using YAML or dictionary definitions
    - **Ring Buffer**: Internal state management for temporal conditions
    - **Integration Support**: Works with any analyzer that produces metrics
    - **Schema Validation**: Ensures event definitions match required format

Typical Usage:
    1. Configure auxiliary analyzers (e.g., ZoneCounter, LineCounter) for required metrics
    2. Create an EventDetector instance with event definitions
    3. Attach it to a model or compound model
    4. Access detected events via result.events_detected
    5. Use EventNotifier for event-based notifications

Integration Notes:
    - Requires metrics from other analyzers to be present in results
    - Event definitions must match the event_definition_schema
    - Supports both YAML and dictionary-based configuration
    - Events are stored in result.events_detected for downstream use

Key Classes:
    - `EventDetector`: Main analyzer class for detecting events
    - `EventDefinitionSchema`: Schema for validating event definitions

Configuration Options:
    - `event_definitions`: YAML file or dictionary containing event definitions
    - `metrics`: Dictionary mapping metric names to their sources
    - `temporal_window`: Default duration for evaluating conditions
    - `quantifier`: Default proportion/duration for event triggers
    - `comparator`: Default operator for comparing metrics to thresholds
"""

import numpy as np
import yaml, time, jsonschema
from .result_analyzer_base import ResultAnalyzerBase
from .ui_support import put_text, color_complement, deduce_text_color, CornerPosition
from .math_support import AnchorPoint, get_image_anchor_point
from collections import deque
from typing import Union, Optional, Callable

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
Metric_Custom = "CustomMetric"

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
        additionalProperties: true
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
    """Computes the number of detected objects inside a specified zone or zones.

    Requires that `result.zone_counts` is present (produced by a `ZoneCounter` analyzer). This function can filter the count by object class and aggregate counts across multiple zones if needed.

    Args:
        result (InferenceResults): Inference results object containing zone count data.
        params (dict, optional): Additional parameters to filter/aggregate the count.
            classes (List[str]): Class labels to include. If None, all classes are counted.
            index (int): Zone index to count. If None, all zones are included.
            aggregation (str): Aggregation function to apply across zones. One of 'sum', 'max', 'min', 'mean', 'std'. Defaults to 'sum'.

    Returns:
        Union[int, float]: Total count of matching objects in the selected zone(s).

    Raises:
        AttributeError: If `result.zone_counts` is missing (no ZoneCounter applied upstream).
        ValueError: If a specified zone index is out of range for the available zones.
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
    """Computes the number of object crossings on a specified line (or across all lines).

    Relies on a `result.line_counts` attribute (produced by a `LineCounter` analyzer). This function can filter count by object class and crossing direction for fine-grained event definitions.

    Args:
        result (InferenceResults): Inference results object containing line crossing counts.
        params (dict, optional): Filter parameters from the event's "with" clause.
            index (int): Line index to count. If None, all lines are considered.
            classes (List[str]): Class labels to include. If None, all detected classes are counted.
            directions (List[str]): Directions of crossing to include. One of 'left', 'right', 'top', 'bottom'. If None, all directions are counted.

    Returns:
        count (int): Number of line-crossing events that match the specified filters.

    Raises:
        AttributeError: If `result.line_counts` is missing (no LineCounter applied upstream).
        ValueError: If a specified line index is out of range for the available lines.
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
    """Counts the detected objects in the result, with optional class and score filtering.

    This metric does not require any auxiliary analyzer; it simply tallies detections, optionally constrained by object class and minimum confidence score.

    Args:
        result (InferenceResults): The inference results object containing detections.
        params (dict, optional): Filter parameters.
            classes (List[str]): Class labels to include. If None, all detected classes are counted.
            min_score (float): Minimum confidence score required for counting. If None, no score threshold is applied.

    Returns:
        count (int): Number of detected objects that meet the specified class and score criteria.
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
    """Analyzes inference results over time to detect high-level events based on metric conditions.

    This analyzer monitors a chosen metric (e.g., ZoneCount, LineCount, or ObjectCount) over a sliding time window and triggers an event when a specified condition is satisfied. The condition consists of a comparison of the metric value against a threshold, and a temporal requirement that the condition holds for a certain duration or proportion of the window. When the condition is met, the event name is added to the `events_detected` set in the result.

    For example, you can detect an event "PersonInZone" when the count of persons in a region (`ZoneCount`) remains above 0 for N seconds, or a "VehicleCountExceeded" event when a line-crossing count exceeds a threshold within a frame. Multiple `EventDetector` instances can be attached to the same model to detect different events in parallel.

    Attributes:
        key_events_detected (str): Name of the result attribute that stores detected event names (defaults to "events_detected").
        comparators (Dict[str, Callable[[float, float], bool]]): Mapping of comparator keywords (e.g., "is greater than") to the corresponding comparison functions used internally.

    Note:
        Ensure that required analyzers (such as `ZoneCounter` or `LineCounter`) are attached before this detector so that the necessary metric data is present in `result`. The `EventDetector` maintains all state internally (using a ring buffer for timing) and is therefore stateless to callers. It is safe to use in a single-threaded inference pipeline (thread-safe per inference thread).
    """

    key_events_detected = "events_detected"

    def __init__(
        self,
        event_description: Union[str, dict],
        *,
        custom_metric: Optional[Callable] = None,
        show_overlay: bool = True,
        annotation_color: Optional[tuple] = None,
        annotation_font_scale: Optional[float] = None,
        annotation_pos: Union[AnchorPoint, tuple] = AnchorPoint.BOTTOM_LEFT,
    ):
        """Initializes an EventDetector with a given event description and overlay settings.

        The `event_description` defines the event's trigger name, metric, comparison, and timing requirements.
        It can be provided as a YAML string or an equivalent dictionary and must conform to the expected schema (`event_definition_schema`).

        The description includes these key components:
        - Trigger: Name of the event to detect
        - when: Metric to evaluate (one of "ZoneCount", "LineCount", or "ObjectCount")
        - Comparator: A comparison operator (e.g., "is greater than") with a threshold value
        - during: Duration of the sliding window as `[value, unit]` (unit can be "frames" or "seconds")
        - for at least / for at most (optional): Required portion of the window that the condition must hold true to trigger the event

        **Event Definition Schema (YAML)**:
        ```yaml
        type: object
        additionalProperties: false
        properties:
            Trigger:
                type: string
                description: The name of event to raise
            when:
                type: string
                enum: [ZoneCount, LineCount, ObjectCount, CustomMetric]
                description: The name of the metric to evaluate
            with:
                type: object
                additionalProperties: false
                properties:
                    classes:
                        type: array
                        items:
                            type: string
                        description: The class labels to count; if not specified, all classes are counted
                    index:
                        type: integer
                        description: The location number (zone or line index) to count; if not specified, all locations are counted
                    directions:
                        type: array
                        items:
                            type: string
                            enum: [left, right, top, bottom]
                        description: The line intersection directions to count; if not specified, all directions are counted
                    min score:
                        type: number
                        description: The minimum score of the object to count
                        minimum: 0
                        maximum: 1
                    aggregation:
                        type: string
                        enum: [sum, max, min, mean, std]
            is equal to:
                type: number
                description: The value to compare against
            is not equal to:
                type: number
                description: The value to compare against
            is greater than:
                type: number
                description: The value to compare against
            is greater than or equal to:
                type: number
                description: The value to compare against
            is less than:
                type: number
                description: The value to compare against
            is less than or equal to:
                type: number
                description: The value to compare against
            during:
                type: array
                prefixItems:
                    - type: number
                    - enum: [seconds, frames, second, frame]
                items: false
                description: Duration to evaluate the metric
            for at least:
                type: array
                prefixItems:
                    - type: number
                    - enum: [percent, frames, frame]
                items: false
                description: Minimum duration the metric must hold true to trigger the event
            for at most:
                type: array
                prefixItems:
                    - type: number
                    - enum: [percent, frames, frame]
                items: false
                description: Maximum duration the metric can hold true for the event to trigger
        required: [Trigger, when, during]
        oneOf:
            - required: [is equal to]
              type: object
            - required: [is not equal to]
              type: object
            - required: [is greater than]
              type: object
            - required: [is greater than or equal to]
              type: object
            - required: [is less than]
              type: object
            - required: [is less than or equal to]
              type: object
        ```

        Args:
            event_description (Union[str, dict]): YAML string or dictionary defining the event conditions (must match the schema above).
            custom_metric (Callable, optional): Custom metric function. Must be provided, if `when` key in `event_description` is set to "CustomMetric". The function accepts inference result and parameters dict of "with" clause and returns a numeric value.
            show_overlay (bool, optional): Whether to draw a label on the frame when the event fires. Defaults to True.
            annotation_color (tuple, optional): RGB color for the label background. If None, a contrasting color is auto-chosen. Defaults to None.
            annotation_font_scale (float, optional): Font scale for the overlay text. If None, uses a default scale. Defaults to None.
            annotation_pos (Union[AnchorPoint, tuple], optional): Position for the overlay label (an `AnchorPoint` or (x,y) coordinate). Defaults to `AnchorPoint.BOTTOM_LEFT`.

        Raises:
            jsonschema.ValidationError: If `event_description` does not conform to the required schema for events.
            ValueError: If no comparison operator is specified in the event description.
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

        self._custom_metric: Optional[Callable] = None
        if self._metric_name == Metric_Custom:
            if custom_metric is None or not callable(custom_metric):
                raise ValueError(
                    "Custom metric function must be provided when 'when' is set to 'CustomMetric'"
                )
            self._custom_metric = custom_metric

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
        """Evaluates the configured event condition on an inference result and updates the result if the event is detected.

        The method computes the metric value, compares it to the threshold, maintains a history of condition states, and triggers the event when the condition holds for the required duration.

        Args:
            result (InferenceResults): The inference result to analyze (must contain necessary metrics from prior analyzers).

        Returns:
            (None): This method modifies the `result` in-place by adding to its `events_detected` attribute.

        Raises:
            AttributeError: If the required metric data is missing in `result` (e.g., using ZoneCount without attaching `ZoneCounter`).
        """

        # evaluate metric and compare with threshold

        if self._custom_metric is not None:
            metric_value = self._custom_metric(result, self._metric_params)
        else:
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
        """Draws the event label onto the image frame if the event is active for this result.

        The overlay text is rendered only when all of the following conditions are true:
            - `self._show_overlay` is True for this EventDetector.
            - The event's name is present in `result.events_detected` for the current frame.

        If these conditions are met, the event name is drawn on the image at the configured position. The label's background color defaults to the complement of the model's overlay color (if no `annotation_color` was specified), and the text color is automatically chosen for optimal contrast.

        Args:
            result (InferenceResults): The result object from the model (after analysis), which may contain the event in its `events_detected` set.
            image (np.ndarray): The BGR image frame to annotate.

        Returns:
            np.ndarray: The same image frame with the event label overlay (if the event was detected). If no overlay was added, the image is returned unmodified.
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
