#
# event_detector.py: event detector analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements analyzer class to detect various events by analyzing inference results
#

import yaml, time, jsonschema
from .result_analyzer_base import ResultAnalyzerBase
from collections import deque
from typing import Union

#
# Schema for event definition
#

# keys
Key_Trigger = "Trigger"
Key_When = "When"
Key_With = "With"
Key_Is = "Is"
Key_For = "For"
Key_Always = "Always"
Key_Mostly = "Mostly"
Key_Rarely = "Rarely"
Key_Sometimes = "Sometimes"
Key_To = "To"
Key_Than = "Than"
Key_Classes = "Classes"
Key_Index = "Index"
Key_Directions = "Directions"
Key_MinScore = "MinScore"

# operators
Op_Equal = "Equal"
Op_NotEqual = "NotEqual"
Op_Greater = "Greater"
Op_GreaterOrEqual = "GreaterOrEqual"
Op_Less = "Less"
Op_LessOrEqual = "LessOrEqual"

# units
Unit_Seconds = "seconds"
Unit_Frames = "frames"

# directions
Direction_Left = "left"
Direction_Right = "right"
Direction_Top = "top"
Direction_Bottom = "bottom"


# metrics
Metric_ZoneCount = "ZoneCount"
Metric_LineCount = "LineCount"
Metric_ObjectCount = "ObjectCount"

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
        enum: [{Metric_ZoneCount}, {Metric_LineCount}, {Metric_ObjectCount}]
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
    {Key_Is}:
        type: object
        additionalProperties: false
        properties:
            {Key_Always}:
                type: string
                enum: [{Op_Equal}, {Op_NotEqual}, {Op_Greater}, {Op_GreaterOrEqual}, {Op_Less}, {Op_LessOrEqual}]
                description: Metric comparison operator condition is always true
            {Key_Sometimes}:
                type: string
                enum: [{Op_Equal}, {Op_NotEqual}, {Op_Greater}, {Op_GreaterOrEqual}, {Op_Less}, {Op_LessOrEqual}]
                description: Metric comparison operator condition is true at least once
            {Key_Mostly}:
                type: string
                enum: [{Op_Equal}, {Op_NotEqual}, {Op_Greater}, {Op_GreaterOrEqual}, {Op_Less}, {Op_LessOrEqual}]
                description: Metric comparison operator condition is true more than half of the time
            {Key_Rarely}:
                type: string
                enum: [{Op_Equal}, {Op_NotEqual}, {Op_Greater}, {Op_GreaterOrEqual}, {Op_Less}, {Op_LessOrEqual}]
                description: Metric comparison operator condition is true less than half of the time
            {Key_To}:
                type: number
                description: The value to compare against
            {Key_Than}:
                type: number
                description: The value to compare against
        oneOf:
            - required: [{Key_Always}, {Key_To}]
            - required: [{Key_Always}, {Key_Than}]
            - required: [{Key_Sometimes}, {Key_To}]
            - required: [{Key_Sometimes}, {Key_Than}]
            - required: [{Key_Mostly}, {Key_To}]
            - required: [{Key_Mostly}, {Key_Than}]
            - required: [{Key_Rarely}, {Key_To}]
            - required: [{Key_Rarely}, {Key_Than}]
    {Key_For}:
        type: array
        prefixItems:
            - type: number
            - enum: [{Unit_Seconds}, {Unit_Frames}]
        items: false
        description: Duration to evaluate the metric
required: [{Key_Trigger}, {Key_When}, {Key_Is}, {Key_For}]
"""

event_definition_schema = yaml.safe_load(event_definition_schema_text)

#
# Metric functions
# Function name should match the metric name defined in the schema
#


def ZoneCount(result, params):
    """
    Get the number of objects in the specified zone.
    It uses the following result attributes:
        result.zone_counts

    Args:
        result: PySDK model result object
        params: metric parameters as defined in "With" field of the event description

    Returns:
        number of objects in the zone(s) belonging to the specified classes
    """

    index = None
    classes = None
    if params is not None:
        index = params.get(Key_Index)
        classes = params.get(Key_Classes)

    if not hasattr(result, "zone_counts"):
        raise ValueError(
            "Zone counts are not available in the result: insert ZoneCounter analyzer in a chain"
        )

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
    count = 0
    for zone in zone_counts:
        if classes is not None:
            count += sum(zone[key] for key in classes if key in zone)
        else:
            count += sum(zone.values())

    return count


def LineCount(result, params):
    """
    Count the number of objects in the specified zone.
    It uses the following result attributes:
        result.line_counts

    Args:
        result: PySDK model result object
        params: metric parameters

    Returns:
        float: number of objects in the zone
    """

    index = None
    classes = None
    directions = None

    if params is not None:
        index = params.get(Key_Index)
        classes = params.get(Key_Classes)
        directions = params.get(Key_Directions)

    if not hasattr(result, "line_counts"):
        raise ValueError(
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
                    getattr(line.for_class[key], dir)
                    for dir in (all_dirs if directions is None else directions)
                )
                for key in classes
                if key in line.for_class
            )
        else:
            count += sum(
                getattr(line, dir)
                for dir in (all_dirs if directions is None else directions)
            )

    return count


def ObjectCount(result, params):
    """
    Get the number of detected objects satisfying the specified conditions.
    It uses the following result attributes:
        result.results[]["label"]
        result.results[]["score"]

    Args:
        result: PySDK model result object
        params: metric parameters as defined in "With" field of the event description

    Returns:
        number of detected objects satisfying the specified conditions
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
    Class to detect various events by analyzing inference results

    """

    def __init__(self, event_description: Union[str, dict]):
        """
        Constructor

        Args:
            event_description: event description. It is a dictionary, which defines
                how to detect the event. The dictionary should conform to the schema provided
                in the `event_definition_schema` variable.
                Alternatively, it can be a string in YAML format, which will be parsed into a dictionary.

        """

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
        self._duration = event_desc[Key_For][0]
        duration_unit = event_desc[Key_For][1]
        compare_spec = event_desc[Key_Is]

        self._threshold = (
            compare_spec[Key_To] if Key_To in compare_spec else compare_spec[Key_Than]
        )

        if Key_Always in compare_spec:
            self._op = compare_spec[Key_Always]
            self._quantifier = Key_Always
        elif Key_Sometimes in compare_spec:
            self._op = compare_spec[Key_Sometimes]
            self._quantifier = Key_Sometimes
        elif Key_Mostly in compare_spec:
            self._op = compare_spec[Key_Mostly]
            self._quantifier = Key_Mostly
        elif Key_Rarely in compare_spec:
            self._op = compare_spec[Key_Rarely]
            self._quantifier = Key_Rarely

        if self._duration <= 0:
            raise ValueError("Duration should be greater than zero")

        self._event_history: deque = deque(
            maxlen=self._duration if duration_unit == Unit_Frames else None
        )

    comparators = {
        Op_Equal: lambda a, b: a == b,
        Op_NotEqual: lambda a, b: a != b,
        Op_Greater: lambda a, b: a > b,
        Op_GreaterOrEqual: lambda a, b: a >= b,
        Op_Less: lambda a, b: a < b,
        Op_LessOrEqual: lambda a, b: a <= b,
    }

    def analyze(self, result):
        """
        Detect event analyzing given result according to rules provided in the event description.

        If event is detected, the event name will be appended to the `events_detected` set in the result object.
        If result object does not have `events_detected` attribute, it will be created
        and initialized with the single element set containing detected event name.

        Args:
            result: PySDK model result object
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
        if self._quantifier == Key_Always:
            event_is_detected = (
                all(x[1] for x in self._event_history) if self._event_history else False
            )
        elif self._quantifier == Key_Sometimes:
            event_is_detected = any(x[1] for x in self._event_history)
        elif self._quantifier == Key_Mostly:
            event_is_detected = (
                sum(x[1] for x in self._event_history) > len(self._event_history) / 2
            )
        elif self._quantifier == Key_Rarely:
            s = sum(x[1] for x in self._event_history)
            event_is_detected = s < len(self._event_history) / 2 and s > 0
        else:
            event_is_detected = False

        # add detected event to the result object
        if not hasattr(result, "events_detected"):
            result.events_detected = set()
        if event_is_detected:
            result.events_detected.add(self._event_name)
