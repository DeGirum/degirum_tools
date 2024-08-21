#
# test_event_detector.py: unit tests for event detector analyzer functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test event detector analyzer functionality
#

import numpy as np
from typing import List
import pytest


def test_event_detector():
    """
    Test for EventDetector analyzer
    """

    import degirum_tools, degirum as dg

    class Result:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    test_cases: List[dict] = [
        # ----------------------------------------------------------------
        # General errors
        # ----------------------------------------------------------------
        # syntax error in params
        {
            "params": "Some incorrect text",
        },
        # params schema incomplete
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
            """
        },
        # ----------------------------------------------------------------
        # ObjectCount metric
        # ----------------------------------------------------------------
        # no objects
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is: 
                    Always: Greater
                    Than: 0
                For: [1, frames]
            """,
            "inp": [{"results": []}],
            "res": [set()],
        },
        # some objects
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is: 
                    Always: Equal
                    To: 2
                For: [1, frames]
            """,
            "inp": [
                {
                    "results": [
                        {"label": "person", "score": 0.5},
                        {"label": "person", "score": 0.6},
                    ]
                }
            ],
            "res": [{"MyEvent"}],
        },
    ]

    for ci, case in enumerate(test_cases):
        if "res" not in case:
            # expected to fail
            with pytest.raises(Exception):
                event_detector = degirum_tools.EventDetector(case["params"])
            continue

        event_detector = degirum_tools.EventDetector(case["params"])

        for i, input in enumerate(case["inp"]):
            result = Result(**input)

            if case["res"] is None:
                with pytest.raises(Exception):
                    event_detector.analyze(result)
            else:
                event_detector.analyze(result)
                assert (
                    result.events_detected == case["res"][i]
                ), f"Case {ci} failed at step {i}: detected events do not match expected."
