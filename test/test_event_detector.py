#
# test_event_detector.py: unit tests for event detector analyzer functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test event detector analyzer functionality
#

from typing import List
import pytest


def test_event_detector():
    """
    Test for EventDetector analyzer
    """

    import degirum_tools

    class Result:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    test_cases: List[dict] = [
        # ----------------------------------------------------------------
        # Basic tests
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
        # ----------------------------------------------------------------
        # Tests for conditions (Equal, Greater, etc.)
        # ----------------------------------------------------------------
        # Equal To: 0
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: Equal
                    To: 0
                For: [1, frames]
            """,
            "inp": [{"results": []}],
            "res": [{"MyEvent"}],
        },
        # Op_NotEqual To: 0
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: NotEqual
                    To: 0
                For: [1, frames]
            """,
            "inp": [{"results": [{"label": "person", "score": 0.5}]}],
            "res": [{"MyEvent"}],
        },
        # Greater Than: 0
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: Greater
                    To: 0
                For: [1, frames]
            """,
            "inp": [{"results": [{"label": "person", "score": 0.5}]}],
            "res": [{"MyEvent"}],
        },
        # GreaterOrEqual Than: 1
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: GreaterOrEqual
                    To: 1
                For: [1, frames]
            """,
            "inp": [{"results": [{"label": "person", "score": 0.5}]}],
            "res": [{"MyEvent"}],
        },
        # Less Than: 1
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: Less
                    Than: 1
                For: [1, frames]
            """,
            "inp": [{"results": []}],
            "res": [{"MyEvent"}],
        },
        # LessOrEqual Than: 2
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: LessOrEqual
                    Than: 1
                For: [1, frames]
            """,
            "inp": [{"results": [{"label": "person", "score": 0.5}]}],
            "res": [{"MyEvent"}],
        },
        # ----------------------------------------------------------------
        # Tests for quantifiers (Always, Sometimes, etc.)
        # ----------------------------------------------------------------
        # Always
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: Equal
                    To: 1
                For: [3, frames]
            """,
            "inp": [
                {"results": []},
                {"results": []},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
            ],
            "res": [set(), set(), set(), set(), {"MyEvent"}],
        },
        # Sometimes
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Sometimes: Equal
                    To: 1
                For: [3, frames]
            """,
            "inp": [
                {"results": []},
                {"results": []},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
            ],
            "res": [set(), set(), {"MyEvent"}, {"MyEvent"}, {"MyEvent"}],
        },
        # Mostly
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Mostly: Equal
                    To: 1
                For: [3, frames]
            """,
            "inp": [
                {"results": []},
                {"results": []},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
            ],
            "res": [set(), set(), set(), {"MyEvent"}, {"MyEvent"}],
        },
        # Rarely
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Rarely: Equal
                    To: 1
                For: [3, frames]
            """,
            "inp": [
                {"results": []},
                {"results": []},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
            ],
            "res": [set(), set(), {"MyEvent"}, set(), set()],
        },
        # ----------------------------------------------------------------
        # Tests for duration (frames, seconds)
        # ----------------------------------------------------------------
        # 0 frames: expect to fail
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: Equal
                    To: 1
                For: [0, frames]
            """,
        },
        # 1 frame
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: Equal
                    To: 1
                For: [1, frames]
            """,
            "inp": [
                {"results": []},
                {"results": []},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": []},
            ],
            "res": [set(), set(), {"MyEvent"}, {"MyEvent"}, {"MyEvent"}, set()],
        },
        # 2 frames
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: Equal
                    To: 1
                For: [2, frames]
            """,
            "inp": [
                {"results": []},
                {"results": []},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": []},
            ],
            "res": [set(), set(), set(), {"MyEvent"}, {"MyEvent"}, set()],
        },
        # 4 frames (never satisfied)
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: Equal
                    To: 1
                For: [4, frames]
            """,
            "inp": [
                {"results": []},
                {"results": []},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": []},
            ],
            "res": [set(), set(), set(), set(), set(), set()],
        },
        # seconds: always satisfied
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: Equal
                    To: 1
                For: [2, seconds]
            """,
            "inp": [
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
            ],
            "res": [{"MyEvent"}, {"MyEvent"}, {"MyEvent"}],
        },
        # seconds: not always satisfied -> no events detected
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                Is:
                    Always: Equal
                    To: 1
                For: [2, seconds]
            """,
            "inp": [
                {"results": []},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
            ],
            "res": [set(), set(), set()],
        },
        # ----------------------------------------------------------------
        # Tests for ObjectCount metric
        # ----------------------------------------------------------------
        # score filter
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                With:
                    MinScore: 0.5
                Is:
                    Always: Equal
                    To: 1
                For: [1, frames]
            """,
            "inp": [
                {
                    "results": [
                        {"label": "person", "score": 0.4},
                        {"label": "person", "score": 0.6},
                    ]
                }
            ],
            "res": [{"MyEvent"}],
        },
        # class filter
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                With:
                    Classes: ["cat"]
                Is:
                    Always: Equal
                    To: 1
                For: [1, frames]
            """,
            "inp": [
                {
                    "results": [
                        {"label": "person", "score": 0.4},
                        {"label": "cat", "score": 0.6},
                    ]
                }
            ],
            "res": [{"MyEvent"}],
        },
        # class and score filter
        {
            "params": """
                Trigger: MyEvent
                When: ObjectCount
                With:
                    MinScore: 0.5
                    Classes: ["cat"]
                Is:
                    Always: Equal
                    To: 1
                For: [1, frames]
            """,
            "inp": [
                {
                    "results": [
                        {"label": "person", "score": 0.4},
                        {"label": "cat", "score": 0.6},
                        {"label": "cat", "score": 0.3},
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
                    result.events_detected == case["res"][i]  # type: ignore[attr-defined]
                ), (
                    f"Case {ci} failed at step {i}: "
                    + f"detected events `{result.events_detected}` "  # type: ignore[attr-defined]
                    + f"do not match expected `{case['res'][i]}`."
                    + f"\nConfig: {case['params']}"
                )
