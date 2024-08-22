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

    # helper class to convert dictionary to object
    class D2C:
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
        # ----------------------------------------------------------------
        # Tests for ZoneCount metric
        # ----------------------------------------------------------------
        # Missing zone counts: expect to fail
        {
            "params": """
                Trigger: MyEvent
                When: ZoneCount
                Is:
                    Always: Equal
                    To: 1
                For: [1, frames]
            """,
            "inp": [{}],
            "res": None,
        },
        # No zone counts
        {
            "params": """
                Trigger: MyEvent
                When: ZoneCount
                Is:
                    Always: Equal
                    To: 0
                For: [1, frames]
            """,
            "inp": [{"zone_counts": []}],
            "res": [{"MyEvent"}],
        },
        # One zone, one count, no parameters
        {
            "params": """
                Trigger: MyEvent
                When: ZoneCount
                Is:
                    Always: Equal
                    To: 1
                For: [1, frames]
            """,
            "inp": [{"zone_counts": [{"total": 1}]}],
            "res": [{"MyEvent"}],
        },
        # Multiple zones, no parameters
        {
            "params": """
                Trigger: MyEvent
                When: ZoneCount
                Is:
                    Always: Equal
                    To: 6
                For: [1, frames]
            """,
            "inp": [{"zone_counts": [{"total": 1}, {"total": 2}, {"total": 3}]}],
            "res": [{"MyEvent"}],
        },
        # Multiple zones, zone filter out of range
        {
            "params": """
                Trigger: MyEvent
                When: ZoneCount
                With:
                    Index: 3
                Is:
                    Always: Equal
                    To: 3
                For: [1, frames]
            """,
            "inp": [{"zone_counts": [{"total": 1}, {"total": 2}, {"total": 3}]}],
            "res": None,
        },
        # Multiple zones, zone filter
        {
            "params": """
                Trigger: MyEvent
                When: ZoneCount
                With:
                    Index: 2
                Is:
                    Always: Equal
                    To: 3
                For: [1, frames]
            """,
            "inp": [{"zone_counts": [{"total": 1}, {"total": 2}, {"total": 3}]}],
            "res": [{"MyEvent"}],
        },
        # Multiple zones, class filter
        {
            "params": """
                Trigger: MyEvent
                When: ZoneCount
                With:
                    Classes: ["cat", "dog"]
                Is:
                    Always: Equal
                    To: 10
                For: [1, frames]
            """,
            "inp": [
                {
                    "zone_counts": [
                        {"cat": 1, "dog": 2},
                        {"dog": 3},
                        {"cat": 4},
                        {"person": 3},
                    ]
                }
            ],
            "res": [{"MyEvent"}],
        },
        # Multiple zones, zone and class filters
        {
            "params": """
                Trigger: MyEvent
                When: ZoneCount
                With:
                    Classes: ["cat", "dog"]
                    Index: 0
                Is:
                    Always: Equal
                    To: 2
                For: [1, frames]
            """,
            "inp": [
                {
                    "zone_counts": [
                        {"cat": 1, "dog": 1},
                        {"dog": 3},
                        {"cat": 4},
                        {"person": 3},
                    ]
                }
            ],
            "res": [{"MyEvent"}],
        },
        # ----------------------------------------------------------------
        # Tests for LineCount metric
        # ----------------------------------------------------------------
        # Missing line counts: expect to fail
        {
            "params": """
                Trigger: MyEvent
                When: LineCount
                Is:
                    Always: Equal
                    To: 1
                For: [1, frames]
            """,
            "inp": [{}],
            "res": None,
        },
        # No line counts
        {
            "params": """
                Trigger: MyEvent
                When: LineCount
                Is:
                    Always: Equal
                    To: 0
                For: [1, frames]
            """,
            "inp": [{"line_counts": []}],
            "res": [{"MyEvent"}],
        },
        # One line, zero counts
        {
            "params": """
                Trigger: MyEvent
                When: LineCount
                Is:
                    Always: Equal
                    To: 0
                For: [1, frames]
            """,
            "inp": [{"line_counts": [degirum_tools.LineCounts()]}],
            "res": [{"MyEvent"}],
        },
        # Multiple line, some counts, no filters
        {
            "params": """
                Trigger: MyEvent
                When: LineCount
                Is:
                    Always: Equal
                    To: 110
                For: [1, frames]
            """,
            "inp": [
                {
                    "line_counts": [
                        D2C(left=1, right=2, top=3, bottom=4),
                        D2C(left=10, right=20, top=30, bottom=40),
                    ]
                }
            ],
            "res": [{"MyEvent"}],
        },
        # Multiple line, some counts, line filter
        {
            "params": """
                Trigger: MyEvent
                When: LineCount
                With:
                    Index: 1
                Is:
                    Always: Equal
                    To: 100
                For: [1, frames]
            """,
            "inp": [
                {
                    "line_counts": [
                        D2C(left=1, right=2, top=3, bottom=4),
                        D2C(left=10, right=20, top=30, bottom=40),
                    ]
                }
            ],
            "res": [{"MyEvent"}],
        },
        # Multiple line, some counts, direction filter
        {
            "params": """
                Trigger: MyEvent
                When: LineCount
                With:
                    Directions: [left, right]
                Is:
                    Always: Equal
                    To: 33
                For: [1, frames]
            """,
            "inp": [
                {
                    "line_counts": [
                        D2C(left=1, right=2, top=3, bottom=4),
                        D2C(left=10, right=20, top=30, bottom=40),
                    ]
                }
            ],
            "res": [{"MyEvent"}],
        },
        # Multiple line, some counts, class filter
        {
            "params": """
                Trigger: MyEvent
                When: LineCount
                With:
                    Classes: ["cat", "dog"]
                Is:
                    Always: Equal
                    To: 200
                For: [1, frames]
            """,
            "inp": [
                {
                    "line_counts": [
                        D2C(
                            left=1,
                            right=2,
                            top=3,
                            bottom=4,
                            for_class={
                                "cat": D2C(left=5, right=6, top=7, bottom=8),
                                "dog": D2C(left=9, right=10, top=11, bottom=12),
                                "person": D2C(left=99, right=99, top=99, bottom=99),
                            },
                        ),
                        D2C(
                            left=10,
                            right=20,
                            top=30,
                            bottom=40,
                            for_class={
                                "cat": D2C(left=13, right=14, top=15, bottom=16),
                                "dog": D2C(left=17, right=18, top=19, bottom=20),
                                "person": D2C(left=99, right=99, top=99, bottom=99),
                            },
                        ),
                    ]
                }
            ],
            "res": [{"MyEvent"}],
        },
        # Multiple line, some counts, all filters
        {
            "params": """
                Trigger: MyEvent
                When: LineCount
                With:
                    Index: 1
                    Classes: ["cat", "dog"]
                    Directions: [top, bottom]
                Is:
                    Always: Equal
                    To: 70
                For: [1, frames]
            """,
            "inp": [
                {
                    "line_counts": [
                        D2C(
                            left=1,
                            right=2,
                            top=3,
                            bottom=4,
                            for_class={
                                "cat": D2C(left=5, right=6, top=7, bottom=8),
                                "dog": D2C(left=9, right=10, top=11, bottom=12),
                                "person": D2C(left=99, right=99, top=99, bottom=99),
                            },
                        ),
                        D2C(
                            left=10,
                            right=20,
                            top=30,
                            bottom=40,
                            for_class={
                                "cat": D2C(left=13, right=14, top=15, bottom=16),
                                "dog": D2C(left=17, right=18, top=19, bottom=20),
                                "person": D2C(left=99, right=99, top=99, bottom=99),
                            },
                        ),
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
            result = D2C(**input)

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
