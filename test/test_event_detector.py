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
        {
            "case": "basic: syntax error in params",
            "params": "Some incorrect text",
        },
        {
            "case": "basic: params schema incomplete",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
            """,
        },
        {
            "case": "basic: no objects",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is greater than: 0
                during: [1, frame]
            """,
            "inp": [{"results": []}],
            "res": [set()],
        },
        {
            "case": "basic: some objects",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 2
                during: [1, frame]
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
        {
            "case": "conditions: is equal to: 0",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 0
                during: [1, frame]
            """,
            "inp": [{"results": []}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "conditions: is not equal to: 0",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is not equal to: 0
                during: [1, frame]
            """,
            "inp": [{"results": [{"label": "person", "score": 0.5}]}],
            "res": [{"MyEvent"}],
        },
        # Greater Than: 0
        {
            "case": "conditions: greater than: 0",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is greater than: 0
                during: [1, frame]
            """,
            "inp": [{"results": [{"label": "person", "score": 0.5}]}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "conditions: greater than or equal to : 1",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is greater than or equal to: 1
                during: [1, frame]
            """,
            "inp": [{"results": [{"label": "person", "score": 0.5}]}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "conditions: less than: 1",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is less than: 1
                during: [1, frame]
            """,
            "inp": [{"results": []}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "conditions: less than or equal to: 1",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is less than or equal to: 1
                during: [1, frame]
            """,
            "inp": [{"results": [{"label": "person", "score": 0.5}]}],
            "res": [{"MyEvent"}],
        },
        # ----------------------------------------------------------------
        # Tests for quantifiers (Always, Sometimes, etc.)
        # ----------------------------------------------------------------
        {
            "case": "quantifiers: default case (always)",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [3, frames]
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
        {
            "case": "quantifiers: for at least 2 frames",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [3, frames]
                for at least: [2, frames]
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
        {
            "case": "quantifiers: for at least 50 percent",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [3, frames]
                for at least: [50, percent]
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
        {
            "case": "quantifiers: for at most 2 frames",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [3, frames]
                for at most: [2, frames]
            """,
            "inp": [
                {"results": []},
                {"results": []},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
            ],
            "res": [{"MyEvent"}, {"MyEvent"}, {"MyEvent"}, {"MyEvent"}, set()],
        },
        {
            "case": "quantifiers: for at most 50 percent",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [3, frames]
                for at most: [50, percent]
            """,
            "inp": [
                {"results": []},
                {"results": []},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
            ],
            "res": [{"MyEvent"}, {"MyEvent"}, {"MyEvent"}, set(), set()],
        },
        # ----------------------------------------------------------------
        # Tests for duration (frames, seconds)
        # ----------------------------------------------------------------
        {
            "case": "duration: 0 frames: expect to fail",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [0, frames]
            """,
        },
        {
            "case": "duration: 1 frame",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [1, frame]
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
        {
            "case": "duration: 2 frames",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [2, frames]
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
        {
            "case": "duration: 4 frames (never satisfied)",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [4, frames]
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
        {
            "case": "duration: seconds: always satisfied",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [2, seconds]
            """,
            "inp": [
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
                {"results": [{"label": "person", "score": 0.5}]},
            ],
            "res": [{"MyEvent"}, {"MyEvent"}, {"MyEvent"}],
        },
        {
            "case": "duration: seconds: not always satisfied -> no events detected",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                is equal to: 1
                during: [2, seconds]
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
        {
            "case": "ObjectCount: score filter",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                with:
                    min score: 0.5
                is equal to: 1
                during: [1, frame]
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
        {
            "case": "ObjectCount: class filter",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                with:
                    classes: ["cat"]
                is equal to: 1
                during: [1, frame]
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
        {
            "case": "ObjectCount: class and score filter",
            "params": """
                Trigger: MyEvent
                when: ObjectCount
                with:
                    min score: 0.5
                    classes: ["cat"]
                is equal to: 1
                during: [1, frame]
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
        {
            "case": "ZoneCount: missing zone counts: expect to fail",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                is equal to: 1
                during: [1, frame]
            """,
            "inp": [{}],
            "res": None,
        },
        {
            "case": "ZoneCount: no zone counts",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                is equal to: 0
                during: [1, frame]
            """,
            "inp": [{"zone_counts": []}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount: one zone, one count, no parameters",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                is equal to: 1
                during: [1, frame]
            """,
            "inp": [{"zone_counts": [{"total": 1}]}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount: multiple zones, no parameters",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                is equal to: 6
                during: [1, frame]
            """,
            "inp": [{"zone_counts": [{"total": 1}, {"total": 2}, {"total": 3}]}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount: multiple zones, zone filter out of range",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    index: 3
                is equal to: 3
                during: [1, frame]
            """,
            "inp": [{"zone_counts": [{"total": 1}, {"total": 2}, {"total": 3}]}],
            "res": None,
        },
        {
            "case": "ZoneCount: multiple zones, zone filter",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    index: 2
                is equal to: 3
                during: [1, frame]
            """,
            "inp": [{"zone_counts": [{"total": 1}, {"total": 2}, {"total": 3}]}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount: multiple zones, class filter",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    classes: ["cat", "dog"]
                is equal to: 10
                during: [1, frame]
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
        {
            "case": "ZoneCount: multiple zones, aggregation max",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    aggregation: max
                is equal to: 3
                during: [1, frame]
            """,
            "inp": [{"zone_counts": [{"total": 1}, {"total": 2}, {"total": 3}]}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount: multiple zones, aggregation min",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    aggregation: min
                is equal to: 1
                during: [1, frame]
            """,
            "inp": [{"zone_counts": [{"total": 1}, {"total": 2}, {"total": 3}]}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount: multiple zones, aggregation mean",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    aggregation: mean
                is equal to: 1.5
                during: [1, frame]
            """,
            "inp": [{"zone_counts": [{"total": 1}, {"total": 2}]}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount: multiple zones, aggregation std",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    aggregation: std
                is equal to: 0
                during: [1, frame]
            """,
            "inp": [{"zone_counts": [{"total": 1}, {"total": 1}, {"total": 1}]}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount: multiple zones, zone and class filters, aggregation sum",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    classes: ["cat", "dog"]
                    index: 0
                    aggregation: sum
                is equal to: 2
                during: [1, frame]
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
        # Tests for ZoneCount metric - DICT FORMAT (Named Zones)
        # ----------------------------------------------------------------
        {
            "case": "ZoneCount (dict): empty zones dict",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                is equal to: 0
                during: [1, frame]
            """,
            "inp": [{"zone_counts": {}}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount (dict): single named zone, no parameters",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                is equal to: 5
                during: [1, frame]
            """,
            "inp": [{"zone_counts": {"entrance": {"total": 5}}}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount (dict): multiple named zones, no parameters (sum all)",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                is equal to: 10
                during: [1, frame]
            """,
            "inp": [
                {
                    "zone_counts": {
                        "entrance": {"total": 3},
                        "exit": {"total": 2},
                        "parking": {"total": 5},
                    }
                }
            ],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount (dict): zone filter by name (string index)",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    index: exit
                is equal to: 7
                during: [1, frame]
            """,
            "inp": [
                {
                    "zone_counts": {
                        "entrance": {"total": 3},
                        "exit": {"total": 7},
                        "parking": {"total": 5},
                    }
                }
            ],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount (dict): zone filter by numeric index (position)",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    index: 1
                is equal to: 7
                during: [1, frame]
            """,
            "inp": [
                {
                    "zone_counts": {
                        "entrance": {"total": 3},
                        "exit": {"total": 7},
                        "parking": {"total": 5},
                    }
                }
            ],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount (dict): zone filter by name - zone not found",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    index: nonexistent_zone
                is equal to: 1
                during: [1, frame]
            """,
            "inp": [
                {
                    "zone_counts": {
                        "entrance": {"total": 3},
                        "exit": {"total": 7},
                    }
                }
            ],
            "res": None,
        },
        {
            "case": "ZoneCount (dict): numeric index out of range",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    index: 5
                is equal to: 1
                during: [1, frame]
            """,
            "inp": [
                {
                    "zone_counts": {
                        "entrance": {"total": 3},
                        "exit": {"total": 7},
                    }
                }
            ],
            "res": None,
        },
        {
            "case": "ZoneCount (dict): class filter on named zone",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    index: entrance
                    classes: ["person", "bicycle"]
                is equal to: 8
                during: [1, frame]
            """,
            "inp": [
                {
                    "zone_counts": {
                        "entrance": {"person": 5, "bicycle": 3, "car": 2},
                        "exit": {"person": 1},
                    }
                }
            ],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount (dict): aggregation max across named zones",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    aggregation: max
                is equal to: 8
                during: [1, frame]
            """,
            "inp": [
                {
                    "zone_counts": {
                        "zone_A": {"total": 3},
                        "zone_B": {"total": 8},
                        "zone_C": {"total": 5},
                    }
                }
            ],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount (dict): aggregation min across named zones",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    aggregation: min
                is equal to: 2
                during: [1, frame]
            """,
            "inp": [
                {
                    "zone_counts": {
                        "zone_A": {"total": 5},
                        "zone_B": {"total": 2},
                        "zone_C": {"total": 8},
                    }
                }
            ],
            "res": [{"MyEvent"}],
        },
        {
            "case": "ZoneCount (dict): named zone with class filter and aggregation",
            "params": """
                Trigger: MyEvent
                when: ZoneCount
                with:
                    index: parking
                    classes: ["car", "truck"]
                    aggregation: sum
                is equal to: 12
                during: [1, frame]
            """,
            "inp": [
                {
                    "zone_counts": {
                        "entrance": {"person": 5, "car": 1},
                        "parking": {"car": 8, "truck": 4, "bicycle": 2},
                        "exit": {"person": 3},
                    }
                }
            ],
            "res": [{"MyEvent"}],
        },
        # ----------------------------------------------------------------
        # Tests for LineCount metric
        # ----------------------------------------------------------------
        {
            "case": "LineCount: missing line counts: expect to fail",
            "params": """
                Trigger: MyEvent
                when: LineCount
                is equal to: 1
                during: [1, frame]
            """,
            "inp": [{}],
            "res": None,
        },
        {
            "case": "LineCount: no line counts",
            "params": """
                Trigger: MyEvent
                when: LineCount
                is equal to: 0
                during: [1, frame]
            """,
            "inp": [{"line_counts": []}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "LineCount: one line, zero counts",
            "params": """
                Trigger: MyEvent
                when: LineCount
                is equal to: 0
                during: [1, frame]
            """,
            "inp": [{"line_counts": [degirum_tools.LineCounts()]}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "LineCount: multiple lines, some counts, no filters",
            "params": """
                Trigger: MyEvent
                when: LineCount
                is equal to: 110
                during: [1, frame]
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
        {
            "case": "LineCount: multiple lines, some vector counts, no filters",
            "params": """
                Trigger: MyEvent
                when: LineCount
                is equal to: 33
                during: [1, frame]
            """,
            "inp": [
                {
                    "line_counts": [
                        D2C(left=1, right=2),
                        D2C(left=10, right=20),
                    ]
                }
            ],
            "res": [{"MyEvent"}],
        },
        {
            "case": "LineCount: multiple lines, some counts, line filter",
            "params": """
                Trigger: MyEvent
                when: LineCount
                with:
                    index: 1
                is equal to: 100
                during: [1, frame]
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
        {
            "case": "LineCount: multiple lines, some counts, direction filter",
            "params": """
                Trigger: MyEvent
                when: LineCount
                with:
                    directions: [left, right]
                is equal to: 33
                during: [1, frame]
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
        {
            "case": "LineCount: multiple lines, some counts, class filter",
            "params": """
                Trigger: MyEvent
                when: LineCount
                with:
                    classes: ["cat", "dog"]
                is equal to: 200
                during: [1, frame]
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
        {
            "case": "LineCount: multiple lines, some counts, all filters",
            "params": """
                Trigger: MyEvent
                when: LineCount
                with:
                    index: 1
                    classes: ["cat", "dog"]
                    directions: [top, bottom]
                is equal to: 70
                during: [1, frame]
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
        # ----------------------------------------------------------------
        # Tests for Custom metric
        # ----------------------------------------------------------------
        {
            "case": "Custom metric: no metric - expect to fail",
            "params": """
                Trigger: MyEvent
                when: CustomMetric
                is equal to: 1
                during: [1, frame]
            """,
        },
        {
            "case": "Custom metric: simple case with no event",
            "params": """
                Trigger: MyEvent
                when: CustomMetric
                is equal to: 1
                during: [1, frame]
            """,
            "custom_metric": lambda result, params: 0,
            "inp": [{"results": []}],
            "res": [set()],
        },
        {
            "case": "Custom metric: simple case with event",
            "params": """
                Trigger: MyEvent
                when: CustomMetric
                is equal to: 1
                during: [1, frame]
            """,
            "custom_metric": lambda result, params: 1,
            "inp": [{"results": []}],
            "res": [{"MyEvent"}],
        },
        {
            "case": "Custom metric: custom params",
            "params": """
                Trigger: MyEvent
                when: CustomMetric
                with:
                    some_param: 42
                is equal to: 42
                during: [1, frame]
            """,
            "custom_metric": lambda result, params: params.get("some_param", 0),
            "inp": [{"results": []}],
            "res": [{"MyEvent"}],
        },
    ]

    for ci, case in enumerate(test_cases):
        print(f"\n[{ci + 1}/{len(test_cases)}] Testing: {case['case']}")

        if "res" not in case:
            # expected to fail
            with pytest.raises(Exception):
                event_detector = degirum_tools.EventDetector(
                    case["params"], custom_metric=case.get("custom_metric")
                )
            print("  ✓ Expected failure caught")
            continue

        try:
            event_detector = degirum_tools.EventDetector(
                case["params"], custom_metric=case.get("custom_metric")
            )
        except Exception as e:
            raise Exception(
                f"Case `{case['case']}` failed: {e}\nConfig: {case['params']}"
            )

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
                    f"Case `{case['case']}` failed at step {i}: "
                    + f"detected events `{result.events_detected}` "  # type: ignore[attr-defined]
                    + f"do not match expected `{case['res'][i]}`."
                    + f"\nConfig: {case['params']}"
                )
        print("  ✓ Passed")
