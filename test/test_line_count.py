#
# test_line_counter.py: unit tests for line counter analyzer functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test line counter analyzer functionality
#

import numpy as np
from typing import List


def test_line_counter():
    """
    Test for LineCounter analyzer
    """

    import degirum_tools
    from degirum_tools import AnchorPoint

    # helper class to convert dictionary to object
    class D2C:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    lines = [(40, 50, 110, 140)]

    test_cases: List[dict] = [
        {"case": "No trails", "params": {"lines": lines}, "inp": [{}]},
        {
            "case": "Incomplete trail",
            "params": {"lines": lines},
            "inp": [
                {
                    "trails": {1: [np.array([20, 35, 30, 55])]},
                    "trail_classes": {1: "label1"},
                }
            ],
            "res": [
                [
                    {
                        "bottom": 0,
                        "for_class": {},
                        "left": 0,
                        "right": 0,
                        "top": 0,
                    }
                ]
            ],
        },
        {
            "case": "One trail, does not cross line",
            "params": {"lines": lines, "anchor_point": AnchorPoint.TOP_CENTER},
            "inp": [
                {
                    "trails": {
                        1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]
                    },
                    "trail_classes": {1: "label1"},
                }
            ],
            "res": [
                [
                    {
                        "bottom": 0,
                        "for_class": {},
                        "left": 0,
                        "right": 0,
                        "top": 0,
                    }
                ]
            ],
        },
        {
            "case": "One trail, crosses line",
            "params": {"lines": lines},
            "inp": [
                {
                    "trails": {
                        1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]
                    },
                    "trail_classes": {1: "label1"},
                }
            ],
            "res": [
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 0,
                        "right": 1,
                        "top": 0,
                    }
                ]
            ],
        },
        {
            "case": "Two trails",
            "params": {"lines": lines},
            "inp": [
                {
                    "trails": {
                        1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]
                    },
                    "trail_classes": {1: "label1"},
                },
                {
                    "trails": {
                        2: [np.array([80, 60, 90, 80]), np.array([30, 50, 40, 70])]
                    },
                    "trail_classes": {2: "label2"},
                },
            ],
            "res": [
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 0,
                        "right": 1,
                        "top": 0,
                    }
                ],
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 1,
                        "right": 1,
                        "top": 1,
                    }
                ],
            ],
        },
        {
            "case": "Two trails, display per-class counts",
            "params": {"lines": lines, "per_class_display": True},
            "inp": [
                {
                    "trails": {
                        1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]
                    },
                    "trail_classes": {1: "label1"},
                },
                {
                    "trails": {
                        2: [np.array([80, 60, 90, 80]), np.array([30, 50, 40, 70])]
                    },
                    "trail_classes": {2: "label2"},
                },
            ],
            "res": [
                [
                    {
                        "bottom": 1,
                        "for_class": {
                            "label1": {"bottom": 1, "left": 0, "right": 1, "top": 0}
                        },
                        "left": 0,
                        "right": 1,
                        "top": 0,
                    }
                ],
                [
                    {
                        "bottom": 1,
                        "for_class": {
                            "label1": {"bottom": 1, "left": 0, "right": 1, "top": 0},
                            "label2": {"bottom": 0, "left": 1, "right": 0, "top": 1},
                        },
                        "left": 1,
                        "right": 1,
                        "top": 1,
                    }
                ],
            ],
        },
        {
            "case": "Two trails, do not accumulate",
            "params": {"lines": lines, "accumulate": False},
            "inp": [
                {
                    "trails": {
                        1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]
                    },
                    "trail_classes": {1: "label1"},
                },
                {
                    "trails": {
                        2: [np.array([80, 60, 90, 80]), np.array([30, 50, 40, 70])]
                    },
                    "trail_classes": {2: "label2"},
                },
            ],
            "res": [
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 0,
                        "right": 1,
                        "top": 0,
                    }
                ],
                [
                    {
                        "bottom": 0,
                        "for_class": {},
                        "left": 1,
                        "right": 0,
                        "top": 1,
                    }
                ],
            ],
        },
        {
            "case": "Two trails, directions relative to line",
            "params": {"lines": lines, "absolute_directions": False},
            "inp": [
                {
                    "trails": {
                        1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]
                    },
                    "trail_classes": {1: "label1"},
                },
                {
                    "trails": {
                        2: [np.array([80, 60, 90, 80]), np.array([50, 120, 60, 140])]
                    },
                    "trail_classes": {2: "label2"},
                },
            ],
            "res": [
                [
                    {
                        "bottom": 0,
                        "for_class": {},
                        "left": 1,
                        "right": 0,
                        "top": 1,
                    }
                ],
                [
                    {
                        "bottom": 0,
                        "for_class": {},
                        "left": 1,
                        "right": 1,
                        "top": 2,
                    }
                ],
            ],
        },
        {
            "case": "Zigzag trail",
            "params": {"lines": lines},
            "inp": [
                {
                    "trails": {
                        1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]
                    },
                    "trail_classes": {1: "label1"},
                },
                {
                    "trails": {
                        1: [
                            np.array([20, 35, 30, 55]),
                            np.array([70, 40, 80, 60]),
                            np.array([20, 65, 30, 85]),
                        ]
                    },
                    "trail_classes": {1: "label1"},
                },
                {
                    "trails": {
                        1: [
                            np.array([20, 35, 30, 55]),
                            np.array([70, 40, 80, 60]),
                            np.array([20, 65, 30, 85]),
                            np.array([90, 85, 100, 105]),
                        ]
                    },
                    "trail_classes": {1: "label1"},
                },
            ],
            "res": [
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 0,
                        "right": 1,
                        "top": 0,
                    }
                ],
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 0,
                        "right": 1,
                        "top": 0,
                    }
                ],
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 0,
                        "right": 1,
                        "top": 0,
                    }
                ],
            ],
        },
        {
            "case": "Zigzag trail, count all crossings",
            "params": {"lines": lines, "count_first_crossing": False},
            "inp": [
                {
                    "trails": {
                        1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]
                    },
                    "trail_classes": {1: "label1"},
                },
                {
                    "trails": {
                        1: [
                            np.array([20, 35, 30, 55]),
                            np.array([70, 40, 80, 60]),
                            np.array([20, 65, 30, 85]),
                        ]
                    },
                    "trail_classes": {1: "label1"},
                },
                {
                    "trails": {
                        1: [
                            np.array([20, 35, 30, 55]),
                            np.array([70, 40, 80, 60]),
                            np.array([20, 65, 30, 85]),
                            np.array([90, 85, 100, 105]),
                        ]
                    },
                    "trail_classes": {1: "label1"},
                },
            ],
            "res": [
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 0,
                        "right": 1,
                        "top": 0,
                    }
                ],
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 0,
                        "right": 1,
                        "top": 0,
                    }
                ],
                [
                    {
                        "bottom": 2,
                        "for_class": {},
                        "left": 0,
                        "right": 2,
                        "top": 0,
                    }
                ],
            ],
        },
        {
            "case": "Zigzag trail, count crossing based on last segment of trail",
            "params": {"lines": lines, "whole_trail": False},
            "inp": [
                {
                    "trails": {
                        1: [
                            np.array([20, 35, 30, 55]),
                            np.array([70, 40, 80, 60]),
                            np.array([20, 65, 30, 85]),
                        ]
                    },
                    "trail_classes": {1: "label1"},
                }
            ],
            "res": [
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 1,
                        "right": 0,
                        "top": 0,
                    }
                ]
            ],
        },
        {
            "case": "Zigzag trail, count all crossings and count crossing based on last segment of trail",
            "params": {
                "lines": lines,
                "count_first_crossing": False,
                "whole_trail": False,
            },
            "inp": [
                {
                    "trails": {
                        1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]
                    },
                    "trail_classes": {1: "label1"},
                },
                {
                    "trails": {
                        1: [
                            np.array([20, 35, 30, 55]),
                            np.array([70, 40, 80, 60]),
                            np.array([20, 65, 30, 85]),
                        ]
                    },
                    "trail_classes": {1: "label1"},
                },
                {
                    "trails": {
                        1: [
                            np.array([20, 35, 30, 55]),
                            np.array([70, 40, 80, 60]),
                            np.array([20, 65, 30, 85]),
                            np.array([90, 85, 100, 105]),
                        ]
                    },
                    "trail_classes": {1: "label1"},
                },
            ],
            "res": [
                [
                    {
                        "bottom": 1,
                        "for_class": {},
                        "left": 0,
                        "right": 1,
                        "top": 0,
                    }
                ],
                [
                    {
                        "bottom": 2,
                        "for_class": {},
                        "left": 1,
                        "right": 1,
                        "top": 0,
                    }
                ],
                [
                    {
                        "bottom": 3,
                        "for_class": {},
                        "left": 1,
                        "right": 2,
                        "top": 0,
                    }
                ],
            ],
        },
    ]

    for ci, case in enumerate(test_cases):
        line_counter = degirum_tools.LineCounter(**case["params"])

        for i, input in enumerate(case["inp"]):
            result = D2C(**input)

            line_counter.analyze(result)

            if "res" not in case:
                assert not hasattr(
                    result, "line_counts"
                ), f"Case {ci} failed at step {i}: result has line_counts field."
            else:
                line_count_fields = case["res"][i][0]
                res = degirum_tools.LineCounts()
                res.bottom = line_count_fields["bottom"]
                res.for_class = line_count_fields["for_class"]
                res.left = line_count_fields["left"]
                res.right = line_count_fields["right"]
                res.top = line_count_fields["top"]
                assert result.line_counts[0] == res, (  # type: ignore[attr-defined]
                    f"Case `{case['case']}` failed at step {i}: "
                    + f"line counts `{result.line_counts}` "  # type: ignore[attr-defined]
                    + f"do not match expected `{case['res'][i]}`."
                    + f"\nConfig: {case['params']}"
                )
