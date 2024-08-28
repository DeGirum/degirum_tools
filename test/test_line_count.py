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

    import degirum_tools, degirum as dg
    from degirum_tools import AnchorPoint

    lines = [(40, 50, 110, 140)]

    test_cases: List[dict] = [
        # no trails
        {"params": {"lines": lines}, "inp": [{}]},
        # incomplete trail
        {
            "params": {"lines": lines},
            "inp": [{1: [np.array([20, 35, 30, 55])]}],
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
        # one trail, does not cross line
        {
            "params": {"lines": lines, "anchor_point": AnchorPoint.TOP_CENTER},
            "inp": [{1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]}],
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
        # one trail, crosses line
        {
            "params": {"lines": lines},
            "inp": [{1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]}],
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
        # two trails
        {
            "params": {"lines": lines},
            "inp": [
                {1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]},
                {2: [np.array([80, 60, 90, 80]), np.array([30, 50, 40, 70])]},
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
        # two trails, do not accumulate
        {
            "params": {"lines": lines, "accumulate": False},
            "inp": [
                {1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]},
                {2: [np.array([80, 60, 90, 80]), np.array([30, 50, 40, 70])]},
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
        # two trails, directions relative to line
        {
            "params": {"lines": lines, "absolute_directions": False},
            "inp": [
                {1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]},
                {2: [np.array([80, 60, 90, 80]), np.array([50, 120, 60, 140])]},
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
        # zigzag trail
        {
            "params": {"lines": lines},
            "inp": [
                {1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]},
                {
                    1: [
                        np.array([20, 35, 30, 55]),
                        np.array([70, 40, 80, 60]),
                        np.array([20, 65, 30, 85]),
                    ]
                },
                {
                    1: [
                        np.array([20, 35, 30, 55]),
                        np.array([70, 40, 80, 60]),
                        np.array([20, 65, 30, 85]),
                        np.array([90, 85, 100, 105]),
                    ]
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
        # zigzag trail, count all crossings
        {
            "params": {"lines": lines, "count_first_crossing": False},
            "inp": [
                {1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]},
                {
                    1: [
                        np.array([20, 35, 30, 55]),
                        np.array([70, 40, 80, 60]),
                        np.array([20, 65, 30, 85]),
                    ]
                },
                {
                    1: [
                        np.array([20, 35, 30, 55]),
                        np.array([70, 40, 80, 60]),
                        np.array([20, 65, 30, 85]),
                        np.array([90, 85, 100, 105]),
                    ]
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
        # zigzag trail, count crossing based on last segment of trail
        {
            "params": {"lines": lines, "whole_trail": False},
            "inp": [
                {
                    1: [
                        np.array([20, 35, 30, 55]),
                        np.array([70, 40, 80, 60]),
                        np.array([20, 65, 30, 85]),
                    ]
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
        # zigzag trail, count all crossings and count crossing based on last segment of trail
        {
            "params": {
                "lines": lines,
                "count_first_crossing": False,
                "whole_trail": False,
            },
            "inp": [
                {1: [np.array([20, 35, 30, 55]), np.array([70, 40, 80, 60])]},
                {
                    1: [
                        np.array([20, 35, 30, 55]),
                        np.array([70, 40, 80, 60]),
                        np.array([20, 65, 30, 85]),
                    ]
                },
                {
                    1: [
                        np.array([20, 35, 30, 55]),
                        np.array([70, 40, 80, 60]),
                        np.array([20, 65, 30, 85]),
                        np.array([90, 85, 100, 105]),
                    ]
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
            result = dg.postprocessor.InferenceResults(
                model_params=None,
                input_image=np.zeros((200, 200)),
                inference_results=[],
                conversion=None,
            )
            result.trails = input

            line_counter.analyze(result)
            if ci == 0:
                assert not hasattr(
                    result, "line_counts"
                ), f"Case {ci} failed at step {i}: result has line_counts field."
            else:
                line_count_fields = case["res"][i][0]
                assert (
                    result.line_counts[0].bottom == line_count_fields["bottom"]
                ), f"Case {ci} failed at step {i}: 'bottom' line counts do not match."
                assert (
                    result.line_counts[0].left == line_count_fields["left"]
                ), f"Case {ci} failed at step {i}: 'left' line counts do not match."
                assert (
                    result.line_counts[0].right == line_count_fields["right"]
                ), f"Case {ci} failed at step {i}: 'right' line counts do not match."
                assert (
                    result.line_counts[0].top == line_count_fields["top"]
                ), f"Case {ci} failed at step {i}: 'top' line counts do not match."
