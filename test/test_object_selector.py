#
# test_object_selector.py: unit tests for object selector analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test object selector analyzer
#

from typing import List


def test_object_selector():
    """
    Test for ObjectSelector analyzer
    """

    import degirum_tools, degirum as dg

    lil_box = [0, 0, 10, 10]
    big_box = [20, 20, 100, 100]

    test_cases: List[dict] = [
        # ----------------------------------------------------------------
        # No tracking
        # ----------------------------------------------------------------
        # no objects
        {
            "params": {"use_tracking": False},
            "inp": [[]],
            "res": [[]],
        },
        # one object, highest score
        {
            "params": {
                "top_k": 1,
                "selection_strategy": degirum_tools.ObjectSelectionStrategies.HIGHEST_SCORE,
                "use_tracking": False,
            },
            "inp": [[{"bbox": lil_box, "score": 1}, {"bbox": big_box, "score": 0.5}]],
            "res": [[{"bbox": lil_box, "score": 1}]],
        },
        # one object, largest area
        {
            "params": {
                "top_k": 1,
                "selection_strategy": degirum_tools.ObjectSelectionStrategies.LARGEST_AREA,
                "use_tracking": False,
            },
            "inp": [[{"bbox": lil_box, "score": 1}, {"bbox": big_box, "score": 0.5}]],
            "res": [[{"bbox": big_box, "score": 0.5}]],
        },
        # two objects, highest score
        {
            "params": {
                "top_k": 2,
                "selection_strategy": degirum_tools.ObjectSelectionStrategies.HIGHEST_SCORE,
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": lil_box, "score": 1},
                    {"bbox": big_box, "score": 0.5},
                    {"bbox": big_box, "score": 0.2},
                ]
            ],
            "res": [[{"bbox": lil_box, "score": 1}, {"bbox": big_box, "score": 0.5}]],
        },
        # two objects, largest area
        {
            "params": {
                "top_k": 2,
                "selection_strategy": degirum_tools.ObjectSelectionStrategies.LARGEST_AREA,
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": lil_box, "score": 1},
                    {"bbox": big_box, "score": 0.5},
                    {"bbox": lil_box, "score": 0.2},
                ]
            ],
            "res": [[{"bbox": big_box, "score": 0.5}, {"bbox": lil_box, "score": 1}]],
        },
        # too many objects, largest area
        {
            "params": {
                "top_k": 10,
                "selection_strategy": degirum_tools.ObjectSelectionStrategies.LARGEST_AREA,
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": lil_box, "score": 1},
                    {"bbox": big_box, "score": 0.5},
                    {"bbox": lil_box, "score": 0.2},
                ]
            ],
            "res": [
                [
                    {"bbox": big_box, "score": 0.5},
                    {"bbox": lil_box, "score": 1},
                    {"bbox": lil_box, "score": 0.2},
                ]
            ],
        },
        # ----------------------------------------------------------------
        # With tracking
        # ----------------------------------------------------------------
        # no objects
        {
            "params": {"use_tracking": True},
            "inp": [[]],
            "res": [[]],
        },
        # no tracking info
        {
            "params": {"top_k": 1, "use_tracking": True},
            "inp": [[{"bbox": lil_box, "score": 1}, {"bbox": big_box, "score": 0.5}]],
            "res": [[]],
        },
        # one object, highest score
        {
            "params": {
                "top_k": 1,
                "selection_strategy": degirum_tools.ObjectSelectionStrategies.HIGHEST_SCORE,
                "use_tracking": True,
            },
            "inp": [
                [
                    {"bbox": lil_box, "score": 1, "track_id": 0},
                    {"bbox": big_box, "score": 0.5, "track_id": 1},
                ]
            ],
            "res": [[{"bbox": lil_box, "score": 1, "track_id": 0}]],
        },
        # one object, largest area
        {
            "params": {
                "top_k": 1,
                "selection_strategy": degirum_tools.ObjectSelectionStrategies.LARGEST_AREA,
                "use_tracking": True,
            },
            "inp": [
                [
                    {"bbox": lil_box, "score": 1, "track_id": 0},
                    {"bbox": big_box, "score": 0.5, "track_id": 1},
                ]
            ],
            "res": [[{"bbox": big_box, "score": 0.5, "track_id": 1}]],
        },
        # one object, which disappears and appears again
        {
            "params": {
                "top_k": 1,
                "selection_strategy": degirum_tools.ObjectSelectionStrategies.LARGEST_AREA,
                "use_tracking": True,
            },
            "inp": [
                [
                    {"bbox": lil_box, "score": 1, "track_id": 0},
                    {"bbox": big_box, "score": 0.5, "track_id": 1},
                ],
                [{"bbox": lil_box, "score": 1, "track_id": 0}],
                [{"bbox": big_box, "score": 0.8, "track_id": 1}],
            ],
            "res": [
                [{"bbox": big_box, "score": 0.5, "track_id": 1}],
                [{"bbox": big_box, "score": 0.5, "track_id": 1}],
                [{"bbox": big_box, "score": 0.8, "track_id": 1}],
            ],
        },
        # one object, which disappears and appears again but too late
        {
            "params": {
                "top_k": 1,
                "selection_strategy": degirum_tools.ObjectSelectionStrategies.LARGEST_AREA,
                "use_tracking": True,
                "tracking_timeout": 1,
            },
            "inp": [
                [
                    {"bbox": lil_box, "score": 1, "track_id": 0},
                    {"bbox": big_box, "score": 0.5, "track_id": 1},
                ],
                [{"bbox": lil_box, "score": 0.7, "track_id": 0}],
                [{"bbox": lil_box, "score": 1, "track_id": 0}],
                [{"bbox": big_box, "score": 0.8, "track_id": 1}],
            ],
            "res": [
                [{"bbox": big_box, "score": 0.5, "track_id": 1}],
                [{"bbox": big_box, "score": 0.5, "track_id": 1}],
                [{"bbox": lil_box, "score": 1, "track_id": 0}],
                [{"bbox": lil_box, "score": 1, "track_id": 0}],
            ],
        },
    ]

    for ci, case in enumerate(test_cases):
        selector = degirum_tools.ObjectSelector(**case["params"])

        for i, input in enumerate(case["inp"]):
            result = dg.postprocessor.InferenceResults(
                inference_results=input, conversion=None
            )

            selector.analyze(result)
            assert (
                result._inference_results == case["res"][i]
            ), f"Case {ci} failed at step {i}"
