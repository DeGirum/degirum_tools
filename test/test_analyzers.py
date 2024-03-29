#
# test_analyzers.py: unit tests for analyzer functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test analyzer functionality
#

import numpy as np, pytest
from typing import List


def test_attach_analyzers():
    """Test for attach_analyzers() function"""

    import degirum_tools, degirum as dg

    class MyAnalyzer(degirum_tools.ResultAnalyzerBase):

        def __init__(self, color: tuple, level: int):
            self.mycolor = color
            self.mylevel = level

        def analyze(self, result):
            if not hasattr(result, "mycolor"):
                result.mycolor = [self.mycolor]
            else:
                result.mycolor.append(self.mycolor)

        def annotate(self, result, image: np.ndarray) -> np.ndarray:
            image[self.mylevel, self.mylevel] = self.mycolor
            return image

    model = dg.connect(dg.LOCAL, "test/model-zoo/dummy/dummy.json").load_model("dummy")
    model._model_parameters.SimulateParams = True
    model.output_postprocess_type = "Classification"

    data = np.zeros((10, 10, 3), dtype=np.uint8)
    black = (0, 0, 0)

    # test with no analyzers
    assert model.custom_postprocessor is None
    result = model(data)
    assert np.array_equal(result.image_overlay[0, 0], black)

    # test with one analyzer
    red = (255, 0, 0)
    red_analyzer = MyAnalyzer(red, 0)
    degirum_tools.attach_analyzers(model, red_analyzer)
    assert model.custom_postprocessor is not None
    result = model(data)
    assert hasattr(result, "mycolor") and red in result.mycolor
    assert np.array_equal(result.image_overlay[0, 0], red)

    # test with two analyzers
    green = (0, 255, 0)
    green_analyzer = MyAnalyzer(green, 1)
    degirum_tools.attach_analyzers(model, green_analyzer)
    assert model.custom_postprocessor is not None
    result = model(data)
    assert (
        hasattr(result, "mycolor") and green in result.mycolor and red in result.mycolor
    )
    assert np.array_equal(result.image_overlay[0, 0], red)
    assert np.array_equal(result.image_overlay[1, 1], green)

    # remove second analyzer
    degirum_tools.attach_analyzers(model, None)
    assert model.custom_postprocessor is not None
    result = model(data)
    assert (
        hasattr(result, "mycolor")
        and green not in result.mycolor
        and red in result.mycolor
    )
    assert np.array_equal(result.image_overlay[0, 0], red)
    assert np.array_equal(result.image_overlay[1, 1], black)

    # remove first analyzer
    degirum_tools.attach_analyzers(model, None)
    assert model.custom_postprocessor is None
    result = model(data)
    assert not hasattr(result, "mycolor")
    assert np.array_equal(result.image_overlay[0, 0], black)
    assert np.array_equal(result.image_overlay[1, 1], black)

    # check that compound model cannot have analyzers
    compound_model = degirum_tools.CombiningCompoundModel(model, model)
    with pytest.raises(Exception):
        degirum_tools.attach_analyzers(compound_model, red_analyzer)


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
                model_params=None, inference_results=input, conversion=None
            )

            selector.analyze(result)
            assert (
                result._inference_results == case["res"][i]
            ), f"Case {ci} failed at step {i}"
