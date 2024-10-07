#
# test_analyzers.py: unit tests for analyzer functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test general analyzer functionality
#

import numpy as np


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

    # test result attributes propagation
    result._inference_results = []
    assert len(result.results) == 0

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

    # check that compound model can have analyzers
    model2 = dg.connect(dg.LOCAL, "test/model-zoo/dummy/dummy.json").load_model("dummy")
    model2._model_parameters.SimulateParams = True
    model2.output_postprocess_type = "Classification"

    class SampleCompoundModel(degirum_tools.CompoundModelBase):

        def queue_result1(self, result1):
            self.queue.put(
                (result1.image, degirum_tools.compound_models._FrameInfo(result1, -1))
            )

        def transform_result2(self, result2):
            return result2

    compound_model = SampleCompoundModel(model, model2)
    degirum_tools.attach_analyzers(compound_model, red_analyzer)
    result = compound_model(data)
    assert hasattr(result, "mycolor") and red in result.mycolor
    assert np.array_equal(result.image_overlay[0, 0], red)
