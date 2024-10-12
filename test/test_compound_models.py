#
# test_compound_models.py: unit tests for compound models
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test the compound models functionality.
#


def test_compound_model_properties(zoo_dir, detection_model, classification_model):
    """Test for compound model properties propagation"""

    import degirum_tools, degirum as dg

    model = degirum_tools.CroppingAndClassifyingCompoundModel(
        detection_model, classification_model
    )

    assert model._master_model == detection_model

    all_props = {
        k
        for cls in model._master_model.__class__.__mro__
        for k, v in cls.__dict__.items()
        if isinstance(v, property) and not k.startswith("__")
    }

    # check that all properties can be read from the compound model
    # and they are equal to the master model
    for prop in all_props:
        master_value = getattr(model._master_model, prop)
        compound_value = getattr(model, prop)
        if not isinstance(master_value, dg.aiclient.ModelParams):
            assert master_value == compound_value

    # check few properties that they can be set in the compound model
    assert not model._master_model.measure_time
    model.measure_time = True
    assert model._master_model.measure_time
    assert not classification_model.measure_time

    assert model._master_model.input_pad_method == "letterbox"
    model.input_pad_method = "stretch"
    assert model._master_model.input_pad_method == "stretch"
    assert classification_model.input_pad_method == "letterbox"

    # TODO:
    # CombiningCompoundModel
    # CroppingAndClassifyingCompoundModel
    # CroppingAndDetectingCompoundModel
    # RegionExtractionPseudoModel


def test_combining_compound_model(zoo_dir, detection_model, short_video):
    """Test for CombiningCompoundModel class"""

    import degirum_tools

    roi = [[1, 2, 30, 40]]
    extractor = degirum_tools.RegionExtractionPseudoModel(roi, detection_model)
    model = degirum_tools.CombiningCompoundModel(extractor, detection_model)

    roi_class_label = "ROI0"
    detected_classes: set = set()

    with degirum_tools.open_video_stream(short_video) as stream:
        for res in model.predict_batch(degirum_tools.video_source(stream)):
            bboxes = {r["label"]: r["bbox"] for r in res.results}
            assert roi_class_label in bboxes and bboxes[roi_class_label] == roi[0]
            detected_classes |= bboxes.keys()

    assert "Car" in detected_classes and roi_class_label in detected_classes
