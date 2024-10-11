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
