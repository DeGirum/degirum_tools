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

    assert model.model1 == detection_model
    assert model.model2 == classification_model
    master_model = model.model1

    all_props = {
        k
        for cls in master_model.__class__.__mro__
        for k, v in cls.__dict__.items()
        if isinstance(v, property) and not k.startswith("__")
    }

    # check that all properties can be read from the compound model
    # and they are equal to the master model
    for prop in all_props:
        master_value = getattr(master_model, prop)
        compound_value = getattr(model, prop)
        if not isinstance(master_value, dg.aiclient.ModelParams):
            assert master_value == compound_value

    # check few properties that they can be set in the compound model
    assert not master_model.measure_time
    model.measure_time = True
    assert master_model.measure_time
    assert not classification_model.measure_time

    assert master_model.input_pad_method == "letterbox"
    model.input_pad_method = "stretch"
    assert master_model.input_pad_method == "stretch"
    assert classification_model.input_pad_method == "letterbox"


def test_compound_model_combining(zoo_dir, detection_model, short_video):
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


def test_compound_model_cropping(
    zoo_dir, detection_model, classification_model, short_video
):
    """Test for CroppingAndClassifyingCompoundModel and CroppingAndDetectingCompoundModel classes"""

    import degirum_tools

    model1 = degirum_tools.CroppingAndClassifyingCompoundModel(
        detection_model, classification_model
    )

    with degirum_tools.open_video_stream(short_video) as stream:
        detection_results = list(
            detection_model.predict_batch(degirum_tools.video_source(stream))
        )

    with degirum_tools.open_video_stream(short_video) as stream:
        compound_results1 = list(
            model1.predict_batch(degirum_tools.video_source(stream))
        )

    assert len(detection_results) == len(compound_results1)
    for d, c in zip(detection_results, compound_results1):
        assert len(d.results) == len(c.results)
        for dr, cr in zip(d.results, c.results):
            assert dr["label"] == cr["label"]
            assert [round(x) for x in dr["bbox"]] == cr["bbox"]
            assert dr["score"] != cr["score"]
            assert dr["category_id"] != cr["category_id"]

    tiles = [[0, 0, 640, 720], [640, 0, 1280, 720]]
    tiler = degirum_tools.RegionExtractionPseudoModel(tiles, detection_model)
    model2 = degirum_tools.CroppingAndDetectingCompoundModel(tiler, detection_model)

    with degirum_tools.open_video_stream(short_video) as stream:
        compound_results2 = list(
            model2.predict_batch(degirum_tools.video_source(stream))
        )

    assert len(detection_results) == len(compound_results2)
    for d, c in zip(detection_results, compound_results2):
        assert len(d.results) <= len(c.results)
        assert all(cr["label"] == "Car" for cr in c.results)
        assert all("crop_index" not in cr for cr in c.results)

    model2a = degirum_tools.CroppingAndDetectingCompoundModel(
        tiler, detection_model, add_model1_results=True
    )

    with degirum_tools.open_video_stream(short_video) as stream:
        compound_results2a = list(
            model2a.predict_batch(degirum_tools.video_source(stream))
        )

    def is_rect_inside(inner, outer):
        """
        Returns True if the rectangle 'inner' is fully inside rectangle 'outer'.
        Rectangles are [x1, y1, x2, y2].
        """
        return (
            inner[0] >= outer[0]
            and inner[1] >= outer[1]
            and inner[2] <= outer[2]
            and inner[3] <= outer[3]
        )

    for d, c in zip(detection_results, compound_results2a):
        assert all("crop_index" in cr for cr in c.results if cr["label"] == "Car")
        assert all(
            is_rect_inside(cr["bbox"], c.results[cr["crop_index"]]["bbox"])
            for cr in c.results
            if cr["label"] == "Car"
        )
