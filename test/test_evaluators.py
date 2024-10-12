#
# test_evaluators.py: unit tests for model evaluators functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test model evaluators
#

import pytest


def test_ObjectDetectionModelEvaluator(zoo_dir):
    """Test for ObjectDetectionModelEvaluator class"""

    import degirum_tools, degirum as dg
    import os, io, json

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # load COCO detection and segmentation models
    object_detection_model = dg.load_model(
        "mobilenet_v2_ssd_coco--300x300_quant_n2x_cpu_1", dg.LOCAL, zoo_dir
    )
    object_segmentation_model = dg.load_model(
        "yolov8n_relu6_coco_seg--640x640_quant_tflite_cpu_1", dg.LOCAL, zoo_dir
    )

    #
    # test evaluator creation
    #

    # default parameters
    evaluator_default = degirum_tools.ObjectDetectionModelEvaluator(
        object_detection_model
    )
    assert not evaluator_default.show_progress
    assert evaluator_default.classmap is None
    assert evaluator_default.pred_path is None
    assert object_detection_model.input_pad_method == "letterbox"
    assert object_detection_model.image_backend == "opencv"

    # handling invalid parameters
    with pytest.raises(Exception):
        degirum_tools.ObjectDetectionModelEvaluator.init_from_yaml(
            object_detection_model, io.StringIO("non_existent_parameter: 0")
        )

    # test parameters
    evaluator1 = degirum_tools.ObjectDetectionModelEvaluator.init_from_yaml(
        object_detection_model,
        io.StringIO(
            """
                show_progress: true
                classmap: {1: 2}
                pred_path: "test.json"
                input_pad_method: "crop-last"
                image_backend: "pil"
            """
        ),
    )
    assert evaluator1.show_progress
    assert evaluator1.classmap is not None and evaluator1.classmap.get(1) == 2
    assert evaluator1.pred_path == "test.json"
    assert object_detection_model.input_pad_method == "crop-last"
    assert object_detection_model.image_backend == "pil"

    #
    # test object detection model evaluation
    #
    predictions_file = "test/ObjectDetectionModelEvaluator_predictions.json"
    predictions_cnt = 20
    try:
        evaluator = degirum_tools.ObjectDetectionModelEvaluator(
            object_detection_model, pred_path=predictions_file
        )
        res = evaluator.evaluate(
            "test/sample_dataset", "test/sample_dataset/labels.json", predictions_cnt
        )

        # validate results
        assert len(res) == 1
        assert len(res[0]) == 12
        assert os.path.exists(predictions_file)

        saved_predictions = json.load(open(predictions_file))
        assert (
            isinstance(saved_predictions, list)
            and len(saved_predictions) >= predictions_cnt
        )
    finally:
        if os.path.exists(predictions_file):
            os.remove(predictions_file)

    #
    # test object segmentation model evaluation
    #
    predictions_file = "test/ObjectSegmentationModelEvaluator_predictions.json"
    predictions_cnt = 20
    try:
        evaluator = degirum_tools.ObjectDetectionModelEvaluator(
            object_segmentation_model, pred_path=predictions_file
        )
        res = evaluator.evaluate(
            "test/sample_dataset", "test/sample_dataset/labels.json", predictions_cnt
        )

        # validate results
        assert len(res) == 2
        assert len(res[1]) == len(res[0]) == 12
        assert os.path.exists(predictions_file)

        saved_predictions = json.load(open(predictions_file))
        assert (
            isinstance(saved_predictions, list)
            and len(saved_predictions) >= predictions_cnt
        )
    finally:
        if os.path.exists(predictions_file):
            os.remove(predictions_file)


def test_ImageClassificationModelEvaluator(classification_model):
    """Test for ImageClassificationModelEvaluator class"""

    import degirum_tools, io, json

    model = classification_model

    #
    # test evaluator creation
    #

    # default parameters
    evaluator_default = degirum_tools.ImageClassificationModelEvaluator(model)
    assert not evaluator_default.show_progress
    assert evaluator_default.top_k == [1, 5]
    assert evaluator_default.foldermap is None
    assert model.input_pad_method == "letterbox"
    assert model.image_backend == "opencv"

    # handling invalid parameters
    with pytest.raises(Exception):
        degirum_tools.ImageClassificationModelEvaluator.init_from_yaml(
            model, io.StringIO("non_existent_parameter: 0")
        )

    # test parameters
    evaluator1 = degirum_tools.ImageClassificationModelEvaluator.init_from_yaml(
        model,
        io.StringIO(
            """
                show_progress: true
                top_k: [2,4]
                foldermap: {0: "zero"}
                input_pad_method: "crop-last"
                image_backend: "pil"
            """
        ),
    )
    assert evaluator1.show_progress
    assert evaluator1.top_k == [2, 4]
    assert evaluator1.foldermap is not None and evaluator1.foldermap.get(0) == "zero"
    assert model.input_pad_method == "crop-last"
    assert model.image_backend == "pil"

    #
    # test model evaluation
    #

    dataset_root = "test/sample_dataset"
    dataset = json.load(open(dataset_root + "/labels.json"))

    # deduce foldermap from the dataset
    model_categories = {
        label.lower(): id for id, label in model.label_dictionary.items()
    }
    foldermap = {
        model_categories[dataset_category["name"].lower()]: dataset_category["name"]
        for dataset_category in dataset["categories"]
        if dataset_category["name"].lower() in model_categories
    }

    # run evaluation
    predictions_cnt = 50
    evaluator = degirum_tools.ImageClassificationModelEvaluator(
        model, foldermap=foldermap
    )
    res = evaluator.evaluate(dataset_root, "", predictions_cnt)

    # validate results
    assert isinstance(res, list) and len(res) == 2
    assert isinstance(res[0], list) and len(res[0]) == len(evaluator.top_k)
    assert (
        isinstance(res[1], list) and len(res[1]) > 0 and len(res[1]) <= len(foldermap)
    )
    for v in res[1]:
        assert isinstance(v, list) and len(v) == len(evaluator.top_k)


def test_ImageRegressionModelEvaluator(zoo_dir):
    """Test for ImageRegressionModelEvaluator class"""

    import degirum_tools, degirum as dg
    import os, io

    # load regression model
    model = dg.load_model(
        "yolov8n_relu6_age--256x256_quant_tflite_cpu_1", dg.LOCAL, zoo_dir
    )

    #
    # test evaluator creation
    #

    # default parameters
    evaluator_default = degirum_tools.ImageRegressionModelEvaluator(model)
    assert not evaluator_default.show_progress
    assert model.input_pad_method == "crop-last"
    assert model.image_backend == "opencv"

    # handling invalid parameters
    with pytest.raises(Exception):
        degirum_tools.ImageRegressionModelEvaluator.init_from_yaml(
            model, io.StringIO("non_existent_parameter: 0")
        )

    # test parameters
    evaluator1 = degirum_tools.ImageRegressionModelEvaluator.init_from_yaml(
        model,
        io.StringIO(
            """
                show_progress: true
                input_pad_method: "letterbox"
                image_backend: "pil"
            """
        ),
    )
    assert evaluator1.show_progress
    assert model.input_pad_method == "letterbox"
    assert model.image_backend == "pil"

    #
    # test model evaluation
    #

    dataset_root = "test/sample_regression_dataset"

    # run evaluation
    predictions_cnt = 50
    evaluator = degirum_tools.ImageRegressionModelEvaluator(model)
    res = evaluator.evaluate(
        dataset_root, dataset_root + "/annotations.json", predictions_cnt
    )

    # validate results
    assert isinstance(res, list) and len(res) == 1
    assert isinstance(res[0], list) and len(res[0]) == 2
