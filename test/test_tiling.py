from degirum_tools.tile_compound_models import TileExtractorPseudoModel
from degirum_tools.math_support import edge_box_fusion
import numpy as np, pytest


def test_bf():
    """Test for box fusion cases"""
    res_list = [
        {"bbox": [10, 40, 20, 50], "score": 0.8, "category_id": "1"},
        {"bbox": [10, 90, 20, 100], "score": 0.7, "category_id": "1"},
        {"bbox": [50, 90, 60, 100], "score": 0.6, "category_id": "1"},
        {"bbox": [55, 95, 65, 105], "score": 0.5, "category_id": "1"},
        {"bbox": [11, 45, 21, 55], "score": 0.9, "category_id": "1"},
    ]

    res_list_2 = [
        {"bbox": [0, 0, 10, 10], "score": 0.8, "category_id": "1"},
        {"bbox": [9, 0, 19, 10], "score": 0.8, "category_id": "1"},
        {"bbox": [0, 9, 10, 19], "score": 0.8, "category_id": "1"},
        {"bbox": [9, 9, 19, 19], "score": 0.8, "category_id": "1"},
    ]

    # Normalize the widths and heights for box fusion
    for res in res_list:
        res["wbf_info"] = [res["bbox"][0] / 500, res["bbox"][1] / 500, res["bbox"][2] / 500, res["bbox"][3] / 500]  # type: ignore[index]
    for res in res_list_2:
        res["wbf_info"] = [res["bbox"][0] / 500, res["bbox"][1] / 500, res["bbox"][2] / 500, res["bbox"][3] / 500]  # type: ignore[index]

    # case where the overlap > 0.8 in the y dimension, no overlap in the x dimension, expect no fusion
    fusion_result = edge_box_fusion([res_list[1], res_list[2]], 0.8, 0.3)
    assert len(fusion_result) == 2

    # case where overlap in both dimensions, 1D-IOU < 0.8 for both, expect no fusion
    fusion_result = edge_box_fusion([res_list[2], res_list[3]], 0.8, 0.3)
    assert len(fusion_result) == 2

    # case where overlap in both dimensions, x dimension 1D-IOU > 0.8, expect fusion
    fusion_result = edge_box_fusion([res_list[0], res_list[4]], 0.8, 0.3)
    assert len(fusion_result) == 1

    # case where overlap in both dimensions, x dimension 1D-IOU > 0.8, objects are not the same class, expect no fusion
    res_list[4]["category_id"] = 2
    fusion_result = edge_box_fusion([res_list[0], res_list[4]], 0.8, 0.3)
    assert len(fusion_result) == 2
    res_list[4]["category_id"] = 1

    # case where overlap both dimensions, x dim 1D-IOU > 0.8, one box score is less than the score threshold, expect no fusion
    res_list[4]["score"] = 0.2
    fusion_result = edge_box_fusion(
        [res_list[0], res_list[4]], 0.8, 0.3, destructive=False
    )
    assert len(fusion_result) == 2
    res_list[4]["score"] = 0.8

    # All boxes (order matters to check if masking feature works in the IoU matching)
    res_list.append(res_list[1])
    res_list[1] = res_list[4]
    res_list.pop(4)
    fusion_result = edge_box_fusion(res_list, 0.8, 0.3)
    assert len(fusion_result) == 4

    # Corner case, fusion of four boxes at corners
    fusion_result = edge_box_fusion(res_list_2, 0.8, 0.3)
    assert len(fusion_result) == 1


def test_generate_tiles():
    """
    Tests for generating tiles.
    Includes variations of aspect ratios for the model and tiles, rows/cols, and overlap percentages.
    """
    # Tolerances account for rounding errors due to discrete nature of pixels.
    overlap_tolerance = 0.01
    aspect_ratio_tolerance = 0.002

    class DummyModelParams:
        InputW = [640]
        InputH = [640]
        InputC = [3]
        InputImgFmt = ["RAW"]
        InputRawDataType = ["DG_UINT8"]
        InputColorSpace = ["RGB"]

    class DummyModel:
        image_backend = "auto"
        input_image_format = "RAW"
        _model_parameters = DummyModelParams()
        model_info = _model_parameters
        overlay_color = (0, 0, 0)
        overlay_line_width = 1.0
        overlay_show_labels = True
        overlay_show_probabilities = True
        overlay_alpha = False
        overlay_font_scale = 1.0
        input_letterbox_fill_color = (0, 0, 0)
        label_dictionary = {0: 0}

    dummy_model = DummyModel()

    # 1 x 1 no overlap square, matching aspect ratio, aspect ratio = 1
    tile_extractor = TileExtractorPseudoModel(1, 1, 0.0, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((100, 100, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]
    assert len(tiles) == 1
    assert tiles[0] == [0, 0, 100, 100]

    # 2 x 2 no overlap square, matching aspect ratio, aspect ratio = 1
    tile_extractor = TileExtractorPseudoModel(2, 2, 0.0, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((100, 100, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]
    assert len(tiles) == 4
    assert tiles[0] == [0, 0, 50, 50]

    # 2 x 2 10% overlap, matching aspect ratio, aspect ratio = 1
    tile_extractor = TileExtractorPseudoModel(2, 2, 0.1, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((640, 640, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]

    width = tiles[0][2]
    height = tiles[0][3]

    tile2_x = tiles[1][0]
    tile3_y = tiles[2][1]

    assert abs(((width - tile2_x) / width) - 0.1) <= overlap_tolerance
    assert abs(((height - tile3_y) / height) - 0.1) <= overlap_tolerance

    # 2 x 2 rectangle, 10% overlap, matching aspect ratio, aspect ratio = 1.66
    dummy_model._model_parameters.InputH = [384]
    tile_extractor = TileExtractorPseudoModel(2, 2, 0.1, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((384, 640, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]

    assert len(tiles) == 4

    width = tiles[0][2]
    height = tiles[0][3]

    tile2_x = tiles[1][0]
    tile3_y = tiles[2][1]

    assert abs(((width - tile2_x) / width) - 0.1) <= overlap_tolerance
    assert abs(((height - tile3_y) / height) - 0.1) <= overlap_tolerance

    # 2 x 2 rectangle, model aspect ratio > image aspect ratio, w >= h
    # model aspect ratio = 1, image aspect ratio = 1.666666
    # expect forced overlap in the y dimension
    dummy_model._model_parameters.InputH = [640]
    tile_extractor = TileExtractorPseudoModel(2, 2, 0.0, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((384, 640, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]

    width = tiles[0][2]
    height = tiles[0][3]

    tile2_x = tiles[1][0]
    tile3_y = tiles[2][1]

    assert width - tile2_x == 0
    assert (height - tile3_y) / height > overlap_tolerance

    for tile in tiles:
        assert tile[0] >= 0 and tile[0] <= 640
        assert tile[1] >= 0 and tile[1] <= 384
        assert tile[2] >= 0 and tile[2] <= 640
        assert tile[3] >= 0 and tile[3] <= 384

    # 2 x 2 rectangle, model aspect ratio < image aspect ratio, w >= h
    # model aspect ratio = 1.6666, image aspect ratio = 1
    # expect forced overlap in the x dimension
    dummy_model._model_parameters.InputH = [384]
    tile_extractor = TileExtractorPseudoModel(2, 2, 0.0, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((640, 640, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]

    width = tiles[0][2]
    height = tiles[0][3]

    tile2_x = tiles[1][0]
    tile3_y = tiles[2][1]

    assert (width - tile2_x) / width > overlap_tolerance
    assert height - tile3_y == 0

    for tile in tiles:
        assert tile[0] >= 0 and tile[0] <= 640
        assert tile[1] >= 0 and tile[1] <= 640
        assert tile[2] >= 0 and tile[2] <= 640
        assert tile[3] >= 0 and tile[3] <= 640

    # too many columns, overlap percent due to aspect aware reshaping is greater than 100
    with pytest.raises(Exception):
        dummy_model._model_parameters.InputH = [640]
        tile_extractor = TileExtractorPseudoModel(3, 2, 0.1, dummy_model)
        list(tile_extractor.predict_batch([np.zeros((640, 100, 3), dtype=np.uint8)]))

    # too many rows, overlap percent due to aspect aware reshaping is greater than 100
    with pytest.raises(Exception):
        tile_extractor = TileExtractorPseudoModel(3, 2, 0.1, dummy_model)
        list(tile_extractor.predict_batch([np.zeros((100, 640, 3), dtype=np.uint8)]))

    # All aspect ratio expansion checks (combinations where the aspect ratios are > or < 1)
    # model aspect ratio <1 (0.666), tile aspect ratio <1 (0.9), tile aspect_ratio > model_aspect ratio
    dummy_model._model_parameters.InputH = [600]
    dummy_model._model_parameters.InputW = [400]
    tile_extractor = TileExtractorPseudoModel(2, 2, 0.1, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((1000, 900, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]
    assert abs(tiles[0][2] / tiles[0][3] - 2 / 3) < aspect_ratio_tolerance

    # model aspect ratio <1 (0.666), tile aspect ratio <1 (0.611), tile aspect_ratio < model_aspect ratio
    dummy_model._model_parameters.InputH = [600]
    dummy_model._model_parameters.InputW = [400]
    tile_extractor = TileExtractorPseudoModel(3, 2, 0.1, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((1000, 900, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]
    assert abs(tiles[0][2] / tiles[0][3] - 2 / 3) < aspect_ratio_tolerance

    # model aspect ratio <1 (0.666), tile aspect ratio > 1 (1.326), tile aspect_ratio > model_aspect ratio
    dummy_model._model_parameters.InputH = [600]
    dummy_model._model_parameters.InputW = [400]
    tile_extractor = TileExtractorPseudoModel(2, 3, 0.1, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((1000, 900, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]
    assert abs(tiles[0][2] / tiles[0][3] - 2 / 3) < aspect_ratio_tolerance

    # model aspect ratio >1 (1.5), tile aspect ratio < 1 (0.611), tile aspect_ratio < model_aspect ratio
    dummy_model._model_parameters.InputH = [400]
    dummy_model._model_parameters.InputW = [600]
    tile_extractor = TileExtractorPseudoModel(3, 2, 0.1, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((1000, 900, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]
    assert abs(tiles[0][2] / tiles[0][3] - 3 / 2) < aspect_ratio_tolerance

    # model aspect ratio >1 (1.5), tile aspect ratio > 1 (1.753), tile aspect_ratio > model_aspect ratio
    dummy_model._model_parameters.InputH = [400]
    dummy_model._model_parameters.InputW = [600]
    tile_extractor = TileExtractorPseudoModel(2, 4, 0.1, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((1000, 900, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]
    assert abs(tiles[0][2] / tiles[0][3] - 3 / 2) < aspect_ratio_tolerance

    # model aspect ratio >1 (1.5), tile aspect ratio > 1 (1.326), tile aspect_ratio < model_aspect ratio
    dummy_model._model_parameters.InputH = [400]
    dummy_model._model_parameters.InputW = [600]
    tile_extractor = TileExtractorPseudoModel(2, 3, 0.1, dummy_model)
    results = list(
        tile_extractor.predict_batch([np.zeros((1000, 900, 3), dtype=np.uint8)])
    )[0]
    tiles = [x["bbox"] for x in results.results]
    assert abs(tiles[0][2] / tiles[0][3] - 3 / 2) < aspect_ratio_tolerance
