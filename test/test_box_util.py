#
# test_box_util.py: unit tests for bbox processing functions
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests for bbox processing functions
#

import numpy as np


def test_area():
    from degirum_tools import area

    unit_square = np.array([0, 0, 1, 1])
    assert np.allclose(area(unit_square), 1)
    unit_squares = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]])
    assert np.allclose(area(unit_squares), 1)


def test_intersection():
    from degirum_tools import intersection

    unit_square_origin0 = np.array([0, 0, 1, 1])
    assert np.allclose(intersection(unit_square_origin0, unit_square_origin0), 1)
    unit_square_origin1 = np.array([1, 1, 2, 2])
    assert np.allclose(intersection(unit_square_origin0, unit_square_origin1), 0)


def test_nms():
    from degirum_tools import nms, NmsBoxSelectionPolicy
    import degirum as dg
    from copy import deepcopy

    res_list = [
        {"bbox": [0, 0, 10, 10], "score": 0.8, "label": ""},
        {"bbox": [10, 10, 20, 20], "score": 0.7, "label": ""},
        {"bbox": [20, 20, 29, 30], "score": 0.6, "label": ""},
        {"bbox": [1, 1, 11, 12], "score": 0.5, "label": ""},
        {"bbox": [22, 19, 30, 29], "score": 0.9, "label": ""},
        {"bbox": [21, 0, 28, 10], "score": 0.4, "label": ""},
        {"bbox": [0, 0, 1, 1], "score": 0.3, "label": ""},
    ]

    def area(res):
        return np.prod(np.array(res["bbox"][2:]) - np.array(res["bbox"][:2]))

    def bbox_is_in(res, res_list):
        return any(np.allclose(res["bbox"], r["bbox"]) for r in res_list)

    res = dg.postprocessor.InferenceResults(
        model_params=None, inference_results=res_list, conversion=None
    )

    # test default case: IoU, no merge
    res_base = deepcopy(res)
    nms(
        res_base,
        iou_threshold=0.3,
        use_iou=True,
    )
    assert len(res_base.results) == 5

    # test max IoU threshold: no suppression
    res_maxthr = deepcopy(res)
    nms(
        res_maxthr,
        iou_threshold=1.0,
        use_iou=True,
    )
    assert len(res_maxthr.results) == len(res_list)

    # test IoS: fully covered box is now suppressed
    res_ios = deepcopy(res)
    nms(
        res_ios,
        iou_threshold=0.3,
        use_iou=False,
    )
    assert len(res_ios.results) == 4
    for i, r in enumerate(res_ios.results):
        assert r == res_base.results[i]

    # test box averaging
    res_avg = deepcopy(res)
    nms(
        res_avg,
        iou_threshold=0.3,
        use_iou=True,
        box_select=NmsBoxSelectionPolicy.AVERAGE,
    )
    assert len(res_avg.results) == len(res_base.results)
    assert (
        sum(
            base["bbox"] != avg["bbox"]
            for base, avg in zip(res_base.results, res_avg.results)
        )
        == 2
    )
    assert (
        sum(bbox_is_in(avg, res_list) for avg in res_avg.results)
        == len(res_avg.results) - 2
    )

    # test largest box
    res_largest = deepcopy(res)
    nms(
        res_largest,
        iou_threshold=0.3,
        use_iou=True,
        box_select=NmsBoxSelectionPolicy.LARGEST_AREA,
    )
    assert len(res_largest.results) == len(res_base.results)
    assert (
        sum(
            area(base) < area(lg)
            for base, lg in zip(res_base.results, res_largest.results)
        )
        == 2
    )
    assert sum(bbox_is_in(lg, res_list) for lg in res_largest.results) == len(
        res_largest.results
    )

    # test merge boxes
    res_merge = deepcopy(res)
    nms(
        res_merge,
        iou_threshold=0.3,
        use_iou=True,
        box_select=NmsBoxSelectionPolicy.MERGE,
    )
    assert len(res_merge.results) == len(res_merge.results)
    assert (
        sum(
            area(base) < area(merge)
            for base, merge in zip(res_base.results, res_merge.results)
        )
        == 2
    )
    assert (
        sum(bbox_is_in(merge, res_list) for merge in res_merge.results)
        == len(res_merge.results) - 2
    )

    # test unique class labels: no suppression
    res_unique = deepcopy(res)
    for i, r in enumerate(res_unique._inference_results):
        r["label"] = str(i)

    nms(
        res_unique,
        iou_threshold=0.3,
        use_iou=True,
    )
    assert len(res_unique.results) == len(res_list)


def test_generate_tiles():
    from degirum_tools import generate_tiles_fixed_size, generate_tiles_fixed_ratio

    # single tile
    assert generate_tiles_fixed_size([200, 150], [200, 150], [0, 0]).shape == (1, 1, 4)

    # 4x2 tiles, zero overlap
    assert generate_tiles_fixed_size([200, 150], [800, 300], [0, 0]).shape == (2, 4, 4)

    # 5x3 tiles, some overlap
    assert generate_tiles_fixed_size([200, 150], [800, 300], [10, 20]).shape == (
        3,
        5,
        4,
    )

    # single tile
    assert generate_tiles_fixed_ratio(2.0, [1, 0], [200, 100], 0).shape == (1, 1, 4)

    # 2x2 tiles
    assert generate_tiles_fixed_ratio(0.5, [2, 0], [200, 300], [10, 20]).shape == (
        2,
        2,
        4,
    )

    # 3x3 tiles
    assert generate_tiles_fixed_ratio(1.5, [0, 3], [600, 400], 10).shape == (3, 4, 4)
