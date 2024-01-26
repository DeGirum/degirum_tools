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
    from degirum_tools import nms
    import degirum as dg
    from copy import deepcopy

    res_list = [
        {"bbox": [0, 0, 10, 10], "score": 0.8, "category_id": 0},
        {"bbox": [10, 10, 20, 20], "score": 0.7, "category_id": 0},
        {"bbox": [20, 20, 29, 30], "score": 0.6, "category_id": 0},
        {"bbox": [1, 1, 11, 11], "score": 0.5, "category_id": 0},
        {"bbox": [22, 19, 30, 29], "score": 0.9, "category_id": 0},
        {"bbox": [21, 0, 28, 10], "score": 0.4, "category_id": 0},
        {"bbox": [0, 0, 1, 1], "score": 0.3, "category_id": 0},
    ]

    res = dg.postprocessor.InferenceResults(
        model_params=None, inference_results=res_list, conversion=None
    )

    res_base = deepcopy(res)
    nms(
        res_base,
        iou_threshold=0.3,
        use_iou=True,
        merge_boxes=False,
    )
    assert len(res_base.results) == 5

    res_ios = deepcopy(res)
    nms(
        res_ios,
        iou_threshold=0.3,
        use_iou=False,
        merge_boxes=False,
    )
    assert len(res_ios.results) == 4
    for i, r in enumerate(res_ios.results):
        assert r == res_base.results[i]

    res_merge = deepcopy(res)
    nms(
        res_merge,
        iou_threshold=0.3,
        use_iou=True,
        merge_boxes=True,
    )

    assert len(res_merge.results) == len(res_base.results)
    assert (
        sum(base != merge for base, merge in zip(res_base.results, res_merge.results))
        == 2
    )
