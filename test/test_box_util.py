import numpy as np
from dgtools import area, intersection


def test_area():
    unit_square = np.array([0, 0, 1, 1])
    assert np.allclose(area(unit_square), 1)
    unit_squares = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]])
    assert np.allclose(area(unit_squares), 1)


def test_intersection():
    unit_square_origin0 = np.array([0, 0, 1, 1])
    assert np.allclose(intersection(unit_square_origin0, unit_square_origin0), 1)
    unit_square_origin1 = np.array([1, 1, 2, 2])
    assert np.allclose(intersection(unit_square_origin0, unit_square_origin1), 0)
