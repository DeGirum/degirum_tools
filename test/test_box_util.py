import numpy as np
from dgtools.dgtools import area, intersection

def test_area():
  unit_square = np.array([0, 0, 1, 1])
  assert np.allclose(area(unit_square), 1)
  
  unit_squares = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]])
  assert np.allclose(area(unit_squares), 1)
