#
# test_figure_annotator.py: unit tests for figure annotator functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test figure annotator functionality
#

import numpy as np
from typing import List


def test_figure_annotator():
    """
    Test for FigureAnnotator
    """

    from degirum_tools import FigureAnnotator, Grid

    #
    # Test Grid class
    #

    test_case = "static lin_func method"
    assert Grid.lin_func(m=0.5, b=1, x=4) == 3, ": ".join(
        [test_case, "lin_func return value does not match expected"]
    )

    test_case = "Freshly-instantiated grid"
    grid = Grid("0")

    assert (
        grid.grid_id == "0"
        and grid.ids == []
        and grid.points == []
        and grid.displayed_points == []
        and grid.top_m == 0.0
        and grid.top_b == 0.0
        and grid.bottom_m == 0.0
        and grid.bottom_b == 0.0
    ), ": ".join([test_case, "grid's initial state is not as expected"])
    assert not grid.complete(), ": ".join([test_case, "grid is complete"])
    assert grid.mostly_horizontal(), ": ".join(
        [test_case, "mostly_horizontal() return value does not match expected"]
    )
    assert grid.get_temp_polygon() == ([], []), ": ".join(
        [test_case, "get_temp_polygon() return value does not match expected"]
    )
    assert grid.get_grid_polygons() == [], ": ".join(
        [test_case, "get_grid_polygons() return value does not match expected"]
    )
    assert grid.get_grid_polygons(display=False) == [], ": ".join(
        [
            test_case,
            "get_grid_polygons(display=False) return value does not match expected",
        ]
    )

    original_width, original_height = 1000, 1500
    current_width, current_height = 2000, 3000
    horizontal_grid_points = [
        (100, 100),
        (100, 300),
        (500, 300),
        (500, 100),
        (200, 50),
        (200, 250),
    ]
    horizontal_grid_points_displayed = [
        (
            point[0] * current_width / original_width,
            point[1] * current_height / original_height,
        )
        for point in horizontal_grid_points
    ]
    vertical_grid_points = [
        (400, 100),
        (200, 100),
        (200, 500),
        (400, 500),
        (500, 300),
        (150, 300),
    ]
    vertical_grid_points_displayed = [
        (
            point[0] * current_width / original_width,
            point[1] * current_height / original_height,
        )
        for point in vertical_grid_points
    ]

    test_case = "Add one point to grid, grid incomplete"
    grid.process_point_addition(horizontal_grid_points[0])
    grid.update_displayed_points(
        current_width, current_height, original_width, original_height
    )

    assert (
        grid.grid_id == "0"
        and grid.points == horizontal_grid_points[0:1]
        and grid.displayed_points == horizontal_grid_points_displayed[0:1]
        and grid.top_m == 0.0
        and grid.top_b == 0.0
        and grid.bottom_m == 0.0
        and grid.bottom_b == 0.0
    ), ": ".join([test_case, "grid's state is not as expected"])
    assert not grid.complete(), ": ".join([test_case, "grid is complete"])
    assert grid.mostly_horizontal(), ": ".join(
        [test_case, "mostly_horizontal() return value does not match expected"]
    )
    assert grid.get_temp_polygon() == (
        horizontal_grid_points_displayed[0:1],
        [0],
    ), ": ".join([test_case, "get_temp_polygon() return value does not match expected"])
    assert grid.get_grid_polygons() == [], ": ".join(
        [test_case, "get_grid_polygons() return value does not match expected"]
    )
    assert grid.get_grid_polygons(display=False) == [], ": ".join(
        [
            test_case,
            "get_grid_polygons(display=False) return value does not match expected",
        ]
    )

    test_case = "Add three points to grid, grid incomplete"
    grid.process_point_addition(horizontal_grid_points[1])
    grid.process_point_addition(horizontal_grid_points[2])
    grid.update_displayed_points(
        current_width, current_height, original_width, original_height
    )

    assert (
        grid.grid_id == "0"
        and grid.points == horizontal_grid_points[0:3]
        and grid.displayed_points == horizontal_grid_points_displayed[0:3]
        and grid.top_m == 0.0
        and grid.top_b == 0.0
        and grid.bottom_m == 0.0
        and grid.bottom_b == 0.0
    ), ": ".join([test_case, "grid's state is not as expected"])
    assert not grid.complete(), ": ".join([test_case, "grid is complete"])
    assert grid.mostly_horizontal(), ": ".join(
        [test_case, "mostly_horizontal() return value does not match expected"]
    )
    assert grid.get_temp_polygon() == (grid.displayed_points, [0, 1, 2]), ": ".join(
        [test_case, "get_temp_polygon() return value does not match expected"]
    )
    assert grid.get_grid_polygons() == [], ": ".join(
        [test_case, "get_grid_polygons() return value does not match expected"]
    )
    assert grid.get_grid_polygons(display=False) == [], ": ".join(
        [
            test_case,
            "get_grid_polygons(display=False) return value does not match expected",
        ]
    )

    test_case = "Add four points to grid, grid complete"
    grid.process_point_addition(horizontal_grid_points[3])
    grid.update_displayed_points(
        current_width, current_height, original_width, original_height
    )

    assert (
        grid.grid_id == "0"
        and grid.points == horizontal_grid_points[0:4]
        and grid.displayed_points == horizontal_grid_points_displayed[0:4]
        and grid.top_m == 0.0
        and grid.top_b == 100.0
        and grid.bottom_m == 0.0
        and grid.bottom_b == 300.0
    ), ": ".join([test_case, "grid's state is not as expected"])
    assert grid.complete(), ": ".join([test_case, "grid is not complete"])
    assert grid.mostly_horizontal(), ": ".join(
        [test_case, "mostly_horizontal() return value does not match expected"]
    )
    assert grid.get_temp_polygon() == ([], []), ": ".join(
        [test_case, "get_temp_polygon() return value does not match expected"]
    )
    assert grid.get_grid_polygons() == [
        [list(point) for point in grid.displayed_points]
    ], ": ".join(
        [test_case, "get_grid_polygons() return value does not match expected"]
    )
    assert grid.get_grid_polygons(display=False) == [
        [list(point) for point in grid.points]
    ], ": ".join(
        [
            test_case,
            "get_grid_polygons(display=False) return value does not match expected",
        ]
    )

    test_case = "Add five points to grid, grid incomplete"
    grid.process_point_addition(horizontal_grid_points[4])
    grid.update_displayed_points(
        current_width, current_height, original_width, original_height
    )

    assert (
        grid.grid_id == "0"
        and grid.points
        == horizontal_grid_points[0:2]
        + horizontal_grid_points[4:5]
        + horizontal_grid_points[2:4]
        and grid.displayed_points
        == horizontal_grid_points_displayed[0:2]
        + horizontal_grid_points_displayed[4:5]
        + horizontal_grid_points_displayed[2:4]
        and grid.top_m == 0.0
        and grid.top_b == 100.0
        and grid.bottom_m == 0.0
        and grid.bottom_b == 300.0
    ), ": ".join([test_case, "grid's state is not as expected"])
    assert not grid.complete(), ": ".join([test_case, "grid is complete"])
    assert grid.mostly_horizontal(), ": ".join(
        [test_case, "mostly_horizontal() return value does not match expected"]
    )
    assert grid.get_temp_polygon() == (grid.displayed_points[2:3], [2]), ": ".join(
        [test_case, "get_temp_polygon() return value does not match expected"]
    )
    assert grid.get_grid_polygons() == [
        [list(point) for point in grid.displayed_points]
    ], ": ".join(
        [test_case, "get_grid_polygons() return value does not match expected"]
    )
    assert grid.get_grid_polygons(display=False) == [
        [list(point) for point in grid.points]
    ], ": ".join(
        [
            test_case,
            "get_grid_polygons(display=False) return value does not match expected",
        ]
    )
