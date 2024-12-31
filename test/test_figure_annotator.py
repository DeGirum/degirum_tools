#
# test_figure_annotator.py: unit tests for figure annotator functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test figure annotator functionality
#


def test_figure_annotator():
    """
    Test for FigureAnnotator
    """

    from degirum_tools import FigureAnnotator, Grid

    #
    # Test Grid class
    #

    test_case = "Static lin_func method"
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
        (100.0, 100.0),
        (100.0, 300.0),
        (500.0, 300.0),
        (500.0, 100.0),
        (200.0, 50.0),
        (200.0, 250.0),
    ]
    horizontal_grid_points_displayed = [
        (
            (
                point[0] * current_width / original_width,
                (
                    point[1]
                    if idx < 4
                    else (
                        horizontal_grid_points[1][1]
                        if idx % 2
                        else horizontal_grid_points[0][1]
                    )
                )
                * current_height
                / original_height,
            )
        )
        for idx, point in enumerate(horizontal_grid_points)
    ]
    vertical_grid_points = [
        (400.0, 100.0),
        (200.0, 100.0),
        (200.0, 500.0),
        (400.0, 500.0),
        (500.0, 300.0),
        (150.0, 300.0),
    ]
    vertical_grid_points_displayed = [
        (
            (
                point[0]
                if idx < 4
                else (
                    vertical_grid_points[1][0]
                    if idx % 2
                    else vertical_grid_points[0][0]
                )
            )
            * current_width
            / original_width,
            point[1] * current_height / original_height,
        )
        for idx, point in enumerate(vertical_grid_points)
    ]

    # Horizontal grid creation

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
        + [(horizontal_grid_points[4][0], float(horizontal_grid_points[0][1]))]
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
        [list(point) for idx, point in enumerate(grid.displayed_points) if idx != 2]
    ], ": ".join(
        [test_case, "get_grid_polygons() return value does not match expected"]
    )
    assert grid.get_grid_polygons(display=False) == [
        [list(point) for idx, point in enumerate(grid.points) if idx != 2]
    ], ": ".join(
        [
            test_case,
            "get_grid_polygons(display=False) return value does not match expected",
        ]
    )

    test_case = "Add six points to grid, grid complete"
    grid.process_point_addition(horizontal_grid_points[5])
    grid.update_displayed_points(
        current_width, current_height, original_width, original_height
    )

    assert (
        grid.grid_id == "0"
        and grid.points
        == horizontal_grid_points[0:2]
        + [
            (horizontal_grid_points[4][0], float(horizontal_grid_points[0][1])),
            (horizontal_grid_points[5][0], float(horizontal_grid_points[1][1])),
        ]
        + horizontal_grid_points[2:4]
        and grid.displayed_points
        == horizontal_grid_points_displayed[0:2]
        + horizontal_grid_points_displayed[4:6]
        + horizontal_grid_points_displayed[2:4]
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
        [list(grid.displayed_points[i]) for i in [0, 1, 4, 5]],
        [list(grid.displayed_points[i]) for i in [0, 1, 3, 2]],
        [list(grid.displayed_points[i]) for i in [2, 3, 4, 5]],
    ], ": ".join(
        [test_case, "get_grid_polygons() return value does not match expected"]
    )
    assert grid.get_grid_polygons(display=False) == [
        [list(grid.points[i]) for i in [0, 1, 4, 5]],
        [list(grid.points[i]) for i in [0, 1, 3, 2]],
        [list(grid.points[i]) for i in [2, 3, 4, 5]],
    ], ": ".join(
        [
            test_case,
            "get_grid_polygons(display=False) return value does not match expected",
        ]
    )

    # Vertical grid creation

    grid = Grid("0")

    test_case = "Add one point to vertical grid, grid incomplete"
    grid.process_point_addition(vertical_grid_points[0])
    grid.update_displayed_points(
        current_width, current_height, original_width, original_height
    )

    assert (
        grid.grid_id == "0"
        and grid.points == vertical_grid_points[0:1]
        and grid.displayed_points == vertical_grid_points_displayed[0:1]
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
        vertical_grid_points_displayed[0:1],
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

    test_case = "Add three points to vertical grid, grid incomplete"
    grid.process_point_addition(vertical_grid_points[1])
    grid.process_point_addition(vertical_grid_points[2])
    grid.update_displayed_points(
        current_width, current_height, original_width, original_height
    )

    assert (
        grid.grid_id == "0"
        and grid.points == vertical_grid_points[0:3]
        and grid.displayed_points == vertical_grid_points_displayed[0:3]
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

    test_case = "Add four points to vertical grid, grid complete"
    grid.process_point_addition(vertical_grid_points[3])
    grid.update_displayed_points(
        current_width, current_height, original_width, original_height
    )

    assert (
        grid.grid_id == "0"
        and grid.points == vertical_grid_points[0:4]
        and grid.displayed_points == vertical_grid_points_displayed[0:4]
        and grid.top_m == 0.0
        and grid.top_b == 400.0
        and grid.bottom_m == 0.0
        and grid.bottom_b == 200.0
    ), ": ".join([test_case, "grid's state is not as expected"])
    assert grid.complete(), ": ".join([test_case, "grid is not complete"])
    assert not grid.mostly_horizontal(), ": ".join(
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

    test_case = "Add five points to vertical grid, grid incomplete"
    grid.process_point_addition(vertical_grid_points[4])
    grid.update_displayed_points(
        current_width, current_height, original_width, original_height
    )

    assert (
        grid.grid_id == "0"
        and grid.points
        == vertical_grid_points[0:2]
        + [(vertical_grid_points[0][0], float(vertical_grid_points[4][1]))]
        + vertical_grid_points[2:4]
        and grid.displayed_points
        == vertical_grid_points_displayed[0:2]
        + vertical_grid_points_displayed[4:5]
        + vertical_grid_points_displayed[2:4]
        and grid.top_m == 0.0
        and grid.top_b == 400.0
        and grid.bottom_m == 0.0
        and grid.bottom_b == 200.0
    ), ": ".join([test_case, "grid's state is not as expected"])
    assert not grid.complete(), ": ".join([test_case, "grid is complete"])
    assert not grid.mostly_horizontal(), ": ".join(
        [test_case, "mostly_horizontal() return value does not match expected"]
    )
    assert grid.get_temp_polygon() == (grid.displayed_points[2:3], [2]), ": ".join(
        [test_case, "get_temp_polygon() return value does not match expected"]
    )
    assert grid.get_grid_polygons() == [
        [list(point) for idx, point in enumerate(grid.displayed_points) if idx != 2]
    ], ": ".join(
        [test_case, "get_grid_polygons() return value does not match expected"]
    )
    assert grid.get_grid_polygons(display=False) == [
        [list(point) for idx, point in enumerate(grid.points) if idx != 2]
    ], ": ".join(
        [
            test_case,
            "get_grid_polygons(display=False) return value does not match expected",
        ]
    )

    test_case = "Add six points to vertical grid, grid complete"
    grid.process_point_addition(vertical_grid_points[5])
    grid.update_displayed_points(
        current_width, current_height, original_width, original_height
    )

    assert (
        grid.grid_id == "0"
        and grid.points
        == vertical_grid_points[0:2]
        + [
            (vertical_grid_points[0][0], float(vertical_grid_points[4][1])),
            (vertical_grid_points[1][0], float(vertical_grid_points[5][1])),
        ]
        + vertical_grid_points[2:4]
        and grid.displayed_points
        == vertical_grid_points_displayed[0:2]
        + vertical_grid_points_displayed[4:6]
        + vertical_grid_points_displayed[2:4]
        and grid.top_m == 0.0
        and grid.top_b == 400.0
        and grid.bottom_m == 0.0
        and grid.bottom_b == 200.0
    ), ": ".join([test_case, "grid's state is not as expected"])
    assert grid.complete(), ": ".join([test_case, "grid is not complete"])
    assert not grid.mostly_horizontal(), ": ".join(
        [test_case, "mostly_horizontal() return value does not match expected"]
    )
    assert grid.get_temp_polygon() == ([], []), ": ".join(
        [test_case, "get_temp_polygon() return value does not match expected"]
    )
    assert grid.get_grid_polygons() == [
        [list(grid.displayed_points[i]) for i in [0, 1, 4, 5]],
        [list(grid.displayed_points[i]) for i in [0, 1, 3, 2]],
        [list(grid.displayed_points[i]) for i in [2, 3, 4, 5]],
    ], ": ".join(
        [test_case, "get_grid_polygons() return value does not match expected"]
    )
    assert grid.get_grid_polygons(display=False) == [
        [list(grid.points[i]) for i in [0, 1, 4, 5]],
        [list(grid.points[i]) for i in [0, 1, 3, 2]],
        [list(grid.points[i]) for i in [2, 3, 4, 5]],
    ], ": ".join(
        [
            test_case,
            "get_grid_polygons(display=False) return value does not match expected",
        ]
    )

    #
    # Test FigureAnnotator class
    #

    # Create line annotator

    test_case = "Freshly-instantiated line annotator"
    line_annotator = FigureAnnotator(2, test_mode=True)

    assert (
        line_annotator.num_vertices == 2
        and line_annotator.figure_type == "line"
        and not line_annotator.with_grid
        and line_annotator.points == []
        and line_annotator.displayed_points == []
        and line_annotator.original_width == 0
        and line_annotator.original_height == 0
        and line_annotator.current_width == 0
        and line_annotator.current_height == 0
    ), ": ".join([test_case, "annotator's state is not as expected"])

    test_case = "Test figures_empty, line annotator does not have any annotations"
    assert line_annotator.figures_empty(), ": ".join(
        {test_case, "figures_empty returns false"}
    )

    test_case = "Test figures_complete, line annotator does not have any annotations"
    assert line_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns false"}
    )

    test_case = "Freshly-instantiated zone annotator"
    zone_annotator = FigureAnnotator(4, test_mode=True)

    assert (
        zone_annotator.num_vertices == 4
        and zone_annotator.figure_type == "zone"
        and zone_annotator.with_grid
        and zone_annotator.points == []
        and zone_annotator.displayed_points == []
        and zone_annotator.original_width == 0
        and zone_annotator.original_height == 0
        and zone_annotator.current_width == 0
        and zone_annotator.current_height == 0
    ), ": ".join([test_case, "annotator's state is not as expected"])

    test_case = "Test figures_empty, zone annotator does not have any annotations"
    assert zone_annotator.figures_empty(), ": ".join(
        {test_case, "figures_empty returns false"}
    )

    test_case = "Test figures_complete, zone annotator does not have any annotations"
    assert zone_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns false"}
    )

    test_case = "Test figures_complete, line annotator has incomplete annotations"
    line_annotator.points.append(horizontal_grid_points[0])

    assert not line_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns true"}
    )

    test_case = "Test figures_complete, line annotator has complete annotations"
    line_annotator.points.append(horizontal_grid_points[1])

    assert line_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns false"}
    )

    test_case = "Test figures_empty, line annotator has annotations"
    assert not line_annotator.figures_empty(), ": ".join(
        {test_case, "figures_empty returns true"}
    )

    test_case = "Test figures_complete, zone annotator has incomplete grid annotations"
    zone_annotator.grids[0] = Grid("0")
    zone_annotator.grids[0].process_point_addition(horizontal_grid_points[0])

    assert not zone_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns true"}
    )

    test_case = "Test figures_complete, zone annotator has complete grid annotations"
    zone_annotator.grids[0].points = horizontal_grid_points[:4]

    assert zone_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns false"}
    )

    test_case = "Test figures_complete, zone annotator has incomplete zone annotations"
    zone_annotator.points = horizontal_grid_points[:2]

    assert not zone_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns true"}
    )

    test_case = "Test figures_complete, zone annotator has complete zone annotations"
    zone_annotator.points = horizontal_grid_points[:4]

    assert zone_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns false"}
    )

    test_case = "Test figures_complete, zone annotator has incomplete zone and incomplete grid annotations"
    zone_annotator.grids[0].points = horizontal_grid_points[:2]
    zone_annotator.points = horizontal_grid_points[:2]

    assert not zone_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns true"}
    )

    test_case = "Test figures_complete, zone annotator has complete zone and incomplete grid annotations"
    zone_annotator.grids[0].points = horizontal_grid_points[:2]
    zone_annotator.points = horizontal_grid_points[:4]

    assert not zone_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns true"}
    )

    test_case = "Test figures_complete, zone annotator has incomplete zone and complete grid annotations"
    zone_annotator.grids[0].points = horizontal_grid_points[:4]
    zone_annotator.points = horizontal_grid_points[:2]

    assert not zone_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns true"}
    )

    test_case = "Test figures_complete, zone annotator has complete zone and complete grid annotations"
    zone_annotator.grids[0].points = horizontal_grid_points[:4]
    zone_annotator.points = horizontal_grid_points[:4]

    assert zone_annotator.figures_complete(), ": ".join(
        {test_case, "figures_complete returns false"}
    )

    test_case = "Test figures_empty, zone annotator has annotations"
    assert not zone_annotator.figures_empty(), ": ".join(
        {test_case, "figures_empty returns true"}
    )

    test_case = "Test update_displayed_points"
    line_annotator.original_width = original_width
    line_annotator.original_height = original_height
    line_annotator.current_width = current_width
    line_annotator.current_height = current_height

    line_annotator.update_displayed_points()

    assert (
        line_annotator.displayed_points == horizontal_grid_points_displayed[:2]
    ), ": ".join([test_case, "displayed points are not as expected"])

    test_case = "Test find_point, cursor point is not near any annotated points"
    assert line_annotator.find_point(110, 115, line_annotator.points) == (
        None,
        None,
    ), ": ".join([test_case, "find_point's return value does not match expected"])

    test_case = "Test find_point, cursor point is near an annotated point"
    assert line_annotator.find_point(104, 96, line_annotator.points) == (
        (100, 100),
        0,
    ), ": ".join([test_case, "find_point's return value does not match expected"])

    test_case = "Test is_near_line, point is not near the line"
    assert not line_annotator.is_near_line(
        110, 200, [*horizontal_grid_points[0], *horizontal_grid_points[1]]
    ), ": ".join([test_case, "point is near the line"])

    test_case = "Test is_near_line, point is near the line"
    assert line_annotator.is_near_line(
        104, 250, [*horizontal_grid_points[0], *horizontal_grid_points[1]]
    ), ": ".join([test_case, "point is not near the line"])
