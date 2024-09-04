#
# test_figure_annotator.py: unit tests for geometric figure annotator functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test geometric figure annotator functionality
#


def test_figure_annotator_region():
    """
    Test for FigureAnnotator Region class
    """

    import degirum_tools
    from degirum_tools import Region

    #
    # test Region class
    #

    test_case = "freshly-instantiated object, manual-mode"
    region = Region()
    assert region.is_empty(), ": ".join([test_case, "region is not empty"])
    assert region.auto_mode is False, ": ".join(
        [test_case, "region is not in manual mode"]
    )

    test_case = "freshly-instantiated object, auto-mode"
    region = Region(auto_mode=True)
    assert region.is_empty(), ": ".join([test_case, "region is not empty"])
    assert region.auto_mode is True, ": ".join(
        [test_case, "region is not in auto mode"]
    )
    assert (
        region.point_selection_state == degirum_tools.PointSelState.RightAnchors
    ), ": ".join([test_case, "region's point selection state is not in expected state"])

    test_case = "non-empty object"
    region.current_selection.append((20, 30))
    assert not region.is_empty(), ": ".join([test_case, "region is empty"])


def test_figure_annotator():
    """
    Test for FigureAnnotator
    """

    import degirum_tools

    #
    # test FigureAnnotator
    #

    pt_1 = (30, 40)
    pt_2 = (70, 50)
    pt_3 = (10, 20)
    pt_4 = (50, 40)
    pt_5 = (25, 40)
    pt_6 = (65, 70)
    pt_7 = (15, 30)
    pt_8 = (55, 60)

    test_case = "freshly-instantiated object"
    figure_annotator = degirum_tools.FigureAnnotator(debug=True)
    assert len(figure_annotator.regions) == 1, ": ".join(
        [test_case, "new annotator does not contain only one region"]
    )
    assert figure_annotator.regions[0].is_empty(), ": ".join(
        [test_case, "new annotator's region is not empty"]
    )

    test_case = "region parameters computation"
    region = degirum_tools.Region(auto_mode=True)
    region.anchor_points.append([pt_1, pt_2])
    region.anchor_points.append([pt_3, pt_4])
    m_top, b_top, m_bot, b_bot = degirum_tools.FigureAnnotator.region_parameters(region)
    assert (m_top, b_top, m_bot, b_bot) == (1.0, 10.0, 0.5, 15.0), ": ".join(
        [test_case, "region parameters calculated incorrectly"]
    )

    test_case = "linear function computation"
    y = degirum_tools.FigureAnnotator.lin_func(m_bot, b_bot, 30)
    assert y == 30, ": ".join([test_case, "linear function computed incorrectly"])

    test_case = "point addition, region in manual mode"
    current_region = figure_annotator.regions[0]
    current_region.current_selection.append(pt_1)
    title, message = figure_annotator._process_point_addition(
        figure_annotator.regions[0]
    )
    assert not title, ": ".join(
        [test_case, "addition of point resulted in non-empty title"]
    )
    assert not message, ": ".join(
        [test_case, "addition of point resulted in non-empty message"]
    )
    assert len(current_region.figures) == 0, ": ".join(
        [test_case, "figures list is not empty"]
    )
    assert len(current_region.current_selection) == 1, ": ".join(
        [test_case, "region's current selection does not have length 1"]
    )

    current_region.current_selection.append(pt_2)
    figure_annotator._process_point_addition(figure_annotator.regions[0])

    current_region.current_selection.append(pt_3)
    figure_annotator._process_point_addition(figure_annotator.regions[0])

    test_case = "zone addition, region in manual mode"
    current_region.current_selection.append(pt_4)
    title, message = figure_annotator._process_point_addition(
        figure_annotator.regions[0]
    )
    assert title == "Success", ": ".join(
        [test_case, "title does not match expected value"]
    )
    assert message == f"{figure_annotator.figure_type.capitalize()} added.", ": ".join(
        [test_case, "message does not match expected value"]
    )
    assert (
        len(current_region.figures) == 1
        and len(current_region.figures[0]) == figure_annotator.num_vertices
    ), ": ".join([test_case, "figures list does not match expected length"])
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )

    test_case = "point addition, region in auto mode"
    figure_annotator = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator.regions[0]
    current_region.current_selection.append(pt_1)
    title, message = figure_annotator._process_point_addition(
        figure_annotator.regions[0]
    )
    assert not title, ": ".join(
        [test_case, "addition of point resulted in non-empty title"]
    )
    assert not message, ": ".join(
        [test_case, "addition of point resulted in non-empty message"]
    )
    assert len(current_region.anchor_points) == 0, ": ".join(
        [test_case, "anchor points list is not empty"]
    )
    assert len(current_region.current_selection) == 1, ": ".join(
        [test_case, "region's current selection does not have length 1"]
    )
    assert (
        current_region.point_selection_state == degirum_tools.PointSelState.RightAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    current_region.current_selection.append(pt_2)
    title, message = figure_annotator._process_point_addition(
        figure_annotator.regions[0]
    )
    assert title == "Info", ": ".join(
        [test_case, "title does not match expected value"]
    )
    assert message == "Please select two left anchor points.", ": ".join(
        [test_case, "message does not match expected value"]
    )
    assert (
        len(current_region.anchor_points) == 1
        and len(current_region.anchor_points[0]) == 2
    ), ": ".join([test_case, "anchor points list does not match expected length"])
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state == degirum_tools.PointSelState.LeftAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    current_region.current_selection.append(pt_3)
    title, message = figure_annotator._process_point_addition(
        figure_annotator.regions[0]
    )
    assert not title, ": ".join(
        [test_case, "addition of point resulted in non-empty title"]
    )
    assert not message, ": ".join(
        [test_case, "addition of point resulted in non-empty message"]
    )
    assert (
        len(current_region.anchor_points) == 1
        and len(current_region.anchor_points[0]) == 2
    ), ": ".join([test_case, "anchor points list does not match expected length"])
    assert len(current_region.current_selection) == 1, ": ".join(
        [test_case, "region's current selection does not have length 1"]
    )
    assert (
        current_region.point_selection_state == degirum_tools.PointSelState.LeftAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    current_region.current_selection.append(pt_4)
    title, message = figure_annotator._process_point_addition(
        figure_annotator.regions[0]
    )
    assert title == "Info", ": ".join(
        [test_case, "title does not match expected value"]
    )
    assert (
        message
        == "Please select intermediate boundary points, going from right to left."
    ), ": ".join([test_case, "message does not match expected value"])
    assert (
        len(current_region.anchor_points) == 2
        and len(current_region.anchor_points[1]) == 2
    ), ": ".join([test_case, "anchor points list does not match expected length"])
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state
        == degirum_tools.PointSelState.IntermediateAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    current_region.current_selection.append(pt_5)
    title, message = figure_annotator._process_point_addition(
        figure_annotator.regions[0]
    )
    assert not title, ": ".join(
        [test_case, "addition of point resulted in non-empty title"]
    )
    assert not message, ": ".join(
        [test_case, "addition of point resulted in non-empty message"]
    )
    assert (
        len(current_region.anchor_points) == 2
        and len(current_region.anchor_points[1]) == 2
    ), ": ".join([test_case, "anchor points list does not match expected length"])
    assert len(current_region.current_selection) == 1, ": ".join(
        [test_case, "region's current selection does not have length 1"]
    )
    assert (
        current_region.point_selection_state
        == degirum_tools.PointSelState.IntermediateAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    current_region.current_selection.append(pt_6)
    title, message = figure_annotator._process_point_addition(
        figure_annotator.regions[0]
    )
    assert not title, ": ".join(
        [test_case, "addition of point resulted in non-empty title"]
    )
    assert not message, ": ".join(
        [test_case, "addition of point resulted in non-empty message"]
    )
    assert (
        len(current_region.anchor_points) == 2
        and len(current_region.anchor_points[1]) == 2
    ), ": ".join([test_case, "anchor points list does not match expected length"])
    assert (
        len(current_region.intermediate_points) == 1
        and len(current_region.intermediate_points[0]) == 2
    ), ": ".join(test_case, "intermediate points list does not match expected length")
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state
        == degirum_tools.PointSelState.IntermediateAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    current_region.current_selection.append(pt_7)
    title, message = figure_annotator._process_point_addition(
        figure_annotator.regions[0]
    )
    assert not title, ": ".join(
        [test_case, "addition of point resulted in non-empty title"]
    )
    assert not message, ": ".join(
        [test_case, "addition of point resulted in non-empty message"]
    )
    assert (
        len(current_region.anchor_points) == 2
        and len(current_region.anchor_points[1]) == 2
    ), ": ".join([test_case, "anchor points list does not match expected length"])
    assert (
        len(current_region.intermediate_points) == 1
        and len(current_region.intermediate_points[0]) == 2
    ), ": ".join(test_case, "intermediate points list does not match expected length")
    assert len(current_region.current_selection) == 1, ": ".join(
        [test_case, "region's current selection does not have length 1"]
    )
    assert (
        current_region.point_selection_state
        == degirum_tools.PointSelState.IntermediateAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    current_region.current_selection.append(pt_8)
    title, message = figure_annotator._process_point_addition(
        figure_annotator.regions[0]
    )
    assert not title, ": ".join(
        [test_case, "addition of point resulted in non-empty title"]
    )
    assert not message, ": ".join(
        [test_case, "addition of point resulted in non-empty message"]
    )
    assert (
        len(current_region.anchor_points) == 2
        and len(current_region.anchor_points[1]) == 2
    ), ": ".join([test_case, "anchor points list does not match expected length"])
    assert (
        len(current_region.intermediate_points) == 2
        and len(current_region.intermediate_points[1]) == 2
    ), ": ".join(test_case, "intermediate points list does not match expected length")
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state
        == degirum_tools.PointSelState.IntermediateAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    test_case = "region completeness check, one region in manual mode, empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region = degirum_tools.Region()
    figure_annotator_test.regions = [region]
    message_res = (
        f"Cannot compute {figure_annotator_test.figure_type}s because not enough points are specified in\n"
        + "\n  - manual-mode region(s) 0"
    )
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = "region completeness check, one region in manual mode, current selection is non-empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region = degirum_tools.Region()
    region.current_selection.append(pt_1)
    figure_annotator_test.regions = [region]
    message_res = (
        f"Cannot compute {figure_annotator_test.figure_type}s because not enough points are specified in\n"
        + "\n  - manual-mode region(s) 0"
    )
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = "region completeness check, one region in manual mode, complete"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region = degirum_tools.Region()
    region.figures.append([pt_1, pt_2, pt_3, pt_4])
    figure_annotator_test.regions = [region]
    message_res = ""
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = "region completeness check, two regions in manual mode, both empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region1 = degirum_tools.Region()
    region2 = degirum_tools.Region()
    figure_annotator_test.regions = [region1, region2]
    message_res = (
        f"Cannot compute {figure_annotator_test.figure_type}s because not enough points are specified in\n"
        + "\n  - manual-mode region(s) 0, 1"
    )
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = "region completeness check, one region in auto mode, empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region = degirum_tools.Region(auto_mode=True)
    figure_annotator_test.regions = [region]
    message_res = (
        f"Cannot compute {figure_annotator_test.figure_type}s because not enough points are specified in\n"
        + "\n  - automatic-mode region(s) 0"
    )
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = "region completeness check, one region in auto mode, current selection is non-empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region = degirum_tools.Region(auto_mode=True)
    region.current_selection.append(pt_1)
    figure_annotator_test.regions = [region]
    message_res = (
        f"Cannot compute {figure_annotator_test.figure_type}s because not enough points are specified in\n"
        + "\n  - automatic-mode region(s) 0"
    )
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = (
        "region completeness check, one region in auto mode, one pair of anchor points"
    )
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region = degirum_tools.Region(auto_mode=True)
    region.anchor_points.append([pt_1, pt_2])
    figure_annotator_test.regions = [region]
    message_res = (
        f"Cannot compute {figure_annotator_test.figure_type}s because not enough points are specified in\n"
        + "\n  - automatic-mode region(s) 0"
    )
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = (
        "region completeness check, one region in auto mode, incomplete anchor points"
    )
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region = degirum_tools.Region(auto_mode=True)
    region.anchor_points.append([pt_1, pt_2])
    region.anchor_points.append([pt_3])
    figure_annotator_test.regions = [region]
    message_res = (
        f"Cannot compute {figure_annotator_test.figure_type}s because not enough points are specified in\n"
        + "\n  - automatic-mode region(s) 0"
    )
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = "region completeness check, one region in auto mode, incomplete intermediate points"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region = degirum_tools.Region(auto_mode=True)
    region.anchor_points.append([pt_1, pt_2])
    region.anchor_points.append([pt_3, pt_4])
    region.intermediate_points.append([pt_5])
    figure_annotator_test.regions = [region]
    message_res = (
        f"Cannot compute {figure_annotator_test.figure_type}s because not enough points are specified in\n"
        + "\n  - automatic-mode region(s) 0"
    )
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = "region completeness check, one region in auto mode, complete"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region = degirum_tools.Region(auto_mode=True)
    region.anchor_points.append([pt_1, pt_2])
    region.anchor_points.append([pt_3, pt_4])
    region.intermediate_points.append([pt_5, pt_6])
    figure_annotator_test.regions = [region]
    message_res = ""
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = "region completeness check, two regions in auto mode, both empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region1 = degirum_tools.Region(auto_mode=True)
    region2 = degirum_tools.Region(auto_mode=True)
    figure_annotator_test.regions = [region1, region2]
    message_res = (
        f"Cannot compute {figure_annotator_test.figure_type}s because not enough points are specified in\n"
        + "\n  - automatic-mode region(s) 0, 1"
    )
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = "region completeness check, one region in auto mode and one region in manual mode, both empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    region1 = degirum_tools.Region(auto_mode=True)
    region2 = degirum_tools.Region(auto_mode=False)
    figure_annotator_test.regions = [region1, region2]
    message_res = (
        f"Cannot compute {figure_annotator_test.figure_type}s because not enough points are specified in\n"
        + "\n  - automatic-mode region(s) 0"
        + "\n  - manual-mode region(s) 1"
    )
    message = figure_annotator_test._check_regions_completeness()
    assert message == message_res, ": ".join(
        [test_case, "completeness message does not match expected value"]
    )

    test_case = "point deletion, region in manual mode, empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point is None, ": ".join(
        [test_case, "removed point does not match expected value"]
    )

    test_case = "point deletion, region in manual mode, current selection is non-empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions[0].current_selection.append(pt_1)
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point == pt_1, ": ".join(
        [test_case, "removed point does not match expected value"]
    )

    test_case = "point deletion, region in manual mode, figures list is non-empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions[0].figures.append([pt_1, pt_2, pt_3, pt_4])
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point == pt_4, ": ".join(
        [test_case, "removed point does not match expected value"]
    )

    test_case = "point deletion, region in auto mode, empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point is None, ": ".join(
        [test_case, "removed point does not match expected value"]
    )

    test_case = "point deletion, region in auto mode, point selection state is PointSelState.RightAnchors, current selection is not empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.current_selection.append(pt_1)
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point == pt_1, ": ".join(
        [test_case, "removed point does not match expected value"]
    )
    assert current_region.is_empty(), ": ".join([test_case, "region is not empty"])

    test_case = "point deletion, region in auto mode, point selection state is PointSelState.RightAnchors, anchor points list is not empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.anchor_points.append([pt_1])
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point == pt_1, ": ".join(
        [test_case, "removed point does not match expected value"]
    )
    assert current_region.is_empty(), ": ".join([test_case, "region is not empty"])

    test_case = "point deletion, region in auto mode, point selection state is PointSelState.LeftAnchors, current selection is not empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.anchor_points.append([pt_1, pt_2])
    current_region.current_selection.append(pt_3)
    current_region.point_selection_state = degirum_tools.PointSelState.LeftAnchors
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point == pt_3, ": ".join(
        [test_case, "removed point does not match expected value"]
    )
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state == degirum_tools.PointSelState.LeftAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    test_case = "point deletion, region in auto mode, point selection state is PointSelState.LeftAnchors, left anchor points list is not empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.anchor_points.append([pt_1, pt_2])
    current_region.anchor_points.append([pt_3])
    current_region.point_selection_state = degirum_tools.PointSelState.LeftAnchors
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point == pt_3, ": ".join(
        [test_case, "removed point does not match expected value"]
    )
    assert len(current_region.anchor_points) == 1, ": ".join(
        [test_case, "region's anchor points list does not have expected length"]
    )
    assert (
        current_region.point_selection_state == degirum_tools.PointSelState.LeftAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    test_case = "point deletion, region in auto mode, point selection state is PointSelState.LeftAnchors, left anchor points list is empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.anchor_points.append([pt_1, pt_2])
    current_region.point_selection_state = degirum_tools.PointSelState.LeftAnchors
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point == pt_2, ": ".join(
        [test_case, "removed point does not match expected value"]
    )
    assert len(current_region.anchor_points[0]) == 1, ": ".join(
        [test_case, "region's anchor points list does not have expected length"]
    )
    assert (
        current_region.point_selection_state == degirum_tools.PointSelState.RightAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    test_case = "point deletion, region in auto mode, point selection state is PointSelState.IntermediateAnchors, current selection is not empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.anchor_points.append([pt_1, pt_2])
    current_region.anchor_points.append([pt_3, pt_4])
    current_region.current_selection.append(pt_5)
    current_region.point_selection_state = (
        degirum_tools.PointSelState.IntermediateAnchors
    )
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point == pt_5, ": ".join(
        [test_case, "removed point does not match expected value"]
    )
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state
        == degirum_tools.PointSelState.IntermediateAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    test_case = "point deletion, region in auto mode, point selection state is PointSelState.IntermediateAnchors, intermediate anchors list is not empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.anchor_points.append([pt_1, pt_2])
    current_region.anchor_points.append([pt_3, pt_4])
    current_region.intermediate_points.append([pt_5])
    current_region.point_selection_state = (
        degirum_tools.PointSelState.IntermediateAnchors
    )
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point == pt_5, ": ".join(
        [test_case, "removed point does not match expected value"]
    )
    assert len(current_region.intermediate_points) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state
        == degirum_tools.PointSelState.IntermediateAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    test_case = "point deletion, region in auto mode, point selection state is PointSelState.IntermediateAnchors, intermediate anchors list is empty"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.anchor_points.append([pt_1, pt_2])
    current_region.anchor_points.append([pt_3, pt_4])
    current_region.point_selection_state = (
        degirum_tools.PointSelState.IntermediateAnchors
    )
    removed_point = figure_annotator_test._process_point_deletion(0)
    assert removed_point == pt_4, ": ".join(
        [test_case, "removed point does not match expected value"]
    )
    assert len(current_region.anchor_points[1]) == 1, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state == degirum_tools.PointSelState.LeftAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    test_case = "point addition after deletion, region in auto mode, point selection state is PointSelState.RightAnchors"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.anchor_points.append([pt_1, pt_2])
    current_region.point_selection_state = degirum_tools.PointSelState.LeftAnchors
    removed_point = figure_annotator_test._process_point_deletion(0)
    current_region.current_selection.append(pt_3)
    title, message = figure_annotator_test._process_point_addition(
        figure_annotator_test.regions[0]
    )
    assert title == "Info", ": ".join(
        [test_case, "title does not match expected value"]
    )
    assert message == "Please select two left anchor points.", ": ".join(
        [test_case, "message does not match expected value"]
    )
    assert (
        len(current_region.anchor_points) == 1
        and len(current_region.anchor_points[0]) == 2
    ), ": ".join([test_case, "anchor points list does not match expected length"])
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state == degirum_tools.PointSelState.LeftAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    test_case = "point addition after deletion, region in auto mode, point selection state is PointSelState.LeftAnchors"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.anchor_points.append([pt_1, pt_2])
    current_region.anchor_points.append([pt_3, pt_4])
    current_region.point_selection_state = (
        degirum_tools.PointSelState.IntermediateAnchors
    )
    removed_point = figure_annotator_test._process_point_deletion(0)
    current_region.current_selection.append(pt_4)
    title, message = figure_annotator_test._process_point_addition(
        figure_annotator_test.regions[0]
    )
    assert title == "Info", ": ".join(
        [test_case, "title does not match expected value"]
    )
    assert (
        message
        == "Please select intermediate boundary points, going from right to left."
    ), ": ".join([test_case, "message does not match expected value"])
    assert (
        len(current_region.anchor_points) == 2
        and len(current_region.anchor_points[1]) == 2
    ), ": ".join([test_case, "anchor points list does not match expected length"])
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state
        == degirum_tools.PointSelState.IntermediateAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    test_case = "point addition after deletion, region in auto mode, point selection state is PointSelState.IntermediateAnchors"
    figure_annotator_test = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_test.regions = [degirum_tools.Region(auto_mode=True)]
    current_region = figure_annotator_test.regions[0]
    current_region.anchor_points.append([pt_1, pt_2])
    current_region.anchor_points.append([pt_3, pt_4])
    current_region.intermediate_points.append([pt_5, pt_6])
    current_region.point_selection_state = (
        degirum_tools.PointSelState.IntermediateAnchors
    )
    removed_point = figure_annotator_test._process_point_deletion(0)
    current_region.current_selection.append(pt_6)
    title, message = figure_annotator_test._process_point_addition(
        figure_annotator_test.regions[0]
    )
    assert not title, ": ".join([test_case, "title does not match expected value"])
    assert not message, ": ".join([test_case, "message does not match expected value"])
    assert (
        len(current_region.intermediate_points) == 1
        and len(current_region.intermediate_points[0]) == 2
    ), ": ".join([test_case, "intermediate points list does not match expected length"])
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
    assert (
        current_region.point_selection_state
        == degirum_tools.PointSelState.IntermediateAnchors
    ), ": ".join(
        [test_case, "region's point selection state does not match expected value"]
    )

    test_case = "compute figures, one region in auto mode"
    res_regions_json = [
        {
            "figures": [
                {
                    "points": [
                        (
                            30,
                            40.0,
                        ),
                        (
                            70,
                            50.0,
                        ),
                        (
                            65,
                            47.5,
                        ),
                        (
                            25,
                            35.0,
                        ),
                    ],
                },
                {
                    "points": [
                        (
                            25,
                            35.0,
                        ),
                        (
                            65,
                            47.5,
                        ),
                        (
                            55,
                            42.5,
                        ),
                        (
                            15,
                            25.0,
                        ),
                    ],
                },
                {
                    "points": [
                        (
                            15,
                            25.0,
                        ),
                        (
                            55,
                            42.5,
                        ),
                        (
                            50,
                            40.0,
                        ),
                        (
                            10,
                            20.0,
                        ),
                    ],
                },
            ],
        },
    ]

    regions_json = degirum_tools.FigureAnnotator.compute_figures(
        figure_annotator.regions
    )
    assert (
        regions_json == res_regions_json
    ), "regions JSON does not match expected value"

    test_case = "compute figures, one region in manual mode"

    res_regions_json = [
        {
            "figures": [
                {
                    "points": [
                        (
                            30,
                            40,
                        ),
                        (
                            70,
                            50,
                        ),
                        (
                            10,
                            20,
                        ),
                        (
                            50,
                            40,
                        ),
                    ],
                },
            ],
        },
    ]

    region = degirum_tools.Region()
    region.figures.append([pt_1, pt_2, pt_3, pt_4])
    figure_annotator_manual = degirum_tools.FigureAnnotator(debug=True)
    figure_annotator_manual.regions = [region]
    regions_json = degirum_tools.FigureAnnotator.compute_figures(
        figure_annotator_manual.regions
    )
    assert (
        regions_json == res_regions_json
    ), "regions JSON does not match expected value"

    test_case = "compute figures, one region in auto mode and one region in manual mode"

    res_regions_json = [
        {
            "figures": [
                {
                    "points": [
                        (
                            30,
                            40.0,
                        ),
                        (
                            70,
                            50.0,
                        ),
                        (
                            65,
                            47.5,
                        ),
                        (
                            25,
                            35.0,
                        ),
                    ],
                },
                {
                    "points": [
                        (
                            25,
                            35.0,
                        ),
                        (
                            65,
                            47.5,
                        ),
                        (
                            55,
                            42.5,
                        ),
                        (
                            15,
                            25.0,
                        ),
                    ],
                },
                {
                    "points": [
                        (
                            15,
                            25.0,
                        ),
                        (
                            55,
                            42.5,
                        ),
                        (
                            50,
                            40.0,
                        ),
                        (
                            10,
                            20.0,
                        ),
                    ],
                },
            ],
        },
        {
            "figures": [
                {
                    "points": [
                        (
                            30,
                            40,
                        ),
                        (
                            70,
                            50,
                        ),
                        (
                            10,
                            20,
                        ),
                        (
                            50,
                            40,
                        ),
                    ],
                },
            ],
        },
    ]

    figure_annotator.regions.append(region)
    regions_json = degirum_tools.FigureAnnotator.compute_figures(
        figure_annotator.regions
    )
    assert (
        regions_json == res_regions_json
    ), "regions JSON does not match expected value"

    test_case = "regions to JSON conversion, DeGirum-compatible format"
    regions_json = [
        {
            "figures": [
                {
                    "points": [
                        (
                            30,
                            40.0,
                        ),
                        (
                            70,
                            50.0,
                        ),
                        (
                            65,
                            47.5,
                        ),
                        (
                            25,
                            35.0,
                        ),
                    ],
                },
                {
                    "points": [
                        (
                            25,
                            35.0,
                        ),
                        (
                            65,
                            47.5,
                        ),
                        (
                            55,
                            42.5,
                        ),
                        (
                            15,
                            25.0,
                        ),
                    ],
                },
                {
                    "points": [
                        (
                            15,
                            25.0,
                        ),
                        (
                            55,
                            42.5,
                        ),
                        (
                            50,
                            40.0,
                        ),
                        (
                            10,
                            20.0,
                        ),
                    ],
                },
            ],
        },
        {
            "figures": [
                {
                    "points": [
                        (
                            30,
                            40,
                        ),
                        (
                            70,
                            50,
                        ),
                        (
                            10,
                            20,
                        ),
                        (
                            50,
                            40,
                        ),
                    ],
                },
            ],
        },
    ]

    res_regions_formatted = [
        [
            (60, 60),
            (
                140,
                75,
            ),
            (
                130,
                71,
            ),
            (
                50,
                52,
            ),
        ],
        [
            (
                50,
                52,
            ),
            (
                130,
                71,
            ),
            (
                110,
                63,
            ),
            (
                30,
                37,
            ),
        ],
        [
            (
                30,
                37,
            ),
            (
                110,
                63,
            ),
            (
                100,
                60,
            ),
            (
                20,
                30,
            ),
        ],
        [
            (
                60,
                60,
            ),
            (
                140,
                75,
            ),
            (
                20,
                30,
            ),
            (
                100,
                60,
            ),
        ],
    ]

    regions_formatted = degirum_tools.FigureAnnotator.convert_regions_to_json(
        regions_json, 2.0, 1.5, "DeGirum-compatible"
    )
    assert regions_formatted == res_regions_formatted, ": ".join(
        [test_case, "converted regions list does not match expected value"]
    )

    test_case = "regions to JSON conversion, Ultralytics-compatible format"
    res_regions_formatted = [
        {
            "points": [
                (60, 60),
                (
                    140,
                    75,
                ),
                (
                    130,
                    71,
                ),
                (
                    50,
                    52,
                ),
            ],
        },
        {
            "points": [
                (
                    50,
                    52,
                ),
                (
                    130,
                    71,
                ),
                (
                    110,
                    63,
                ),
                (
                    30,
                    37,
                ),
            ],
        },
        {
            "points": [
                (
                    30,
                    37,
                ),
                (
                    110,
                    63,
                ),
                (
                    100,
                    60,
                ),
                (
                    20,
                    30,
                ),
            ],
        },
        {
            "points": [
                (
                    60,
                    60,
                ),
                (
                    140,
                    75,
                ),
                (
                    20,
                    30,
                ),
                (
                    100,
                    60,
                ),
            ],
        },
    ]

    regions_formatted = degirum_tools.FigureAnnotator.convert_regions_to_json(
        regions_json, 2.0, 1.5, "Ultralytics-compatible"
    )
    assert regions_formatted == res_regions_formatted, ": ".join(
        [test_case, "converted regions list does not match expected value"]
    )

    test_case = "lines annotator (num_vertices = 2), freshly-instantiated object"
    line_annotator = degirum_tools.FigureAnnotator(num_vertices=2, debug=True)
    assert line_annotator.figure_type == "line", ": ".join(
        [test_case, "figure type does not match expected value"]
    )

    test_case = "lines annotator point addition"
    current_region = line_annotator.regions[0]
    current_region.current_selection.append(pt_1)
    line_annotator._process_point_addition(current_region)
    current_region.current_selection.append(pt_2)
    title, message = line_annotator._process_point_addition(current_region)
    assert title == "Success", ": ".join(
        [test_case, "title does not match expected value"]
    )
    assert message == f"{line_annotator.figure_type.capitalize()} added.", ": ".join(
        [test_case, "message does not match expected value"]
    )
    assert (
        len(current_region.figures) == 1
        and len(current_region.figures[0]) == line_annotator.num_vertices
    ), ": ".join([test_case, "figures list does not match expected length"])
    assert len(current_region.current_selection) == 0, ": ".join(
        [test_case, "region's current selection is not empty"]
    )
