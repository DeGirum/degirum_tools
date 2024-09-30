#
# test_zone_counter.py: unit tests for zone counter analyzer functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test zone counter analyzer functionality
#

import numpy as np
from typing import List


def test_zone_counter():
    """
    Test for ZoneCounter analyzer
    """

    import degirum_tools, degirum as dg
    from degirum_tools import AnchorPoint

    box_1_zone_1 = [5, 40, 35, 70]
    box_1_zone_1_shifted = [55, 40, 85, 70]
    box_2_zone_1 = [50, 50, 90, 100]
    box_1_zone_2 = [155, 50, 185, 110]
    zone_1 = [[30, 30], [80, 30], [80, 80], [30, 80]]
    zone_2 = [[150, 70], [190, 70], [190, 100], [150, 100]]
    zone_3 = [[55, 55], [105, 55], [105, 105], [55, 105]]
    zone_4 = [[85, 85], [135, 85], [135, 135], [85, 135]]
    zones = [zone_1, zone_2]
    overlapping_zones = [zone_1, zone_3]

    test_cases: List[dict] = [
        # ----------------------------------------------------------------
        # No tracking
        # ----------------------------------------------------------------
        {
            "case": "No objects",
            "params": {
                "count_polygons": zones,
                "triggering_position": AnchorPoint.TOP_RIGHT,
                "use_tracking": False,
            },
            "inp": [[]],
            "res": [[[], [{"total": 0}, {"total": 0}]]],
        },
        {
            "case": "One trigger",
            "params": {
                "count_polygons": zones,
                "triggering_position": AnchorPoint.TOP_RIGHT,
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": box_1_zone_1, "label": "label1"},
                    {"bbox": box_2_zone_1, "label": "label2"},
                    {"bbox": box_1_zone_2, "label": "label3"},
                ]
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ]
            ],
        },
        {
            "case": "One trigger, overlapping zones",
            "params": {
                "count_polygons": overlapping_zones,
                "triggering_position": AnchorPoint.CENTER,
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": box_1_zone_1, "label": "label1"},
                    {"bbox": box_2_zone_1, "label": "label2"},
                    {"bbox": box_1_zone_2, "label": "label3"},
                ]
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "in_zone": [True, True],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 1}],
                ]
            ],
        },
        {
            "case": "Multiple triggers",
            "params": {
                "count_polygons": zones,
                "triggering_position": [AnchorPoint.TOP_RIGHT, AnchorPoint.TOP_LEFT],
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": box_1_zone_1, "label": "label1"},
                    {"bbox": box_2_zone_1, "label": "label2"},
                    {"bbox": box_1_zone_2, "label": "label3"},
                ]
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 2}, {"total": 0}],
                ]
            ],
        },
        {
            "case": "Multiple triggers, object in multiple zones",
            "params": {
                "count_polygons": [zone_1, zone_4],
                "triggering_position": [AnchorPoint.TOP_LEFT, AnchorPoint.BOTTOM_RIGHT],
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": box_1_zone_1, "label": "label1"},
                    {"bbox": box_2_zone_1, "label": "label2"},
                    {"bbox": box_1_zone_2, "label": "label3"},
                ]
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "in_zone": [True, True],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 2}, {"total": 1}],
                ]
            ],
        },
        {
            "case": "Multiple triggers, defined class list",
            "params": {
                "class_list": ["label1"],
                "count_polygons": zones,
                "triggering_position": [AnchorPoint.TOP_RIGHT, AnchorPoint.TOP_LEFT],
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": box_1_zone_1, "label": "label1"},
                    {"bbox": box_2_zone_1, "label": "label2"},
                    {"bbox": box_1_zone_2, "label": "label3"},
                ]
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "in_zone": [True, False],
                        },
                        {"bbox": box_2_zone_1, "label": "label2"},
                        {"bbox": box_1_zone_2, "label": "label3"},
                    ],
                    [{"total": 1}, {"total": 0}],
                ]
            ],
        },
        {
            "case": "Multiple triggers, defined class list and per class display",
            "params": {
                "class_list": ["label1"],
                "count_polygons": zones,
                "triggering_position": [AnchorPoint.TOP_RIGHT, AnchorPoint.TOP_LEFT],
                "use_tracking": False,
                "per_class_display": True,
            },
            "inp": [
                [
                    {"bbox": box_1_zone_1, "label": "label1"},
                    {"bbox": box_2_zone_1, "label": "label2"},
                    {"bbox": box_1_zone_2, "label": "label3"},
                ]
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "in_zone": [True, False],
                        },
                        {"bbox": box_2_zone_1, "label": "label2"},
                        {"bbox": box_1_zone_2, "label": "label3"},
                    ],
                    [{"label1": 1, "total": 1}, {"label1": 0, "total": 0}],
                ]
            ],
        },
        {
            "case": "One trigger, bbox scaling",
            "params": {
                "count_polygons": zones,
                "triggering_position": AnchorPoint.TOP_RIGHT,
                "bounding_box_scale": 0.5,
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": box_1_zone_1, "label": "label1"},
                    {"bbox": box_2_zone_1, "label": "label2"},
                    {"bbox": box_1_zone_2, "label": "label3"},
                ]
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ]
            ],
        },
        {
            "case": "IoPA",
            "params": {
                "count_polygons": zones,
                "triggering_position": None,
                "iopa_threshold": 0.2,
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": box_1_zone_1, "label": "label1"},
                    {"bbox": box_2_zone_1, "label": "label2"},
                    {"bbox": box_1_zone_2, "label": "label3"},
                ]
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "in_zone": [False, True],
                        },
                    ],
                    [{"total": 1}, {"total": 1}],
                ]
            ],
        },
        {
            "case": "IoPA, bbox scaling",
            "params": {
                "count_polygons": zones,
                "triggering_position": None,
                "bounding_box_scale": 0.5,
                "iopa_threshold": 0.2,
                "use_tracking": False,
            },
            "inp": [
                [
                    {"bbox": box_1_zone_1, "label": "label1"},
                    {"bbox": box_2_zone_1, "label": "label2"},
                    {"bbox": box_1_zone_2, "label": "label3"},
                ]
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "in_zone": [False, True],
                        },
                    ],
                    [{"total": 0}, {"total": 1}],
                ]
            ],
        },
        # ----------------------------------------------------------------
        # With tracking
        # ----------------------------------------------------------------
        {
            "case": "No objects",
            "params": {
                "count_polygons": zones,
                "triggering_position": AnchorPoint.TOP_RIGHT,
                "use_tracking": True,
            },
            "inp": [[[]]],
            "res": [[[], [{"total": 0}, {"total": 0}]]],
        },
        {
            "case": "One trigger, no tracking info",
            "params": {
                "count_polygons": zones,
                "triggering_position": AnchorPoint.TOP_RIGHT,
                "use_tracking": True,
            },
            "inp": [
                [
                    [
                        {"bbox": box_1_zone_1, "label": "label1"},
                        {"bbox": box_2_zone_1, "label": "label2"},
                        {"bbox": box_1_zone_2, "label": "label3"},
                    ]
                ]
            ],
            "res": [
                [
                    [
                        {"bbox": box_1_zone_1, "label": "label1"},
                        {"bbox": box_2_zone_1, "label": "label2"},
                        {"bbox": box_1_zone_2, "label": "label3"},
                    ],
                    [{"total": 0}, {"total": 0}],
                ]
            ],
        },
        {
            "case": "One trigger, object disappears and appears again",
            "params": {
                "count_polygons": zones,
                "triggering_position": AnchorPoint.TOP_RIGHT,
                "use_tracking": True,
            },
            "inp": [
                [
                    [
                        {"bbox": box_1_zone_1, "label": "label1", "track_id": 0},
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                        {"bbox": box_1_zone_2, "label": "label3", "track_id": 2},
                    ],
                    [
                        {
                            0: [np.array(box_1_zone_1, dtype=np.int32)],
                            1: [np.array(box_2_zone_1, dtype=np.int32)],
                            2: [np.array(box_1_zone_2, dtype=np.int32)],
                        }
                    ],
                    [{0: "label1", 1: "label2", 2: "label3"}],
                ],
                [
                    [
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                        {"bbox": box_1_zone_2, "label": "label3", "track_id": 2},
                    ],
                    [
                        {
                            0: [np.array(box_1_zone_1, dtype=np.int32)],
                            1: [
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                            ],
                            2: [
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                            ],
                        }
                    ],
                    [{0: "label1", 1: "label2", 2: "label3"}],
                ],
                [
                    [
                        {"bbox": box_1_zone_1, "label": "label1", "track_id": 0},
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                        {"bbox": box_1_zone_2, "label": "label3", "track_id": 2},
                    ],
                    [
                        {
                            0: [
                                np.array(box_1_zone_1, dtype=np.int32),
                                np.array(box_1_zone_1, dtype=np.int32),
                            ],
                            1: [
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                            ],
                            2: [
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                            ],
                        }
                    ],
                    [{0: "label1", 1: "label2", 2: "label3"}],
                ],
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "track_id": 0,
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
                [
                    [
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "track_id": 0,
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
            ],
        },
        {
            "case": "One trigger, object exits and re-enters zone within timeout period",
            "params": {
                "count_polygons": zones,
                "triggering_position": AnchorPoint.TOP_RIGHT,
                "use_tracking": True,
                "timeout_frames": 1,
            },
            "inp": [
                [
                    [
                        {"bbox": box_1_zone_1, "label": "label1", "track_id": 0},
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                        {"bbox": box_1_zone_2, "label": "label3", "track_id": 2},
                    ],
                    [
                        {
                            0: [np.array(box_1_zone_1, dtype=np.int32)],
                            1: [np.array(box_2_zone_1, dtype=np.int32)],
                            2: [np.array(box_1_zone_2, dtype=np.int32)],
                        }
                    ],
                    [{0: "label1", 1: "label2", 2: "label3"}],
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1_shifted,
                            "label": "label1",
                            "track_id": 0,
                        },
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                        {"bbox": box_1_zone_2, "label": "label3", "track_id": 2},
                    ],
                    [
                        {
                            0: [
                                np.array(box_1_zone_1, dtype=np.int32),
                                np.array(box_1_zone_1_shifted, dtype=np.int32),
                            ],
                            1: [
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                            ],
                            2: [
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                            ],
                        }
                    ],
                    [{0: "label1", 1: "label2", 2: "label3"}],
                ],
                [
                    [
                        {"bbox": box_1_zone_1, "label": "label1", "track_id": 0},
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                        {"bbox": box_1_zone_2, "label": "label3", "track_id": 2},
                    ],
                    [
                        {
                            0: [
                                np.array(box_1_zone_1, dtype=np.int32),
                                np.array(box_1_zone_1_shifted, dtype=np.int32),
                                np.array(box_1_zone_1, dtype=np.int32),
                            ],
                            1: [
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                            ],
                            2: [
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                            ],
                        }
                    ],
                    [{0: "label1", 1: "label2", 2: "label3"}],
                ],
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "track_id": 0,
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1_shifted,
                            "label": "label1",
                            "track_id": 0,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "track_id": 0,
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
            ],
        },
        {
            "case": "One trigger, object exits and re-enters the zone late",
            "params": {
                "count_polygons": zones,
                "triggering_position": AnchorPoint.TOP_RIGHT,
                "use_tracking": True,
                "timeout_frames": 1,
            },
            "inp": [
                [
                    [
                        {"bbox": box_1_zone_1, "label": "label1", "track_id": 0},
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                        {"bbox": box_1_zone_2, "label": "label3", "track_id": 2},
                    ],
                    [
                        {
                            0: [np.array(box_1_zone_1, dtype=np.int32)],
                            1: [np.array(box_2_zone_1, dtype=np.int32)],
                            2: [np.array(box_1_zone_2, dtype=np.int32)],
                        }
                    ],
                    [{0: "label1", 1: "label2", 2: "label3"}],
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1_shifted,
                            "label": "label1",
                            "track_id": 0,
                        },
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                        {"bbox": box_1_zone_2, "label": "label3", "track_id": 2},
                    ],
                    [
                        {
                            0: [
                                np.array(box_1_zone_1, dtype=np.int32),
                                np.array(box_1_zone_1_shifted, dtype=np.int32),
                            ],
                            1: [
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                            ],
                            2: [
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                            ],
                        }
                    ],
                    [{0: "label1", 1: "label2", 2: "label3"}],
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1_shifted,
                            "label": "label1",
                            "track_id": 0,
                        },
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                        {"bbox": box_1_zone_2, "label": "label3", "track_id": 2},
                    ],
                    [
                        {
                            0: [
                                np.array(box_1_zone_1, dtype=np.int32),
                                np.array(box_1_zone_1_shifted, dtype=np.int32),
                                np.array(box_1_zone_1_shifted, dtype=np.int32),
                            ],
                            1: [
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                            ],
                            2: [
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                            ],
                        }
                    ],
                    [{0: "label1", 1: "label2", 2: "label3"}],
                ],
                [
                    [
                        {"bbox": box_1_zone_1, "label": "label1", "track_id": 0},
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                        {"bbox": box_1_zone_2, "label": "label3", "track_id": 2},
                    ],
                    [
                        {
                            0: [
                                np.array(box_1_zone_1, dtype=np.int32),
                                np.array(box_1_zone_1_shifted, dtype=np.int32),
                                np.array(box_1_zone_1_shifted, dtype=np.int32),
                                np.array(box_1_zone_1, dtype=np.int32),
                            ],
                            1: [
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                            ],
                            2: [
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                                np.array(box_1_zone_2, dtype=np.int32),
                            ],
                        }
                    ],
                    [{0: "label1", 1: "label2", 2: "label3"}],
                ],
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "track_id": 0,
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1_shifted,
                            "label": "label1",
                            "track_id": 0,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1_shifted,
                            "label": "label1",
                            "track_id": 0,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 0}, {"total": 0}],
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "track_id": 0,
                            "in_zone": [True, False],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
            ],
        },
    ]

    for case in test_cases:
        zone_counter = degirum_tools.ZoneCounter(**case["params"])

        for i, input in enumerate(case["inp"]):
            inference_results = input[0] if case["params"]["use_tracking"] else input
            result = dg.postprocessor.InferenceResults(
                model_params=None,
                input_image=np.zeros((200, 200)),
                inference_results=inference_results,
                conversion=None,
            )
            if case["params"]["use_tracking"]:
                if len(input) == 3:
                    result.trails = input[1][0]
                    result.trail_classes = input[2][0]

            zone_counter.analyze(result)
            assert result._inference_results == case["res"][i][0], (
                f"Case `{case['case']}` failed at step {i}: "
                + f"inference results `{result._inference_results}` "
                + f"do not match expected `{case['res'][i][0]}`."
                + f"\nConfig: {case['params']}"
            )
            assert result.zone_counts == case["res"][i][1], (
                f"Case `{case['case']}` failed at step {i}: "
                + f"zone counts `{result.zone_counts}` "
                + f"do not match expected `{case['res'][i][1]}`."
                + f"\nConfig: {case['params']}"
            )
