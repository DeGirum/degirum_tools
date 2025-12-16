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
            "case": "One trigger, object disappears and appears again (no timeout)",
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
                    ],
                    [
                        {
                            0: [np.array(box_1_zone_1, dtype=np.int32)],
                            1: [np.array(box_2_zone_1, dtype=np.int32)],
                        }
                    ],
                    [{0: "label1", 1: "label2"}],
                ],
                [
                    [
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
                    ],
                    [
                        {
                            0: [np.array(box_1_zone_1, dtype=np.int32)],
                            1: [
                                np.array(box_2_zone_1, dtype=np.int32),
                                np.array(box_2_zone_1, dtype=np.int32),
                            ],
                        }
                    ],
                    [{0: "label1", 1: "label2"}],
                ],
                [
                    [
                        {"bbox": box_1_zone_1, "label": "label1", "track_id": 0},
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
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
                        }
                    ],
                    [{0: "label1", 1: "label2"}],
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
                            "frames_in_zone": [1, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
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
                            "frames_in_zone": [0, 0],
                        },
                    ],
                    [
                        {"total": 0},
                        {"total": 0},
                    ],  # Object 0 missing, immediately deleted (timeout_frames=0)
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "track_id": 0,
                            "in_zone": [True, False],
                            "frames_in_zone": [
                                1,
                                0,
                            ],  # Track re-created, starts fresh at 1
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
            ],
        },
        {
            "case": "One trigger, object leaves and re-enters zone (no timeout)",
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
                    ],
                    [
                        {
                            0: [np.array(box_1_zone_1, dtype=np.int32)],
                            1: [np.array(box_2_zone_1, dtype=np.int32)],
                        }
                    ],
                    [{0: "label1", 1: "label2"}],
                ],
                [
                    [
                        {
                            "bbox": box_1_zone_1_shifted,
                            "label": "label1",
                            "track_id": 0,
                        },
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
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
                        }
                    ],
                    [{0: "label1", 1: "label2"}],
                ],
                [
                    [
                        {"bbox": box_1_zone_1, "label": "label1", "track_id": 0},
                        {"bbox": box_2_zone_1, "label": "label2", "track_id": 1},
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
                        }
                    ],
                    [{0: "label1", 1: "label2"}],
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
                            "frames_in_zone": [1, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
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
                            "frames_in_zone": [0, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
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
                            "frames_in_zone": [1, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
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
                            "frames_in_zone": [1, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
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
                            "in_zone": [True, False],
                            "frames_in_zone": [2, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
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
                            "frames_in_zone": [1, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
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
                            "frames_in_zone": [1, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
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
                            "in_zone": [True, False],
                            "frames_in_zone": [2, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
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
                            "frames_in_zone": [0, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
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
                            "frames_in_zone": [1, 0],
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "track_id": 1,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "track_id": 2,
                            "in_zone": [False, False],
                            "frames_in_zone": [0, 0],
                        },
                    ],
                    [{"total": 1}, {"total": 0}],
                ],
            ],
        },
        # ----------------------------------------------------------------
        # IoPA threshold list tests
        # ----------------------------------------------------------------
        {
            "case": "IoPA list - different thresholds per zone",
            "params": {
                "count_polygons": zones,
                "triggering_position": None,
                "iopa_threshold": [0.1, 0.5],  # Different thresholds for each zone
                "use_tracking": False,
            },
            "inp": [
                [
                    {
                        "bbox": box_1_zone_1,
                        "label": "label1",
                    },  # Small overlap with zone 1
                    {
                        "bbox": box_2_zone_1,
                        "label": "label2",
                    },  # Larger overlap with zone 1
                    {"bbox": box_1_zone_2, "label": "label3"},  # Overlap with zone 2
                ]
            ],
            "res": [
                [
                    [
                        {
                            "bbox": box_1_zone_1,
                            "label": "label1",
                            "in_zone": [
                                False,
                                False,
                            ],  # Low IoPA with zone 1, doesn't meet 0.1 threshold
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "in_zone": [
                                True,
                                False,
                            ],  # Good IoPA with zone 1, meets 0.1 threshold
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "in_zone": [
                                False,
                                True,
                            ],  # IoPA with zone 2 meets 0.5 threshold
                        },
                    ],
                    [{"total": 1}, {"total": 1}],
                ]
            ],
        },
        {
            "case": "IoPA list - scalar vs list equivalence",
            "params": {
                "count_polygons": zones,
                "triggering_position": None,
                "iopa_threshold": [0.2, 0.2],  # Same threshold for both zones
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
            "case": "IoPA list - very strict thresholds",
            "params": {
                "count_polygons": zones,
                "triggering_position": None,
                "iopa_threshold": [0.8, 0.9],  # Very high thresholds
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
                            "in_zone": [False, False],  # Doesn't meet high threshold
                        },
                        {
                            "bbox": box_2_zone_1,
                            "label": "label2",
                            "in_zone": [False, False],  # Doesn't meet high threshold
                        },
                        {
                            "bbox": box_1_zone_2,
                            "label": "label3",
                            "in_zone": [False, False],  # Doesn't meet high threshold
                        },
                    ],
                    [{"total": 0}, {"total": 0}],
                ]
            ],
        },
    ]

    keys_to_ignore = ["time_in_zone"]

    def cleanup_result(result):
        for r in result._inference_results:
            for key in keys_to_ignore:
                if key in r:
                    del r[key]
        return result

    for case in test_cases:
        zone_counter = degirum_tools.ZoneCounter(**case["params"])

        for i, input in enumerate(case["inp"]):
            inference_results = input[0] if case["params"]["use_tracking"] else input
            result = dg.postprocessor.InferenceResults(
                input_image=np.zeros((200, 200)),
                inference_results=inference_results,
                conversion=None,
            )
            if case["params"]["use_tracking"]:
                if len(input) == 3:
                    result.trails = input[1][0]
                    result.trail_classes = input[2][0]

            zone_counter.analyze(result)
            cleanup_result(result)
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


# ============================================================================
# Additional comprehensive tests for new ZoneCounter features
# ============================================================================


class MockResult:
    """Mock inference result for testing."""

    def __init__(self, detections, frame_number=0):
        self.results = detections
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.image_overlay = self.image.copy()
        self.inference_results = {"frame_number": frame_number}
        self.zone_counts = {}  # Will be replaced by ZoneCounter with list or dict
        self.zone_events = []
        # Attributes for annotate() method
        self.overlay_color = (255, 0, 0)  # Red
        self.overlay_line_width = 2
        self.overlay_font_scale = 0.5


def test_entry_delay_frames():
    """Test entry_delay_frames parameter for entry smoothing."""
    from degirum_tools.analyzers.zone_count import ZoneCounter

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = ZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=3,
        timeout_frames=2,
        enable_zone_events=True,
    )

    # Frame 1: Object enters
    result1 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 1
    )
    counter.analyze(result1)
    assert result1.zone_counts["test_zone"]["total"] == 0

    # Frame 2: Still in zone
    result2 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 2
    )
    counter.analyze(result2)
    assert result2.zone_counts["test_zone"]["total"] == 0

    # Frame 3: Established
    result3 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 3
    )
    counter.analyze(result3)
    assert result3.zone_counts["test_zone"]["total"] == 1
    assert len(result3.zone_events) == 2  # entry + occupied


def test_multi_zone_tracking():
    """Test tracking across multiple zones."""
    from degirum_tools.analyzers.zone_count import ZoneCounter

    zones = {
        "zone_A": np.array([[50, 50], [200, 50], [200, 200], [50, 200]]),
        "zone_B": np.array([[250, 50], [400, 50], [400, 200], [250, 200]]),
        "zone_C": np.array([[150, 250], [300, 250], [300, 400], [150, 400]]),
    }

    counter = ZoneCounter(
        zones=zones,
        use_tracking=True,
        entry_delay_frames=1,
        timeout_frames=1,
        enable_zone_events=True,
    )

    # Frame 1: Object in zone_A, another in zone_B
    result1 = MockResult(
        [
            {"bbox": [75, 75, 125, 125], "track_id": 1, "label": "person"},
            {"bbox": [275, 75, 325, 125], "track_id": 2, "label": "car"},
        ],
        1,
    )
    counter.analyze(result1)
    assert result1.zone_counts["zone_A"]["total"] == 1
    assert result1.zone_counts["zone_B"]["total"] == 1
    assert result1.zone_counts["zone_C"]["total"] == 0


def test_per_class_counting():
    """Test per-class display mode."""
    from degirum_tools.analyzers.zone_count import ZoneCounter

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = ZoneCounter(
        zones={"test_zone": zone_polygon},
        class_list=["person", "car", "bicycle"],
        per_class_display=True,
        use_tracking=True,
        entry_delay_frames=1,
    )

    # Frame 1: Mixed objects
    result = MockResult(
        [
            {"bbox": [120, 120, 170, 170], "track_id": 1, "label": "person"},
            {"bbox": [180, 180, 230, 230], "track_id": 2, "label": "person"},
            {"bbox": [240, 240, 290, 290], "track_id": 3, "label": "car"},
        ],
        1,
    )
    counter.analyze(result)

    counts = result.zone_counts["test_zone"]
    assert counts["total"] == 3
    assert counts["person"] == 2
    assert counts["car"] == 1
    assert counts["bicycle"] == 0


def test_zone_events():
    """Test zone event generation."""
    from degirum_tools.analyzers.zone_count import ZoneCounter

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = ZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=1,
        timeout_frames=1,
        enable_zone_events=True,
    )

    # Frame 1: Object enters
    result1 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 1
    )
    counter.analyze(result1)
    entry_events = [e for e in result1.zone_events if e["event_type"] == "zone_entry"]
    occupied_events = [
        e for e in result1.zone_events if e["event_type"] == "zone_occupied"
    ]
    assert len(entry_events) == 1
    assert len(occupied_events) == 1

    # Frame 2: Object exits (grace period)
    result2 = MockResult([], 2)
    counter.analyze(result2)
    assert len(result2.zone_events) == 0

    # Frame 3: Grace period expires
    result3 = MockResult([], 3)
    counter.analyze(result3)
    exit_events = [e for e in result3.zone_events if e["event_type"] == "zone_exit"]
    empty_events = [e for e in result3.zone_events if e["event_type"] == "zone_empty"]
    assert len(exit_events) == 1
    assert len(empty_events) == 1


def test_parameter_validation():
    """
    Test parameter validation for ZoneCounter
    """
    import degirum_tools
    import pytest

    zone = [[10, 10], [90, 10], [90, 90], [10, 90]]

    # Test valid parameters (should not raise)
    degirum_tools.ZoneCounter(
        count_polygons=[zone],
        timeout_frames=0,
        entry_delay_frames=1,
        use_tracking=True,
    )

    degirum_tools.ZoneCounter(
        count_polygons=[zone],
        timeout_frames=10,
        entry_delay_frames=5,
        use_tracking=True,
    )

    # Test invalid timeout_frames (negative)
    with pytest.raises(
        ValueError, match="timeout_frames must be a non-negative integer"
    ):
        degirum_tools.ZoneCounter(
            count_polygons=[zone],
            timeout_frames=-1,
            use_tracking=True,
        )

    # Test invalid timeout_frames (float)
    with pytest.raises(
        ValueError, match="timeout_frames must be a non-negative integer"
    ):
        degirum_tools.ZoneCounter(
            count_polygons=[zone],
            timeout_frames=1.5,
            use_tracking=True,
        )

    # Test invalid entry_delay_frames (zero)
    with pytest.raises(
        ValueError, match="entry_delay_frames must be a positive integer"
    ):
        degirum_tools.ZoneCounter(
            count_polygons=[zone],
            entry_delay_frames=0,
            use_tracking=True,
        )

    # Test invalid entry_delay_frames (negative)
    with pytest.raises(
        ValueError, match="entry_delay_frames must be a positive integer"
    ):
        degirum_tools.ZoneCounter(
            count_polygons=[zone],
            entry_delay_frames=-1,
            use_tracking=True,
        )

    # Test invalid entry_delay_frames (float)
    with pytest.raises(
        ValueError, match="entry_delay_frames must be a positive integer"
    ):
        degirum_tools.ZoneCounter(
            count_polygons=[zone],
            entry_delay_frames=2.5,
            use_tracking=True,
        )
