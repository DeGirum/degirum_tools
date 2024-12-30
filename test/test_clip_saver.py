#
# test_clip_saver.py: unit tests for video clip saving analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test video clip saving analyzer
#

import os, cv2, json
import numpy as np
from typing import List, Optional, Set
from dataclasses import dataclass
import degirum_tools


@dataclass
class _TestCase:
    name: str
    frames: int
    triggers: list
    events: list
    notifications: list
    args: dict
    expected_triggers: Optional[list] = None
    expect_fail: bool = False
    w = 64
    h = 32


test_cases: List[_TestCase] = [
    _TestCase(
        name="too big pre_trigger_delay",
        expect_fail=True,
        frames=10,
        triggers=[],
        events=["event1"],
        notifications=[],
        args=dict(
            clip_duration=5,
            pre_trigger_delay=6,
        ),
    ),
    _TestCase(
        name="no events",
        expect_fail=True,
        frames=10,
        triggers=[],
        events=[],
        notifications=[],
        args=dict(
            clip_duration=5,
        ),
    ),
    _TestCase(
        name="no triggers",
        frames=10,
        triggers=[],
        events=["event1"],
        notifications=[],
        args=dict(
            clip_duration=5,
        ),
    ),
    _TestCase(
        name="single trigger on event, no delay",
        frames=10,
        triggers=[3],
        events=["event1"],
        notifications=[],
        args=dict(
            clip_duration=4,
        ),
    ),
    _TestCase(
        name="single trigger on event, with delay",
        frames=12,
        triggers=[7],
        events=["event1"],
        notifications=[],
        args=dict(
            clip_duration=5,
            pre_trigger_delay=2,
        ),
    ),
    _TestCase(
        name="single trigger on notification, with delay, no json",
        frames=12,
        triggers=[7],
        events=[],
        notifications=["noti1"],
        args=dict(
            clip_duration=5,
            pre_trigger_delay=2,
            save_ai_result_json=False,
        ),
    ),
    _TestCase(
        name="single trigger on event, with delay, no ai annotations",
        frames=12,
        triggers=[7],
        events=["event1"],
        notifications=[],
        args=dict(
            clip_duration=5,
            pre_trigger_delay=2,
            embed_ai_annotations=False,
        ),
    ),
    _TestCase(
        name="many triggers on notification, with delay, another FPS",
        frames=25,
        triggers=[4, 10, 17],
        events=[],
        notifications=["noti1"],
        args=dict(
            clip_duration=5,
            pre_trigger_delay=1,
            target_fps=10,
        ),
    ),
    _TestCase(
        name="overlapping triggers on notification, with delay",
        frames=25,
        triggers=[7, 9, 11, 13],
        expected_triggers=[7, 13],
        events=[],
        notifications=["noti1"],
        args=dict(
            clip_duration=7,
            pre_trigger_delay=2,
        ),
    ),
    _TestCase(
        name="triggers on early events, back-to-back",
        frames=15,
        triggers=[0, 1, 2, 3, 4, 5],
        events=["event1"],
        notifications=[],
        args=dict(
            clip_duration=5,
            pre_trigger_delay=4,
        ),
    ),
    _TestCase(
        name="triggers on early events, overlapping",
        frames=15,
        triggers=[0, 2, 3],
        expected_triggers=[0, 3],
        events=["event1"],
        notifications=[],
        args=dict(
            clip_duration=5,
            pre_trigger_delay=2,
        ),
    ),
    _TestCase(
        name="trigger on late event",
        frames=10,
        triggers=[2, 8],
        expected_triggers=[2, 8],
        events=["event1"],
        notifications=[],
        args=dict(
            clip_duration=5,
            pre_trigger_delay=1,
        ),
    ),
]


class FakeResult:
    def __init__(self, index: int, case: _TestCase):
        self.index = index
        self.color = 255 - index
        dim = (case.h, case.w, 3)
        self.image = np.full(dim, (self.color, 0, 0), dtype=np.uint8)
        self.image_overlay = np.full(dim, (0, 0, self.color), dtype=np.uint8)
        self.events_detected: Set[str] = set()
        self.notifications: Set[str] = set()


def generate_results(N: int, case: _TestCase):
    return [FakeResult(i, case) for i in range(N)]


def apply_analyzer(analyzer, results):
    for r in results:
        analyzer.analyze(r)


def verify_results(case, clip_saver, triggers, results):
    case_name = f"Case '{case.name}'"
    nthreads = clip_saver.join_all_saver_threads()
    assert nthreads == len(
        triggers
    ), f"{case_name}: expected {len(triggers)} threads, but got {nthreads}."
    dir_path = clip_saver._saver._dir_path
    assert os.path.exists(
        clip_saver._saver._dir_path
    ), f"{case_name}: directory {clip_saver._saver._dir_path} does not exist."

    files = os.listdir(dir_path)
    file_count = len(files)
    expected_file_count = len(triggers) * (
        2 if clip_saver._saver._save_ai_result_json else 1
    )
    assert (
        file_count == expected_file_count
    ), f"{case_name}: expected {expected_file_count} files, but found {file_count}\n({files})."

    for t in triggers:
        start = t - clip_saver._saver._pre_trigger_delay
        clip_len = clip_saver._saver._clip_duration
        if start < 0:
            clip_len += start
        if start + clip_len > case.frames:
            clip_len = case.frames - start

        clip_path = os.path.join(
            dir_path, f"{clip_saver._saver._file_prefix}_{start:08d}.mp4"
        )
        assert os.path.exists(
            clip_path
        ), f"{case_name}/{t}: file {clip_path} does not exist."
        with degirum_tools.open_video_stream(clip_path) as stream:
            width, height, fps = degirum_tools.get_video_stream_properties(stream)
            assert (
                width == case.w
            ), f"{case_name}/{t}: expected width {case.w}, but got {width}."
            assert (
                height == case.h
            ), f"{case_name}/{t}: expected height {case.h}, but got {height}."
            assert (
                fps == clip_saver._saver._target_fps
            ), f"{case_name}/{t}: expected fps {clip_saver._saver._target_fps}, but got {fps}."

            cnt = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
            assert (
                cnt == clip_len
            ), f"{case_name}/{t}: expected {clip_len} frames, but got {cnt}."

        json_path = os.path.join(
            dir_path, f"{clip_saver._saver._file_prefix}_{start:08d}.json"
        )

        if clip_saver._saver._save_ai_result_json:
            assert os.path.exists(
                json_path
            ), f"{case_name}/{t}: file {json_path} does not exist."
            loaded_json = json.load(open(json_path))
            assert "properties" in loaded_json, f"{case_name}/{t}: missing properties."
            loaded_properties = loaded_json["properties"]
            assert isinstance(
                loaded_properties, dict
            ), f"{case_name}/{t}: properties is not a dict."

            assert (
                "timestamp" in loaded_properties
            ), f"{case_name}/{t}: missing timestamp."
            assert (
                "start_frame" in loaded_properties
            ), f"{case_name}/{t}: missing start_frame."
            assert (
                loaded_properties["start_frame"] == start
            ), f"{case_name}/{t}: start frame does not match."
            assert (
                "triggered_by" in loaded_properties
            ), f"{case_name}/{t}: missing triggered_by."
            assert loaded_properties["triggered_by"] == list(
                set(case.events + case.notifications)
            ), f"{case_name}/{t}: triggered set does not match."
            assert (
                "duration" in loaded_properties
            ), f"{case_name}/{t}: missing duration."
            assert (
                loaded_properties["duration"] == clip_len
            ), f"{case_name}/{t}: duration does not match."
            assert (
                "pre_trigger_delay" in loaded_properties
            ), f"{case_name}/{t}: missing pre_trigger_delay."
            assert (
                loaded_properties["pre_trigger_delay"]
                == clip_saver._saver._pre_trigger_delay
            ), f"{case_name}/{t}: pre-trigger delay does not match."
            assert (
                "target_fps" in loaded_properties
            ), f"{case_name}/{t}: missing target_fps."
            assert (
                loaded_properties["target_fps"] == clip_saver._saver._target_fps
            ), f"{case_name}/{t}: target fps does not match."

            assert "results" in loaded_json, f"{case_name}/{t}: missing results."
            loaded_results = loaded_json["results"]
            assert isinstance(
                loaded_results, list
            ), f"{case_name}/{t}: results is not a list."
            assert (
                len(loaded_results) == clip_len
            ), f"{case_name}/{t}: expected {clip_len} JSON results, but got {len(loaded_results)}."
            rslice = results[max(0, start) : start + clip_len]
            assert all(
                r1.index == r2["index"] for r1, r2 in zip(rslice, loaded_results)
            ), f"{case_name}/{t}: JSON results do not match expected."
        else:
            assert not os.path.exists(
                json_path
            ), f"{case_name}/{t}: file {json_path} should not exist."


def test_clip_saver(temp_dir):
    """
    Test for ClipSavingAnalyzer
    """

    for case in test_cases:
        file_prefix = str(temp_dir / case.name / "clip")
        results = generate_results(case.frames, case)
        for pos in case.triggers:
            if case.events:
                results[pos].events_detected = set(case.events)
            if case.notifications:
                results[pos].notifications = set(case.notifications)

        expect_fail = getattr(case, "expect_fail", False)

        try:
            clip_saver = degirum_tools.ClipSavingAnalyzer(
                triggers=set(case.events + case.notifications),
                file_prefix=file_prefix,
                **case.args,
            )
            assert not expect_fail, f"Expected failure for case {case.name}"
        except Exception as e:
            assert expect_fail, f"Unexpected failure for case {case.name}: {str(e)}"
            continue

        apply_analyzer(clip_saver, results)
        verify_results(
            case,
            clip_saver,
            (
                case.expected_triggers
                if case.expected_triggers is not None
                else case.triggers
            ),
            results,
        )
