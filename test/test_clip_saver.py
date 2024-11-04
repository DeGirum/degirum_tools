#
# test_clip_saver.py: unit tests for video clip saving analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test video clip saving analyzer
#

import os, cv2, json
from typing import List, Optional
from dataclasses import dataclass


def test_clip_saver(temp_dir):
    """
    Test for ClipSaver analyzer
    """

    import degirum_tools
    import numpy as np

    w, h = 64, 32

    class FakeResult:
        def __init__(self, index):
            self.index = index
            self.color = 255 - index
            self.image = np.full((h, w, 3), (self.color, 0, 0), dtype=np.uint8)
            self.overlay_image = np.full((h, w, 3), (0, 0, self.color), dtype=np.uint8)
            self.events_detected = set()
            self.notifications = set()

    def generate_results(N):
        return [FakeResult(i) for i in range(N)]

    def apply_analyzer(analyzer, results):
        for r in results:
            analyzer.analyze(r)

    def verify_results(case, clip_saver, triggers, results):
        case_name = f"Case '{case.name}'"
        clip_saver.join_all_saver_threads()
        dir_path = clip_saver._dir_path
        assert os.path.exists(
            clip_saver._dir_path
        ), f"{case_name}: directory {clip_saver._dir_path} does not exist."

        file_count = len(os.listdir(dir_path))
        expected_file_count = len(triggers) * (
            2 if clip_saver._save_ai_result_json else 1
        )
        assert (
            file_count == expected_file_count
        ), f"{case_name}: expected {expected_file_count} files, but found {file_count}."

        for t in triggers:
            start = t - clip_saver._pre_trigger_delay
            clip_len = clip_saver._clip_duration
            if start < 0:
                clip_len += start + 1
                start = 0

            clip_path = os.path.join(
                dir_path, f"{clip_saver._file_prefix}_{start:08d}.mp4"
            )
            assert os.path.exists(
                clip_path
            ), f"{case_name}: file {clip_path} does not exist."
            with degirum_tools.open_video_stream(clip_path) as stream:
                width, height, fps = degirum_tools.get_video_stream_properties(stream)
                assert width == w, f"{case_name}: expected width {w}, but got {width}."
                assert (
                    height == h
                ), f"{case_name}: expected height {h}, but got {height}."
                assert (
                    fps == clip_saver._target_fps
                ), f"{case_name}: expected fps {clip_saver._target_fps}, but got {fps}."

                cnt = stream.get(cv2.CAP_PROP_FRAME_COUNT)
                assert (
                    cnt == clip_len
                ), f"{case_name}: expected {clip_len} frames, but got {cnt}."

            json_path = os.path.join(
                dir_path, f"{clip_saver._file_prefix}_{start:08d}.json"
            )

            if clip_saver._save_ai_result_json:
                assert os.path.exists(
                    json_path
                ), f"{case_name}: file {json_path} does not exist."
                loaded_json = json.load(open(json_path))
                assert "properties" in loaded_json, f"{case_name}: missing properties."
                loaded_properties = loaded_json["properties"]
                assert isinstance(
                    loaded_properties, dict
                ), f"{case_name}: properties is not a dict."

                assert (
                    "timestamp" in loaded_properties
                ), f"{case_name}: missing timestamp."
                assert (
                    "start_frame" in loaded_properties
                ), f"{case_name}: missing start_frame."
                assert (
                    loaded_properties["start_frame"] == start
                ), f"{case_name}: start frame does not match."
                assert (
                    "triggered_by" in loaded_properties
                ), f"{case_name}: missing triggered_by."
                assert loaded_properties["triggered_by"] == list(
                    set(case.events + case.notifications)
                ), f"{case_name}: triggered set does not match."
                assert (
                    "duration" in loaded_properties
                ), f"{case_name}: missing duration."
                assert (
                    loaded_properties["duration"] == clip_saver._clip_duration
                ), f"{case_name}: duration does not match."
                assert (
                    "pre_trigger_delay" in loaded_properties
                ), f"{case_name}: missing pre_trigger_delay."
                assert (
                    loaded_properties["pre_trigger_delay"]
                    == clip_saver._pre_trigger_delay
                ), f"{case_name}: pre-trigger delay does not match."
                assert (
                    "target_fps" in loaded_properties
                ), f"{case_name}: missing target_fps."
                assert (
                    loaded_properties["target_fps"] == clip_saver._target_fps
                ), f"{case_name}: target fps does not match."

                assert "results" in loaded_json, f"{case_name}: missing results."
                loaded_results = loaded_json["results"]
                assert isinstance(
                    loaded_results, list
                ), f"{case_name}: results is not a list."
                assert (
                    len(loaded_results) == clip_len
                ), f"{case_name}: expected {clip_len} JSON results, but got {len(loaded_results)}."
                rslice = results[start : start + clip_len]
                assert all(
                    r1.index == r2["index"] for r1, r2 in zip(rslice, loaded_results)
                ), f"{case_name}: JSON results do not match expected."
            else:
                assert not os.path.exists(
                    json_path
                ), f"{case_name}: file {json_path} should not exist."

    @dataclass
    class TestCase:
        name: str
        frames: int
        triggers: list
        events: list
        notifications: list
        args: dict
        expected_triggers: Optional[list] = None
        expect_fail: bool = False

    test_cases: List[TestCase] = [
        TestCase(
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
        TestCase(
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
        TestCase(
            name="no triggers",
            frames=10,
            triggers=[],
            events=["event1"],
            notifications=[],
            args=dict(
                clip_duration=5,
            ),
        ),
        TestCase(
            name="single trigger on event, no delay",
            frames=10,
            triggers=[3],
            events=["event1"],
            notifications=[],
            args=dict(
                clip_duration=4,
            ),
        ),
        TestCase(
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
        TestCase(
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
        TestCase(
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
        TestCase(
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
        TestCase(
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
        TestCase(
            name="trigger on early event, overlapping, and normal",
            frames=10,
            triggers=[0, 7, 8],
            expected_triggers=[0, 8],
            events=["event1"],
            notifications=[],
            args=dict(
                clip_duration=5,
                pre_trigger_delay=4,
            ),
        ),
        TestCase(
            name="trigger on late event",
            frames=10,
            triggers=[2, 8],
            expected_triggers=[2],
            events=["event1"],
            notifications=[],
            args=dict(
                clip_duration=5,
                pre_trigger_delay=1,
            ),
        ),
    ]

    for case in test_cases:
        file_prefix = str(temp_dir / case.name / "clip")
        results = generate_results(case.frames)
        for pos in case.triggers:
            if case.events:
                results[pos].events_detected = set(case.events)
            if case.notifications:
                results[pos].notifications = set(case.notifications)

        expect_fail = getattr(case, "expect_fail", False)

        try:
            clip_saver = degirum_tools.ClipSaver(
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