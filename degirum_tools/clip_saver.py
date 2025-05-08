#
# clip_saver.py: video clip saving analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements analyzer class to save video clips triggered by event or notification
# with a specified duration before and after the event.
#

"""
Clip-Saving Analyzer Module Overview
====================================

`ClipSavingAnalyzer` is a [`ResultAnalyzerBase`](result_analyzer_base.md)
sub-class that records short video snippets when user-defined trigger
names appear in an `InferenceResults`.

Typical flow:
    1. Up-stream gizmos (e.g. [`EventDetector`](event_detector.md), [`EventNotifier`](event_notifier.md))
       attach event/notification strings to each result.
    2. `ClipSavingAnalyzer` watches for a match in its trigger set.
    3. When a trigger fires, an internal [`ClipSaver`](video_support.md)
       back-fills N frames before the trigger and records M frames after, saving:
           YYYYMMDD_HHMMSS.mp4 and YYYYMMDD_HHMMSS.json.

Key features:
    - Pre/Post buffering: configurable frame count before/after trigger
    - Optional overlays: embed AI bounding-boxes / labels in the clip
    - Side-car JSON: save raw inference results alongside the video
    - Thread-safe: each clip is written by its own worker thread

See the class-level documentation below for constructor parameters and
usage patterns.
"""

from typing import Set
from .result_analyzer_base import ResultAnalyzerBase
from .notifier import EventNotifier
from .event_detector import EventDetector
from .video_support import ClipSaver


class ClipSavingAnalyzer(ResultAnalyzerBase):
    """
    Result-analyzer that records short video clips whenever one of the configured trigger names appears
    in an `InferenceResults`. It delegates internally to [`ClipSaver`](video_support.md), which maintains
    a circular buffer, so every clip contains both pre-trigger and post-trigger context.

    Args:
        clip_duration (int): Total length of the output clip in frames (pre-buffer + post-buffer).
        triggers (Set[str]): Names that fire the recorder when found in either [`EventDetector`](event_detector.md#key_events_detected) or [`EventNotifier`](event_notifier.md#key_notifications).
        file_prefix (str): Path and filename prefix for generated files (timestamp & extension are appended automatically).
        pre_trigger_delay (int, optional): Frames to include before the trigger. Defaults to 0.
        embed_ai_annotations (bool, optional): If True, use `InferenceResults.image_overlay` so bounding boxes/labels
                                               are burned into the clip. Defaults to True.
        save_ai_result_json (bool, optional): If True, dump a JSON file with raw inference results alongside the video. Defaults to True.
        target_fps (float, optional): Frame rate of the output file. Defaults to 30.0.
    """

    def __init__(
        self,
        clip_duration: int,
        triggers: Set[str],
        file_prefix: str,
        *,
        pre_trigger_delay: int = 0,
        embed_ai_annotations: bool = True,
        save_ai_result_json: bool = True,
        target_fps=30.0,
    ):

        if not triggers or not isinstance(triggers, set):
            raise ValueError("`triggers` should be non-empty set of string")

        self._saver = ClipSaver(
            clip_duration,
            file_prefix,
            pre_trigger_delay=pre_trigger_delay,
            embed_ai_annotations=embed_ai_annotations,
            save_ai_result_json=save_ai_result_json,
            target_fps=target_fps,
        )
        self._triggers = triggers

    def analyze(self, result):
        """
        Inspect a single `InferenceResults` and forward it to internal [`ClipSaver`](video_support.md)
        if any trigger names are matched.

        This method is called automatically for each frame when attached via [`attach_analyzers`](inference_support.md).

        Args:
            result (InferenceResults): Current model output to scan for events/notifications.
        """

        # check trigger
        triggered = set()
        notifications = getattr(result, EventNotifier.key_notifications, None)
        if notifications is not None:
            intersection = self._triggers & notifications
            if intersection:
                triggered |= intersection

        events = getattr(result, EventDetector.key_events_detected, None)
        if events is not None:
            intersection = self._triggers & events
            if intersection:
                triggered |= intersection

        self._saver.forward(result, list(triggered))

    def join_all_saver_threads(self) -> int:
        """
        Block until all background clip-writer threads finish.

        Returns:
            int: Number of threads that were joined.
        """
        return self._saver.join_all_saver_threads()
