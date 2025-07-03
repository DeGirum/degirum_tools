#
# clip_saver.py: video clip saving analyzer
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements analyzer class to save video clips triggered by event or notification
# with a specified duration before and after the event.
#

"""
Clip Saving Analyzer Module Overview
====================================

This module provides an analyzer (`ClipSavingAnalyzer`) for recording video snippets triggered by events
or notifications. It captures frames before and after trigger events, saving them as video clips with
optional AI annotations and metadata.

Key Features:
    - **Pre/Post Buffering**: Configurable frame count before and after trigger events
    - **Optional Overlays**: Embed AI bounding boxes and labels in the saved clips
    - **Side-car JSON**: Save raw inference results alongside video files
    - **Thread-Safe**: Each clip is written by its own worker thread
    - **Frame Rate Control**: Configurable target FPS for saved clips
    - **Event Integration**: Works with EventDetector and EventNotifier triggers
    - **Storage Support**: Optional integration with object storage for clip uploads

Typical Usage:
    1. Create a `ClipSavingAnalyzer` instance with desired buffer and output settings
    2. Process inference results through the analyzer chain
    3. When triggers occur, clips are automatically saved with pre/post frames
    4. Access saved clips and their associated metadata files
    5. Optionally upload clips to object storage for remote access

Integration Notes:
    - Works with any analyzer that adds trigger names to results
    - Requires video frames to be available in the result object
    - Supports both local file storage and object storage uploads
    - Thread-safe for concurrent clip saving operations

Key Classes:
    - `ClipSavingAnalyzer`: Main analyzer class for saving video clips
    - `ClipSaver`: Internal class handling clip writing and buffering

Configuration Options:
    - `clip_duration`: Number of frames to save after trigger
    - `clip_prefix`: Base path for saved clip files
    - `pre_trigger_delay`: Frames to include before trigger
    - `embed_ai_annotations`: Enable/disable AI overlays in clips
    - `save_ai_result_json`: Enable/disable metadata saving
    - `target_fps`: Frame rate for saved video clips
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
        """
        Constructor.

        Args:
            clip_duration (int): Total length of the output clip in frames (pre-buffer + post-buffer).
            triggers (Set[str]): Names that fire the recorder when found in either [`EventDetector`](event_detector.md#key_events_detected) or [`EventNotifier`](notifier.md#key_notifications).
            file_prefix (str): Path and filename prefix for generated files (frame number & extension are appended automatically).
            pre_trigger_delay (int, optional): Frames to include before the trigger. Defaults to 0.
            embed_ai_annotations (bool, optional): If True, use `InferenceResults.image_overlay` so bounding boxes/labels
                                                   are burned into the clip. Defaults to True.
            save_ai_result_json (bool, optional): If True, dump a JSON file with raw inference results alongside the video. Defaults to True.
            target_fps (float, optional): Frame rate of the output file. Defaults to 30.0.
        """
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
