#
# clip_saver.py: video clip saving analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements analyzer class to save video clips triggered by event or notification
# with a specified duration before and after the event.
#

from typing import Set
from .result_analyzer_base import ResultAnalyzerBase
from .notifier import EventNotifier
from .event_detector import EventDetector
from .video_support import ClipSaver


class ClipSavingAnalyzer(ResultAnalyzerBase):
    """
    Class to to save video clips triggered by event or notification
    with a specified duration before and after the event.
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
        Constructor

        Args:
            clip_duration: duration of the video clip to save (in frames)
            triggers: a set of event names or notifications which trigger video clip saving
            file_prefix: path and file prefix for video clip files
            pre_trigger_delay: delay before the event to start clip saving (in frames)
            embed_ai_annotations: True to embed AI inference annotations into video clip, False to use original image
            save_ai_result_json: True to save AI result JSON file along with video clip
            target_fps: target frames per second for saved videos
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
        Analyze inference result and save video clip if event or notification trigger happens

        Args:
            result: PySDK model result object
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
        Join all threads started by this instance
        """
        return self._saver.join_all_saver_threads()
