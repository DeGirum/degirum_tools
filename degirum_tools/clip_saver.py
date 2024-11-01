#
# clip_saver.py: video clip saving analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements analyzer class to save video clips triggered by event or notification
# with a specified duration before and after the event.
#

import os
import threading
from collections import deque
from typing import Set
from .result_analyzer_base import ResultAnalyzerBase
from .notifier import EventNotifier
from .event_detector import EventDetector
from .image_tools import image_size
from .video_support import open_video_writer


class ClipSaver(ResultAnalyzerBase):
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
        generate_ai_overlay: bool = True,
        target_fps=30.0,
    ):
        """
        Constructor

        Args:
            clip_duration: duration of the video clip to save (in frames)
            triggers: a set of event names or notifications which trigger video clip saving
            file_prefix: path and file prefix for video clip files
            pre_trigger_delay: delay before the event to start clip saving (in frames)
            generate_ai_overlay: True to generate AI inference overlay image, False to use original image
            target_fps: target frames per second for saved videos
        """

        if pre_trigger_delay >= clip_duration:
            raise ValueError("`pre_trigger_delay` should be less than `clip_duration`")
        if not triggers or not isinstance(triggers, set):
            raise ValueError("`triggers` should be non-empty set of string")
        self._clip_duration = clip_duration
        self._triggers = triggers
        self._pre_trigger_delay = pre_trigger_delay
        self._generate_ai_overlay = generate_ai_overlay
        self._file_prefix = file_prefix
        self._target_fps = target_fps

        # extract dir from file prefix and make it if it does not exist
        dir_path = os.path.dirname(file_prefix)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self._clip_buffer: deque = deque()
        self._end_counter = -1
        self._frame_counter = 0

    def analyze(self, result):
        """
        Analyze inference result and save video clip if event or notification trigger happens

        Args:
            result: PySDK model result object
        """

        # add result to the clip buffer
        self._clip_buffer.append(result)
        if len(self._clip_buffer) > self._clip_duration:
            self._clip_buffer.popleft()

        if self._end_counter < 0:
            # if not in the middle of accumulating the clip from the previous trigger...

            # check trigger
            triggered = False
            notifications = getattr(result, EventNotifier.key_notifications, None)
            if notifications is not None and self._triggers & notifications:
                triggered = True
            else:
                events = getattr(result, EventDetector.key_events_detected, None)
                if events is not None and self._triggers & events:
                    triggered = True

            # if triggered, set down-counting timer
            if triggered:
                self._end_counter = self._clip_duration - self._pre_trigger_delay
        else:
            # otherwise, continue accumulating the clip

            # decrement the timer, and if the timer is over, save the clip
            self._end_counter -= 1
            if self._end_counter <= 0:
                self._save_clip()
                self._end_counter = -1

        self._frame_counter += 1

    def _save_clip(self):
        """
        Save video clip from the buffer
        """

        def save(clip_buffer):
            if clip_buffer:
                w, h = image_size(clip_buffer[0].image)
                filename = f"{self._file_prefix}_{self._frame_counter - self._clip_duration:08d}.mp4"
                with open_video_writer(filename, w, h, self._target_fps) as writer:
                    for frame in clip_buffer:
                        if self._generate_ai_overlay:
                            writer.write(frame.overlay_image)
                        else:
                            writer.write(frame.image)

        # preserve a shallow copy of the clip buffer
        clip_buffer = deque(self._clip_buffer)

        # save the clip in a separate thread
        threading.Thread(target=save, args=(clip_buffer,)).start()
