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
import json
import uuid
import time
import copy
from collections import deque
from typing import Set, Dict, Any
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

        if pre_trigger_delay >= clip_duration:
            raise ValueError("`pre_trigger_delay` should be less than `clip_duration`")
        if not triggers or not isinstance(triggers, set):
            raise ValueError("`triggers` should be non-empty set of string")
        self._clip_duration = clip_duration
        self._triggers = triggers
        self._pre_trigger_delay = pre_trigger_delay
        self._embed_ai_annotations = embed_ai_annotations
        self._save_ai_result_json = save_ai_result_json
        self._file_prefix = file_prefix
        self._target_fps = target_fps

        # extract dir from file prefix and make it if it does not exist
        self._dir_path = os.path.dirname(file_prefix)
        if not os.path.exists(self._dir_path):
            os.makedirs(self._dir_path)

        self._clip_buffer: deque = deque()
        self._end_counter = -1
        self._triggered: set = set()
        self._frame_counter = 0
        self._thread_name = "dgtools_ClipSaverThread_" + str(uuid.uuid4())

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

            # if triggered, set down-counting timer
            if triggered:
                self._end_counter = self._clip_duration - self._pre_trigger_delay - 1
                self._triggered = triggered
        else:
            # otherwise, continue accumulating the clip

            # decrement the timer, and if the timer is over, save the clip
            self._end_counter -= 1
            if self._end_counter <= 0:
                self._save_clip()
                self._end_counter = -1
                self._triggered.clear()

        self._frame_counter += 1

    def _save_clip(self):
        """
        Save video clip from the buffer
        """

        def save(context):
            if context._clip_buffer:
                w, h = image_size(context._clip_buffer[0].image)
                start = max(0, context._frame_counter + 1 - context._clip_duration)
                filename = f"{context._file_prefix}_{start:08d}"

                with open_video_writer(
                    filename + ".mp4", w, h, context._target_fps
                ) as writer:

                    json_result: Dict[str, Any] = {}
                    if context._save_ai_result_json:
                        json_result["properties"] = dict(
                            timestamp=time.ctime(),
                            start_frame=start,
                            triggered_by=context._triggered,
                            duration=context._clip_duration,
                            pre_trigger_delay=context._pre_trigger_delay,
                            target_fps=context._target_fps,
                        )
                        json_result["results"] = []

                    for result in context._clip_buffer:
                        if context._embed_ai_annotations:
                            writer.write(result.overlay_image)
                        else:
                            writer.write(result.image)

                        if context._save_ai_result_json:
                            json_result["results"].append(
                                {
                                    k: v
                                    for k, v in result.__dict__.items()
                                    if not k.startswith("__")
                                    and isinstance(
                                        v,
                                        (int, float, str, bool, tuple, list, dict, set),
                                    )
                                }
                            )

                    if context._save_ai_result_json:

                        def custom_serializer(obj):
                            if isinstance(obj, set):
                                return list(obj)
                            raise TypeError(
                                f"Object of type {obj.__class__.__name__} is not JSON serializable"
                            )

                        with open(filename + ".json", "w") as f:
                            json.dump(
                                json_result,
                                f,
                                indent=2,
                                default=custom_serializer,
                            )

        # preserve a shallow copy of self to use in thread
        context = copy.copy(self)
        context._clip_buffer = deque(self._clip_buffer)
        context._triggered = set(self._triggered)

        # save the clip in a separate thread
        threading.Thread(target=save, args=(context,), name=self._thread_name).start()

    def join_all_saver_threads(self):
        """
        Join all threads started by this instance
        """
        for thread in threading.enumerate():
            if thread.name == self._thread_name:
                thread.join()
