#
# notifier.py: notification analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements analyzer class to generate notifications based on triggered events.
# It works with conjunction with EventDetector analyzer.
#

import numpy as np
import multiprocessing
import time
import os
import queue
from typing import Tuple, Union, Optional
from .result_analyzer_base import ResultAnalyzerBase
from .ui_support import put_text, color_complement, deduce_text_color, CornerPosition
from .math_support import AnchorPoint, get_image_anchor_point


class NotificationServer:
    """
    Notification server class to asynchronously send notifications via apprise
    """

    def __init__(self, title: str, config: str, tags: Optional[str]):
        """
        Constructor

        Args:
            config: path to the apprise configuration file or notification server URL in apprise format
            tags: tags to use for cloud notifications
        """
        import atexit

        # create message queues and start child process
        self._in_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._out_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._process: Optional[multiprocessing.Process] = multiprocessing.Process(
            target=NotificationServer._process_commands,
            args=(self._in_queue, self._out_queue, title, config, tags),
        )
        self._process.start()

        response = self._out_queue.get()  # wait for the child process to be ready
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, bool) or not response:
            raise RuntimeError("Failed to initialize notification server")

        # register atexit handler to send the poison pill
        atexit.register(self._send_poison)

    def send_notification(self, message: str):
        """
        Send notification to cloud service

        Args:
            message: notification message
        """
        self._in_queue.put(message)
        try:
            exc = self._out_queue.get_nowait()
            if isinstance(exc, Exception):
                raise exc
        except queue.Empty:
            pass

    @staticmethod
    def _process_commands(
        in_queue: multiprocessing.Queue,
        out_queue: multiprocessing.Queue,
        title: str,
        config: str,
        tags: str,
    ):
        """
        Process commands from the queue. Runs in separate process.
        """
        from apprise import Apprise, AppriseConfig

        try:
            # initialize Apprise object
            apprise_obj = Apprise()
            if os.path.isfile(config):
                # treat config as a path to the configuration file
                conf = AppriseConfig()
                if not conf.add(config):
                    raise ValueError(f"Invalid configuration file: {config}")
                apprise_obj.add(conf)
            else:
                # treat config as a single server URL
                if not apprise_obj.add(config):
                    raise ValueError(f"Invalid configuration URL: {config}")
        except Exception as e:
            out_queue.put(e)
            return

        out_queue.put(True)  # to indicate that the child process is ready

        # process messages from the queue
        while True:
            message = in_queue.get()
            if message is None:
                # poison pill received, exit the loop
                break

            try:
                if "unittest" in config:  # special case for unit tests
                    continue

                if not apprise_obj.notify(body=message, title=title, tag=tags):
                    raise Exception(f"Notification failed: {title} - {message}")
            except Exception as e:
                out_queue.put(e)

    def _send_poison(self):
        """Send the poison pill to stop the child process"""
        if self._process is not None:
            self._in_queue.put(None)
            self._process.join()
            self._process = None

    def __del__(self):
        # Send the poison pill to stop the child process
        self._send_poison()


class EventNotifier(ResultAnalyzerBase):
    """
    Class to generate notifications based on triggered events.
    It works with conjunction with EventDetector analyzer, analyzing `events_detected` set
    in the `result` object.

    Adds `notifications` dictionary to the `result` object, where keys are names of generated
    notifications and values are notification messages.
    """

    def __init__(
        self,
        name: str,
        condition: str,
        *,
        message: str = "",
        holdoff: Union[Tuple[float, str], int, float] = 0,
        notification_config: Union[str, NotificationServer, None] = None,
        notification_tags: Optional[str] = None,
        show_overlay: bool = True,
        annotation_color: Optional[tuple] = None,
        annotation_font_scale: Optional[float] = None,
        annotation_pos: Union[AnchorPoint, tuple] = AnchorPoint.BOTTOM_LEFT,
        annotation_cool_down: float = 3.0,
    ):
        """
        Constructor

        Args:
            name: name of the notification event
            condition: condition to trigger notification; may be any valid Python expression, referencing
                event names, as generated by preceding EventDetector analyzers.
            holdoff: holdoff time to suppress repeated notifications; it is either integer holdoff value in frames,
                floating-point holdoff value in seconds, or a tuple in a form (holdoff, unit), where unit is either
                "seconds" or "frames".
            message: message to display in the notification; may be valid Python format string, in which you can use
                `{result}` placeholder with any valid derivatives to access current inference result and its attributes.
                For example: "Objects detected: {result.results}"
            notification_config: optional configuration of cloud notification service:
                it can be already constructed notification server object (taken from another notifier);
                or path to the configuration file for notification service; or single notification service URL
            notification_tags: optional tags to use for cloud notifications
                Tags can be separated by "and" (or commas) and "or" (or spaces).
                For example:
                "Tag1, Tag2" (equivalent to "Tag1 and Tag2"
                "Tag1 Tag2" (equivalent to "Tag1 or Tag2")
            apprise_config_path: The config file (either in yaml or txt) for the apprise library on github which is
                used by us to send notifications. [https://github.com/caronc/apprise]
            show_overlay: if True, annotate image; if False, send through original image
            annotation_color: Color to use for annotations, None to use complement to result overlay color
            annotation_font_scale: font scale to use for annotations or None to use model default
            annotation_pos: position to place annotation text (either predefined point or (x,y) tuple)
            annotation_cool_down: time in seconds to keep notification on the screen
        """

        self._name = name
        self._message = message if message else f"Notification triggered: {name}"
        self._show_overlay = show_overlay
        self._annotation_color = annotation_color
        self._annotation_font_scale = annotation_font_scale
        self._annotation_pos = annotation_pos
        self._annotation_cool_down = annotation_cool_down

        # setting up notification server
        self.notification_server: Optional[NotificationServer] = None
        if isinstance(notification_config, NotificationServer):
            # externally provided notification server
            self.notification_server = notification_config
        elif isinstance(notification_config, str) and notification_config:
            self.notification_server = NotificationServer(
                self._name, notification_config, notification_tags
            )

        # compile condition to evaluate it later
        self._condition = compile(condition, "<string>", "eval")

        # parse holdoff duration
        self._holdoff_frames = 0
        self._holdoff_sec = 0.0
        if isinstance(holdoff, int):
            self._holdoff_frames = holdoff
        elif isinstance(holdoff, float):
            self._holdoff_sec = holdoff
        elif isinstance(holdoff, tuple) or isinstance(holdoff, list):
            if holdoff[1] == "seconds":
                self._holdoff_sec = holdoff[0]
            elif holdoff[1] == "frames":
                self._holdoff_frames = int(holdoff[0])
            else:
                raise ValueError(
                    f"Invalid unit in holdoff time {holdoff[1]}, must be 'seconds' or 'frames'"
                )
        else:
            raise TypeError(f"Invalid holdoff time type: {holdoff}")

        self._frame = 0
        self._prev_cond = False
        self._prev_frame = -1_000_000_000  # arbitrary big negative number
        self._prev_time = -1_000_000_000.0
        self._last_notifications: dict = {}
        self._last_display_time = -1_000_000_000.0

    def analyze(self, result):
        """
        Generate notification by analyzing given result according to the condition expression.

        If condition is met the first time, the notification is generated.

        If condition is met repeatedly on every consecutive frame, the notification is generated only once, when
        condition is met the first time.

        If condition is not met for a period less than holdoff time and then met again, the notification
        is not generated to reduce the number of notifications.

        When notification is generated, the notification message is stored in the `result.notifications` dictionary
        under the key equal to the notification name.

        Args:
            result: PySDK model result object
        """

        if not hasattr(result, "events_detected"):
            raise AttributeError(
                "Detected events info is not available in the result: insert EventDetector analyzer in a chain"
            )

        # evaluate condition using detected event names as variables in the condition expression
        var_dict = {v: (v in result.events_detected) for v in self._condition.co_names}
        cond = eval(self._condition, var_dict)

        if not hasattr(result, "notifications"):
            result.notifications = {}

        if cond and not self._prev_cond:  # condition is met for the first time
            # check for holdoff time
            if (
                (self._holdoff_frames == 0 and self._holdoff_sec == 0)  # no holdoff
                or (
                    self._holdoff_frames > 0
                    and (self._frame - self._prev_frame > self._holdoff_frames)
                )  # holdoff in frames is passed
                or (
                    self._holdoff_sec > 0
                    and (time.time() - self._prev_time > self._holdoff_sec)
                )  # holdoff in seconds is passed
            ):
                result.notifications[self._name] = self._message.format(result=result)

                # send notifications to cloud service
                if self.notification_server is not None:
                    self.notification_server.send_notification(
                        result.notifications[self._name]
                    )

        # update holdoff timestamp and frame number if condition is met
        if cond:
            self._prev_frame = self._frame
            self._prev_time = time.time()

        self._prev_cond = cond
        self._frame += 1

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Display active notifications on a given image

        Args:
            result: PySDK result object to display (should be the same as used in analyze() method)
            image (np.ndarray): image to display on

        Returns:
            np.ndarray: annotated image
        """

        if not self._show_overlay:
            return image

        if hasattr(result, "notifications") and result.notifications:
            self._last_notifications = {
                k: v for k, v in result.notifications.items() if self._name == k
            }
            self._last_display_time = time.time()
        else:
            if (
                not self._last_notifications
                or time.time() - self._last_display_time > self._annotation_cool_down
            ):
                return image

        bg_color = (
            color_complement(result.overlay_color)
            if self._annotation_color is None
            else self._annotation_color
        )
        text_color = deduce_text_color(bg_color)

        if isinstance(self._annotation_pos, AnchorPoint):
            pos = get_image_anchor_point(
                image.shape[1], image.shape[0], self._annotation_pos
            )
        else:
            pos = self._annotation_pos

        return put_text(
            image,
            "\n".join(self._last_notifications.values()),
            pos,
            font_color=text_color,
            bg_color=bg_color,
            font_scale=(
                result.overlay_font_scale
                if self._annotation_font_scale is None
                else self._annotation_font_scale
            ),
            corner_position=CornerPosition.AUTO,
        )
