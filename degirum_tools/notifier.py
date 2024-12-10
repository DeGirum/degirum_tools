#
# notifier.py: notification analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements analyzer class to generate notifications based on triggered events.
# It works with conjunction with EventDetector analyzer.
#


import numpy as np, multiprocessing, threading, time, os, queue, tempfile, shutil
from typing import Tuple, Union, Optional, List
from dataclasses import dataclass
from . import logger_get
from .result_analyzer_base import ResultAnalyzerBase
from .ui_support import put_text, color_complement, deduce_text_color, CornerPosition
from .math_support import AnchorPoint, get_image_anchor_point
from .event_detector import EventDetector
from .environment import import_optional_package
from .video_support import ClipSaver
from .object_storage_support import ObjectStorageConfig, ObjectStorage


class NotificationServer:
    """
    Notification server class to asynchronously send notifications via apprise
    """

    @dataclass
    class Job:
        """
        Job dataclass to store file upload request
        """

        payload: str  # job payload (it is job-dependent)
        timestamp: float = time.time()  # timestamp of the job
        is_done: bool = False
        error: Optional[Exception] = None

    def __init__(
        self,
        notification_cfg: Optional[str],
        notification_title: Optional[str],
        notification_tags: Optional[str],
        storage_cfg: Optional[ObjectStorageConfig],
        pending_timeout_s: float = 5.0,
    ):
        """
        Constructor

        Args:
            notification_cfg: path to the apprise configuration file or notification server URL in apprise format
            notification_title: title of the notification
            notification_tags: tags to use for cloud notifications
            storage_cfg: object storage configuration for file uploads
            pending_timeout_s: timeout for pending notifications and file uploads
        """

        # create message queues and start child process
        self._message_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._file_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._response_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._process: Optional[multiprocessing.Process] = multiprocessing.Process(
            target=NotificationServer._process_commands,
            args=(
                self._message_queue,
                self._file_queue,
                self._response_queue,
                notification_cfg,
                notification_title,
                notification_tags,
                storage_cfg,
                pending_timeout_s,
            ),
        )
        self._process.start()

        response = self._response_queue.get()  # wait for the child process to be ready
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, bool) or not response:
            raise RuntimeError("Failed to initialize notification server")

    def _process_response_queue(self):
        """
        Process response queue to handle exceptions
        """
        logger = logger_get()
        while True:
            try:
                exc = self._response_queue.get_nowait()
                if isinstance(exc, Exception):
                    logger.error(f"Notification error: {exc}")
            except queue.Empty:
                break

    def send_notification(self, message: str):
        """
        Send notification to cloud service

        Args:
            message: notification message
        """
        self._message_queue.put(message)
        self._process_response_queue()

    def send_file_upload_req(self, local_filepaths: List[str]):
        """
        Send file upload request

        Args:
            local_filepaths: list of local file paths to upload
        """
        self._file_queue.put(local_filepaths)
        self._process_response_queue()

    @staticmethod
    def _process_commands(
        message_queue: multiprocessing.Queue,
        file_queue: multiprocessing.Queue,
        response_queue: multiprocessing.Queue,
        notification_cfg: Optional[str],
        notification_title: Optional[str],
        notification_tags: Optional[str],
        storage_cfg: Optional[ObjectStorageConfig],
        pending_timeout_s: float,
    ):
        """
        Process commands from the queue. Runs in separate process.
        """

        response_queue.put(True)  # to indicate that the child process is ready

        # process notification messages from the queue
        def process_notifications():
            if not notification_cfg:
                return

            apprise = import_optional_package(
                "apprise",
                custom_message="`apprise` package is required for notifications. "
                + "Please run `pip install degirum_tools[notifications]` to install required dependencies.",
            )

            try:
                # initialize Apprise object
                apprise_obj = apprise.Apprise()
                if os.path.isfile(notification_cfg):
                    # treat notification_cfg as a path to the configuration file
                    conf = apprise.AppriseConfig()
                    if not conf.add(notification_cfg):
                        raise ValueError(
                            f"Invalid configuration file: {notification_cfg}"
                        )
                    apprise_obj.add(conf)
                else:
                    # treat notification_cfg as a single server URL
                    if not apprise_obj.add(notification_cfg):
                        raise ValueError(
                            f"Invalid configuration URL: {notification_cfg}"
                        )
            except Exception as e:
                response_queue.put(e)
                return

            while True:
                message = message_queue.get()
                if message is None:
                    # poison pill received, exit the loop
                    break

                try:
                    if "unittest" in notification_cfg:  # special case for unit tests
                        continue

                    if not apprise_obj.notify(
                        body=message, title=notification_title, tag=notification_tags
                    ):
                        raise Exception(
                            f"Notification failed: {notification_title} - {message}"
                        )
                except Exception as e:
                    response_queue.put(e)

        # process file upload requests from the queue
        def process_file_uploads():
            if not storage_cfg:
                return

            queue_poll_interval_s = 0.1

            storage = ObjectStorage(storage_cfg)
            storage.ensure_bucket_exists()

            pending_jobs: List[NotificationServer.Job] = []
            queue_is_active = True

            while queue_is_active or pending_jobs:
                if queue_is_active:
                    try:
                        files_to_upload = file_queue.get(
                            timeout=queue_poll_interval_s if pending_jobs else None
                        )
                    except queue.Empty:
                        files_to_upload = []

                    if files_to_upload is None:
                        queue_is_active = False

                    if files_to_upload:
                        pending_jobs.extend(
                            [NotificationServer.Job(f) for f in files_to_upload]
                        )

                # upload file to the cloud
                for job in pending_jobs:
                    file_name = job.payload
                    parent_dir = os.path.basename(os.path.dirname(file_name))
                    filename_with_parent_dir = (
                        parent_dir + "/" + os.path.basename(file_name)
                    )

                    if not os.path.exists(file_name):
                        if time.time() - job.timestamp > pending_timeout_s:
                            job.error = TimeoutError(f"File not found: {file_name}")
                            job.is_done = True
                        continue

                    try:
                        storage.upload_file_to_object_storage(
                            file_name, filename_with_parent_dir
                        )
                    except Exception as e:
                        job.error = e

                    job.is_done = True

                # remove completed and stalled jobs
                for job in pending_jobs[:]:  # use shallow copy
                    if job.is_done:
                        if os.path.exists(job.payload):
                            os.remove(job.payload)
                        if job.error:
                            response_queue.put(job.error)
                        pending_jobs.remove(job)

        # start processing notifications and file uploads in separate threads
        threads = []
        if notification_cfg:
            threads.append(threading.Thread(target=process_notifications))
        if storage_cfg:
            threads.append(threading.Thread(target=process_file_uploads))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def terminate(self):
        """Send the poison pill to stop the child process"""
        if self._process is not None:
            self._message_queue.put(None)
            self._file_queue.put(None)
            self._process.join()
            self._process = None
            self._process_response_queue()

    def __del__(self):
        # Send the poison pill to stop the child process
        self.terminate()


class EventNotifier(ResultAnalyzerBase):
    """
    Class to generate notifications based on triggered events.
    It works with conjunction with EventDetector analyzer, analyzing `events_detected` set
    in the `result` object.

    Adds `notifications` dictionary to the `result` object, where keys are names of generated
    notifications and values are notification messages.
    """

    key_notifications = "notifications"

    def __init__(
        self,
        name: str,
        condition: str,
        *,
        message: str = "",
        holdoff: Union[Tuple[float, str], int, float] = 0,
        notification_config: Optional[str] = None,
        notification_tags: Optional[str] = None,
        show_overlay: bool = True,
        annotation_color: Optional[tuple] = None,
        annotation_font_scale: Optional[float] = None,
        annotation_pos: Union[AnchorPoint, tuple] = AnchorPoint.BOTTOM_LEFT,
        annotation_cool_down: float = 3.0,
        clip_save: bool = False,
        clip_sub_dir: str = "",
        clip_duration: int = 0,
        clip_pre_trigger_delay: int = 0,
        clip_embed_ai_annotations: bool = True,
        clip_target_fps: float = 30.0,
        storage_config: Optional[ObjectStorageConfig] = None,
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
                it can be path to the configuration file for notification service
                or single notification service URL
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
            clip_save: if True, save video clips when notification is triggered
            clip_sub_dir: the name of subdirectory in the bucket for video clip files
            clip_duration: duration of the video clip to save (in frames)
            clip_pre_trigger_delay: delay before the event to start clip saving (in frames)
            clip_embed_ai_annotations: True to embed AI inference annotations into video clip, False to use original image
            clip_target_fps: target frames per second for saved videos
            storage_config: The object storage configuration (to save video clips)
        """

        self._frame = 0
        self._prev_cond = False
        self._prev_frame = -1_000_000_000  # arbitrary big negative number
        self._prev_time = -1_000_000_000.0
        self._last_notifications: dict = {}
        self._last_display_time = -1_000_000_000.0

        self._name = name
        self._message = message if message else f"Notification triggered: {name}"
        self._show_overlay = show_overlay
        self._annotation_color = annotation_color
        self._annotation_font_scale = annotation_font_scale
        self._annotation_pos = annotation_pos
        self._annotation_cool_down = annotation_cool_down
        self._clip_save = clip_save
        self.notification_server: Optional[NotificationServer] = None

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

        # compile condition to evaluate it later
        self._condition = compile(condition, "<string>", "eval")

        # instantiate clip saver if required
        if clip_save and storage_config:
            self._clip_path = tempfile.mkdtemp()
            full_clip_prefix = self._clip_path + "/" + clip_sub_dir + "/"

            self._clip_saver = ClipSaver(
                clip_duration,
                full_clip_prefix,
                pre_trigger_delay=clip_pre_trigger_delay,
                embed_ai_annotations=clip_embed_ai_annotations,
                save_ai_result_json=True,
                target_fps=clip_target_fps,
            )
            self._storage_cfg = storage_config

        # setting up notification server
        if (isinstance(notification_config, str) and notification_config) or clip_save:
            self.notification_server = NotificationServer(
                notification_config,
                self._name,
                notification_tags,
                self._storage_cfg if clip_save else None,
            )

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

        if not hasattr(result, EventDetector.key_events_detected):
            raise AttributeError(
                "Detected events info is not available in the result: insert EventDetector analyzer in a chain"
            )

        # evaluate condition using detected event names as variables in the condition expression
        var_dict = {v: (v in result.events_detected) for v in self._condition.co_names}
        cond = eval(self._condition, var_dict)

        if not hasattr(result, self.key_notifications):
            result.notifications = {}

        fired = False
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
                fired = True

        # save video clip if required
        clip_url = ""
        if self._clip_save:
            clip_filenames = self._clip_saver.forward(
                result, [self._name] if fired else []
            )
            if clip_filenames:
                if self.notification_server is not None:
                    self.notification_server.send_file_upload_req(clip_filenames)
                    clip_url = self._storage_cfg.construct_direct_url(
                        os.path.basename(clip_filenames[0])
                    )

        if fired:
            message = self._message.format(result=result, clip_url=clip_url)
            result.notifications[self._name] = message

            # send notifications to cloud service
            if self.notification_server is not None:
                self.notification_server.send_notification(message)

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

        if hasattr(result, self.key_notifications) and result.notifications:
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

    def __del__(self):
        if self._clip_save:
            self._clip_saver.join_all_saver_threads()

        if self.notification_server:
            self.notification_server.terminate()

        if self._clip_save and os.path.exists(self._clip_path):
            shutil.rmtree(self._clip_path)
