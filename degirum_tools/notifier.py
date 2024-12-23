#
# notifier.py: notification analyzer
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements analyzer class to generate notifications based on triggered events.
# It works with conjunction with EventDetector analyzer.
#


import numpy as np, sys, multiprocessing, threading, time, os, queue, tempfile, shutil, datetime
from typing import Tuple, List, Union, Optional, Dict
from contextvars import ContextVar
from . import logger_get
from .result_analyzer_base import ResultAnalyzerBase
from .ui_support import put_text, color_complement, deduce_text_color, CornerPosition
from .math_support import AnchorPoint, get_image_anchor_point
from .event_detector import EventDetector
from .environment import import_optional_package
from .video_support import ClipSaver
from .object_storage_support import ObjectStorageConfig, ObjectStorage


# special notification configuration for console output
notification_config_console = "json://console"


class NotificationServer:
    """
    Notification server class to asynchronously send notifications via apprise
    and upload files to the object storage.
    """

    class DefaultDict(dict):
        def __missing__(self, key):
            return f"{{{key}}}"  # Return the literal placeholder if key is missing

    class Job:
        """
        Notification server job base class
        """

        _id: int = 0  # job ID counter
        _lock = threading.Lock()

        # job types
        job_file_upload = "upload file"
        job_file_reference = "reference file"
        job_notification = "send notification"

        @staticmethod
        def new_id() -> int:
            """Generate new job ID"""
            with NotificationServer.Job._lock:
                NotificationServer.Job._id += 1
                return NotificationServer.Job._id

        def __init__(
            self, job_type: str, payload: str, dependent: Optional[int] = None
        ):
            """
            Constructor

            Args:
                job_type: job type
                payload: job payload (it is job-dependent)
                dependent: job ID of the job which depends on this one
            """

            self.id = NotificationServer.Job.new_id()  # job ID
            self.job_type = job_type  # job type
            self.dependent = dependent  # job ID of dependent job
            self.payload = payload  # job payload (it is job-dependent)
            self.timestamp: float = time.time()  # timestamp of the job
            self.is_done: bool = False
            self.error: Optional[Exception] = None

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
        self._job_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._response_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._process: Optional[multiprocessing.Process] = multiprocessing.Process(
            target=NotificationServer._process_commands,
            args=(
                self._job_queue,
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

    def send_job(self, job_type, payload: str, dependent: Optional[int] = None):
        """
        Send job to notification server

        Args:
            job_type: job type
            payload: job payload
            dependent: job ID of another job which depends on this one

        Returns:
            job ID of posted job
        """

        job = NotificationServer.Job(job_type, payload, dependent)
        self._job_queue.put(job)
        self._process_response_queue()
        return job.id

    @staticmethod
    def _process_commands(
        job_queue: multiprocessing.Queue,
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

        def configure_file_uploads():
            if storage_cfg:
                try:
                    storage = ObjectStorage(storage_cfg)
                    storage.ensure_bucket_exists()
                    return storage
                except Exception as e:
                    response_queue.put(e)
            return None

        def configure_notifications():
            if notification_cfg:
                try:
                    apprise = import_optional_package(
                        "apprise",
                        custom_message="`apprise` package is required for notifications. "
                        + "Please run `pip install degirum_tools[notifications]` to install required dependencies.",
                    )

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
                    return apprise_obj
                except Exception as e:
                    response_queue.put(e)

            return None

        storage = configure_file_uploads()
        notifier = configure_notifications()

        param_key_url = "url"  # key to access the file URL in the job results

        #
        # Run file upload job
        #
        def run_file_upload_job(job, do_upload):
            if storage is None:
                job.is_done = True
                return

            file_name = job.payload
            parent_dir = os.path.basename(os.path.dirname(file_name))
            filename_with_parent_dir = parent_dir + "/" + os.path.basename(file_name)
            url: Optional[str] = None

            if do_upload and not os.path.exists(file_name):
                job.is_done = False  # no local file to upload yet: job is not done
                return None

            try:
                if do_upload:
                    storage.upload_file_to_object_storage(
                        file_name, filename_with_parent_dir
                    )
                    os.remove(file_name)  # remove local file after upload
                url = storage.generate_presigned_url(filename_with_parent_dir)

            except Exception as e:
                job.error = e

            job.is_done = True
            return {param_key_url: url} if url else None

        #
        # Run notification job
        #
        def run_notification_job(job, params):

            message = job.payload
            need_params = "{" in message and "}" in message

            if need_params and not params:
                job.is_done = False  # no params available yet: job is not done
                return None

            job.is_done = True
            if (
                notifier is None
                or notification_cfg is None
                or "unittest" in notification_cfg  # special case for unit tests
            ):
                return None

            try:

                # replace placeholders in the message
                if need_params:
                    message = message.format_map(NotificationServer.DefaultDict(params))

                if notification_config_console in notification_cfg:
                    print(
                        f"{datetime.datetime.now()}: {notification_title} - {message}"
                    )
                    sys.stdout.flush()
                else:
                    if not notifier.notify(
                        body=message, title=notification_title, tag=notification_tags
                    ):
                        raise Exception(
                            f"Notification failed: {notification_title} - {message}"
                        )
            except Exception as e:
                job.error = e

            return None

        #
        # Job processing loop
        #
        pending_jobs: Dict[int, NotificationServer.Job] = {}
        job_results: Dict[int, dict] = {}
        queue_poll_interval_s = 0.5
        queue_is_active = True

        while queue_is_active or pending_jobs:

            # process queue
            if queue_is_active:
                try:
                    job = job_queue.get(
                        timeout=queue_poll_interval_s if pending_jobs else None
                    )
                    if job is None:  # poison pill received
                        queue_is_active = False
                except queue.Empty:
                    job = None

                if job is not None:
                    pending_jobs[job.id] = job

            if not pending_jobs:
                continue

            # process pending jobs
            for job in pending_jobs.values():
                # prepare job parameters
                params: Optional[dict] = job_results.get(job.id)
                results: Optional[dict] = None

                # run job
                if job.job_type == NotificationServer.Job.job_file_upload:
                    results = run_file_upload_job(job, True)
                elif job.job_type == NotificationServer.Job.job_file_reference:
                    results = run_file_upload_job(job, False)
                elif job.job_type == NotificationServer.Job.job_notification:
                    results = run_notification_job(job, params)
                else:
                    job.error = NotImplementedError(f"Invalid job type: {job.job_type}")
                    job.is_done = True

                # handle job timeouts
                if not job.is_done and time.time() - job.timestamp > pending_timeout_s:
                    job.error = TimeoutError(
                        f"Job {job.job_type} '{job.payload}' is not completed in {pending_timeout_s} sec"
                    )
                    job.is_done = True

                # handle errors
                if job.error:
                    response_queue.put(job.error)

                # save results for dependent job
                if results and job.dependent:
                    job_results[job.dependent] = results

                # remove results of completed job
                if job.is_done:
                    job_results.pop(job.id, None)

            # cleanup completed jobs
            pending_jobs = {k: v for k, v in pending_jobs.items() if not v.is_done}

    def terminate(self):
        """Send the poison pill to stop the child process"""
        if self._process is not None:
            self._job_queue.put(None)
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

    It sends notifications to the notification service specified in `notification_config`.

    If `clip_save` is set to True, it saves video clips when notification is triggered.
    Saved clips are uploaded to the object storage if `storage_config` is provided.
    """

    key_notifications = "notifications"  # extra result key

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
        annotation_pos: Union[
            AnchorPoint, Tuple[int, int], List[int]
        ] = AnchorPoint.BOTTOM_LEFT,
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
                or single notification service URL (see [https://github.com/caronc/apprise])
            notification_tags: optional tags to use for cloud notifications
                Tags can be separated by "and" (or commas) and "or" (or spaces).
                For example:
                "Tag1, Tag2" (equivalent to "Tag1 and Tag2"
                "Tag1 Tag2" (equivalent to "Tag1 or Tag2")
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
        self._last_notification = ContextVar(
            f"last_notification_{id(self)}", default=""
        )
        self._last_display_time = ContextVar(
            f"last_display_time_{id(self)}", default=-1_000_000_000.0
        )

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

        # send notification if event is fired
        notification_job_id: Optional[int] = None
        if fired:
            message = self._message.format_map(
                NotificationServer.DefaultDict(result=result, time=time.asctime())
            )
            result.notifications[self._name] = message

            # send notifications to cloud service
            if self.notification_server is not None:
                notification_job_id = self.notification_server.send_job(
                    NotificationServer.Job.job_notification, message
                )

        # save video clip if required
        if self._clip_save:
            clip_filenames, new_files = self._clip_saver.forward(
                result, [self._name] if fired else []
            )
            if clip_filenames:
                if self.notification_server is not None:
                    for clip_filename in clip_filenames:

                        job_type = (
                            NotificationServer.Job.job_file_upload
                            if new_files
                            else NotificationServer.Job.job_file_reference
                        )
                        dependent_job = (
                            notification_job_id if ".mp4" in clip_filename else None
                        )

                        if new_files or dependent_job is not None:
                            self.notification_server.send_job(
                                job_type,
                                clip_filename,
                                dependent_job,
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

        # handle cool down time
        if hasattr(result, self.key_notifications) and result.notifications:
            msg = result.notifications.get(self._name)
            if msg:
                self._last_notification.set(msg)
                self._last_display_time.set(time.time())

        last_notification = self._last_notification.get()
        last_display_time = self._last_display_time.get()
        if (
            not last_notification
            or time.time() - last_display_time > self._annotation_cool_down
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
            pos = tuple(self._annotation_pos)

        return put_text(
            image,
            last_notification,
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

    def finalize(self):
        """
        Perform finalization/cleanup actions
        """
        if self._clip_save:
            self._clip_saver.join_all_saver_threads()

        if self.notification_server is not None:
            self.notification_server.terminate()

        if self._clip_save and os.path.exists(self._clip_path):
            shutil.rmtree(self._clip_path)
