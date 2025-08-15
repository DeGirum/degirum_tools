#
# notifier.py: notification analyzer
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements analyzer class to generate notifications based on triggered events.
# It works with conjunction with EventDetector analyzer.
#
"""
Notification Analyzer Module Overview
====================================

This module provides tools for generating and delivering notifications based on AI inference events.
It implements the `EventNotifier` analyzer for triggering notifications and optional clip saving on events.

Key Features:
    - Event-Based Triggers: Generates notifications when user-defined event conditions are met
    - Message Formatting: Supports Python format strings for dynamic notification content
    - Holdoff Control: Configurable time/frame windows to suppress repeat notifications
    - Video Clip Saving: Optional video clip saving with local or cloud storage
    - Visual Overlay: Annotates active notification status on images
    - File Management: Handles temporary file cleanup and storage integration

Typical Usage:
    1. Configure notification service. For external services, use Apprise URL or config file. For console output, use "json://console" as notification_config
    2. Define event conditions and create `EventNotifier` instances
    3. Process inference results through the notifier chain
    4. Notifications are sent when conditions are met

Integration Notes:
    - Requires `EventDetector` analyzer in the chain to provide event detection
    - Optional dependencies (e.g., apprise) must be installed for external notification services
    - Storage configuration required for clip saving (supports both local and cloud storage)
    - Supports both frame-based and time-based notification holdoff periods

Key Classes:
    - `EventNotifier`: Analyzer for triggering notifications based on event conditions

Configuration Options:
    - `notification_config`: Apprise URL or config file for notification service, or "json://console" for stdout output
    - `notification_title`: Default title for notifications
    - `holdoff_frames`: Number of frames to wait between notifications
    - `holdoff_seconds`: Time in seconds to wait between notifications
    - `clip_save`: Enable/disable video clip saving
    - `storage_config`: Storage configuration for clip saving (supports local and cloud storage)
    - `show_overlay`: Enable/disable visual annotations

Message Formatting:
    - Use Python format strings in the `message` parameter to include dynamic content (e.g., `{time}` for the time the notification was sent)
    - Use markdown formatting for rich text notifications (e.g. `**bold**`, `*italic*`, `[link](url)`)
    - Supported placeholders include:
        - `{result}`: The inference result
        - `{time}`: The time the notification was sent
        - `{url}`: The URL of the uploaded file
        - `{filename}`: The name of the uploaded file

Example:
    For local storage configuration:
    ```python
    clip_storage_config = ObjectStorageConfig(
        endpoint=".",  # path to local folder
        access_key="",  # not needed for local storage
        secret_key="",  # not needed for local storage
        bucket="my_bucket_dir",  # subdirectory name for local storage
    )
    ```
"""
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
    """Server to asynchronously send notifications and upload files.

    Runs as a background process handling jobs such as uploading files,
    sending notifications, and referencing previously uploaded files.

    Attributes:
        _job_queue (multiprocessing.Queue): Queue for sending jobs to the background process.
        _response_queue (multiprocessing.Queue): Queue for receiving responses from the background process.
        _process (multiprocessing.Process): Background process handling notifications.

    Jobs:
        * **job_file_upload**: Uploads a local file to object storage.
        * **job_file_reference**: References a file that has already been uploaded (generates a link).
        * **job_notification**: Sends a notification message via the configured service.

    Usage:
        Instantiate with the desired configuration, then send jobs using `send_job()`.
    """

    class DefaultDict(dict):
        def __missing__(self, key):
            return f"{{{key}}}"  # Return the literal placeholder if key is missing

    class Job:
        """Encapsulates a notification job for the NotificationServer.

        Attributes:
            id (int): Unique job identifier.
            job_type (str): Type of job (one of `job_file_upload`, `job_file_reference`, or `job_notification`).
            payload (str): Job payload data (e.g., file path or message text, depending on the job type).
            dependent (int or None): ID of another job that depends on this one (if any).
            timestamp (float): Timestamp when the job was created.
            is_done (bool): Indicates whether the job has completed.
            error (Exception or None): Captured exception if an error occurred during processing.
        """

        _id: int = 0  # job ID counter
        _lock = threading.Lock()

        # job types
        job_file_upload = "upload file"
        job_file_reference = "reference file"
        job_notification = "send notification"

        @staticmethod
        def new_id() -> int:
            """Generate a new job ID."""
            with NotificationServer.Job._lock:
                NotificationServer.Job._id += 1
                return NotificationServer.Job._id

        def __init__(
            self, job_type: str, payload: str, dependent: Optional[int] = None
        ):
            """Constructor.

            Args:
                job_type (str): The job type.
                payload (str): The job payload (content depends on the job type).
                dependent (int, optional): Job ID of another job that should run after this one. Defaults to None.
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
        """Constructor.

        Args:
            notification_cfg (str, optional): Path to an Apprise configuration file or notification URL (Apprise format). If None, no external notification service is used.
            notification_title (str, optional): Title for the notifications. If None, notifications are sent without a title.
            notification_tags (str, optional): Tags to attach to notifications for filtering. If None, no tags are applied.
            storage_cfg (ObjectStorageConfig, optional): Object storage configuration for uploading files (used for clip uploads). If None, file upload jobs are disabled.
            pending_timeout_s (float, optional): Maximum time in seconds to wait for a job to complete before marking it as timed out. Default is 5.0 seconds.

        Raises:
            RuntimeError: If the notification server fails to initialize.
            ImportError: If required optional packages (e.g., apprise) are not installed.
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
        """Process the internal response queue and log any exceptions."""
        logger = logger_get()
        while True:
            try:
                exc = self._response_queue.get_nowait()
                if isinstance(exc, Exception):
                    logger.error(f"Notification error: {exc}")
            except queue.Empty:
                break

    def send_job(self, job_type, payload: str, dependent: Optional[int] = None):
        """Queue a new job for the notification server.

        Args:
            job_type (str): The type of job to queue (use `NotificationServer.Job.job_*` constants).
            payload (str): The job payload (e.g., file path for uploads or message text for notifications).
            dependent (int, optional): ID of a job that should run after this job (to enforce order). Defaults to None.

        Returns:
            id (int): The ID of the posted job.
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
        """Process commands from the queue in a separate process.

        This method runs in a background process and handles all notification and file upload jobs.
        It maintains job state, handles dependencies between jobs, and manages timeouts.
        The process creates background threads for file uploads, sends notifications via configured
        services, and manages temporary files and cleanup.

        Args:
            job_queue (multiprocessing.Queue): Queue for receiving jobs from the main process.
            response_queue (multiprocessing.Queue): Queue for sending responses back to the main process.
            notification_cfg (str | None): Notification service configuration.
            notification_title (str | None): Title for notifications.
            notification_tags (str | None): Tags for notifications.
            storage_cfg (ObjectStorageConfig | None): Storage configuration for file uploads.
            pending_timeout_s (float): Timeout duration for pending jobs.
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

        param_key_url = "url"  # key to access the file URL in job results
        param_key_filename = "filename"  # key to access the file name in job results

        #
        # Run file upload job
        #
        def run_file_upload_job(job, do_upload):
            if storage is None:
                job.is_done = True
                return

            filepath = job.payload
            filename = os.path.basename(filepath)
            url: Optional[str] = None

            if do_upload and not os.path.exists(filepath):
                job.is_done = False  # no local file to upload yet: job is not done
                return None

            try:
                if do_upload:
                    storage.upload_file_to_object_storage(filepath, filename)
                    os.remove(filepath)  # remove local file after upload
                url = storage.generate_presigned_url(filename)

            except Exception as e:
                job.error = e

            job.is_done = True
            ret = {param_key_filename: filename}
            if url:
                ret[param_key_url] = url
            return ret

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
                except KeyboardInterrupt:
                    queue_is_active = False
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
        """Sends a poison pill to stop the child process.

        Raises:
            RuntimeError: If the process termination fails.
            Exception: Any exception that occurs during response queue processing.
        """
        if self._process is not None:
            self._job_queue.put(None)
            self._process.join()
            self._process = None
            self._process_response_queue()

    def __del__(self):
        # Send the poison pill to stop the child process
        self.terminate()


class EventNotifier(ResultAnalyzerBase):
    """Analyzer for event-based notifications.

    Works in conjunction with an `EventDetector` analyzer by examining the `events_detected` set in the inference results.
    Generates notifications when user-defined event conditions are met.

    Features:
        * Message formatting using Python format strings (e.g., `{result}` for inference results)
        * Holdoff to suppress repeat notifications within a specified time/frame window
        * Optional video clip saving upon notification trigger with local or cloud storage
        * Records triggered notifications in the result object's `notifications` dictionary
        * Overlay annotation of active notification status on images
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
        """Constructor.

        Args:
            name (str): Name of the notification.
            condition (str): Python expression defining the condition to trigger the notification (references event names from `EventDetector`).
            message (str, optional): Notification message format string. If empty, uses "Notification triggered: {name}". Default is "".
            holdoff (int | float | Tuple[float, str], optional): Holdoff duration to suppress repeated notifications. If int, interpreted as frames; if float, as seconds; if tuple (value, "seconds"/"frames"), uses the specified unit. Default is 0 (no holdoff).
            notification_config (str, optional): Notification service config file path, Apprise URL, or "json://console" for stdout output. If None, notifications are not sent to any external service.
            notification_tags (str, optional): Tags to attach to notifications for filtering. Multiple tags can be separated by commas (for logical AND) or spaces (for logical OR).
            show_overlay (bool, optional): Whether to overlay notification text on images. Default is True.
            annotation_color (tuple, optional): RGB color for the annotation text background. If None, uses a complementary color to the result overlay.
            annotation_font_scale (float, optional): Font scale for the annotation text. If None, uses the default model font scale.
            annotation_pos (AnchorPoint | Tuple[int, int] | List[int], optional): Position to place annotation text (either an AnchorPoint or an (x,y) coordinate). Default is AnchorPoint.BOTTOM_LEFT.
            annotation_cool_down (float, optional): Time in seconds to display the notification text on the image. Default is 3.0.
            clip_save (bool, optional): If True, save a video clip when the notification triggers. Default is False.
            clip_sub_dir (str, optional): Subdirectory name in the storage bucket for saved clips. Default is "" (no subdirectory).
            clip_duration (int, optional): Length of the saved video clip in frames. Default is 0 (uses available frames around event).
            clip_pre_trigger_delay (int, optional): Number of frames to include before the trigger event in the saved clip. Default is 0.
            clip_embed_ai_annotations (bool, optional): If True, embed AI annotations in the saved clip. Default is True.
            clip_target_fps (float, optional): Frame rate (FPS) for the saved video clip. Default is 30.0.
            storage_config (ObjectStorageConfig, optional): Storage configuration for clip saving. For local storage, use endpoint="./" and local directory as bucket. For cloud storage, use S3-compatible endpoint and credentials. If None, clips are only saved locally.

        Raises:
            ValueError: If holdoff unit is not "seconds" or "frames".
            ImportError: If required optional packages are not installed.
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
            full_clip_prefix = self._clip_path + (
                ("/" + clip_sub_dir + "/") if clip_sub_dir else "/"
            )

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
                notification_title=self._name,
                notification_tags=notification_tags,
                storage_cfg=self._storage_cfg if clip_save else None,
                pending_timeout_s=2 * clip_duration / clip_target_fps,
            )

    def analyze(self, result):
        """Evaluate the notification condition on the given inference result.

        If the condition is satisfied (and not within a holdoff period), generates a notification message and stores it in `result.notifications`.
        Optionally saves a video clip when a notification is triggered, and schedules that clip for upload if storage is configured.
        This method modifies the input result object in-place.

        Args:
            result (InferenceResults): The inference result to analyze, which should include events detected by an `EventDetector`.

        Returns:
            (None): This method modifies the input result object in-place.

        Raises:
            AttributeError: If the result does not contain events_detected (EventDetector not in chain).
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
        """Draws the active notification message on the image.

        Only draws the message if the notification is currently active and within its cool-down period.

        Args:
            result (InferenceResults): The inference result object (may contain a notification entry).
            image (np.ndarray): The image frame (BGR format) to annotate.

        Returns:
            np.ndarray: The annotated image.
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
        """Finalize and clean up resources.

        Waits for all background clip-saving threads to finish, stops the notification server, and removes the temporary clip directory if it was used.
        """
        if self._clip_save:
            self._clip_saver.join_all_saver_threads()

        if self.notification_server is not None:
            self.notification_server.terminate()

        if self._clip_save and os.path.exists(self._clip_path):
            shutil.rmtree(self._clip_path)
