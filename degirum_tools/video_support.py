#
# video_support.py: video stream handling classes and functions
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements classes and functions to handle video streams for capturing and saving
#

"""
Video Support Module Overview
============================

This module provides comprehensive video stream handling capabilities, including
capturing from various sources, saving to files, and managing video clips. It
supports local cameras, IP cameras, video files, and YouTube videos.

Key Features:
    - **Multi-Source Support**: Capture from local cameras, IP cameras, video files, and YouTube
    - **Video Writing**: Save video streams with configurable quality and format
    - **Frame Extraction**: Convert video files to JPEG sequences
    - **Clip Management**: Save video clips triggered by events with pre/post buffers
    - **FPS Control**: Frame rate management for both capture and writing
    - **Stream Properties**: Query video stream dimensions and frame rate

Typical Usage:
    1. Open video streams with `open_video_stream()`
    2. Process frames using `video_source()` generator
    3. Save videos with `VideoWriter` or `open_video_writer()`
    4. Extract frames using `video2jpegs()`
    5. Save event-triggered clips with `ClipSaver`

Integration Notes:
    - Works with OpenCV's VideoCapture and VideoWriter
    - Supports YouTube videos through pafy
    - Handles both real-time and file-based video sources
    - Provides context managers for safe resource handling
    - Thread-safe for concurrent video operations

Key Classes:
    - `VideoWriter`: Main class for saving video streams
    - `ClipSaver`: Manages saving video clips with pre/post buffers

Configuration Options:
    - Video quality and format settings
    - Frame rate control
    - Clip duration and buffer size
    - Output file naming and paths
"""

import platform
import shutil
import subprocess
import time, os, threading, cv2, urllib, copy, json, uuid
import ffmpeg
import numpy as np
from collections import deque
from contextlib import contextmanager
from functools import cmp_to_key
from pathlib import Path
from . import environment as env
from .ui_support import Progress
from .image_tools import ImageType, image_size, resize_image, to_opencv
from typing import Union, Generator, Optional, Callable, Any, List, Dict, Tuple


def create_video_stream(
    video_source: Union[int, str, Path, None] = None, max_yt_quality: int = 0
) -> cv2.VideoCapture:
    """Create a video stream from various sources.

    This function creates and returns video stream object working from different
    sources, including local cameras, IP cameras, video files, and YouTube videos.
    The stream is automatically closed when the context is exited.
    Use `open_video_stream()` to create a context manager for this function for
    automatic cleanup.

    Args:
        video_source (Union[int, str, Path, None], optional): Video source specification:
            - int: 0-based index for local cameras
            - str: IP camera URL (rtsp://user:password@hostname)
            - str: Local video file path
            - str: URL to mp4 video file
            - str: YouTube video URL
            - None: Use environment variable or default camera
        max_yt_quality (int, optional): Maximum video quality for YouTube videos in
            pixels (height). If 0, use best quality. Defaults to 0.

    Returns:
        cv2.VideoCapture: OpenCV video capture object.

    Raises:
        Exception: If the video stream cannot be opened.

    """

    if env.get_test_mode() or video_source is None:
        video_source = env.get_var(env.var_VideoSource, 0)
        if isinstance(video_source, str) and video_source.isnumeric():
            video_source = int(video_source)

    if isinstance(video_source, Path):
        video_source = str(video_source)

    if isinstance(video_source, str) and urllib.parse.urlparse(
        video_source
    ).hostname in (
        "www.youtube.com",
        "youtube.com",
        "youtu.be",
    ):  # if source is YouTube video
        import pafy

        if max_yt_quality == 0:
            video_source = pafy.new(video_source).getbest(preftype="mp4").url
        else:
            # Ignore DASH/HLS YouTube videos because we cannot download them trivially w/ OpenCV or ffmpeg.
            # Format ids are from pafy backend https://github.com/ytdl-org/youtube-dl/blob/master/youtube_dl/extractor/youtube.py
            dash_hls_formats = [
                91,
                92,
                93,
                94,
                95,
                96,
                132,
                151,
                133,
                134,
                135,
                136,
                137,
                138,
                160,
                212,
                264,
                298,
                299,
                266,
            ]

            video_qualities = pafy.new(video_source).videostreams
            # Sort descending based on vertical pixel count.
            video_qualities = sorted(video_qualities, key=cmp_to_key(lambda a, b: b.dimensions[1] - a.dimensions[1]))  # type: ignore[attr-defined]

            for video in video_qualities:
                if video.dimensions[1] <= max_yt_quality and video.extension == "mp4":
                    if video.itag not in dash_hls_formats:
                        video_source = video.url
                        break
            else:
                video_source = pafy.new(video_source).getbest(preftype="mp4").url

    stream = cv2.VideoCapture(video_source)  # type: ignore[arg-type]
    if not stream.isOpened():
        raise Exception(f"Error opening '{video_source}' video stream")
    return stream


@contextmanager
def open_video_stream(
    video_source: Union[int, str, Path, None] = None, max_yt_quality: int = 0
) -> Generator[cv2.VideoCapture, None, None]:
    """Open a video stream from various sources.

    This function provides a context manager for opening video streams from different
    sources, including local cameras, IP cameras, video files, and YouTube videos.
    The stream is automatically closed when the context is exited.
    Internally it calls `create_video_stream` to create the stream.

    Args:
        video_source (Union[int, str, Path, None], optional): Video source specification:
            - int: 0-based index for local cameras
            - str: IP camera URL (rtsp://user:password@hostname)
            - str: Local video file path
            - str: URL to mp4 video file
            - str: YouTube video URL
            - None: Use environment variable or default camera
        max_yt_quality (int, optional): Maximum video quality for YouTube videos in
            pixels (height). If 0, use best quality. Defaults to 0.

    Yields:
        cv2.VideoCapture: OpenCV video capture object.

    Raises:
        Exception: If the video stream cannot be opened.

    """
    stream = create_video_stream(video_source, max_yt_quality)
    try:
        yield stream
    finally:
        stream.release()


def get_video_stream_properties(
    video_source: Union[int, str, Path, None, cv2.VideoCapture],
) -> tuple:
    """Return the dimensions and frame rate of a video source.

    Args:
        video_source (Union[int, str, Path, None, cv2.VideoCapture]): Video
            source identifier or an already opened ``VideoCapture`` object.

    Returns:
        ``(width, height, fps)`` describing the video stream.
    """

    def get_props(stream: cv2.VideoCapture) -> tuple:
        return (
            int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            stream.get(cv2.CAP_PROP_FPS),
        )

    if isinstance(video_source, cv2.VideoCapture):
        return get_props(video_source)
    else:
        with open_video_stream(video_source) as stream:
            return get_props(stream)


def video_source(
    stream: cv2.VideoCapture, fps: Optional[float] = None
) -> Generator[np.ndarray, None, None]:
    """Yield frames from a video stream.

    Args:
        stream (cv2.VideoCapture): Open video stream.
        fps (Optional[float], optional): Target frame rate cap.

    Yields:
        Frames from the stream.
    """

    is_file = stream.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    # do not report errors for files and in test mode;
    # report errors only for camera streams
    report_error = False if env.get_test_mode() or is_file else True

    if fps:
        # Decimate if file
        if is_file:
            _, _, video_fps = get_video_stream_properties(stream)
            # Do not decimate if target fps > video fps
            if video_fps <= fps:
                fps = None
            else:
                drop_frames_count = int(video_fps - fps)
                drop_indices = np.linspace(
                    0, video_fps - 1, drop_frames_count, dtype=int
                )
                frame_id = -1
        # Throttle if camera feed
        else:
            minimum_elapsed_time = 1.0 / fps
            prev_time = time.time()

    while True:
        ret, frame = stream.read()
        if not ret:
            if report_error:
                raise Exception(
                    "Fail to capture camera frame. May be camera was opened by another notebook?"
                )
            else:
                break
        if fps:
            if is_file:
                frame_id += 1

                if frame_id % video_fps in drop_indices:
                    continue

                yield frame
            else:
                curr_time = time.time()
                elapsed_time = curr_time - prev_time

                if elapsed_time < minimum_elapsed_time:
                    continue

                prev_time = curr_time - (elapsed_time - minimum_elapsed_time)

                yield frame
        else:
            yield frame


class VideoWriter:
    """Video stream writer with configurable quality and format.

    This class provides functionality to save video streams to files with
    configurable dimensions, frame rate, and format. It supports both
    OpenCV and PIL image formats as input.

    Use `open_video_writer()` to create a video writer instance with proper cleanup.

    Attributes:
        filename (str): Output video file path.
        width (int): Video width in pixels.
        height (int): Video height in pixels.
        fps (float): Target frame rate.
        count (int): Number of frames written.

    """

    def __init__(self, fname: str, w: int = 0, h: int = 0, fps: float = 30.0):
        """Initialize the video writer.

        Args:
            fname (str): Output video file path.
            w (int, optional): Video width in pixels. If 0, use input frame width.
                Defaults to 0.
            h (int, optional): Video height in pixels. If 0, use input frame height.
                Defaults to 0.
            fps (float, optional): Target frame rate. Defaults to 30.0.

        Raises:
            Exception: If the video writer cannot be created.
        """
        import platform

        self._count = 0
        self._writer: Any = None
        self._use_ffmpeg = platform.system() != "Windows"
        self._fps = fps
        self._fname = fname
        self._wh: tuple = (w, h)
        if w > 0 and h > 0:
            self._writer = self._create_writer()

    def _create_writer(self) -> Any:
        if self._use_ffmpeg:
            import ffmpegcv

            # use ffmpeg-wrapped VideoWriter on other platforms;
            # reason: OpenCV VideoWriter does not support H264 on Linux
            return ffmpegcv.VideoWriter(
                self._fname, codec=None, fps=self._fps, resize=self._wh
            )
        else:
            # use OpenCV VideoWriter on Windows
            return cv2.VideoWriter(
                self._fname,
                int.from_bytes("H264".encode(), byteorder="little"),
                self._fps,
                self._wh,
            )

    def write(self, img: ImageType):
        """Write a frame to the video file.

        This method writes a single frame to the video file. The frame can be
        in either OpenCV (BGR) or PIL format.

        Args:
            img (ImageType): Frame to write. Can be:
                - OpenCV image (np.ndarray)
                - PIL Image

        Raises:
            Exception: If the frame cannot be written.
        """
        im_sz = image_size(img)
        if self._writer is None:
            self._wh = im_sz
            self._writer = self._create_writer()
        self._count += 1

        if self._wh != im_sz:
            img = resize_image(img, *self._wh)
        self._writer.write(to_opencv(img))

    def release(self):
        """Release the video writer resources.

        This method should be called when finished writing to ensure all
        resources are properly released.
        """
        if self._writer is None:
            return

        if self._use_ffmpeg:
            # workaround for bug in ffmpegcv
            self._writer.process.stdin.close()
            self._writer.process.wait()
            delattr(self._writer, "process")
        else:
            self._writer.release()

    def __enter__(self):
        """Enter the context manager.

        Returns:
            VideoWriter: The current instance.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager.

        This method ensures the video writer is properly released when
        exiting the context.
        """
        self.release()

    @property
    def count(self):
        """Get the number of frames written.

        Returns:
            (int): Number of frames written to the video file.
        """
        return self._count


def create_video_writer(
    fname: str, w: int = 0, h: int = 0, fps: float = 30.0
) -> VideoWriter:
    """Create and return a video writer.

    Args:
        fname (str): Output filename for the video file.
        w (int, optional): Frame width in pixels. ``0`` uses the width of the
            first frame. Defaults to ``0``.
        h (int, optional): Frame height in pixels. ``0`` uses the height of the
            first frame. Defaults to ``0``.
        fps (float, optional): Target frames per second. Defaults to ``30.0``.

    Returns:
        VideoWriter: Open video writer instance.
    """

    directory = Path(fname).parent
    if not directory.is_dir():
        directory.mkdir(parents=True)

    return VideoWriter(str(fname), int(w), int(h), fps)  # create stream writer


@contextmanager
def open_video_writer(
    fname: str, w: int = 0, h: int = 0, fps: float = 30.0
) -> Generator[VideoWriter, None, None]:
    """Context manager for ``VideoWriter``.

    This function creates a video writer, yields it for use inside the context,
    and releases it automatically on exit.

    Args:
        fname (str): Output filename for the video file.
        w (int, optional): Frame width in pixels. ``0`` uses the width of the
            first frame. Defaults to ``0``.
        h (int, optional): Frame height in pixels. ``0`` uses the height of the
            first frame. Defaults to ``0``.
        fps (float, optional): Target frames per second. Defaults to ``30.0``.

    Yields:
        VideoWriter: Open video writer instance ready for use.
    """

    writer = create_video_writer(fname, w, h, fps)
    try:
        yield writer
    finally:
        writer.release()


def video2jpegs(
    video_file: str,
    jpeg_path: str,
    *,
    jpeg_prefix: str = "frame_",
    preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> int:
    """Convert a video file into a sequence of JPEG images.

    Args:
        video_file (str): Path to the input video file.
        jpeg_path (str): Directory where JPEG files will be stored.
        jpeg_prefix (str, optional): Prefix for generated image filenames.
            Defaults to ``"frame_"``.
        preprocessor (Callable[[np.ndarray], np.ndarray], optional): Optional
            function applied to each frame before saving.

    Returns:
        int: Number of frames written to ``jpeg_path``.
    """

    path_to_jpeg = Path(jpeg_path)
    if not path_to_jpeg.exists():  # create directory for annotated images
        path_to_jpeg.mkdir()

    with open_video_stream(video_file) as stream:  # open video stream form file
        nframes = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = Progress(nframes)
        # decode video stream into files resized to model input size
        fi = 0
        for img in video_source(stream):
            if preprocessor is not None:
                img = preprocessor(img)
            fname = str(path_to_jpeg / f"{jpeg_prefix}{fi:05d}.jpg")
            cv2.imwrite(fname, img)
            progress.step()
            fi += 1

        return fi


def detect_rtsp_cameras(subnet_cidr, *, timeout_s=0.5, port=554, max_workers=16):
    """Scan given subnet for RTSP cameras by probing given port with OPTIONS request.
    Args:
        subnet_cidr (str): Subnet in CIDR notation (e.g., '192.168.0.0/24').
        timeout_s (float): Timeout for each connection attempt in seconds.
        port (int): Port to probe for RTSP cameras (default is 554).
        max_workers (int): Maximum number of concurrent threads for scanning (default is 16).
    Returns:
        dict: Dictionary with IP addresses as keys and properties as values.
              Properties include 'require_auth' indicating if authentication is required.
    """

    import ipaddress, socket
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def send_rtsp_options(ip, timeout_s, port):
        """Send RTSP OPTIONS request and check for valid response."""
        try:
            with socket.create_connection((ip, port), timeout=timeout_s) as sock:
                sock.settimeout(timeout_s)
                request = (
                    f"OPTIONS rtsp://{ip}/ RTSP/1.0\r\n"
                    f"CSeq: 1\r\n"
                    f"User-Agent: PythonRTSPScanner\r\n"
                    f"\r\n"
                )
                sock.sendall(request.encode("utf-8"))
                response = sock.recv(4096).decode("utf-8", errors="ignore")
                if response.startswith("RTSP/1.0"):
                    props = {}
                    props["require_auth"] = (
                        "401 Unauthorized" in response or "WWW-Authenticate" in response
                    )
                    return ip, props

        except (socket.timeout, socket.error):
            pass
        return None

    network = ipaddress.ip_network(subnet_cidr, strict=False)
    ips = [str(ip) for ip in network.hosts()]
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(send_rtsp_options, ip, timeout_s, port): ip for ip in ips
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                ip, info = result
                results[ip] = info
    return results


class ClipSaver:
    """Video clip saver with pre/post trigger buffering.

    This class provides functionality to save video clips triggered by events,
    with configurable pre-trigger and post-trigger buffers. It maintains a
    circular buffer of frames and saves clips when triggers occur.

    This class is primarily used by two other components in DeGirum Tools.
    1. ClipSavingAnalyzer wraps ClipSaver and triggers clips from event names found in EventNotifier or EventDetector results.
    2. EventNotifier can instantiate and use ClipSaver to record clips when a notification fires, optionally uploading those clips through NotificationServer.

    Attributes:
        clip_duration (int): Total length of output clips in frames.
        file_prefix (str): Base path for saved clip files.
        pre_trigger_delay (int): Frames to include before trigger.
        embed_ai_annotations (bool): Whether to include AI annotations in clips.
        save_ai_result_json (bool): Whether to save AI results as JSON.
        target_fps (float): Frame rate for saved clips.

    """

    def __init__(
        self,
        clip_duration: int,
        file_prefix: str,
        *,
        pre_trigger_delay: int = 0,
        embed_ai_annotations: bool = True,
        save_ai_result_json: bool = True,
        target_fps=30.0,
    ):
        """Initialize the clip saver.

        Args:
            clip_duration (int): Total length of output clips in frames (pre-buffer + post-buffer).
            file_prefix (str): Base path for saved clip files. Frame number and extension
                are appended automatically.
            pre_trigger_delay (int, optional): Frames to include before trigger. Defaults to 0.
            embed_ai_annotations (bool, optional): If True, use InferenceResults.image_overlay
                to include bounding boxes/labels in the clip. Defaults to True.
            save_ai_result_json (bool, optional): If True, save a JSON file with raw
                inference results alongside the video. Defaults to True.
            target_fps (float, optional): Frame rate for saved clips. Defaults to 30.0.

        Raises:
            ValueError: If clip_duration is not positive.
            ValueError: If pre_trigger_delay is negative or exceeds clip_duration.
        """

        if pre_trigger_delay >= clip_duration:
            raise ValueError("`pre_trigger_delay` should be less than `clip_duration`")
        self._clip_duration = clip_duration
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
        self._triggered_by: list = []
        self._filenames: list = []

        self._end_counter = -1
        self._start_frame = 0
        self._frame_counter = 0
        self._thread_name = "dgtools_ClipSaverThread_" + str(uuid.uuid4())

    def forward(self, result, triggers: List[str] = []) -> Tuple[List[str], bool]:
        """Process a frame and save clips if triggers occur.

        This method adds the current frame to the buffer and saves clips if any
        triggers are present. The saved clips include pre-trigger frames from
        the buffer.

        Args:
            result (Any): InferenceResults object containing the current frame and
                detection results.
            triggers (List[str], optional): List of trigger names that occurred
                in this frame. Defaults to [].

        Returns:
            List of saved clip filenames and whether any clips were saved.

        Raises:
            Exception: If the frame cannot be saved.
        """

        # add result to the clip buffer
        self._clip_buffer.append(result)
        if len(self._clip_buffer) > self._clip_duration:
            self._clip_buffer.popleft()

        triggered = True if triggers else False
        new_files = False

        if self._end_counter < 0:
            # if triggered, set down-counting timer and generate filenames
            if triggered:
                self._end_counter = self._clip_duration - self._pre_trigger_delay - 1
                self._start_frame = self._frame_counter - self._pre_trigger_delay
                self._triggered_by = triggers
                filename = f"{self._file_prefix}{'' if self._file_prefix.endswith('/') else '_'}{self._start_frame:08d}"
                self._filenames = [filename + ".mp4"]
                if self._save_ai_result_json:
                    self._filenames.append(filename + ".json")
                new_files = True
        else:
            # continue accumulating the clip: decrement the timer
            self._end_counter -= 1

        if self._end_counter == 0:
            self._save_clip()
            self._end_counter = -1
            self._triggered_by.clear()

        self._frame_counter += 1

        return (self._filenames if triggered else [], new_files)

    def _save_clip(self):
        """Save a video clip with pre-trigger frames.

        This method creates a new thread to save the clip, including frames
        from the buffer before the trigger and the current frame.
        """

        builtin_types = (int, float, str, bool, tuple, list, dict, set)

        def save(context):
            if context._clip_buffer:
                w, h = image_size(context._clip_buffer[0].image)

                tempfilename = str(
                    Path(context._filenames[0]).parent / str(uuid.uuid4())
                )
                try:
                    with open_video_writer(
                        tempfilename + ".mp4", w, h, context._target_fps
                    ) as writer:

                        json_result: Dict[str, Any] = {}
                        if context._save_ai_result_json:
                            json_result["properties"] = dict(
                                timestamp=time.ctime(),
                                start_frame=context._start_frame,
                                triggered_by=context._triggered_by,
                                duration=len(context._clip_buffer),
                                pre_trigger_delay=context._pre_trigger_delay,
                                target_fps=context._target_fps,
                            )
                            json_result["results"] = []

                        for result in context._clip_buffer:
                            if context._embed_ai_annotations:
                                writer.write(result.image_overlay)
                            else:
                                writer.write(result.image)

                            if context._save_ai_result_json:
                                json_result["results"].append(
                                    {
                                        k: v
                                        for k, v in result.__dict__.items()
                                        if (
                                            not k.startswith("_")
                                            or k == "_inference_results"
                                        )
                                        and isinstance(v, builtin_types)
                                    }
                                )

                        if context._save_ai_result_json:

                            def custom_serializer(obj):
                                if isinstance(obj, set):
                                    return list(obj)
                                if isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                if isinstance(obj, np.integer):
                                    return int(obj)
                                if isinstance(obj, np.floating):
                                    return float(obj)
                                if hasattr(obj, "to_dict"):
                                    return obj.to_dict()
                                raise TypeError(
                                    f"Object of type {obj.__class__.__name__} is not JSON serializable: implement to_dict() method"
                                )

                            with open(tempfilename + ".json", "w") as f:
                                json.dump(
                                    json_result,
                                    f,
                                    indent=2,
                                    default=custom_serializer,
                                )

                finally:
                    if os.path.exists(tempfilename + ".mp4"):
                        os.rename(tempfilename + ".mp4", context._filenames[0])
                    if context._save_ai_result_json and os.path.exists(
                        tempfilename + ".json"
                    ):
                        os.rename(tempfilename + ".json", context._filenames[1])

        # preserve a shallow copy of self to use in thread
        context = copy.copy(self)
        context._clip_buffer = deque(self._clip_buffer)
        context._triggered_by = list(self._triggered_by)
        context._filenames = list(self._filenames)

        # save the clip in a separate thread
        threading.Thread(target=save, args=(context,), name=self._thread_name).start()

    def join_all_saver_threads(self) -> int:
        """Wait for all clip saving threads to complete.

        This method blocks until all background clip saving threads have
        finished. It's useful to call this before exiting to ensure all
        clips are properly saved.

        Returns:
            Number of threads that were joined.
        """

        # save unfinished clip if any
        if self._end_counter > 0:
            for _ in range(self._end_counter):
                self._clip_buffer.popleft()
            self._save_clip()

        nthreads = 0
        for thread in threading.enumerate():
            if thread.name == self._thread_name:
                thread.join()
                nthreads += 1
        return nthreads


class MediaServer:
    """Manages the MediaMTX media server as a subprocess.

    Starts MediaMTX using a provided config file path. If no config path is given,
    it runs from the MediaMTX binary's directory.

    MediaMTX binary must be installed and available in the system path.
    Refer to https://github.com/bluenviron/mediamtx for installation instructions.
    """

    def __init__(self, *, config_path: Optional[str] = None, verbose: bool = False):
        """Initializes and starts the server.

        Args:
            config_path: Path to an existing MediaMTX YAML config file.
                         If not provided, runs with config file from binary directory.
            verbose: If True, shows media server output in the console.
        """
        self._verbose: bool = verbose
        self._process: Optional[subprocess.Popen] = None
        self._config_path: Optional[str] = config_path
        self._binary: str = (
            "mediamtx.exe" if platform.system() == "Windows" else "mediamtx"
        )

        binary_path = shutil.which(self._binary)
        if not binary_path:
            raise FileNotFoundError(
                f"Cannot find {self._binary} in PATH. MediaMTX binary must be installed and available in the system path."
            )

        # Determine working directory
        self._working_dir: str = os.path.dirname(
            os.path.abspath(config_path if config_path else binary_path)
        )
        self._start()

    def _start(self):
        """Starts the media server subprocess."""
        cmd = [self._binary]
        if self._config_path:
            cmd.append(self._config_path)

        stdout = None if self._verbose else subprocess.DEVNULL
        stderr = None if self._verbose else subprocess.DEVNULL

        self._process = subprocess.Popen(
            cmd, cwd=self._working_dir, stdout=stdout, stderr=stderr
        )

    def stop(self):
        """Stops the media server process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def __del__(self):
        """Destructor to ensure the media server is stopped."""
        self.stop()

    def __enter__(self):
        """Enables use with context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops server when context exits."""
        self.stop()


class VideoStreamer:
    """Streams video frames to an RTSP server using FFmpeg.
    This class uses FFmpeg to stream video frames to an RTSP server.
    FFmpeg must be installed and available in the system path.
    """

    def __init__(
        self,
        rtsp_url: str,
        width: int,
        height: int,
        *,
        fps: float = 30.0,
        pix_fmt="bgr24",
        gop_size: int = 50,
        verbose: bool = False,
    ):
        """Initializes the video streamer.

        Args:
            rtsp_url (str): RTSP URL to stream to (e.g., 'rtsp://user:password@hostname:port/stream').
                            Typically you use `MediaServer` class to start media server and
                            then use its RTSP URL like `rtsp://localhost:8554/mystream`
            width (int): Width of the video frames in pixels.
            height (int): Height of the video frames in pixels.
            fps (float, optional): Frames per second for the stream. Defaults to 30.
            pix_fmt (str, optional): Pixel format for the input frames. Defaults to 'bgr24'. Can be 'rgb24'.
            gop_size (int, optional): GOP size for the video stream. Defaults to 50.
            verbose (bool, optional): If True, shows FFmpeg output in the console. Defaults to False.
        """
        self._width = width
        self._height = height

        self._process = (
            ffmpeg.input(
                "pipe:0",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{width}x{height}",
                framerate=fps,
            )
            .output(
                rtsp_url,
                format="rtsp",
                pix_fmt="yuv420p",
                vcodec="libx264",
                preset="ultrafast",
                tune="zerolatency",
                rtsp_transport="tcp",
                fflags="nobuffer",
                max_delay=0,
                g=gop_size,
            )
            .global_args("-loglevel", "info" if verbose else "quiet")
            .run_async(pipe_stdin=True, quiet=not verbose)
        )

    def write(self, img: ImageType):
        """Writes a frame to the RTSP stream.
        Args:
            img (ImageType): Frame to write. Can be:
                - OpenCV image (np.ndarray)
                - PIL Image

            Pixel format must match the one specified in the constructor (default is 'bgr24').
        """

        if not self._process:
            return

        im_sz = image_size(img)

        if (self._width, self._height) != im_sz:
            img = resize_image(img, self._width, self._height)

        try:
            self._process.stdin.write(img.tobytes())
        except (BrokenPipeError, IOError):
            self.stop()

    def stop(self):
        """Stops the streamer process."""
        if self._process:
            if self._process.stdin:
                try:
                    self._process.stdin.close()
                except Exception:
                    pass
            self._process.wait()
            self._process = None

    def __del__(self):
        """Destructor to ensure the streamer is stopped."""
        self.stop()

    def __enter__(self):
        """Enables use with context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stops streamer when context exits."""
        self.stop()
