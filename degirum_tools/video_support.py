#
# video_support.py: video stream handling classes and functions
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements classes and functions to handle video streams for capturing and saving
#

import time, os, threading, cv2, urllib, copy, json, uuid
import numpy as np
from collections import deque
from contextlib import contextmanager
from functools import cmp_to_key
from pathlib import Path
from . import environment as env
from .ui_support import Progress
from .image_tools import ImageType, image_size, resize_image, to_opencv
from typing import Union, Generator, Optional, Callable, Any, List, Dict, Tuple


@contextmanager
def open_video_stream(
    video_source: Union[int, str, Path, None] = None, max_yt_quality: int = 0
) -> Generator[cv2.VideoCapture, None, None]:
    """Open OpenCV video stream from camera with given identifier.

    video_source - 0-based index for local cameras
       or IP camera URL in the format "rtsp://<user>:<password>@<ip or hostname>",
       or local video file path,
       or URL to mp4 video file,
       or YouTube video URL
    max_yt_quality - The maximum video quality for YouTube videos. The units are
       in pixels for the height of the video. Will open a video with the highest
       resolution less than or equal to max_yt_quality. If 0, open the best quality.

    Returns context manager yielding video stream object and closing it on exit
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

    try:
        yield stream
    finally:
        stream.release()


def get_video_stream_properties(
    video_source: Union[int, str, Path, None, cv2.VideoCapture]
) -> tuple:
    """
    Get video stream properties

    Args:
        video_source - VideoCapture object or argument of open_video_stream() function

    Returns:
        tuple of (width, height, fps)
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
    """Generator function, which returns video frames captured from given video stream.
    Useful to pass to model batch_predict().

    stream - video stream context manager object returned by open_video_stream()
    fps - optional fps cap. If greater than the actual FPS, it will do nothing.
       If less than the current fps, it will decimate frames accordingly.
    Yields video frame captured from given video stream
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
    """
    H264 mp4 video stream writer class
    """

    def __init__(self, fname: str, w: int = 0, h: int = 0, fps: float = 30.0):
        """Create, open, and return video stream writer

        Args:
            fname: filename to save video
            w, h: frame width/height (optional, can be zero to deduce on first frame)
            fps: frames per second
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
        """
        Write image to video stream
        Args:
            img (np.ndarray): image to write
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
        """
        Close video stream
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
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def count(self):
        """
        Returns number of frames written to video stream
        """
        return self._count


def create_video_writer(
    fname: str, w: int = 0, h: int = 0, fps: float = 30.0
) -> VideoWriter:
    """Create, open, and return OpenCV video stream writer

    fname - filename to save video
    w, h - frame width/height (optional, can be zero to deduce on first frame)
    fps - frames per second
    """

    directory = Path(fname).parent
    if not directory.is_dir():
        directory.mkdir(parents=True)

    return VideoWriter(str(fname), int(w), int(h), fps)  # create stream writer


@contextmanager
def open_video_writer(
    fname: str, w: int = 0, h: int = 0, fps: float = 30.0
) -> Generator[VideoWriter, None, None]:
    """Create, open, and yield OpenCV video stream writer; release on exit

    fname - filename to save video
    w, h - frame width/height (optional, can be zero to deduce on first frame)
    fps - frames per second
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
    """Decode video file into a set of jpeg images

    video_file - filename of a video file
    jpeg_path - directory path to store decoded jpeg files
    jpeg_prefix - common prefix for jpeg file names
    preprocessor - optional image preprocessing function to be applied to each frame before saving into file
    Returns number of decoded frames

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


class ClipSaver:
    """
    Class to to save video clips triggered at particular frame
    with a specified duration before and after the event.
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
        """
        Constructor

        Args:
            clip_duration: duration of the video clip to save (in frames)
            file_prefix: path and file prefix for video clip files
            pre_trigger_delay: delay before the event to start clip saving (in frames)
            embed_ai_annotations: True to embed AI inference annotations into video clip, False to use original image
            save_ai_result_json: True to save AI result JSON file along with video clip
            target_fps: target frames per second for saved videos
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
        """
        Buffer given result in internal circular buffer.
        Initiate saving video clip if triggers list if not empty.

        Args:
            result: PySDK model result object
            triggers: list of event names or notifications which trigger video clip saving

        Returns:
            tuple of 2 elements:
                - all filenames related to the video clip if event is triggered in this frame, empty list otherwise
                - True if new files are created, False if files are reused from the previous event
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
        """
        Save video clip from the buffer
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
                                        if not k.startswith("_")
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
        """
        Join all threads started by this instance
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
