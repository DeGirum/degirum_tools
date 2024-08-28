#
# video_support.py: video stream handling classes and functions
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements classes and functions to handle video streams for capturing and saving
#

import time
import cv2, urllib, numpy as np
from contextlib import contextmanager
from functools import cmp_to_key
from pathlib import Path
from . import environment as env
from .ui_support import Progress
from typing import Union, Generator, Optional, Callable, Any


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
    else:
        print(f"Successfully opened video stream '{video_source}'")

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
        if w > 0 and h > 0:
            self._writer = self._create_writer(w, h)

    def _create_writer(self, w: int, h: int) -> Any:
        if self._use_ffmpeg:
            import ffmpegcv

            # use ffmpeg-wrapped VideoWriter on other platforms;
            # reason: OpenCV VideoWriter does not support H264 on Linux
            return ffmpegcv.VideoWriter(
                self._fname, codec=None, fps=self._fps, resize=(w, h)
            )
        else:
            # use OpenCV VideoWriter on Windows
            return cv2.VideoWriter(
                self._fname,
                int.from_bytes("H264".encode(), byteorder="little"),
                self._fps,
                (w, h),
            )

    def write(self, img: np.ndarray):
        """
        Write image to video stream
        Args:
            img (np.ndarray): image to write
        """
        if self._writer is None:
            self._writer = self._create_writer(img.shape[1], img.shape[0])
        self._count += 1
        self._writer.write(img)

    def release(self):
        """
        Close video stream
        """

        if self._writer is None:
            return

        process_to_wait = None
        if self._use_ffmpeg:
            import psutil

            # find ffmpeg process: it is a child process of the writer shell process
            for p in psutil.process_iter([]):
                if self._writer.process.pid == p.ppid():
                    process_to_wait = p

        self._writer.release()

        # wait for ffmpeg process to finish
        if process_to_wait is not None:
            try:
                process_to_wait.wait()
            except Exception:
                pass

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
