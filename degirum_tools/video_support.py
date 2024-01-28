#
# video_support.py: video stream handling classes and functions
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes and functions to handle video streams for capturing and saving
#


import cv2, urllib, numpy as np
from contextlib import contextmanager
from pathlib import Path
from . import environment as env
from .ui_support import Progress
from typing import Union, Generator, Optional, Callable


@contextmanager
def open_video_stream(
    video_source: Union[int, str, Path, None] = None
) -> Generator[cv2.VideoCapture, None, None]:
    """Open OpenCV video stream from camera with given identifier.

    video_source - 0-based index for local cameras
       or IP camera URL in the format "rtsp://<user>:<password>@<ip or hostname>",
       or local video file path,
       or URL to mp4 video file,
       or YouTube video URL

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


def video_source(stream: cv2.VideoCapture) -> Generator[np.ndarray, None, None]:
    """Generator function, which returns video frames captured from given video stream.
    Useful to pass to model batch_predict().

    stream - video stream context manager object returned by open_video_stream()

    Yields video frame captured from given video stream
    """

    # do not report errors for files and in test mode;
    # report errors only for camera streams
    report_error = (
        False
        if env.get_test_mode() or stream.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        else True
    )

    while True:
        ret, frame = stream.read()
        if not ret:
            if report_error:
                raise Exception(
                    "Fail to capture camera frame. May be camera was opened by another notebook?"
                )
            else:
                break
        yield frame


class VideoWriter:
    """
    H264 mp4 video stream writer class
    """

    def __init__(self, fname: str, w: int, h: int, fps: float = 30.0):
        """Create, open, and return video stream writer

        Args:
            fname: filename to save video
            w, h: frame width/height
            fps: frames per second
        """
        self._count = 0

        import platform

        self._use_ffmpeg = platform.system() != "Windows"
        if self._use_ffmpeg:
            import ffmpegcv

            # use ffmpeg-wrapped VideoWriter on other platforms;
            # reason: OpenCV VideoWriter does not support H264 on Linux
            self._writer = ffmpegcv.VideoWriter(fname, None, fps, (w, h))
        else:
            # use OpenCV VideoWriter on Windows
            self._writer = cv2.VideoWriter(
                fname, int.from_bytes("H264".encode(), byteorder="little"), fps, (w, h)
            )

    def write(self, img: np.ndarray):
        """
        Write image to video stream
        Args:
            img (np.ndarray): image to write
        """
        self._count += 1
        self._writer.write(img)

    def release(self):
        """
        Close video stream
        """

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


def create_video_writer(fname: str, w: int, h: int, fps: float = 30.0) -> VideoWriter:
    """Create, open, and return OpenCV video stream writer

    fname - filename to save video
    w, h - frame width/height
    fps - frames per second
    """

    directory = Path(fname).parent
    if not directory.is_dir():
        directory.mkdir(parents=True)

    return VideoWriter(str(fname), int(w), int(h), fps)  # create stream writer


@contextmanager
def open_video_writer(
    fname: str, w: int, h: int, fps: float = 30.0
) -> Generator[VideoWriter, None, None]:
    """Create, open, and yield OpenCV video stream writer; release on exit

    fname - filename to save video
    w, h - frame width/height
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
