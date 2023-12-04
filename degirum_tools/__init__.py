#
# degirum_tools.py: toolkit for PySDK samples
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#


import cv2, sys, os, time, urllib, av, PIL.Image, dotenv, importlib, queue
import numpy as np, supervision as sv
from dataclasses import dataclass
from contextlib import contextmanager, ExitStack
from pathlib import Path
from .compound_models import *  # noqa

# Inference options: parameters for connect_model_zoo
CloudInference = 1  # use DeGirum cloud server for inference
AIServerInference = 2  # use AI server deployed in LAN/VPN
LocalHWInference = 3  # use locally-installed AI HW accelerator

# environment variable names
_var_TestMode = "TEST_MODE"
_var_Token = "DEGIRUM_CLOUD_TOKEN"
_var_CloudUrl = "DEGIRUM_CLOUD_PLATFORM_URL"
_var_AiServer = "AISERVER_HOSTNAME_OR_IP"
_var_CloudZoo = "CLOUD_ZOO_URL"
_var_CameraID = "CAMERA_ID"
_var_AudioID = "AUDIO_ID"


def _reload_env(custom_file="env.ini"):
    """Reload environment variables from file
    custom_file - name of the custom env file to try first;
        CWD, and ../CWD are searched for the file;
        if it is None or does not exist, `.env` file is loaded
    """

    custom_file = dotenv.find_dotenv(custom_file, usecwd=True)
    if not custom_file:
        custom_file = None

    dotenv.load_dotenv(
        dotenv_path=custom_file, override=True
    )  # load environment variables from file


def _get_var(var, default_val=None):
    """Returns environment variable value"""
    if var is not None and var.isupper():  # treat `var` as env. var. name
        ret = os.getenv(var)
        if ret is None:
            if default_val is None:
                raise Exception(
                    f"Please define environment variable {var} in `.env` or `env.ini` file located in your CWD"
                )
            else:
                ret = default_val
    else:  # treat `var` literally
        ret = var
    return ret


def _get_test_mode():
    """Returns enable status of test mode from .env file"""
    _reload_env()  # reload environment variables from file
    return _get_var(_var_TestMode, False)


def get_token():
    """Returns a token from .env file"""
    _reload_env()  # reload environment variables from file
    return _get_var(_var_Token)


def get_ai_server_hostname():
    """Returns a AI server hostname/IP from .env file"""
    _reload_env()  # reload environment variables from file
    return _get_var(_var_AiServer)


def get_cloud_zoo_url():
    """Returns a cloud zoo URL from .env file"""
    _reload_env()  # reload environment variables from file

    cloud_url = "https://" + _get_var(_var_CloudUrl, "cs.degirum.com")
    zoo_url = _get_var(_var_CloudZoo, "")
    if zoo_url:
        cloud_url += "/" + zoo_url
    return cloud_url


def connect_model_zoo(inference_option=CloudInference):
    """Connect to model zoo according to given inference option.

    inference_option: should be one of CloudInference, AIServerInference, or LocalHWInference

    Returns model zoo accessor object
    """
    import degirum as dg  # import DeGirum PySDK

    _reload_env()  # reload environment variables from file

    if inference_option == CloudInference:
        # inference on cloud platform
        zoo = dg.connect(dg.CLOUD, get_cloud_zoo_url(), _get_var(_var_Token))

    elif inference_option == AIServerInference:
        # inference on AI server
        hostname = _get_var(_var_AiServer)
        if _get_var(_var_CloudZoo, ""):
            # use cloud zoo
            zoo = dg.connect(hostname, get_cloud_zoo_url(), _get_var(_var_Token))
        else:
            # use local zoo
            zoo = dg.connect(hostname)

    elif inference_option == LocalHWInference:
        zoo = dg.connect(dg.LOCAL, get_cloud_zoo_url(), _get_var(_var_Token))

    else:
        raise Exception(
            "Invalid value of inference_option parameter. Should be one of CloudInference, AIServerInference, or LocalHWInference"
        )

    return zoo


def _in_notebook():
    """Returns `True` if the module is running in IPython kernel,
    `False` if in IPython shell or other Python shell.
    """
    return "ipykernel" in sys.modules


def _in_colab():
    """Returns `True` if the module is running in Google Colab environment"""
    return "google.colab" in sys.modules


def import_optional_package(pkg_name, is_long=False):
    """Import package with given name.
    Returns the package object.
    Raises error message if the package is not installed"""

    if is_long:
        print(f"Loading '{pkg_name}' package, be patient...")
    try:
        ret = importlib.import_module(pkg_name)
        if is_long:
            print(f"...done; '{pkg_name}' version: {ret.__version__}")
        return ret
    except ModuleNotFoundError as e:
        print(f"\n*** Error loading '{pkg_name}' package: {e}. Not installed?\n")
        return None


def configure_colab(*, video_file=None, audio_file=None):
    """
    Configure Google Colab environment

    Args:
        video_file - path to video file to use instead of camera
        audio_file - path to wave file to use instead of microphone
    """

    # check if running under Colab
    if not _in_colab():
        return

    import subprocess

    # define directories
    repo = "PySDKExamples"
    colab_root_dir = "/content"
    repo_dir = f"{colab_root_dir}/{repo}"
    work_dir = f"{repo_dir}/examples/workarea"

    if not os.path.exists(repo_dir):
        # request API token in advance
        def token_request():
            return input("\n\nEnter cloud API access token from cs.degirum.com:\n")

        token = token_request()

        def run_cmd(prompt, cmd):
            print(prompt + "... ", end="")
            result = subprocess.run(
                [cmd],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode != 0:
                print(result.stdout)
                raise Exception(f"{prompt} FAILS")
            print("DONE!")

        # clone PySDKExamples repo if not done yet
        os.chdir(colab_root_dir)
        run_cmd(
            "Cloning DeGirum/PySDKExamples repo",
            f"git clone https://github.com/DeGirum/{repo}",
        )

        # make repo root dir as CWD
        os.chdir(repo_dir)

        # install PySDKExamples requirements
        req_file = "requirements.txt"
        run_cmd(
            "Installing requirements (this will take a while)",
            f"pip install -r {req_file}",
        )

        # validate token
        print("Validating token...", end="")
        import degirum as dg

        while True:
            try:
                dg.connect(dg.CLOUD, "https://cs.degirum.com", token)
                break
            except Exception:
                print("\nProvided token is not valid!\n")
                token = token_request()
        print("DONE!")

        # configure env.ini
        env_file = "env.ini"
        print(f"Configuring {env_file} file...", end="")
        with open(env_file, "a") as file:
            file.write(f'DEGIRUM_CLOUD_TOKEN = "{token}"\n')
            file.write(
                f'CAMERA_ID = {video_file if video_file is not None else "../../images/colab_example.mp4"}\n'
            )
            file.write(
                f'AUDIO_ID = {audio_file if audio_file is not None else "../../images/colab_example.wav"}\n'
            )
        print("DONE!")

    # make working dir as CWD
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)


@contextmanager
def open_video_stream(camera_id=None):
    """Open OpenCV video stream from camera with given identifier.

    camera_id - 0-based index for local cameras
       or IP camera URL in the format "rtsp://<user>:<password>@<ip or hostname>",
       or local video file path,
       or URL to mp4 video file,
       or YouTube video URL

    Returns context manager yielding video stream object and closing it on exit
    """

    if camera_id is None or _get_test_mode():
        _reload_env()  # reload environment variables from file
        camera_id = _get_var(_var_CameraID, 0)
        if isinstance(camera_id, str) and camera_id.isnumeric():
            camera_id = int(camera_id)

    if isinstance(camera_id, Path):
        camera_id = str(camera_id)

    if isinstance(camera_id, str) and urllib.parse.urlparse(camera_id).hostname in (
        "www.youtube.com",
        "youtube.com",
        "youtu.be",
    ):  # if source is YouTube video
        import pafy

        camera_id = pafy.new(camera_id).getbest(preftype="mp4").url

    stream = cv2.VideoCapture(camera_id)
    if not stream.isOpened():
        raise Exception(f"Error opening '{camera_id}' video stream")
    else:
        print(f"Successfully opened video stream '{camera_id}'")

    try:
        yield stream
    finally:
        stream.release()


def video_source(stream):
    """Generator function, which returns video frames captured from given video stream.
    Useful to pass to model batch_predict().

    stream - video stream context manager object returned by open_video_stream()

    Yields video frame captured from given video stream
    """

    # do not report errors for files and in test mode;
    # report errors only for camera streams
    report_error = (
        False if _get_test_mode() or stream.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else True
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

    def __init__(self, fname, w, h, fps=30):
        """Create, open, and return video stream writer

        Args:
            fname: filename to save video
            w, h: frame width/height
            fps: frames per second
        """

        self._container = av.open(fname, "w")
        self._stream = self._container.add_stream("h264", fps)
        self._stream.width = w
        self._stream.height = h
        self._count = 0

    def write(self, img):
        """
        Write image to video stream
        Args:
            img (np.ndarray): image to write
        """
        self._count += 1
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def release(self):
        """
        Close video stream
        """
        if self._container is not None:
            # flush stream
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
            self._container = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def count(self):
        """
        Returns number of frames written to video stream
        """
        return self._count


def create_video_writer(fname, w, h, fps=30):
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
def open_video_writer(fname, w, h, *, fps=30):
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
    video_file, jpeg_path, *, jpeg_prefix="frame_", preprocessor=None
) -> int:
    """Decode video file into a set of jpeg images

    video_file - filename of a video file
    jpeg_path - directory path to store decoded jpeg files
    jpeg_prefix - common prefix for jpeg file names
    preprocessor - optional image preprocessing function to be applied to each frame before saving into file
    Returns number of decoded frames

    """

    jpeg_path = Path(jpeg_path)
    if not jpeg_path.exists():  # create directory for annotated images
        jpeg_path.mkdir()

    with open_video_stream(video_file) as stream:  # open video stream form file
        nframes = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = Progress(nframes)
        # decode video stream into files resized to model input size
        fi = 0
        for img in video_source(stream):
            if preprocessor is not None:
                img = preprocessor(img)
            fname = str(jpeg_path / f"{jpeg_prefix}{fi:05d}.jpg")
            cv2.imwrite(fname, img)
            progress.step()
            fi += 1

        return fi


@contextmanager
def open_audio_stream(sampling_rate_hz, buffer_size, audio_id=None):
    """Open PyAudio audio stream

    Args:
        sampling_rate_hz - desired sample rate in Hz
        buffer_size - read buffer size
        audio_id - 0-based index for local microphones or local WAV file path
    Returns context manager yielding audio stream object and closing it on exit
    """

    pyaudio = import_optional_package("pyaudio")

    if audio_id is None or _get_test_mode():
        _reload_env()  # reload environment variables from file
        audio_id = _get_var(_var_AudioID, 0)
        if isinstance(audio_id, str) and audio_id.isnumeric():
            audio_id = int(audio_id)

    if isinstance(audio_id, int):
        # microphone

        class MicStream:
            def __init__(self, mic_id, sampling_rate_hz, buffer_size):
                self._audio = pyaudio.PyAudio()
                self._result_queue = queue.Queue()  # type: queue.Queue

                def callback(
                    in_data,  # recorded data if input=True; else None
                    frame_count,  # number of frames
                    time_info,  # dictionary
                    status_flags,
                ):  # PaCallbackFlags
                    self._result_queue.put(in_data)
                    return (None, pyaudio.paContinue)

                self.frames_per_buffer = int(buffer_size)
                self._stream = self._audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=int(sampling_rate_hz),
                    input=True,
                    input_device_index=audio_id,
                    frames_per_buffer=self.frames_per_buffer,
                    stream_callback=callback,
                )

            def __enter__(self):
                pass

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._stream.stop_stream()  # stop audio streaming
                self._stream.close()  # close audio stream
                self._audio.terminate()  # terminate audio library

            def get(self, no_wait=False):
                if no_wait:
                    return self._result_queue.get_nowait()
                else:
                    return self._result_queue.get()

        yield MicStream(audio_id, sampling_rate_hz, buffer_size)

    else:
        # file
        import wave

        class WavStream:
            def __init__(self, filename, sampling_rate_hz, buffer_size):
                self._wav = wave.open(filename, "rb")

                if self._wav.getnchannels() != 1:
                    raise Exception(f"{filename} should be mono WAV file")

                if self._wav.getsampwidth() != 2:
                    raise Exception(f"{filename} should have 16-bit samples")

                if self._wav.getframerate() != sampling_rate_hz:
                    raise Exception(
                        f"{filename} should have {sampling_rate_hz} Hz sampling rate"
                    )

                self.frames_per_buffer = buffer_size

            def __enter__(self):
                pass

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._wav.close()

            def get(self, no_wait=False):
                buf = self._wav.readframes(self.frames_per_buffer)
                if len(buf) < self.frames_per_buffer:
                    raise StopIteration
                return buf

        yield WavStream(audio_id, sampling_rate_hz, buffer_size)


def audio_source(stream, check_abort, non_blocking=False):
    """Generator function, which returns audio frames captured from given audio stream.
    Useful to pass to model batch_predict().

    stream - audio stream context manager object returned by open_audio_stream()
    check_abort - check-for-abort function or lambda; stream will be terminated when it returns True
    non_blocking - True for non-blocking mode (immediately yields None if a block is not captured yet)
        False for blocking mode (waits for the end of the block capture and always yields captured block)

    Yields audio waveform captured from given audio stream
    """

    try:
        while not check_abort():
            if non_blocking:
                try:
                    block = stream.get(True)
                except queue.Empty:
                    block = None
            else:
                block = stream.get()

            yield None if block is None else np.frombuffer(block, dtype=np.int16)
    except StopIteration:
        pass


def audio_overlapped_source(stream, check_abort, non_blocking=False):
    """Generator function, which returns audio frames captured from given audio stream with half-length overlap.
    Useful to pass to model batch_predict().

    stream - audio stream context manager object returned by open_audio_stream()
    check_abort - check-for-abort function or lambda; stream will be terminated when it returns True
    non_blocking - True for non-blocking mode (immediately yields None if a block is not captured yet)
        False for blocking mode (waits for the end of the block capture and always yields captured block)

    Yields audio waveform captured from given audio stream with half-length overlap.
    """

    chunk_length = stream.frames_per_buffer
    data = np.zeros(2 * chunk_length, dtype=np.int16)
    try:
        while not check_abort():
            if non_blocking:
                try:
                    block = stream.get(True)
                except queue.Empty:
                    block = None
            else:
                block = stream.get()

            if block is None:
                yield None
            else:
                data[:chunk_length] = data[chunk_length:]
                data[chunk_length:] = np.frombuffer(block, dtype=np.int16)
                yield data
    except StopIteration:
        pass


class FPSMeter:
    """Simple FPS meter class"""

    def __init__(self, avg_len=100):
        """Constructor

        avg_len - number of samples to average
        """
        self._avg_len = avg_len
        self.reset()

    def reset(self):
        """Reset accumulators"""
        self._timestamp_ns = -1
        self._duration_ns = -1
        self._count = 0

    def record(self):
        """Record timestamp and update average duration.

        Returns current average FPS"""
        t = time.time_ns()
        if self._timestamp_ns > 0:
            cur_dur_ns = t - self._timestamp_ns
            self._count = min(self._count + 1, self._avg_len)
            self._duration_ns = (
                self._duration_ns * (self._count - 1) + cur_dur_ns
            ) // self._count
        self._timestamp_ns = t
        return self.fps()

    def fps(self):
        """Return current average FPS"""
        return 1e9 / self._duration_ns if self._duration_ns > 0 else 0


class Display:
    """Class to handle OpenCV image display"""

    def __init__(self, capt="<image>", show_fps=True, w=None, h=None):
        """Constructor

        capt - window title
        show_fps - True to show FPS
        show_embedded - True to show graph embedded into the notebook when possible
        w, h - initial window width/hight in pixels; None for autoscale
        """
        self._fps = FPSMeter()

        if not capt:
            raise Exception("Window title must be non-empty")

        self._capt = capt
        self._show_fps = show_fps
        self._window_created = False
        self._no_gui = not Display._check_gui() or _get_test_mode()
        self._w = w
        self._h = h
        self._video_writer = None
        self._video_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close OpenCV window in any
        if self._window_created:
            cv2.destroyWindow(self._capt)

        # close video writer if any, and show video in Colab
        if self._video_writer is not None:
            self._video_writer.release()
            if _in_colab():
                import IPython

                IPython.display.display(
                    IPython.display.Video(self._video_file, embed=True)
                )

        return exc_type is KeyboardInterrupt  # ignore KeyboardInterrupt errors

    @property
    def window_name(self):
        """
        Returns window name
        """
        return self._capt

    @staticmethod
    def crop(img, bbox):
        """Crop and return OpenCV image to given bbox"""
        return img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

    @staticmethod
    def put_text(
        img,
        text,
        position,
        text_color,
        back_color=None,
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        font_scale=1,
    ):
        """Draw given text on given OpenCV image at given point with given color

        Args:
            img - numpy array with image
            text - text to draw
            position - text top left coordinate tuple (x,y)
            text_color - text color (BGR)
            back_color - background color (BGR) or None for transparent
            font - font to use
            font_scale - font scale factor to use
        """

        text_size = cv2.getTextSize(text, font, 1, 1)
        text_w = text_size[0][0]
        text_h = text_size[0][1] + text_size[1]
        margin = int(text_h / 4)
        bl_corner = (position[0], position[1] + text_h + 2 * margin)
        if back_color is not None:
            tr_corner = (
                bl_corner[0] + text_w + 2 * margin,
                bl_corner[1] - text_h - 2 * margin,
            )
            cv2.rectangle(img, bl_corner, tr_corner, back_color, cv2.FILLED)
        cv2.putText(
            img,
            text,
            (bl_corner[0] + margin, bl_corner[1] - margin),
            font,
            font_scale,
            text_color,
        )

    @staticmethod
    def _check_gui():
        """Check if graphical display is supported

        Returns False if not supported
        """
        import os, platform

        if platform.system() == "Linux":
            return os.environ.get("DISPLAY") is not None
        return True

    @staticmethod
    def _display_fps(img, fps):
        """Helper method to display FPS"""
        Display.put_text(img, f"{fps:5.1f} FPS", (0, 0), (0, 0, 0), (255, 255, 255))

    def show(self, img, waitkey_delay=1):
        """Show OpenCV image or model result

        img - numpy array with valid OpenCV image, or PIL image, or model result object
        waitkey_delay - delay in ms for waitKey() call; use 0 to show still images, use 1 for streaming video
        """

        import IPython.display

        # show image in notebook
        def show_in_notebook(img):
            IPython.display.display(PIL.Image.fromarray(img[..., ::-1]), clear=True)

        if hasattr(img, "image_overlay"):
            # special case for model results: call it recursively
            self.show(img.image_overlay, waitkey_delay)
            return

        if isinstance(img, PIL.Image.Image):
            # PIL image: convert to OpenCV format
            img = np.array(img)[:, :, ::-1]

        if isinstance(img, np.ndarray):
            fps = self._fps.record()
            if self._show_fps and fps > 0:
                Display._display_fps(img, fps)

            if _in_colab():
                # special case for Colab environment
                if waitkey_delay == 0:
                    # show still image in notebook
                    show_in_notebook(img)
                else:
                    # save videos to file
                    if self._video_writer is None:
                        self._video_file = f"{os.getcwd()}/{self._capt}.mp4"
                        self._video_writer = create_video_writer(
                            self._video_file, img.shape[1], img.shape[0]
                        )
                    self._video_writer.write(img)

                    class printer(str):
                        def __repr__(self):
                            return self

                    if self._video_writer.count % 10 == 0:
                        IPython.display.display(
                            printer(
                                f"{self._video_file}: frame {self._video_writer.count}, {fps:.1f} FPS"
                            ),
                            clear=True,
                        )

            elif self._no_gui and _in_notebook():
                # show image in notebook when possible
                show_in_notebook(img)
            else:
                # show image in OpenCV window
                if not self._window_created:
                    cv2.namedWindow(self._capt, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(self._capt, cv2.WND_PROP_TOPMOST, 1)
                    if self._w is not None and self._h is not None:
                        cv2.resizeWindow(self._capt, self._w, self._h)
                    else:
                        cv2.resizeWindow(self._capt, img.shape[1], img.shape[0])

                cv2.imshow(self._capt, img)
                self._window_created = True
                key = cv2.waitKey(waitkey_delay) & 0xFF
                if key == ord("x") or key == ord("q"):
                    if self._fps:
                        self._fps.reset()
                    raise KeyboardInterrupt
                elif key == 43 or key == 45:  # +/-
                    _, _, w, h = cv2.getWindowImageRect(self._capt)
                    factor = 1.25 if key == 43 else 0.75
                    new_w = max(100, int(w * factor))
                    new_h = int(new_w * img.shape[0] / img.shape[1])
                    cv2.resizeWindow(self._capt, new_w, new_h)

        else:
            raise Exception("Unsupported image type")

    def show_image(self, img):
        """Show OpenCV image or model result

        img - numpy array with valid OpenCV image, or PIL image, or model result object
        """
        self.show(img, 0)


class Timer:
    """Simple timer class"""

    def __init__(self):
        """Constructor. Records start time."""
        self._start_time = time.time_ns()

    def __call__(self):
        """Call method.

        Returns time elapsed (in seconds, since object construction)."""
        return (time.time_ns() - self._start_time) * 1e-9


class Progress:
    """Simple progress indicator"""

    def __init__(self, last_step=None, *, start_step=0, bar_len=15, speed_units="FPS"):
        """Constructor
        last_step - last step
        start_step - starting step
        bar_len - progress bar length in symbols
        """
        self._display_id = None
        self._len = bar_len
        self._last_step = last_step
        self._start_step = start_step
        self._time_to_refresh = lambda: time.time() - self._last_update_time > 0.5
        self._speed_units = speed_units
        self.reset()

    def reset(self):
        self._start_time = time.time()
        self._step = self._start_step
        self._percent = 0.0
        self._last_updated_percent = self._percent
        self._last_update_time = 0.0
        self._tip_phase = 0
        self._update()

    def step(self, steps=1):
        """Update progress by given number of steps
        steps - number of steps to advance
        """
        assert (
            self._last_step is not None
        ), "Progress indicator: to do stepping last step must be assigned on construction"
        self._step += steps
        self._percent = (
            100 * (self._step - self._start_step) / (self._last_step - self._start_step)
        )
        if (
            self._percent - self._last_updated_percent >= 100 / self._len
            or self._percent >= 100
            or self._time_to_refresh()
        ):
            self._update()

    @property
    def step_range(self):
        """Get start-end step range (if defined)"""
        if self._last_step is not None:
            return (self._start_step, self._last_step)
        else:
            return None

    @property
    def percent(self):
        return self._percent

    @percent.setter
    def percent(self, value):
        v = float(value)
        delta = abs(self._last_updated_percent - v)
        self._percent = v
        if self._last_step is not None:
            self._step = round(
                0.01 * self._percent * (self._last_step - self._start_step)
                + self._start_step
            )
        if delta >= 100 / self._len or self._time_to_refresh():
            self._update()

    def _update(self):
        """Update progress bar"""
        self._last_updated_percent = self._percent
        bars = int(self._percent / 100 * self._len)
        elapsed_s = time.time() - self._start_time

        tips = "−\\/"
        tip = tips[self._tip_phase] if bars < self._len else ""
        self._tip_phase = (self._tip_phase + 1) % len(tips)

        prog_str = f"{round(self._percent):4d}% |{'█' * bars}{tip}{'-' * (self._len - bars - 1)}|"
        if self._last_step is not None:
            prog_str += f" {self._step}/{self._last_step}"

        prog_str += f" [{elapsed_s:.1f}s elapsed"
        if self._percent > 0 and self._percent <= 100:
            remaining_est_s = elapsed_s * (100 - self._percent) / self._percent
            prog_str += f", {remaining_est_s:.1f}s remaining"
        if self._last_step is not None and elapsed_s > 0:
            prog_str += f", {(self._step - self._start_step) / elapsed_s:.1f} {self._speed_units}]"
        else:
            prog_str += "]"

        class printer(str):
            def __repr__(self):
                return self

        prog_str = printer(prog_str)

        if _in_notebook():
            import IPython.display

            if self._display_id is None:
                self._display_id = "dg_progress_" + str(time.time_ns())
                IPython.display.display(prog_str, display_id=self._display_id)
            else:
                IPython.display.update_display(prog_str, display_id=self._display_id)
        else:
            print(prog_str, end="\r")
        self._last_update_time = time.time()


def area(box):
    """
    Computes bbox(es) area: is vectorized.

    Parameters
    ----------
    box : np.array
        Box(es) in format (x0, y0, x1, y1)

    Returns
    -------
    np.array
        area(s)
    """
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])


def intersection(boxA, boxB):
    """
    Compute area of intersection of two boxes

    Parameters
    ----------
    boxA : np.array
        First boxes
    boxB : np.array
        Second box

    Returns
    -------
    float64
        Area of intersection
    """
    xA = max(boxA[..., 0], boxB[..., 0])
    xB = min(boxA[..., 2], boxB[..., 2])
    dx = xB - xA
    if dx <= 0:
        return 0.0

    yA = max(boxA[..., 1], boxB[..., 1])
    yB = min(boxA[..., 3], boxB[..., 3])
    dy = yB - yA
    if dy <= 0.0:
        return 0.0

    # compute the area of intersection rectangle
    return dx * dy


class ZoneCounter:
    """
    Class to count detected object bounding boxes in polygon zones
    """

    # Triggering position within the bounding box
    CENTER = sv.Position.CENTER
    CENTER_LEFT = sv.Position.CENTER_LEFT
    CENTER_RIGHT = sv.Position.CENTER_RIGHT
    TOP_CENTER = sv.Position.TOP_CENTER
    TOP_LEFT = sv.Position.TOP_LEFT
    TOP_RIGHT = sv.Position.TOP_RIGHT
    BOTTOM_LEFT = sv.Position.BOTTOM_LEFT
    BOTTOM_CENTER = sv.Position.BOTTOM_CENTER
    BOTTOM_RIGHT = sv.Position.BOTTOM_RIGHT

    def __init__(
        self,
        count_polygons,
        *,
        class_list=None,
        triggering_position=BOTTOM_CENTER,
        window_name=None,
    ):
        """Constructor

        Args:
            count_polygons - list of polygons to count objects in; each polygon is a list of points (x,y)
            class_list - list of classes to count; if None, all classes are counted
            triggering_position: the position within the bounding box that triggers the zone
            window_name - optional OpenCV window name to configure for interactive zone adjustment
        """

        self._wh = None
        self._zones = None
        self._win_name = window_name
        self._mouse_callback_installed = False
        self._class_list = class_list
        self._triggering_position = triggering_position
        self._polygons = [
            np.array(polygon, dtype=np.int32) for polygon in count_polygons
        ]

    def _lazy_init(self, result):
        """
        Complete deferred initialization steps
            - initialize polygon zones from model result object
            - install mouse callback
        """
        if self._zones is None:
            self._wh = (result.image.shape[1], result.image.shape[0])
            self._zones = [
                sv.PolygonZone(polygon, self._wh, self._triggering_position)
                for polygon in self._polygons
            ]
        if not self._mouse_callback_installed and self._win_name is not None:
            self._install_mouse_callback()

    def window_attach(self, win_name):
        """Attach OpenCV window for interactive zone adjustment by installing mouse callback
        Args:
            win_name - OpenCV window name to attach to
        """

        self._win_name = win_name
        self._mouse_callback_installed = False

    def count(self, result):
        """
        Count detected object bounding boxes in polygon zones

        Args:
            result - model result object
        Returns:
            list of object counts found in each polygon zone
        """

        self._lazy_init(result)

        def in_class_list(obj):
            return (
                True
                if self._class_list is None
                else obj["label"] in self._class_list
                if "label" in obj
                else False
            )

        bboxes = np.array(
            [
                obj["bbox"]
                for obj in result.results
                if "bbox" in obj and in_class_list(obj)
            ]
        )
        if self._zones is not None:
            return [
                (zone.trigger(sv.Detections(bboxes)).sum() if len(bboxes) > 0 else 0)
                for zone in self._zones
            ]
        return None

    def display(self, result, image, zone_counts):
        """
        Display polygon zones and counts on given image

        Args:
            result - result object to take display settings from
            image - image to display on
            zone_counts - list of object counts found in each polygon zone
        Returns:
            annotated image
        """

        def color_complement(color):
            adj_color = (color[0] if isinstance(color, list) else color)[::-1]
            return tuple([255 - c for c in adj_color])

        zone_color = color_complement(result.overlay_color)
        background_color = color_complement(result.overlay_fill_color)

        for zi in range(len(self._polygons)):
            cv2.polylines(
                image, [self._polygons[zi]], True, zone_color, result.overlay_line_width
            )
            Display.put_text(
                image,
                f"Zone {zi}: {zone_counts[zi]}",
                self._polygons[zi][0],
                zone_color,
                background_color,
                cv2.FONT_HERSHEY_PLAIN,
                result.overlay_font_scale,
            )
        return image

    def count_and_display(self, result):
        """
        Count detected object bounding boxes in polygon zones and display them on model result image

        Args:
            result - model result object
        Returns:
            annotated image
        """
        return self.display(result, result.image_overlay, self.count(result))

    def _mouse_callback(event, x, y, flags, self):
        """Mouse callback for OpenCV window for interactive zone operations"""

        click_point = np.array((x, y))

        def zone_update():
            idx = self._gui_state["update"]
            if idx >= 0 and self._wh is not None:
                self._zones[idx] = sv.PolygonZone(
                    self._polygons[idx], self._wh, self._triggering_position
                )

        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, polygon in enumerate(self._polygons):
                if cv2.pointPolygonTest(polygon, (x, y), False) > 0:
                    zone_update()
                    self._gui_state["dragging"] = polygon
                    self._gui_state["offset"] = click_point
                    self._gui_state["update"] = idx
                    break

        if event == cv2.EVENT_RBUTTONDOWN:
            for idx, polygon in enumerate(self._polygons):
                for pt in polygon:
                    if np.linalg.norm(pt - click_point) < 10:
                        zone_update()
                        self._gui_state["dragging"] = pt
                        self._gui_state["offset"] = click_point
                        self._gui_state["update"] = idx
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._gui_state["dragging"] is not None:
                delta = click_point - self._gui_state["offset"]
                self._gui_state["dragging"] += delta
                self._gui_state["offset"] = click_point

        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self._gui_state["dragging"] = None
            zone_update()
            self._gui_state["update"] = -1

    def _install_mouse_callback(self):
        try:
            cv2.setMouseCallback(self._win_name, ZoneCounter._mouse_callback, self)  # type: ignore[attr-defined]
            self._gui_state = {"dragging": None, "update": -1}
            self._mouse_callback_installed = True
        except Exception:
            pass  # ignore errors


def predict_stream(model, input_video_id, *, zone_counter=None):
    """Run a model on a video stream

    Args:
        model - model to run
        input_video_id - identifier of input video stream. It can be:
            - 0-based index for local cameras
            - IP camera URL in the format "rtsp://<user>:<password>@<ip or hostname>",
            - local path or URL to mp4 video file,
            - YouTube video URL
        zone_counter - optional ZoneCount object; when not None, object counting is performed

    Returns:
        generator object yielding model prediction results;
        when zone_counter is not None, each prediction result contains additional attribute
        `zone_counts`, which is a list of object counts for each polygon zone configured in zone_counter
    """

    # select OpenCV backend and matching colorspace
    model.image_backend = "opencv"
    model.input_numpy_colorspace = "BGR"

    do_zone_count = zone_counter is not None

    with open_video_stream(input_video_id) as stream:
        for res in model.predict_batch(video_source(stream)):
            if do_zone_count:

                class ZoneCountResult:
                    def __init__(self, res, zc):
                        self._result = res
                        self.zone_counter = zc
                        self.zone_counts = zc.count(res)

                    def __getattr__(self, item):
                        return getattr(self._result, item)

                    @property
                    def image_overlay(self):
                        return self.zone_counter.display(
                            self._result, self._result.image_overlay, self.zone_counts
                        )

                yield ZoneCountResult(res, zone_counter)

            else:
                yield res


def annotate_video(
    model,
    input_video_id,
    output_video_path,
    *,
    show_progress=True,
    visual_display=True,
    zone_counter=None,
):
    """Annotate video stream by running a model and saving results to video file

    Args:
        model - model to run
        input_video_id - identifier of input video stream. It can be:
        - 0-based index for local cameras
        - IP camera URL in the format "rtsp://<user>:<password>@<ip or hostname>",
        - local path or URL to mp4 video file,
        - YouTube video URL
        show_progress - when True, show text progress indicator
        visual_display - when True, show interactive video display with annotated video stream
        zone_counter - optional ZoneCount object; when not None, object counting is performed
    """

    model.image_backend = "opencv"
    model.input_numpy_colorspace = "BGR"

    win_name = f"Annotating {input_video_id}"

    do_zone_count = zone_counter is not None
    if do_zone_count:
        zone_counter.window_attach(win_name)

    with ExitStack() as stack:
        if visual_display:
            display = stack.enter_context(Display(win_name))

        stream = stack.enter_context(open_video_stream(input_video_id))
        w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = stack.enter_context(open_video_writer(str(output_video_path), w, h))

        if show_progress:
            progress = Progress(int(stream.get(cv2.CAP_PROP_FRAME_COUNT)))

        for res in model.predict_batch(video_source(stream)):
            img = res.image_overlay

            if do_zone_count:
                zone_counter.display(res, img, zone_counter.count(res))

            writer.write(img)

            if visual_display:
                display.show(img)

            if show_progress:
                progress.step()


@dataclass
class ModelTimeProfile:
    """Class to hold model time profiling results"""

    elapsed: float  # elapsed time in seconds
    iterations: int  # number of iterations made
    observed_fps: float  # observed inference performance, frames per second
    max_possible_fps: float  # maximum possible inference performance, frames per second
    parameters: dict  # copy of model parameters
    time_stats: dict  # model time statistics dictionary


def model_time_profile(model, iterations=100) -> ModelTimeProfile:
    """
    Perform time profiling of a given model

    Args:
        model: PySDK model to profile
        iterations: number of iterations to run

    Returns:
        ModelTimeProfile object
    """

    # skip non-image type models
    if model.model_info.InputType[0] != "Image":
        raise NotImplementedError

    saved_params = {
        "input_image_format": model.input_image_format,
        "measure_time": model.measure_time,
        "image_backend": model.image_backend,
    }

    elapsed = 0
    try:
        # configure model
        model.input_image_format = "JPEG"
        model.measure_time = True
        model.image_backend = "opencv"

        # prepare black input frame
        frame = model._preprocessor.forward(np.zeros((10, 10, 3), dtype=np.uint8))[0]

        # define source of frames
        def source():
            for fi in range(iterations):
                yield frame

        with model:
            model(frame)  # run model once to warm up the system

            # run batch prediction
            t = Timer()
            for res in model.predict_batch(source()):
                pass
            elapsed = t()

    finally:
        # restore model parameters
        for k, v in saved_params.items():
            setattr(model, k, v)

    stats = model.time_stats()

    return ModelTimeProfile(
        elapsed=elapsed,
        iterations=iterations,
        observed_fps=iterations / elapsed,
        max_possible_fps=1e3 / stats["CoreInferenceDuration_ms"].avg,
        parameters=model.model_info,
        time_stats=stats,
    )
