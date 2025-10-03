#
# streams.gizmos.py: streaming toolkit: gizmos implementation
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements various gizmos for streaming pipelines
#

import queue, copy, time, cv2, numpy as np
from types import ModuleType
import degirum as dg
from abc import abstractmethod
from typing import Iterator, Optional, List, Union
from contextlib import ExitStack

try:
    Image: Optional[ModuleType] = None
    from PIL import Image
except ImportError:
    Image = None

from .base import Stream, StreamData, StreamMeta, Gizmo

from ..crop_extent import CropExtentOptions, extend_bbox
from ..environment import get_test_mode, get_token
from ..event_detector import EventDetector
from ..image_tools import crop_image, resize_image, image_size, ImageType
from ..notifier import EventNotifier
from ..result_analyzer_base import image_overlay_substitute, clone_result
from ..ui_support import Display
from ..video_support import (
    create_video_stream,
    get_video_stream_properties,
    open_video_writer,
    VideoStreamer,
)

# predefined meta tags
tag_video = "dgt_video"  # tag for video source data
tag_resize = "dgt_resize"  # tag for resizer result
tag_inference = "dgt_inference"  # tag for inference result
tag_preprocess = "dgt_preprocess"  # tag for preprocessor result
tag_crop = "dgt_crop"  # tag for cropping result
tag_analyzer = "dgt_analyzer"  # tag for analyzer result


class VideoSourceGizmo(Gizmo):
    """OpenCV-based video source gizmo.

    Captures frames from a video source (camera, video file, etc.) and outputs them as [StreamData](streams_base.md#streamdata) into the pipeline.
    """

    # meta keys for video frame information
    key_frame_width = "frame_width"  # frame width
    key_frame_height = "frame_height"  # frame height
    key_fps = "fps"  # stream frame rate
    key_frame_count = "frame_count"  # total frame count (if known)
    key_frame_id = "frame_id"  # frame index (sequence number)
    key_timestamp = "timestamp"  # frame timestamp

    def __init__(
        self,
        video_source=None,
        *,
        stop_composition_on_end: bool = False,
        retry_on_error: bool = False,
    ):
        """Constructor.

        Args:
            video_source (int or str, optional): A cv2.VideoCapture-compatible video source
                (device index as int, or file path/URL as str). Defaults to None.
            stop_composition_on_end (bool): If True, stop the [Composition](streams_base.md#composition) when the video source is over. Defaults to False.
            retry_on_error (bool): If True, retry opening the video source on error after some time. Defaults to False.
        """
        super().__init__()
        self._video_source = video_source
        self._stop_composition_on_end = stop_composition_on_end and not get_test_mode()
        self._retry_on_error = retry_on_error
        self._stream: Optional[cv2.VideoCapture] = None

    def get_video_properties(self) -> tuple:
        self._open_video_source()
        return get_video_stream_properties(self._stream)

    def _open_video_source(self):
        """Open the video source if it is not opened."""
        if self._stream is None:
            self._stream = create_video_stream(self._video_source)

    def _close_video_source(self):
        """Open the video source if it is not opened."""
        if self._stream is not None:
            self._stream.release()
            self._stream = None

    def __del__(self):
        """Destructor to ensure video source is released."""
        self._close_video_source()

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo.

        Returns:
            List[str]: Tags for this gizmo (its name and the video tag).
        """
        return [self.name, tag_video]

    def run(self):
        """Run the video capture loop.

        Continuously reads frames from the video source and sends each frame (with metadata) downstream until the source is exhausted or abort is signaled.
        """

        try:
            self._open_video_source()
            meta: Optional[dict] = None
            while True:
                try:
                    src = self._stream
                    assert src is not None, "Video source is not opened"
                    meta = {
                        self.key_frame_width: int(src.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        self.key_frame_height: int(src.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        self.key_fps: src.get(cv2.CAP_PROP_FPS),
                        self.key_frame_count: int(src.get(cv2.CAP_PROP_FRAME_COUNT)),
                    }
                    while not self._abort:
                        ret, data = src.read()
                        if not ret:
                            if self._retry_on_error:
                                raise Exception(
                                    f"Video source {self._video_source} ended or failed."
                                )
                            break
                        else:
                            meta2 = copy.copy(meta)
                            meta2[self.key_frame_id] = self.result_cnt
                            meta2[self.key_timestamp] = time.time()
                            self.send_result(
                                StreamData(data, StreamMeta(meta2, self.get_tags()))
                            )

                except Exception:
                    if not self._retry_on_error or meta is None:
                        # if not retrying or no metadata (first attempt), abort the run
                        raise

                if self._abort or not self._retry_on_error:
                    # if not retrying, break the loop after one run
                    break

            if self._stop_composition_on_end:
                # Stop composition if video source is over (to halt other sources still running).
                if self.composition is not None and not self._abort:
                    self.composition.stop()

        finally:
            self._close_video_source()


class IteratorSourceGizmo(Gizmo):
    """Iterator-based source gizmo.

    Takes an iterator that yields images in various formats (file paths, numpy arrays, or PIL images)
    and outputs them as StreamData into the pipeline, similar to VideoSourceGizmo but for static images.
    """

    key_file_path = "file_path"  # key for file path meta if input is a file

    def __init__(self, iterator: Iterator):
        """Constructor.

        Args:
            iterator: An iterator that yields images. Each yielded item can be:
                - str: path to an image file
                - numpy.ndarray: image as numpy array
                - PIL.Image: PIL image object
        """
        super().__init__()  # No inputs for source gizmo
        self._iterator = iterator

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo.

        Returns:
            List[str]: Tags for this gizmo (its name and the video tag).
        """
        return [self.name, tag_video]

    def _convert_to_numpy(self, image_input) -> np.ndarray:
        """Convert various image formats to numpy array.

        Args:
            image_input: Image in various formats (str path, numpy array, or PIL image)

        Returns:
            np.ndarray: Image as numpy array in BGR format (OpenCV standard)

        Raises:
            ValueError: If the input format is not supported
            FileNotFoundError: If image file path doesn't exist
        """
        if isinstance(image_input, str):
            # Load image from file path
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(
                    f"Could not load image from path: {image_input}"
                )
            return img

        elif isinstance(image_input, np.ndarray):
            # Already a numpy array, just ensure it's in the right format
            if len(image_input.shape) == 2:
                # Grayscale to BGR
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif len(image_input.shape) == 3:
                if image_input.shape[2] == 3:
                    # Assume it's already BGR or RGB - return as is
                    return image_input
                elif image_input.shape[2] == 4:
                    # RGBA/BGRA to BGR
                    return image_input[:, :, :3]
            return image_input

        elif Image is not None and isinstance(image_input, Image.Image):
            # PIL Image to numpy array
            # Convert to RGB first (PIL standard), then to BGR (OpenCV standard)
            if image_input.mode != "RGB":
                image_input = image_input.convert("RGB")
            img_array = np.array(image_input)
            # Convert RGB to BGR for OpenCV compatibility
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        else:
            raise RuntimeError(
                f"Unsupported image input type: {type(image_input)}. "
                f"Supported types: str (file path), numpy.ndarray, PIL.Image"
            )

    def run(self):
        """Run the iterator source loop.

        Iterates over the iterator, converts each item to a numpy array image,
        creates appropriate metadata (similar to VideoSourceGizmo), and sends
        the results downstream.
        """

        for image_input in self._iterator:
            if self._abort:
                break

            # convert input to numpy array
            img_array = self._convert_to_numpy(image_input)

            # create metadata similar to VideoSourceGizmo
            height, width = img_array.shape[:2]
            meta = {
                VideoSourceGizmo.key_frame_width: width,
                VideoSourceGizmo.key_frame_height: height,
                VideoSourceGizmo.key_fps: 0.0,  # No FPS for iterator
                VideoSourceGizmo.key_frame_count: -1,  # Unknown for iterator
                VideoSourceGizmo.key_frame_id: self.result_cnt,
                VideoSourceGizmo.key_timestamp: time.time(),
                self.key_file_path: image_input if isinstance(image_input, str) else "",
            }

            # Send the result
            self.send_result(StreamData(img_array, StreamMeta(meta, self.get_tags())))


class VideoDisplayGizmo(Gizmo):
    """OpenCV-based video display gizmo.

    Displays incoming frames in one or more OpenCV windows.
    """

    def __init__(
        self,
        window_titles: Union[str, List[str]] = "Display",
        *,
        show_ai_overlay: bool = False,
        show_fps: bool = False,
        stream_depth: int = 10,
        allow_drop: bool = False,
        multiplex: bool = False,
    ):
        """Constructor.

        Args:
            window_titles (str or List[str]): Title or list of titles for the display window(s).
                If a list is provided, multiple windows are opened (one per title). Defaults to "Display".
            show_ai_overlay (bool): If True, overlay AI inference results on the displayed frame (when available). Defaults to False.
            show_fps (bool): If True, show the FPS on the display window(s). Defaults to False.
            stream_depth (int): Depth of the input frame queue. Defaults to 10.
            allow_drop (bool): If True, allow dropping frames if the input queue is full. Defaults to False.
            multiplex (bool): If True, use a single input stream and display frames in a round-robin across multiple windows; if False, each window corresponds to its own input stream. Defaults to False.

        Raises:
            Exception: If multiplex is True while allow_drop is also True (unsupported configuration).
        """
        if isinstance(window_titles, str):
            window_titles = [
                window_titles,
            ]
        inp_cnt = 1 if multiplex else len(window_titles)
        super().__init__([(stream_depth, allow_drop)] * inp_cnt)
        self._window_titles = window_titles
        self._show_ai_overlay = show_ai_overlay
        self._show_fps = show_fps
        if multiplex and allow_drop:
            raise Exception("Frame multiplexing with frame dropping is not supported")
        self._multiplex = multiplex

    def run(self):
        """Run the video display loop.

        Fetches frames from the input stream(s) and shows them in the window(s) (with optional overlays and FPS display) until all inputs are exhausted or aborted.
        """
        with ExitStack() as stack:
            ndisplays = len(self._window_titles)
            ninputs = len(self.get_inputs())
            test_mode = get_test_mode()
            displays = [
                stack.enter_context(Display(title, self._show_fps))
                for title in self._window_titles
            ]
            first_run = [True] * ndisplays
            input_done = [False] * ninputs

            di = -1  # index of display window for multiplexing
            try:
                while True:
                    if self._abort or all(input_done):
                        break
                    for ii, input in enumerate(self.get_inputs()):  # ii is input index
                        if input_done[ii]:
                            continue
                        try:
                            if ninputs > 1 and not test_mode:
                                # non-multiplexing multi-input case (do not use it in test mode to avoid race conditions)
                                data = input.get_nowait()
                            else:
                                # single input or multiplexing case
                                data = input.get()
                                self.result_cnt += 1
                        except queue.Empty:
                            continue
                        # Determine which display window to use for this frame
                        di = (di + 1) % ndisplays if self._multiplex else ii
                        if data == Stream._poison:
                            input_done[ii] = True
                            break
                        img = data.data
                        if self._show_ai_overlay:
                            inference_meta = data.meta.find_last(tag_inference)
                            if inference_meta:
                                # show AI inference overlay if possible
                                img = inference_meta.image_overlay
                        displays[di].show(img)
                        if first_run[di] and not displays[di]._no_gui:
                            cv2.setWindowProperty(
                                self._window_titles[di], cv2.WND_PROP_TOPMOST, 1
                            )
                            first_run[di] = False
            except KeyboardInterrupt:
                if self.composition is not None:
                    self.composition.stop()


class VideoSaverGizmo(Gizmo):
    """Video saving gizmo.

    Writes incoming frames to an output video file.
    """

    def __init__(
        self,
        filename: str,
        *,
        show_ai_overlay: bool = False,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        Args:
            filename (str): Path to the output video file.
            show_ai_overlay (bool): If True, overlay AI inference results on frames before saving (when available). Defaults to False.
            stream_depth (int): Depth of the input frame queue. Defaults to 10.
            allow_drop (bool): If True, allow dropping frames if the input queue is full. Defaults to False.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._filename = filename
        self._show_ai_overlay = show_ai_overlay

    def run(self):
        """Run the video saving loop.

        Reads frames from the input stream and writes them to the output file until the stream is exhausted or aborted.
        """

        def get_img(data: StreamData) -> ImageType:
            frame = data.data
            if self._show_ai_overlay:
                inference_meta = data.meta.find_last(tag_inference)
                if inference_meta:
                    frame = inference_meta.image_overlay
            return frame

        img = get_img(self.get_input(0).get())
        w, h = image_size(img)
        with open_video_writer(self._filename, w, h) as writer:
            self.result_cnt += 1
            writer.write(img)
            for data in self.get_input(0):
                if self._abort:
                    break
                writer.write(get_img(data))


class VideoStreamerGizmo(Gizmo):
    """Video streaming gizmo.

    Streams incoming frames to RTSP/RTMP stream using ffmpeg.
    `MediaServer` must be running to accept the stream.
    """

    def __init__(
        self,
        stream_url: str,
        *,
        fps: float = 0,
        show_ai_overlay: bool = False,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        Args:
            stream_url (str): RTMP/RTSP URL to stream to (e.g., 'rtsp://user:password@hostname:port/stream').
                            Typically you use `MediaServer` class to start media server and
                            then use its RTMP/RTSP URL like `rtsp://localhost:8554/mystream`
            fps (float, optional): Frames per second for the stream. Defaults to 0, meaning to deduce from upstream video source.
            show_ai_overlay (bool, optional): If True, overlay AI inference results on frames before saving (when available). Defaults to False.
            stream_depth (int, optional): Depth of the input frame queue. Defaults to 10.
            allow_drop (bool, optional): If True, allow dropping frames if the input queue is full. Defaults to False.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._stream_url = stream_url
        self._fps = fps
        self._show_ai_overlay = show_ai_overlay

    def run(self):
        """Run the video saving loop.

        Reads frames from the input stream and writes them to the output file until the stream is exhausted or aborted.
        """

        def get_img(data: StreamData) -> ImageType:
            frame = data.data
            if self._show_ai_overlay:
                inference_meta = data.meta.find_last(tag_inference)
                if inference_meta:
                    frame = inference_meta.image_overlay
            return frame

        input_q = self.get_input(0)
        data0 = input_q.get()
        if data0 == Stream._poison:
            return
        last_data = data0
        img = get_img(data0)
        w, h = image_size(img)

        if self._fps <= 0:  # deduce FPS
            default_fps = 30.0  # default FPS if not specified
            video_meta = data0.meta.find_last(tag_video)
            if video_meta:
                self._fps = video_meta.get(VideoSourceGizmo.key_fps, default_fps)
            else:
                self._fps = default_fps
        frame_interval_s = 1.0 / self._fps
        read_timeout_s = 0.5 * frame_interval_s
        fps_threshold = 0.8 * self._fps
        alpha = 0.05  # IIR smoothing factor
        avg_duration_s = 1.0 / self._fps  # initialize with target FPS

        with VideoStreamer(
            self._stream_url,
            w,
            h,
            fps=self._fps,
            pix_fmt="bgr24" if isinstance(img, np.ndarray) else "rgb24",
        ) as streamer:

            self.result_cnt += 1
            streamer.write(img)
            prev_time_s = time.time()

            def send_frame(data: StreamData):
                nonlocal avg_duration_s, prev_time_s, last_data

                streamer.write(get_img(data))
                self.result_cnt += 1
                last_data = data
                now = time.time()
                avg_duration_s = (
                    alpha * (now - prev_time_s) + (1 - alpha) * avg_duration_s
                )
                prev_time_s = now

            while not self._abort:
                # try to read a real frame from the input queue
                try:
                    data = input_q.get(timeout=read_timeout_s)
                    if data == Stream._poison:
                        break
                    send_frame(data)
                except queue.Empty:
                    pass  # no new frame, possibly starvation

                # if FPS is too low, send fake frames to catch up
                fps_est = 1.0 / (
                    alpha * (time.time() - prev_time_s) + (1 - alpha) * avg_duration_s
                )
                if fps_est < fps_threshold:
                    send_frame(last_data)
                    while 1.0 / avg_duration_s < fps_threshold:
                        send_frame(last_data)


class ResizingGizmo(Gizmo):
    """OpenCV-based image resizing/padding gizmo.

    Resizes incoming images to a specified width and height, using the chosen padding or cropping method.
    """

    # meta keys for output image parameters
    key_frame_width = "frame_width"  # frame width after resize
    key_frame_height = "frame_height"  # frame height after resize
    key_pad_method = "pad_method"  # padding method used
    key_resize_method = "resize_method"  # resampling method used

    def __init__(
        self,
        w: int,
        h: int,
        pad_method: str = "letterbox",
        resize_method: str = "bilinear",
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        Args:
            w (int): Target width for output images.
            h (int): Target height for output images.
            pad_method (str): Padding method to use ("stretch", "letterbox", "crop-first", "crop-last"). Defaults to "letterbox".
            resize_method (str): Resampling method to use ("nearest", "bilinear", "area", "bicubic", "lanczos"). Defaults to "bilinear".
            stream_depth (int): Depth of the input frame queue. Defaults to 10.
            allow_drop (bool): If True, allow dropping frames if the input queue is full. Defaults to False.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._w = w
        self._h = h
        self._pad_method = pad_method
        self._resize_method = resize_method

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo.

        Returns:
            List[str]: Tags for this gizmo (its name and the resize tag).
        """
        return [self.name, tag_resize]

    def _resize(self, image):
        # Helper: perform resizing with given methods
        return resize_image(
            image,
            self._w,
            self._h,
            resize_method=self._resize_method,
            pad_method=self._pad_method,
        )

    def run(self):
        """Run the resizing loop.

        Resizes each input image according to the configured width, height, padding, and resizing method, then sends the result with updated metadata downstream.
        """
        my_meta = {
            self.key_frame_width: self._w,
            self.key_frame_height: self._h,
            self.key_pad_method: self._pad_method,
            self.key_resize_method: self._resize_method,
        }
        for data in self.get_input(0):
            if self._abort:
                break
            resized = self._resize(data.data)
            new_meta = data.meta.clone()
            new_meta.append(my_meta, self.get_tags())
            self.send_result(StreamData(resized, new_meta))


class AiGizmoBase(Gizmo):
    """Base class for AI model inference gizmos.

    Handles loading the model and iterating over input data for inference in a background thread.
    """

    def __init__(
        self,
        model: Union[dg.model.Model, str],
        *,
        non_blocking_batch_predict_timeout_s: float = 0.01,
        stream_depth: int = 10,
        allow_drop: bool = False,
        inp_cnt: int = 1,
        **kwargs,
    ):
        """Constructor.

        Args:
            model (dg.model.Model or str): A DeGirum model object or model name string to load. If a string is provided, the model will be loaded via `degirum.load_model()` using the given kwargs.
            non_blocking_batch_predict_timeout_s (float): Input queue get timeout for non-blocking batch prediction mode. Defaults to 0.001 seconds.
            stream_depth (int): Depth of the input stream queue. Defaults to 10.
            allow_drop (bool): If True, allow dropping frames on input overflow. Defaults to False.
            inp_cnt (int): Number of input streams (for models requiring multiple inputs). Defaults to 1.
            **kwargs (any): Additional parameters to pass to `degirum.load_model()` when loading the model (if model is given as a name).
        """
        super().__init__([(stream_depth, allow_drop)] * inp_cnt)

        self._non_blocking_batch_predict_timeout_s = (
            non_blocking_batch_predict_timeout_s
        )

        if isinstance(model, str):
            # Load the model by name, adding a token if not provided
            if "token" not in kwargs:
                self.model = dg.load_model(model, token=get_token(""), **kwargs)
            else:
                self.model = dg.load_model(model, **kwargs)
        else:
            self.model = model

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo.

        Returns:
            List[str]: Tags for this gizmo (its name and the inference tag).
        """
        return [self.name, tag_inference]

    def run(self):
        """Run the model inference loop.

        Internally feeds data from the input stream(s) into the model and yields results, invoking `on_result` for each inference result.
        """

        block = not self.model.non_blocking_batch_predict
        timeout_s = None if block else self._non_blocking_batch_predict_timeout_s

        def source():
            has_data = True
            while has_data:
                # Get data from all inputs
                for inp in self.get_inputs():
                    if block:
                        d = inp.get()
                    else:
                        try:
                            d = inp.get(timeout=timeout_s)
                        except queue.Empty:
                            yield None
                            continue

                    if d == Stream._poison:
                        has_data = False
                        break
                    yield (d.data, d.meta)
                if self._abort:
                    break

        for result in self.model.predict_batch(source()):
            if result is None:  # skip empty results (in non-blocking mode)
                continue

            meta = result.info
            # If input was preprocessed bytes, attempt to attach original image for better visualization
            if isinstance(result._input_image, bytes):
                preprocess_meta = meta.find_last(tag_preprocess)
                if preprocess_meta:
                    # Restore original input image for result visualization
                    result._input_image = preprocess_meta[
                        AiPreprocessGizmo.key_image_input
                    ]
                    # Recalculate bbox coordinates to original image space if converter is available
                    converter = preprocess_meta.get(AiPreprocessGizmo.key_converter)
                    if converter is not None:
                        for res in result._inference_results:
                            if "bbox" in res:
                                box = res["bbox"]
                                res["bbox"] = [
                                    *converter(*box[:2]),
                                    *converter(*box[2:]),
                                ]
            self.on_result(result)
            if self._abort:
                break

    @abstractmethod
    def on_result(self, result):
        """Handle a single inference result (to be implemented by subclasses).

        Args:
            result (dg.postprocessor.InferenceResults): The inference result object from the model.
        """


class AiSimpleGizmo(AiGizmoBase):
    """AI inference gizmo with no custom result processing.

    Passes through input frames and attaches the raw inference results to each frame's metadata.
    """

    def on_result(self, result):
        """Append the inference result to the input frame's metadata and send it downstream.

        Args:
            result (dg.postprocessor.InferenceResults): The inference result for the current frame.
        """
        new_meta = result.info.clone()
        new_meta.append(result, self.get_tags())
        self.send_result(StreamData(result.image, new_meta))


class AiObjectDetectionCroppingGizmo(Gizmo):
    """Gizmo that crops detected objects from frames of an object detection model.

    Each input frame with object detection results yields one or more cropped images as output.

    Output:
        - **Image**: The cropped portion of the original image corresponding to a detected object.
        - **Meta-info**: A dictionary containing:
            - `original_result`: Reference to the original detection result (InferenceResults) for the frame.
            - `cropped_result`: The detection result entry for this specific crop.
            - `cropped_index`: The index of this object in the original results list.
            - `is_last_crop`: True if this crop is the last one for the frame.

    Note:
        `cropped_index` and `is_last_crop` are only present if at least one object is detected in the frame.

    The `validate_bbox()` method can be overridden in subclasses to filter out undesirable detections.
    """

    # meta keys for crop result information
    key_original_result = "original_result"  # original detection result object
    key_cropped_result = "cropped_result"  # detection result entry for the crop
    key_cropped_index = "cropped_index"  # index of the object in original results
    key_is_last_crop = "is_last_crop"  # flag indicating last crop in frame

    def __init__(
        self,
        labels: List[str],
        *,
        send_original_on_no_objects: bool = True,
        crop_extent: float = 0.0,
        crop_extent_option: CropExtentOptions = CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
        crop_aspect_ratio: float = 1.0,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        Args:
            labels (List[str]): List of class labels to process. Only objects whose class is in this list will be cropped.
            send_original_on_no_objects (bool): If True, when no objects are detected in a frame, the original frame is sent through. Defaults to True.
            crop_extent (float): Extra padding around the bounding box, as a percentage of the bbox size. Defaults to 0.0.
            crop_extent_option (CropExtentOptions): Method for applying the crop extent (e.g., aspect ratio adjustment). Defaults to CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT.
            crop_aspect_ratio (float): Desired aspect ratio (W/H) for the cropped images. Defaults to 1.0.
            stream_depth (int): Depth of the input frame queue. Defaults to 10.
            allow_drop (bool): If True, allow dropping frames on overflow. Defaults to False.
        """
        # Use an unlimited-depth input stream by default for input 0 (to not block sources)
        super().__init__([(stream_depth, allow_drop)])
        self._labels = labels
        self._send_original_on_no_objects = send_original_on_no_objects
        self._crop_extent = crop_extent
        self._crop_extent_option = crop_extent_option
        self._crop_aspect_ratio = crop_aspect_ratio

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo.

        Returns:
            List[str]: Tags for this gizmo (its name and the crop tag).
        """
        return [self.name, tag_crop]

    def require_tags(self, inp: int) -> List[str]:
        """Get the list of meta tags this gizmo requires in upstream meta for a specific input.

        Returns:
            List[str]: Tags required by this gizmo in upstream meta for the specified input.
        """
        return [tag_inference]

    def run(self):
        """Run the object cropping loop.

        For each input frame, finds all detected objects (matching the specified labels and passing validation) and sends out a cropped image for each. If no objects are detected and `send_original_on_no_objects` is True, the original frame is forwarded.
        """
        for data in self.get_input(0):
            if self._abort:
                break
            img = data.data
            result = data.meta.find_last(tag_inference)
            if result is None:
                raise Exception(
                    f"{self.__class__.__name__}: inference meta not found: you need to have object detection gizmo in upstream"
                )
            # Prepare crops for each detected object
            crops = []
            crop_metas = []
            for i, r in enumerate(result.results):
                bbox = r.get("bbox")
                if not bbox:
                    continue
                # Filter out objects if validate_bbox returns False
                if not self.validate_bbox(result, i):
                    continue
                # Extend the bounding box according to crop_extent settings
                ext_bbox = extend_bbox(
                    bbox,
                    self._crop_extent_option,
                    self._crop_extent,
                    self._crop_aspect_ratio,
                    image_size(result.image),
                )
                # Crop the image region
                crops.append(crop_image(result.image, ext_bbox))
                # Prepare metadata for this crop
                cropped_obj = copy.deepcopy(r)
                cropped_obj["bbox"] = ext_bbox
                crop_meta = {
                    self.key_original_result: result,
                    self.key_cropped_result: cropped_obj,
                    self.key_cropped_index: i,
                    self.key_is_last_crop: False,  # will be set True for the last crop
                }
                new_meta = data.meta.clone()
                new_meta.append(crop_meta, self.get_tags())
                crop_metas.append(new_meta)
            if crop_metas:
                # Mark the last crop's metadata
                crop_metas[-1].find_last(tag_crop)[self.key_is_last_crop] = True
                # Send out all cropped results
                for crop_img, meta in zip(crops, crop_metas):
                    self.send_result(StreamData(crop_img, meta))
            elif self._send_original_on_no_objects:
                # If no objects and configured to do so, send the original frame
                crop_meta = {
                    self.key_original_result: result,
                    self.key_cropped_result: None,
                    self.key_cropped_index: -1,
                    self.key_is_last_crop: True,
                }
                new_meta = data.meta.clone()
                new_meta.append(crop_meta, self.get_tags())
                self.send_result(StreamData(img, new_meta))

    def validate_bbox(
        self, result: dg.postprocessor.InferenceResults, idx: int
    ) -> bool:
        """Decide whether a detected object should be cropped (can be overridden in subclasses).

        Args:
            result (dg.postprocessor.InferenceResults): The detection result for the frame.
            idx (int): The index of the object in `result.results` to validate.

        Returns:
            bool: True if the object should be cropped; False if it should be skipped.
        """
        return True


class CropCombiningGizmo(Gizmo):
    """Gizmo to combine original frames with their after-crop results.

    Expects N+1 inputs: one input stream of original frames (index 0), and N input streams of inference results from cropped images. This gizmo synchronizes and attaches the after-crop inference results back to each original frame's metadata.
    """

    # key for aggregated extra results in a detection result
    key_extra_results = "extra_results"

    def __init__(
        self,
        crop_inputs_num: int = 1,
        *,
        stream_depth: int = 10,
    ):
        """Constructor.

        Args:
            crop_inputs_num (int): Number of crop result input streams (excluding the original frame stream). Defaults to 1.
            stream_depth (int): Depth for each crop input stream's queue. Defaults to 10.
        """
        # Input 0 is for original frames (unlimited depth by using 0)
        # Inputs 1..N are for cropped results (fixed depth, no dropping to avoid desync)
        input_defs = [(0, False)] + [(stream_depth, False)] * crop_inputs_num
        super().__init__(input_defs)

    def require_tags(self, inp: int) -> List[str]:
        """Get the list of meta tags this gizmo requires in upstream meta for a specific input.

        Returns:
            List[str]: Tags required by this gizmo in upstream meta for the specified input.
        """
        if inp == 0:
            return [tag_video]
        else:
            return [tag_video, tag_crop, tag_inference]

    def run(self):
        """Run the crop combining loop.

        Synchronizes original frames with their corresponding after-crop result streams, merges the inference results from crops back into the original frame's metadata, and sends the updated frame downstream.
        """
        frame_input = self.get_input(0)
        crops: List[StreamData] = []
        combined_meta: Optional[StreamMeta] = None

        for full_frame in frame_input:
            if self._abort:
                break
            video_meta = full_frame.meta.find_last(tag_video)
            if video_meta is None:
                raise Exception(
                    f"{self.__class__.__name__}: video meta not found: you need to have {VideoSourceGizmo.__class__.__name__} in upstream of input 0"
                )
            frame_id = video_meta[VideoSourceGizmo.key_frame_id]
            # Collect all crop results for this frame (matching frame_id)
            while True:
                if not crops:
                    # read all crop inputs
                    crops = []
                    for inp in self.get_inputs()[1:]:
                        crop = inp.get()
                        if crop == Stream._poison:
                            self._abort = True
                            break
                        crops.append(crop)
                if self._abort:
                    break
                # Check frame IDs of all pending crop data
                video_metas = [crop.meta.find_last(tag_video) for crop in crops]
                if None in video_metas:
                    raise Exception(
                        f"{self.__class__.__name__}: video meta not found: you need to have {VideoSourceGizmo.__class__.__name__} in upstream of all crop inputs"
                    )
                crop_frame_ids = [
                    meta[VideoSourceGizmo.key_frame_id] for meta in video_metas if meta
                ]
                if len(set(crop_frame_ids)) != 1:
                    raise Exception(
                        f"{self.__class__.__name__}: crop frame IDs are not synchronized. Make sure all crop inputs have the same {AiObjectDetectionCroppingGizmo.__class__.__name__} in upstream"
                    )
                crop_frame_id = crop_frame_ids[0]
                if crop_frame_id > frame_id:
                    # Crop results belong to a future frame: send current frame as is (no more crop results for it)
                    self.send_result(full_frame)
                    break
                if crop_frame_id < frame_id:
                    # Crop results belong to a past frame (missed sync): discard and retry
                    crops = []
                    continue
                # At this point, crop_frame_id == frame_id, meaning we have all crop results for this frame
                crop_metas = [crop.meta.find_last(tag_crop) for crop in crops]
                if None in crop_metas:
                    raise Exception(
                        f"{self.__class__.__name__}: crop meta(s) not found: you need to have {AiObjectDetectionCroppingGizmo.__class__.__name__} in upstream of all crop inputs"
                    )
                crop_meta = crop_metas[0]
                assert crop_meta
                if not all(crop_meta == cm for cm in crop_metas[1:]):
                    raise Exception(
                        f"{self.__class__.__name__}: crop metas are not synchronized. Make sure all crop inputs have the same {AiObjectDetectionCroppingGizmo.__class__.__name__} in upstream"
                    )
                orig_result = crop_meta[
                    AiObjectDetectionCroppingGizmo.key_original_result
                ]
                bbox_idx = crop_meta[AiObjectDetectionCroppingGizmo.key_cropped_index]
                # create combined meta if not done yet
                # Create a combined meta on first result
                if combined_meta is None:
                    combined_meta = crops[0].meta.clone()
                    combined_meta.remove_last(tag_crop)
                    combined_meta.remove_last(tag_inference)
                    combined_meta.append(self._clone_result(orig_result), tag_inference)
                # Attach each crop's inference results into the combined result
                if bbox_idx >= 0:
                    inference_metas = [
                        crop.meta.find_last(tag_inference) for crop in crops
                    ]
                    if any(im is orig_result for im in inference_metas):
                        raise Exception(
                            f"{self.__class__.__name__}: after-crop inference meta(s) not found: you need to have some inference-type gizmo in upstream of all crop inputs"
                        )
                    result = combined_meta.find_last(tag_inference)
                    assert result is not None
                    result._inference_results[bbox_idx][self.key_extra_results] = (
                        self._adjust_results(result, bbox_idx, inference_metas)
                    )
                crops = []  # mark crops as processed
                if crop_meta[AiObjectDetectionCroppingGizmo.key_is_last_crop]:
                    # last crop in the frame: send full frame
                    self.send_result(StreamData(full_frame.data, combined_meta))
                    combined_meta = None  # reset for next frame
                    break

    def _adjust_results(
        self, orig_result, bbox_idx: int, cropped_results: list
    ) -> list:
        """Adjust inference results from a crop to the original image's coordinate space.

        This converts the coordinates (e.g., bounding boxes, landmarks) of inference results obtained on a cropped image back to the coordinate system of the original image.

        Args:
            orig_result (dg.postprocessor.InferenceResults): The original detection result (InferenceResults) for the full frame.
            bbox_idx (int): The index of the object in the original result list.
            cropped_results (list): A list of InferenceResults from the cropped image's inference.

        Returns:
            list: A list of adjusted InferenceResults corresponding to the original image coordinates.
        """
        bbox = orig_result._inference_results[bbox_idx].get("bbox")
        assert bbox
        tl = bbox[:2]
        ret = []
        for crop_res in cropped_results:
            cr = clone_result(crop_res)
            # Adjust all coordinates in the result to original image space
            for r in cr._inference_results:
                if "bbox" in r:
                    r["bbox"][:] = [a + b for a, b in zip(r["bbox"], tl + tl)]
                if "landmarks" in r:
                    for lm_list in r["landmarks"]:
                        lm = lm_list["landmark"]
                        lm[:2] = [lm[0] + tl[0], lm[1] + tl[1]]
            ret.append(cr)
        return ret

    def _clone_result(self, result):
        """Clone an inference result, deep-copying its `_inference_results` list.

        Args:
            result (dg.postprocessor.InferenceResults): The inference result to clone.

        Returns:
            (dg.postprocessor.InferenceResults): A cloned inference result with a deep-copied results list.
        """

        def _overlay_extra_results(result):
            """Produce an image overlay including all extra results."""
            overlay_image = result._orig_image_overlay_extra_results
            for res in result._inference_results:
                if self.key_extra_results in res:
                    bbox = res.get("bbox")
                    if bbox:
                        for extra_res in res[self.key_extra_results]:
                            orig_image = extra_res._input_image
                            extra_res._input_image = overlay_image
                            overlay_image = extra_res.image_overlay
                            extra_res._input_image = orig_image
            return overlay_image

        clone = clone_result(result)
        # redefine `image_overlay` property to `_overlay_extra_results` function so
        # that it will be called instead of the original one to annotate the image with extra results;
        # preserve original `image_overlay` property as `_orig_image_overlay_extra_results` property;
        clone.__class__ = type(
            clone.__class__.__name__ + "_overlay_extra_results",
            (clone.__class__,),
            {
                "image_overlay": property(_overlay_extra_results),
                "_orig_image_overlay_extra_results": clone.__class__.image_overlay,
            },
        )
        return clone


class AiResultCombiningGizmo(Gizmo):
    """Gizmo to combine inference results from multiple AI gizmos of the same type."""

    def __init__(
        self,
        inp_cnt: int,
        *,
        stream_depth: int = 10,
    ):
        """Constructor.

        Args:
            inp_cnt (int): Number of input result streams to combine.
            stream_depth (int): Depth of each input stream's queue. Defaults to 10.
        """
        self._inp_cnt = inp_cnt
        super().__init__([(stream_depth, False)] * inp_cnt)

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo.

        Returns:
            List[str]: Tags for this gizmo (its name and the inference tag).
        """
        return [self.name, tag_inference]

    def require_tags(self, inp: int) -> List[str]:
        """Get the list of meta tags this gizmo requires in upstream meta for a specific input.

        Returns:
            List[str]: Tags required by this gizmo in upstream meta for the specified input.
        """
        return [tag_inference]

    def run(self):
        """Run the result combining loop.

        Collects inference results from all input streams, merges their results into a single combined result, and sends it downstream.
        """
        while True:
            all_data = []
            # Get one item from each input
            for inp in self.get_inputs():
                d = inp.get()
                if d == Stream._poison:
                    self._abort = True
                    break
                all_data.append(d)
            if self._abort:
                break
            base_data: Optional[StreamData] = None
            base_result: Optional[dg.postprocessor.InferenceResults] = None
            for d in all_data:
                inference_meta = d.meta.find_last(tag_inference)
                if inference_meta is None:
                    continue  # skip if no inference results in this data
                if base_result is None:
                    base_data = d
                    base_result = clone_result(inference_meta)
                else:
                    base_result._inference_results += copy.deepcopy(
                        inference_meta._inference_results
                    )
            assert base_data is not None
            new_meta = base_data.meta.clone()
            new_meta.append(base_result, self.get_tags())
            self.send_result(StreamData(base_data.data, new_meta))


class AiPreprocessGizmo(Gizmo):
    """Preprocessing gizmo that applies a model's preprocessor to input images.

    It generates preprocessed image data to be fed into the model.

    Output:
        - **Data**: Preprocessed image bytes ready for model input.
        - **Meta-info**: Dictionary including:
            - `image_input`: The original input image.
            - `converter`: A function to convert coordinates from model output back to the original image.
            - `image_result`: The preprocessed image (present only if the model is configured to provide it).

    Attributes:
        key_image_input (str): Metadata key for the original input image.
        key_converter (str): Metadata key for the coordinate conversion function.
        key_image_result (str): Metadata key for the preprocessed image.
    """

    key_image_input = "image_input"
    key_converter = "converter"
    key_image_result = "image_result"

    def __init__(
        self,
        model,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        Args:
            model (dg.model.Model): The model object (PySDK model) whose preprocessor will be used.
            stream_depth (int): Depth of the input frame queue. Defaults to 10.
            allow_drop (bool): If True, allow dropping frames on overflow. Defaults to False.
        """
        super().__init__([(stream_depth, allow_drop)])
        self._preprocessor = model._preprocessor

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo.

        Returns:
            List[str]: Tags for this gizmo (its name and the preprocess tag).
        """
        return [self.name, tag_preprocess]

    def run(self):
        """Run the preprocessing loop.

        Applies the model's preprocessor to each input frame and sends the resulting data (and updated meta-info) downstream.
        """
        for data in self.get_input(0):
            if self._abort:
                break
            res = self._preprocessor.forward(data.data)
            # processed is expected to be a tuple (preprocessed_bytes, extra_meta)
            new_meta = data.meta.clone()
            new_meta.append(res[1], self.get_tags())
            self.send_result(StreamData(res[0], new_meta))


class AiAnalyzerGizmo(Gizmo):
    """Gizmo to apply a chain of analyzers to an inference result, with optional filtering.

    Each analyzer (e.g., EventDetector, EventNotifier) processes the inference result and may add events or notifications.
    If filters are provided, only results that contain at least one of the specified events/notifications are passed through.
    """

    def __init__(
        self,
        analyzers: list,
        *,
        filters: Optional[set] = None,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        Args:
            analyzers (List): List of analyzer objects to apply (e.g., EventDetector, EventNotifier instances).
            filters (set, optional): A set of event names or notification names to filter results. Only results that have at least one of these events or notifications will be forwarded (others are dropped). Defaults to None (no filtering).
            stream_depth (int): Depth of the input frame queue. Defaults to 10.
            allow_drop (bool): If True, allow dropping frames on overflow. Defaults to False.
        """
        self._analyzers = analyzers
        self._filters = filters
        super().__init__([(stream_depth, allow_drop)])

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo.

        Returns:
            List[str]: Tags for this gizmo (its name, the inference tag, and the analyzer tag).
        """
        return [self.name, tag_inference, tag_analyzer]

    def require_tags(self, inp: int) -> List[str]:
        """Get the list of meta tags this gizmo requires in upstream meta for a specific input.

        Returns:
            List[str]: Tags required by this gizmo in upstream meta for the specified input.
        """
        return [tag_inference]

    def run(self):
        """Run the analyzer processing loop.

        For each input frame, clones its inference result and runs all analyzers on it (which may add events/notifications). If filters are specified, the result is dropped unless it contains at least one of the specified events or notifications. The possibly modified inference result is appended to the frame's metadata and sent downstream. After processing all frames, all analyzers are finalized.
        """
        for data in self.get_input(0):
            if self._abort:
                break
            filter_ok = True
            new_meta = data.meta.clone()
            inference_meta = new_meta.find_last(tag_inference)
            if inference_meta is not None:
                inference_clone = clone_result(inference_meta)
                # Apply all analyzers in sequence
                for analyzer in self._analyzers:
                    analyzer.analyze(inference_clone)
                # Substitute overlay image if needed
                image_overlay_substitute(inference_clone, self._analyzers)
                # Append the modified inference result to metadata
                new_meta.append(inference_clone, self.get_tags())
                # Apply filtering if filters are set
                if self._filters:
                    notifications = getattr(
                        inference_clone, EventNotifier.key_notifications, None
                    )
                    if notifications is None or not self._filters & notifications:
                        events = getattr(
                            inference_clone, EventDetector.key_events_detected, None
                        )
                        if events is None or not self._filters & events:
                            filter_ok = False
            if filter_ok:
                self.send_result(StreamData(data.data, new_meta))
        # Finalize all analyzers after processing
        for analyzer in self._analyzers:
            analyzer.finalize()


class SinkGizmo(Gizmo):
    """Sink gizmo that receives results and accumulates them in an internal queue.

    This gizmo does not send data further down the pipeline. Instead, it stores all incoming results so they can be retrieved (for example, by iterating over the gizmo's output in the main thread).
    """

    def __init__(
        self,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        Args:
            stream_depth (int): Depth of the input queue. Defaults to 10.
            allow_drop (bool): If True, allow dropping frames on overflow. Defaults to False.
        """
        super().__init__([(stream_depth, allow_drop)])

    def run(self):
        """Run gizmo (no operation).

        Immediately returns, as the sink simply collects incoming data without processing.
        """
        return  # no processing; data is stored in the input queue

    def __call__(self):
        """Retrieve the internal queue for iteration.

        Returns:
            Stream (streams_base.Stream): The input Stream (queue) of this sink gizmo, which can be iterated to get collected results.
        """
        return self._inputs[0]


class FPSStabilizingGizmo(Gizmo):
    """FPS stabilizing gizmo that maintains stable frame rate by sending fake frames when input is empty.

    When the input queue has frames available, they are sent through as-is. When the input queue
    is empty, fake frames are generated by gradually shading the last genuine frame to maintain
    the target FPS derived from video metadata.
    """

    def __init__(
        self,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        Args:
            stream_depth (int): Depth of the input frame queue. Defaults to 10.
            allow_drop (bool): If True, allow dropping frames if the input queue is full. Defaults to False.
        """
        super().__init__([(stream_depth, allow_drop)])

    def require_tags(self, inp):
        """Get the list of meta tags this gizmo requires in upstream meta for a specific input.

        Returns:
            List[str]: Tags required by this gizmo in upstream meta for the specified input.
        """
        return [tag_video]

    def run(self):
        """Run the FPS stabilizing loop.

        Continuously reads frames from input queue and sends them downstream. When the input
        queue is empty, generates fake frames by gradually shading the last genuine frame
        to maintain stable FPS derived from video metadata.
        """
        last_data: Optional[StreamData] = None
        fps = frame_interval = 0.0  # not defined
        fake_frame_count = 0
        last_frame_time = -1.0

        input_queue = self.get_input(0)

        time_budget = 0.0
        while not self._abort:

            # try to get a genuine frame first
            try:
                # calculate timeout
                timeout: Optional[float] = None  # wait indefinite for first frame
                if frame_interval > 0:
                    # use 2x frame interval for normal stream, 0.5x for fake frames
                    timeout = (
                        2.0 * frame_interval
                        if fake_frame_count == 0
                        else 0.5 * frame_interval
                    )

                # get data from input queue, with proper timeout
                data = input_queue.get(timeout=timeout)
                if data == Stream._poison:
                    break

                # take FPS from video metadata if available
                if fps <= 0:
                    video_meta = data.meta.find_last(tag_video)
                    if video_meta:
                        fps = video_meta.get(VideoSourceGizmo.key_fps, fps)
                        if fps > 0:
                            frame_interval = 1.0 / fps

                last_data = data
                fake_frame_count = 0

            except queue.Empty:
                # no genuine frame available, check if we need to send a fake frame
                if (
                    frame_interval > 0
                    and last_data is not None
                    and time.time() - last_frame_time >= frame_interval + time_budget
                ):
                    # calculate shading factor with exponential decay
                    shading_factor = max(0.3, np.exp(-0.1 * fake_frame_count))
                    fake_frame_count += 1

                    # create fake frame by shading
                    fake_frame = (last_data.data * shading_factor).astype(
                        last_data.data.dtype
                    )
                    data = StreamData(fake_frame, last_data.meta.clone())
                else:
                    # not time to send a fake frame yet
                    continue

            # update frame ID and timestamp
            video_meta = data.meta.find_last(tag_video)
            if video_meta:
                video_meta[VideoSourceGizmo.key_frame_id] = self.result_cnt
                video_meta[VideoSourceGizmo.key_timestamp] = time.time()

            # send the frame
            self.send_result(data)
            current_time = time.time()
            if last_frame_time >= 0:
                time_budget += frame_interval - (current_time - last_frame_time)
                time_budget = min(3 * frame_interval, time_budget)
            last_frame_time = current_time
