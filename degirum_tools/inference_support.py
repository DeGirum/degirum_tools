# inference_support.py: classes and functions for AI inference
# Copyright DeGirum Corporation 2023
# All rights reserved
# Implements classes and functions to handle AI inferences

"""
Inference Support Overview
==========================

This module provides utility functions and classes for integrating DeGirum PySDK models into
various inference scenarios, including:

- **Connecting** to different model zoo endpoints (cloud, AI server, or local accelerators).
- **Attaching** custom result analyzers to models or compound models, enabling additional
  data processing or custom overlay annotation on inference results.
- **Running** inferences on video sources or streams (local camera, file, RTSP,
  YouTube links, etc.) with optional real-time annotation and saving to output video files.
- **Measuring** model performance using a profiling function that times inference
  runs under various conditions.

Key Concepts
------------

1. **Model Zoo Connections**:
   Functions like `connect_model_zoo` provide a unified way to choose
   between different inference endpoints (cloud, server, local hardware).

2. **Analyzer Attachment**:
   By calling `attach_analyzers` or using specialized classes within the streaming
   toolkit, you can process each inference result through user-defined or library-provided
   analyzers (subclasses of `ResultAnalyzerBase`).

3. **Video Inference and Annotation**:
   Functions `predict_stream` and `annotate_video` demonstrate how to
   run inference on live or file-based video streams. They optionally include steps to
   create overlays (bounding boxes, labels, etc.) and even show a real-time display
   or write to an output video file.

4. **Model Time Profiling**:
   `model_time_profile` provides a convenient way to measure the performance
   (FPS, average inference time, etc.) of a given DeGirum PySDK model under test conditions.

Basic Usage Example
-------------------
```python
from degirum_tools import ModelSpec, remote_assets
from degirum_tools.inference_support import (
    attach_analyzers,
    annotate_video,
    model_time_profile,
)
from degirum_tools.result_analyzer_base import ResultAnalyzerBase


# Define a simple analyzer that draws text on each frame
class MyDummyAnalyzer(ResultAnalyzerBase):
    def analyze(self, result):
        # Optional: add custom logic here, e.g. track or filter detections
        pass

    def annotate(self, result, image):
        \"\"\"
        Draws a simple "Dummy Analyzer" label in the top-left corner of each frame.
        \"\"\"
        import cv2
        cv2.putText(
            image,
            "Dummy Analyzer",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        return image

# Describe the model once so the configuration is reusable.
model_spec = ModelSpec(
    model_name="<your_model_name>",
    zoo_url="degirum/degirum",
    inference_host_address="@cloud",
)

with model_spec.load_model() as model:
    # Attach dummy analyzer
    attach_analyzers(model, MyDummyAnalyzer())

    # Annotate a video with detection results + dummy analyzer and set a path to save the video
    annotate_video(
        model,
        video_source_id=remote_assets.store_short,
        output_video_path="annotated_output.mp4",
        show_progress=True,      # Show a progress bar in console
        visual_display=True,     # Open an OpenCV window to display frames
        show_ai_overlay=True,    # Use model's overlay with bounding boxes
        fps=None                 # Use the source's native frame rate
    )

    # Time-profile the model
    profile = model_time_profile(model, iterations=100)
    print("Time profiling results:")
    print(f"  Elapsed time: {profile.elapsed:.3f} s")
    print(f"  Observed FPS: {profile.observed_fps:.2f}")
    print(f"  Max possible FPS: {profile.max_possible_fps:.2f}")
```
"""

import cv2
import numpy as np
import degirum as dg  # import DeGirum PySDK
from contextlib import ExitStack
from pathlib import Path
from typing import Union, List, Optional, Iterator, Final
from dataclasses import dataclass
from .compound_models import CompoundModelBase
from .video_support import (
    open_video_stream,
    get_video_stream_properties,
    video_source,
    open_video_writer,
)
from .ui_support import Progress, Display, Timer
from .result_analyzer_base import ResultAnalyzerBase, subclass_result_with_analyzers
from . import environment as env


# Inference options: parameters for connect_model_zoo
CloudInference = 1  # use DeGirum cloud server for inference
AIServerInference = 2  # use AI server deployed in LAN/VPN
LocalHWInference = 3  # use locally-installed AI HW accelerator

DTYPE_MAP: Final = {
    "DG_FLT": np.float32,
    "DG_DBL": np.float64,
    "DG_UINT8": np.uint8,
    "DG_INT8": np.int8,
    "DG_UINT16": np.uint16,
    "DG_INT16": np.int16,
    "DG_UINT32": np.uint32,
    "DG_INT32": np.int32,
    "DG_UINT64": np.uint64,
    "DG_INT64": np.int64,
}


def connect_model_zoo(
    inference_option: int = CloudInference,
) -> dg.zoo_manager.ZooManager:
    """
    Connect to a model zoo endpoint based on the specified inference option.

    This function provides a convenient way to switch between:
      - Cloud-based inference (``CloudInference``),
      - AI server on LAN/VPN (``AIServerInference``),
      - Local hardware accelerator (``LocalHWInference``).

    It uses environment variables (see ``degirum_tools.environment``) to resolve
    the zoo address/URL and token as needed.

    Args:
        inference_option (int):
            One of ``CloudInference``, ``AIServerInference``, or ``LocalHWInference``

    Raises:
        Exception: If an invalid ``inference_option`` is provided.

    Returns:
        dg.zoo_manager.ZooManager:
            A model zoo manager connected to the requested endpoint.
    """
    cloud_zoo_url = env.get_cloud_zoo_url()
    token = env.get_var(env.var_Token)

    if inference_option == CloudInference:
        # inference on cloud platform
        zoo = dg.connect(dg.CLOUD, cloud_zoo_url, token)

    elif inference_option == AIServerInference:
        # inference on AI server
        hostname = env.get_var(env.var_AiServer)
        if env.get_var(env.var_CloudZoo, ""):
            # use cloud zoo
            zoo = dg.connect(hostname, cloud_zoo_url, token)
        else:
            # use local zoo
            zoo = dg.connect(hostname)

    elif inference_option == LocalHWInference:
        zoo = dg.connect(dg.LOCAL, cloud_zoo_url, token)

    else:
        raise Exception(
            "Invalid value of inference_option parameter. Should be one of CloudInference, AIServerInference, or LocalHWInference"
        )

    return zoo


def attach_analyzers(
    model: Union[dg.model.Model, CompoundModelBase],
    analyzers: Union[ResultAnalyzerBase, List[ResultAnalyzerBase], None],
):
    """
    Attach or detach analyzer(s) to a model or compound model.

    For single or compound models, analyzers can augment the inference results
    with extra metadata and/or custom overlay. If attaching analyzers to a
    compound model (e.g., `compound_models.CompoundModelBase`),
    the analyzers are invoked at the final stage of each inference result.

    Args:
        model (Union[dg.model.Model, CompoundModelBase]):
            The model or compound model to which analyzers will be attached.
        analyzers (Union[ResultAnalyzerBase, List[ResultAnalyzerBase], None]):
            One or more analyzer objects. Passing None will detach any previously attached analyzers.

    Usage:
        attach_analyzers(my_model, MyAnalyzerSubclass())

    Notes:
        - If ``model`` is a compound model, the call is forwarded to `model.attach_analyzers()`.
        - If ``model`` is a standard PySDK model, this function subclasses the model's
          current result class with a new class that additionally calls each analyzer
          in turn for `analyze()` and `annotate()` steps.
          This subclass is assigned to `model._custom_postprocessor` property.
    """
    if isinstance(model, CompoundModelBase):
        # For a compound model, forward directly
        model.attach_analyzers(analyzers)
    else:
        if analyzers:
            # set model custom postprocessor as analyzing postprocessor, remembering the original custom postprocessor
            result_class = subclass_result_with_analyzers(
                model.get_inference_results_class(), analyzers
            )
            setattr(
                result_class, "_custom_postprocessor_saved", model._custom_postprocessor
            )
            model._custom_postprocessor = result_class
        else:
            if model._custom_postprocessor is not None and hasattr(
                model._custom_postprocessor, "_custom_postprocessor_saved"
            ):
                for analyzer in model._custom_postprocessor._analyzers:
                    analyzer.finalize()
                # restore the original custom postprocessor
                model._custom_postprocessor = (
                    model._custom_postprocessor._custom_postprocessor_saved
                )


def predict_stream(
    model: dg.model.Model,
    video_source_id: Union[int, str, Path, None],
    *,
    fps: Optional[float] = None,
    analyzers: Union[ResultAnalyzerBase, List[ResultAnalyzerBase], None] = None,
) -> Iterator["dg.postprocessor.InferenceResults"]:
    """
    Run a PySDK model on a live or file-based video source, yielding inference results.

    This function is a generator that continuously:
      1. Reads frames from the specified video source.
      2. Runs inference on each frame via `model.predict_batch`.
      3. If analyzers are provided, each result is wrapped in a dynamic postprocessor
         that calls analyzers' `analyze()` and `annotate()` methods.

    Args:
        model (dg.model.Model):
            Model to run on each incoming frame.
        video_source_id (Union[int, str, Path, None]):
            Identifier for the video source. Possible types include:
              - An integer camera index (e.g., 0 for default webcam).
              - A local file path or string/Path, e.g., "video.mp4".
              - A streaming URL (RTSP/YouTube link).
              - None if no source is available (not typical).
        fps (Optional[float]):
            If provided, caps the effective reading/processing rate to the given FPS.
            If the input source is slower, this has no effect. If faster, frames are decimated.
        analyzers (Union[ResultAnalyzerBase, List[ResultAnalyzerBase], None]):
            One or more analyzers to apply to each inference result. If None, no additional
            analysis or annotation is performed beyond the model's standard postprocessing.

    Yields:
        InferenceResults: The inference result for each processed frame. If analyzers are present,
            the result object is wrapped to allow custom annotation in its `.image_overlay`.

    Example:
    ```python
    for res in predict_stream(my_model, "my_video.mp4", fps=15, analyzers=MyAnalyzer()):
        annotated_img = res.image_overlay  # includes custom overlay
        # do something with annotated_img
    ```
    """

    if analyzers is not None:
        attach_analyzers(model, analyzers)

    with open_video_stream(video_source_id) as stream:
        for res in model.predict_batch(video_source(stream, fps=fps)):
            yield res

    if analyzers is not None:
        attach_analyzers(model, None)  # detach analyzers after use


def annotate_video(
    model: dg.model.Model,
    video_source_id: Union[int, str, Path, None, cv2.VideoCapture],
    output_video_path: str,
    *,
    show_progress: bool = True,
    visual_display: bool = True,
    show_ai_overlay: bool = True,
    fps: Optional[float] = None,
    analyzers: Union[ResultAnalyzerBase, List[ResultAnalyzerBase], None] = None,
):
    """
    Run a model on a video source and save the annotated output to a video file.

    This function:
      1. Opens the input video source.
      2. Processes each frame with the specified model.
      3. (Optionally) calls any analyzers to modify or annotate the inference result.
      4. If `show_ai_overlay` is True, retrieves the `image_overlay` from the result (which
         includes bounding boxes, labels, etc.). Otherwise, uses the original frame.
      5. Writes the annotated frame to `output_video_path`.
      6. (Optionally) displays the annotated frames in a GUI window and shows progress.

    Args:
        model (dg.model.Model):
            The model to run on each frame of the video.
        video_source_id (Union[int, str, Path, None, cv2.VideoCapture]):
            The video source, which can be:
              - A cv2.VideoCapture object already opened by `open_video_stream`,
              - An integer camera index (e.g., 0),
              - A file path or a URL (RTSP/YouTube).
        output_video_path (str):
            Path to the output video file. The file is created or overwritten as needed.
        show_progress (bool):
            If True, displays a textual progress bar or frame counter (for local file streams).
        visual_display (bool):
            If True, opens an OpenCV window to show the annotated frames in real time.
        show_ai_overlay (bool):
            If True, uses the result's `image_overlay`. If False, uses the original unannotated frame.
        fps (Optional[float]):
            If provided, caps the effective processing rate to the given FPS. Otherwise,
            uses the native FPS of the video source if known.
        analyzers (Union[ResultAnalyzerBase, List[ResultAnalyzerBase], None]):
            One or more analyzers to apply. Each analyzer's `analyze_and_annotate()`
            is called on the result before writing the frame.

    Example:
    ```python
    annotate_video(my_model, "input.mp4", "output.mp4", analyzers=[MyAnalyzer()])
    ```
    """
    win_name = f"Annotating {video_source_id}"

    analyzer_list = (
        analyzers
        if isinstance(analyzers, list)
        else ([analyzers] if analyzers is not None else [])
    )

    for analyzer in analyzer_list:
        # If an analyzer needs a custom window attachment step
        if hasattr(analyzer, "window_attach"):
            analyzer.window_attach(win_name)

    with ExitStack() as stack:
        if visual_display:
            display = stack.enter_context(Display(win_name))

        if isinstance(video_source_id, cv2.VideoCapture):
            stream = video_source_id
        else:
            stream = stack.enter_context(open_video_stream(video_source_id))

        w, h, video_fps = get_video_stream_properties(stream)

        # Overwrite or limit the stream's FPS if the user specified an fps argument
        if fps:
            video_fps = fps

        writer = stack.enter_context(
            open_video_writer(str(output_video_path), w, h, video_fps)
        )

        if show_progress:
            progress = Progress(int(stream.get(cv2.CAP_PROP_FRAME_COUNT)))

        for res in model.predict_batch(video_source(stream, fps=video_fps)):
            img = res.image_overlay if show_ai_overlay else res.image

            # Apply all analyzers
            for analyzer in analyzer_list:
                img = analyzer.analyze_and_annotate(res, img)

            writer.write(img)

            if visual_display:
                display.show(img)

            if show_progress:
                progress.step()


@dataclass
class ModelTimeProfile:
    """
    Container for model time profiling results.

    Attributes:
        elapsed (float):
            Total elapsed time in seconds for the profiling run.
        iterations (int):
            Number of inference iterations performed.
        observed_fps (float):
            The measured frames per second (iterations / elapsed).
        max_possible_fps (float):
            Estimated maximum possible FPS based on the model's
            core inference duration, ignoring overhead.
        parameters (dict):
            A copy of the model's metadata or parameters for reference.
        time_stats (dict):
            A dictionary containing detailed timing statistics from
            the model's built-in time measurement (if available).
    """

    elapsed: float
    iterations: int
    observed_fps: float
    max_possible_fps: float
    parameters: dict
    time_stats: dict


def model_time_profile(
    model: dg.model.Model, iterations: int = 100, input_image_format: str = "JPEG"
) -> ModelTimeProfile:
    """
    Profile the inference performance of a DeGirum PySDK model by running a specified
    number of iterations on a synthetic (zero-pixel) image.

    This function:
      1. Adjusts the model's settings to measure time (``model.measure_time = True``).
      2. Warms up the model by performing one inference.
      3. Resets time statistics and runs the specified number of iterations.
      4. Restores original model parameters after profiling.

    Args:
        model (dg.model.Model):
            A PySDK model to profile. Must accept images as input.
        iterations (int):
            Number of inference iterations to run (excluding the warm-up iteration).

    Raises:
        NotImplementedError:
            If the model does not accept images (e.g., audio/text).

    Returns:
        ModelTimeProfile:
            An object containing timing details, measured FPS, and other stats.

    Example:
    ```python
    profile = model_time_profile(my_model, iterations=50)
    print("Elapsed seconds:", profile.elapsed)
    print("Observed FPS:", profile.observed_fps)
    print("Core Inference Stats:", profile.time_stats["CoreInferenceDuration_ms"])
    ```
    """

    # Skip non-image type models
    if model.model_info.InputType[0] != "Image":
        raise NotImplementedError
    # Save the original model parameters to restore later
    saved_params = {
        "input_image_format": model.input_image_format,
        "measure_time": model.measure_time,
        "image_backend": model.image_backend,
    }

    elapsed = 0.0

    try:
        # Configure model for time measurement
        model.input_image_format = input_image_format
        model.measure_time = True
        model.image_backend = "opencv"

        # Prepare a small black input frame
        frame = model._preprocessor.forward(np.zeros((10, 10, 3), dtype=np.uint8))[0]

        # Define a generator for repeated frames
        def source():
            for fi in range(iterations):
                yield frame

        # Warm up once outside the measurement
        with model:
            model(frame)
            model.reset_time_stats()

            # Run batch prediction in a timed block
            t = Timer()
            for res in model.predict_batch(source()):
                pass
            elapsed = t()

    finally:
        # Restore original parameters
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


def _build_dummy_input(model: dg.model.Model):
    """Helper to build one dummy input tuple/list matching the model's inputs"""

    info = model.model_info
    dummy_inputs: list = []

    for i, input_type in enumerate(info.InputType):
        shape = model.input_shape[i]

        if input_type == "Image":
            dummy_inputs.append(np.zeros((10, 10, 3), np.uint8))

        elif input_type == "Audio":
            waveform_size = info.InputWaveformSize[i]
            if waveform_size <= 0:
                raise dg.exceptions.DegirumException(
                    f"Cannot create dummy audio for warmup: invalid InputWaveformSize ({waveform_size})."
                )
            raw_dtype = info.InputRawDataType[i]
            dtype = DTYPE_MAP.get(raw_dtype)
            if dtype is None:
                raise dg.exceptions.DegirumException(
                    f"Cannot create dummy audio for warmup: unsupported data type '{raw_dtype}'."
                )
            dummy_inputs.append(np.zeros((waveform_size,), dtype=dtype))

        elif input_type == "Tensor":
            raw_dtype = info.InputRawDataType[i]
            dtype = DTYPE_MAP.get(raw_dtype)
            if dtype is None:
                raise dg.exceptions.DegirumException(
                    f"Cannot create dummy tensor for warmup: unsupported data type '{raw_dtype}'."
                )
            dummy_inputs.append(np.zeros(shape, dtype=dtype))

        else:
            raise dg.exceptions.DegirumException(
                f"Warmup is not implemented for the model input type: '{input_type}'."
            )

    return dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs


def warmup_device(model: dg.model.Model, chosen_device: int):
    """Warm up given model on the given device in devices_available"""
    if chosen_device not in model.devices_available:
        raise dg.exceptions.DegirumException(
            f"Device {chosen_device} is not in model.devices_available: {model.devices_available}"
        )

    previous_selection = model.devices_selected
    try:
        model.devices_selected = [chosen_device]
        model.predict(_build_dummy_input(model))
    finally:
        model.devices_selected = previous_selection


def warmup_model(model: dg.model.Model):
    """
    Warms up a model by performing one inference on a dummy input frame on each device selected.

    This is useful for initializing the model internals and hardware, which can
    reduce latency on the first real inference.

    Args:
        model (dg.model.Model): The DeGirum PySDK model object to warm up.
    """
    # For cloud models, we enforce a single-device warmup.
    if isinstance(model, dg.model._CloudServerModel):
        warmup_device(model, 0)
        return

    for device in model.devices_selected:
        warmup_device(model, device)
