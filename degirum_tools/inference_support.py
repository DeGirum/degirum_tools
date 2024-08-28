#
# inference_support.py: classes and functions for AI inference
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes and functions to handle AI inferences
#

import cv2, numpy as np
import degirum as dg  # import DeGirum PySDK
from contextlib import ExitStack
from pathlib import Path
from typing import Union, List, Optional
from dataclasses import dataclass
from .video_support import (
    open_video_stream,
    get_video_stream_properties,
    video_source,
    open_video_writer,
)
from .ui_support import Progress, Display, Timer
from .result_analyzer_base import ResultAnalyzerBase
from . import environment as env


# Inference options: parameters for connect_model_zoo
CloudInference = 1  # use DeGirum cloud server for inference
AIServerInference = 2  # use AI server deployed in LAN/VPN
LocalHWInference = 3  # use locally-installed AI HW accelerator


def connect_model_zoo(
    inference_option: int = CloudInference,
) -> dg.zoo_manager.ZooManager:
    """Connect to model zoo according to given inference option.

    inference_option: should be one of CloudInference, AIServerInference, or LocalHWInference

    Returns model zoo accessor object
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


def _create_analyzing_postprocessor_class(
    analyzers: Union[ResultAnalyzerBase, List[ResultAnalyzerBase], None],
    model: Optional[dg.model.Model] = None,
):
    """Helper function to create postprocessor class entity which
    wraps original postprocessor and applies analyzers to its results.

    Args:
        analyzers: list of analyzers to apply to postprocessor results
        model: model object (to be used to deduce postprocessor type), optional

    Returns:
        AnalyzingPostprocessor class
    """

    class AnalyzingPostprocessor:
        def __init__(self, *args, **kwargs):
            if AnalyzingPostprocessor._postprocessor_type is not None:
                # create postprocessor of proper type
                self._result = AnalyzingPostprocessor._postprocessor_type(
                    *args, **kwargs
                )
            else:
                self._result = kwargs.get("result")

            # apply all analyzers to analyze result
            for analyzer in AnalyzingPostprocessor._analyzers:
                analyzer.analyze(self._result)

        @property
        def image_overlay(self):
            img = self._result.image_overlay
            if not isinstance(img, np.ndarray):
                raise Exception(
                    "Only OpenCV image backend is supported. Please set model.image_backend = 'opencv'"
                )
            # apply all analyzers to annotate overlay image
            for analyzer in AnalyzingPostprocessor._analyzers:
                img = analyzer.annotate(self._result, img)
            return img

        # delegate all other attributes to wrapped postprocessor
        def __getattr__(self, attr):
            return getattr(self._result, attr)

        def __dir__(self):
            return self._result.__dir__() + [
                d for d in self._result.__dict__ if not d.startswith("_")
            ]

        # store analyzers
        _analyzers = (
            analyzers
            if isinstance(analyzers, list)
            else ([analyzers] if analyzers is not None else [])
        )

        # deduce postprocessor type from model
        if model is not None:
            if model._custom_postprocessor is not None:
                _postprocessor_type = model._custom_postprocessor
                _was_custom = True
            else:
                _postprocessor_type = dg.postprocessor._inference_result_type(
                    model._model_parameters
                )()
                _was_custom = False
        else:
            _postprocessor_type = None
            _was_custom = False

    return AnalyzingPostprocessor


def attach_analyzers(
    model: dg.model.Model,
    analyzers: Union[ResultAnalyzerBase, List[ResultAnalyzerBase], None],
):
    """
    Attach analyzers to given model object.

    Args:
        model: Model object to attach analyzers to
        analyzers: List of analyzer objects to attach to model,
            or `None` to detach all analyzers if any were attached before

    Returns:
        Model object with attached analyzers
    """

    analyzing_postprocessor = _create_analyzing_postprocessor_class(analyzers, model)

    if analyzers:
        # attach custom postprocessor to model
        model._custom_postprocessor = analyzing_postprocessor
    else:
        # remove analyzing custom postprocessor from model if any
        if (
            model._custom_postprocessor is not None
            and isinstance(model._custom_postprocessor, type)
            and model._custom_postprocessor.__name__ == analyzing_postprocessor.__name__
        ):
            if model._custom_postprocessor._was_custom:
                model._custom_postprocessor = (
                    model._custom_postprocessor._postprocessor_type
                )
            else:
                model._custom_postprocessor = None

    return model


def predict_stream(
    model: dg.model.Model,
    video_source_id: Union[int, str, Path, None],
    *,
    fps: Optional[float] = None,
    analyzers: Union[ResultAnalyzerBase, List[ResultAnalyzerBase], None] = None,
):
    """Run a model on a video stream

    Args:
        model - model to run
        video_source_id - identifier of input video stream. It can be:
            - 0-based index for local cameras
            - IP camera URL in the format "rtsp://<user>:<password>@<ip or hostname>",
            - local path or URL to mp4 video file,
            - YouTube video URL
        fps - optional fps cap. If greater than the actual FPS, it will do nothing.
           If less than the current fps, it will decimate frames accordingly.
        analyzers - optional analyzer or list of analyzers to be applied to model inference results

    Returns:
        generator object yielding model prediction results.
        When `analyzers` is not None, each prediction result contains additional keys, added by those analyzers.
        Also prediction result object has overridden `image_overlay` method which additionally displays analyzers' annotations.
    """

    analyzing_postprocessor = _create_analyzing_postprocessor_class(analyzers)

    with open_video_stream(video_source_id) as stream:
        for res in model.predict_batch(video_source(stream, fps=fps)):
            if analyzers is not None:
                yield analyzing_postprocessor(result=res)
            else:
                yield res


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
    """Annotate video stream by running a model and saving results to video file

    Args:
        model - model to run
        video_source_id - identifier of input video stream. It can be:
        - cv2.VideoCapture object, already opened by open_video_stream()
        - 0-based index for local cameras
        - IP camera URL in the format "rtsp://<user>:<password>@<ip or hostname>",
        - local path or URL to mp4 video file,
        - YouTube video URL
        show_progress - when True, show text progress indicator
        visual_display - when True, show interactive video display with annotated video stream
        show_ai_overlay - when True, apply image overlay to frame; when False, use original frame
        fps - optional fps cap. If greater than the actual FPS, it will do nothing.
           If less than the current fps, it will decimate frames accordingly.
        analyzers - optional analyzer or list of analyzers to be applied to model inference results
    """

    win_name = f"Annotating {video_source_id}"

    analyzer_list = (
        analyzers
        if isinstance(analyzers, list)
        else ([analyzers] if analyzers is not None else [])
    )

    for analyzer in analyzer_list:
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

        # Overwrite the video stream's FPS if the fps argument is set.
        if fps:
            video_fps = fps

        writer = stack.enter_context(
            open_video_writer(str(output_video_path), w, h, video_fps)
        )

        if show_progress:
            progress = Progress(int(stream.get(cv2.CAP_PROP_FRAME_COUNT)))

        for res in model.predict_batch(video_source(stream, fps=video_fps)):
            img = res.image_overlay if show_ai_overlay else res.image

            for analyzer in analyzer_list:
                img = analyzer.analyze_and_annotate(res, img)

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


def model_time_profile(
    model: dg.model.Model, iterations: int = 100
) -> ModelTimeProfile:
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

    elapsed = 0.0
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
