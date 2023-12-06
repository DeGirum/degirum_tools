#
# inference_support.py: classes and functions for AI inference
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes and functions to handle AI inferences
#

import cv2, numpy as np
from contextlib import ExitStack
from dataclasses import dataclass
from .video_support import open_video_stream, video_source, open_video_writer
from .ui_support import Progress, Display, Timer
from .zone_count import ZoneCounter
from typing import Union, Optional
from pathlib import Path
from . import environment as env
import degirum as dg  # import DeGirum PySDK


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


def predict_stream(
    model: dg.model.Model,
    input_video_id: Union[int, str, Path, None],
    *,
    zone_counter: Optional[ZoneCounter] = None,
):
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

    with open_video_stream(input_video_id) as stream:
        for res in model.predict_batch(video_source(stream)):
            if zone_counter is not None:

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
    model: dg.model.Model,
    input_video_id: Union[int, str, Path, None],
    output_video_path: str,
    *,
    show_progress: bool = True,
    visual_display: bool = True,
    zone_counter: Optional[ZoneCounter] = None,
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

    if zone_counter is not None:
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

            if zone_counter is not None:
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
