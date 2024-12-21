#
# streams_gizmos.py: streaming toolkit: gizmos implementation
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements various gizmos for streaming pipelines
#

import queue, copy, time, cv2, numpy as np
import degirum as dg
from abc import abstractmethod
from typing import Optional, List, Union
from contextlib import ExitStack
from .streams_base import Stream, StreamData, StreamMeta, Gizmo
from .environment import get_test_mode
from .video_support import open_video_stream, open_video_writer
from .ui_support import Display
from .image_tools import crop_image, resize_image, image_size, to_opencv
from .result_analyzer_base import image_overlay_substitute
from .crop_extent import CropExtentOptions, extend_bbox
from .notifier import EventNotifier
from .event_detector import EventDetector
from .environment import get_token

#
# predefined meta tags
#
tag_video = "dgt_video"  # tag for video source data
tag_resize = "dgt_resize"  # tag for resizer result
tag_inference = "dgt_inference"  # tag for inference result
tag_preprocess = "dgt_preprocess"  # tag for preprocessor result
tag_crop = "dgt_crop"  # tag for cropping result
tag_analyzer = "dgt_analyzer"  # tag for analyzer result


def clone_result(result):
    """Create a clone of PySDK result object. Clone inherits image references, but duplicates inference results."""
    clone = copy.copy(result)
    clone._inference_results = copy.deepcopy(result._inference_results)
    return clone


class VideoSourceGizmo(Gizmo):
    """OpenCV-based video source gizmo"""

    # meta keys
    key_frame_width = "frame_width"  # frame width
    key_frame_height = "frame_height"  # frame height
    key_fps = "fps"  # stream frame rate
    key_frame_count = "frame_count"  # total stream frame count
    key_frame_id = "frame_id"  # frame index
    key_timestamp = "timestamp"  # frame timestamp

    def __init__(self, video_source=None, *, stop_composition_on_end: bool = False):
        """Constructor.

        - video_source: cv2.VideoCapture-compatible video source designator
        - stop_composition_on_end: stop composition when video source is over
        """
        super().__init__()
        self._video_source = video_source
        self._stop_composition_on_end = stop_composition_on_end and not get_test_mode()

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_video]

    def run(self):
        """Run gizmo"""
        with open_video_stream(self._video_source) as src:

            meta = {
                self.key_frame_width: int(src.get(cv2.CAP_PROP_FRAME_WIDTH)),
                self.key_frame_height: int(src.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                self.key_fps: src.get(cv2.CAP_PROP_FPS),
                self.key_frame_count: int(src.get(cv2.CAP_PROP_FRAME_COUNT)),
            }

            while not self._abort:
                ret, data = src.read()
                if not ret:
                    break
                else:
                    meta2 = copy.copy(meta)
                    meta2[self.key_frame_id] = self.result_cnt
                    meta2[self.key_timestamp] = time.time()
                    self.send_result(
                        StreamData(data, StreamMeta(meta2, self.get_tags()))
                    )

            if self._stop_composition_on_end:
                # stop composition if video source is over;
                # needed to stop other branches of the pipeline, like video sources, which are not over yet
                if self.composition is not None and not self._abort:
                    self.composition.stop()


class VideoDisplayGizmo(Gizmo):
    """OpenCV-based video display gizmo"""

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

        - window_titles: window title string or array of title strings for multiple displays
        - show_fps: True to show FPS
        - show_ai_overlay: True to show AI inference overlay image when possible
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        - multiplex: then True, single input data is multiplexed to multiple displays;
          when False, each input is displayed on individual display
        """

        if isinstance(window_titles, str):
            window_titles = [
                window_titles,
            ]

        inp_cnt = 1 if multiplex else len(window_titles)
        super().__init__([(stream_depth, allow_drop)] * inp_cnt)
        self._window_titles = window_titles
        self._show_fps = show_fps
        self._show_ai_overlay = show_ai_overlay
        if multiplex and allow_drop:
            raise Exception("Frame multiplexing with frame dropping is not supported")
        self._multiplex = multiplex

    def run(self):
        """Run gizmo"""

        with ExitStack() as stack:
            ndisplays = len(self._window_titles)
            ninputs = len(self.get_inputs())

            test_mode = get_test_mode()
            displays = [
                stack.enter_context(Display(w, self._show_fps))
                for w in self._window_titles
            ]
            first_run = [True] * ndisplays

            di = -1  # di is display index
            try:
                while True:
                    if self._abort:
                        break

                    for ii, input in enumerate(self.get_inputs()):  # ii is input index
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

                        # select display to show this frame
                        di = (di + 1) % ndisplays if self._multiplex else ii

                        if data == Stream._poison:
                            self._abort = True
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
    """OpenCV-based gizmo to save video to a file"""

    def __init__(
        self,
        filename: str,
        *,
        show_ai_overlay=False,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - filename: output video file name
        - show_ai_overlay: True to show AI inference overlay image when possible
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])
        self._filename = filename
        self._show_ai_overlay = show_ai_overlay

    def run(self):
        """Run gizmo"""

        def get_img(data: StreamData) -> np.ndarray:
            frame = data.data
            if self._show_ai_overlay:
                inference_meta = data.meta.find_last(tag_inference)
                if inference_meta:
                    frame = inference_meta.image_overlay

            return to_opencv(frame)

        img = get_img(self.get_input(0).get())
        w, h = image_size(img)
        with open_video_writer(self._filename, w, h) as writer:
            self.result_cnt += 1
            writer.write(img)
            for data in self.get_input(0):
                if self._abort:
                    break
                writer.write(get_img(data))


class ResizingGizmo(Gizmo):
    """OpenCV-based image resizing/padding gizmo"""

    # meta keys
    key_frame_width = "frame_width"  # frame width
    key_frame_height = "frame_height"  # frame height
    key_pad_method = "pad_method"  # padding method
    key_resize_method = "resize_method"  # resampling method

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

        - w, h: resulting image width/height
        - pad_method: padding method - one of "stretch", "letterbox", "crop-first", "crop-last"
        - resize_method: resampling method - one of "nearest", "bilinear", "area", "bicubic", "lanczos"
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])
        self._h = h
        self._w = w
        self._pad_method = pad_method
        self._resize_method = resize_method

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_resize]

    def _resize(self, image):
        return resize_image(
            image,
            self._w,
            self._h,
            resize_method=self._resize_method,
            pad_method=self._pad_method,
        )

    def run(self):
        """Run gizmo"""

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
    """Base class for AI inference gizmos"""

    def __init__(
        self,
        model: Union[dg.model.Model, str],
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
        inp_cnt: int = 1,
        **kwargs,
    ):
        """Constructor.

        - model: PySDK model object or model name string
            (in later case you need to specify all model loading parameters exactly as in `degirum.load_model()` function)
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)] * inp_cnt)

        if isinstance(model, str):
            if "token" not in kwargs:
                self.model = dg.load_model(model, token=get_token(""), **kwargs)
            else:
                self.model = dg.load_model(model, **kwargs)
        else:
            self.model = model

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_inference]

    def run(self):
        """Run gizmo"""

        def source():
            has_data = True
            while has_data:
                # get data from all inputs
                for inp in self.get_inputs():
                    d = inp.get()
                    if d == Stream._poison:
                        has_data = False
                        break
                    yield (d.data, d.meta)

                if self._abort:
                    break

        for result in self.model.predict_batch(source()):
            meta = result.info
            if isinstance(result._input_image, bytes):
                # most likely, we have preprocessing gizmo in the pipeline
                preprocess_meta = meta.find_last(tag_preprocess)
                if preprocess_meta:
                    # indeed, we have preprocessing gizmo in the pipeline

                    # patch raw bytes image in result when possible to provide better result visualization
                    result._input_image = preprocess_meta[
                        AiPreprocessGizmo.key_image_input
                    ]

                    # recalculate bbox coordinates to original image
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
        """Result handler to be overloaded in derived classes.

        - result: inference result; result.info contains reference to data frame used for inference
        """


class AiSimpleGizmo(AiGizmoBase):
    """AI inference gizmo with no result processing: it passes through input frames
    attaching inference results as meta info"""

    def on_result(self, result):
        """Result handler to be overloaded in derived classes.

        - result: inference result; result.info contains reference to data frame used for inference
        """
        new_meta = result.info.clone()
        new_meta.append(result, self.get_tags())
        self.send_result(StreamData(result.image, new_meta))


class AiObjectDetectionCroppingGizmo(Gizmo):
    """A gizmo, which receives object detection results, then for each detected object
    it crops input image and sends cropped result.

    Output image is the crop of original image.
    Output meta-info is a dictionary with the following keys:

    - "original_result": reference to original AI object detection result
    - "cropped_result": reference to sub-result for particular crop
    - "cropped_index": the number of that sub-result
    - "is_last_crop": the flag indicating that this is the last crop in the frame
    The last two key are present only if at least one object is detected in a frame.

    The validate_bbox() method can be overloaded in derived classes to filter out
    undesirable objects.
    """

    # meta keys
    key_original_result = "original_result"  # original AI object detection result
    key_cropped_result = "cropped_result"  # sub-result for particular crop
    key_cropped_index = "cropped_index"  # the number of that sub-result
    key_is_last_crop = "is_last_crop"  # 'last crop in the frame' flag

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

        - labels: list of class labels to process
        - send_original_on_no_objects: send original frame when no objects detected
        - crop_extent: extent of cropping in percent of bbox size
        - crop_extent_option: method of applying extending crop to the input image
        - crop_aspect_ratio: desired aspect ratio of the cropped image (W/H)
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """

        # we use unlimited frame queue #0 to not block any source gizmo
        super().__init__([(stream_depth, allow_drop)])

        self._labels = labels
        self._send_original_on_no_objects = send_original_on_no_objects
        self._crop_extent = crop_extent
        self._crop_extent_option = crop_extent_option
        self._crop_aspect_ratio = crop_aspect_ratio

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_crop]

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            img = data.data
            result = data.meta.find_last(tag_inference)
            if result is None:
                raise Exception(
                    f"{self.__class__.__name__}: inference meta not found: you need to have object detection gizmo in upstream"
                )

            # prepare all crops
            crops = []
            crop_metas = []
            for i, r in enumerate(result.results):

                bbox = r.get("bbox")
                if not bbox:
                    continue

                # discard objects which do not pass validation
                if not self.validate_bbox(result, i):
                    continue

                # apply crop extent
                ext_bbox = extend_bbox(
                    bbox,
                    self._crop_extent_option,
                    self._crop_extent,
                    self._crop_aspect_ratio,
                    image_size(result.image),
                )

                crops.append(crop_image(result.image, ext_bbox))

                cropped_obj = copy.deepcopy(r)
                cropped_obj["bbox"] = ext_bbox

                crop_meta = {
                    self.key_original_result: result,
                    self.key_cropped_result: cropped_obj,
                    self.key_cropped_index: i,
                    self.key_is_last_crop: False,  # will adjust later
                }
                new_meta = data.meta.clone()
                new_meta.append(crop_meta, self.get_tags())
                crop_metas.append(new_meta)

            if crop_metas:
                # adjust last crop flag
                crop_metas[-1].find_last(tag_crop)[self.key_is_last_crop] = True
                # and send all croped results
                for crop, meta in zip(crops, crop_metas):
                    self.send_result(StreamData(crop, meta))

            elif self._send_original_on_no_objects:
                # no objects detected: send original result
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
        """Validate detected object. To be overloaded in derived classes.

        - result: inference result
        - idx: index of object within `result.results[]` list to validate

        Returns True if object is accepted to be used for crop, False otherwise
        """
        return True


class CropCombiningGizmo(Gizmo):
    """Gizmo to combine results coming after AiObjectDetectionCroppingGizmo cropping gizmo(s).
    It has N+1 inputs: one for a stream of original frames, and N other inputs for streams of cropped
    and AI-processed results. It combines results from all crop streams
    and attaches them to the original frames.
    """

    # key for extra results list in inference detection results
    key_extra_results = "extra_results"

    def __init__(
        self,
        crop_inputs_num: int = 1,
        *,
        stream_depth: int = 10,
    ):
        """Constructor.

        - crop_inputs_num: number of crop inputs
        - stream_depth: input stream depth for crop inputs
        """

        # input 0 is for original frames, it should be unlimited depth
        # we never drop frames in combiner to avoid mis-synchronization between original and crop streams
        input_def = [(0, False)] + [(stream_depth, False)] * crop_inputs_num
        super().__init__(input_def)

    def run(self):
        """Run gizmo"""

        frame_input = self.get_input(0)
        crops: List[StreamData] = []
        combined_meta: Optional[StreamMeta] = None

        for full_frame in frame_input:
            if self._abort:
                break

            # get index of full frame
            video_meta = full_frame.meta.find_last(tag_video)
            if video_meta is None:
                raise Exception(
                    f"{self.__class__.__name__}: video meta not found: you need to have {VideoSourceGizmo.__class__.__name__} in upstream of input 0"
                )
            frame_id = video_meta[VideoSourceGizmo.key_frame_id]

            # collect all crops for this frame
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

                # get index of cropped frame
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
                    # crop is for the next frame: send full frame
                    self.send_result(full_frame)
                    break

                if crop_frame_id < frame_id:
                    # crop is for the previous frame: skip it
                    crops = []
                    continue

                # at this point we do have both crop and after-crop inference metas for the current frame
                # even if there are no real crops (cropped_index==-1), because we just checked that
                # all crop_frame_ids are equal to frame_id, so crop inference was not skipped

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
                if combined_meta is None:
                    combined_meta = crops[0].meta.clone()
                    # remove all crop-related metas
                    combined_meta.remove_last(tag_crop)
                    combined_meta.remove_last(tag_inference)
                    # append original object detection result clone
                    combined_meta.append(self._clone_result(orig_result), tag_inference)

                # append all crop inference results to the combined meta
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
                    combined_meta = None  # mark combined_meta as processed
                    break

    def _adjust_results(
        self, orig_result, bbox_idx: int, cropped_results: list
    ) -> list:
        """Adjust inference results for the crop: recalculates bbox coordinates to original image,
        attach original image to the result, and return the list of adjusted results"""

        bbox = orig_result._inference_results[bbox_idx].get("bbox")
        assert bbox
        tl = bbox[:2]

        ret = []
        for crop_res in cropped_results:
            cr = clone_result(crop_res)
            # adjust all found coordinates to original image
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
        """Clone inference result object with deepcopy of `_inference_results` list"""

        def _overlay_extra_results(result):
            """Produce image overlay with drawing all extra results"""

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
    """Gizmo to combine results from multiple AI gizmos with similar-typed results"""

    def __init__(
        self,
        inp_cnt: int,
        *,
        stream_depth: int = 10,
    ):
        """Constructor.

        - inp_cnt: number of inputs to combine
        - stream_depth: input stream depth
        """
        self._inp_cnt = inp_cnt
        super().__init__([(stream_depth, False)] * inp_cnt)

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_inference]

    def run(self):
        """Run gizmo"""

        while True:
            # get data from all inputs
            all_data = []
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
                    continue  # no inference results

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
    """Preprocessing gizmo. It applies AI model preprocessor to the input images.

    Output data is the result of preprocessing: raw bytes array to be sent to AI model.
    Output meta-info is a dictionary with the following keys
        "image_input": reference to the input image
        "converter": coordinate conversion lambda
        "image_result": pre-processed image (optional, only when model.save_model_image is set)
    """

    key_image_input = "image_input"  # reference to the input image
    key_converter = "converter"  # coordinate conversion lambda
    key_image_result = "image_result"  # pre-processed image

    def __init__(
        self,
        model,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.
        - model: PySDK model object
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])
        self._preprocessor = model._preprocessor

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_preprocess]

    def run(self):
        """Run gizmo"""
        for data in self.get_input(0):
            if self._abort:
                break

            res = self._preprocessor.forward(data.data)
            new_meta = data.meta.clone()
            new_meta.append(res[1], self.get_tags())
            self.send_result(StreamData(res[0], new_meta))


class AiAnalyzerGizmo(Gizmo):
    """Gizmo to apply a chain of analyzers to the inference result"""

    def __init__(
        self,
        analyzers: list,
        *,
        filters: Optional[set] = None,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - analyzers: list of analyzer objects
        - filters: a set of event names or notifications. When filters are specified, only results, which have
            at least one of the specified events and/or notifications, are passed through, other results are discarded.
            Event names are specified in the configuration of `EventDetector` analyzers, and notification names are
            specified in the configuration of `EventNotifier` analyzers. Filtering feature is useful when you need to
            suppress unwanted results to reduce the load on the downstream gizmos.
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """

        self._analyzers = analyzers
        self._filters = filters
        super().__init__([(stream_depth, allow_drop)])

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_inference, tag_analyzer]

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            filter_ok = True
            new_meta = data.meta.clone()
            inference_meta = new_meta.find_last(tag_inference)
            if inference_meta is not None and isinstance(
                inference_meta, dg.postprocessor.InferenceResults
            ):
                inference_clone = clone_result(inference_meta)
                for analyzer in self._analyzers:
                    analyzer.analyze(inference_clone)
                image_overlay_substitute(inference_clone, self._analyzers)
                new_meta.append(inference_clone, self.get_tags())

                # check filters
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

        for analyzer in self._analyzers:
            analyzer.finalize()


class SinkGizmo(Gizmo):
    """Gizmo to receive results and accumulate them in the queue. This queue can be used as a
    generator source for further processing in the main thread."""

    def __init__(
        self,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """

        super().__init__([(stream_depth, allow_drop)])

    def run(self):
        """Run gizmo"""
        # return immediately to not to consume thread resources
        return

    def __call__(self):
        """Return input queue. To be used in for-loops to get results"""
        return self._inputs[0]
