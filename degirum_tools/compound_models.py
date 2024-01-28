#
# compound_models.py: compound model toolkit for PySDK samples
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes for multi-model aggregation.
#

import queue, cv2, numpy as np, degirum as dg
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Optional
from .image_tools import detect_motion
from .math_support import nms


@dataclass
class NmsOptions:
    """
    Options for non-maximum suppression (NMS) algorithm.
    """

    threshold: float  # IoU or IoS threshold for box clustering [0..1]
    use_iou: bool = True  # use IoU for box clustering (otherwise use IoS)
    merge_boxes: bool = False  # merge cluster boxes by score-weighted average


@dataclass
class MotionDetectOptions:
    """
    Options for motion detection algorithm.
    """

    threshold: float  # threshold for motion detection [0..1]: fraction of changed pixels in respect to frame size
    look_back: int = 1  # number of frames to look back to detect motion


class ModelLike(ABC):
    """
    A base class which provides a common interface for all models, similar to PySDK model class
    """

    @abstractmethod
    def predict_batch(self, data):
        """
        Perform whole inference lifecycle for all objects in given iterator object (for example, `list`).

        Args:
            data (iterator): Inference input data iterator object such as list or generator function.
            Each element returned by this iterator should be compatible to that regular PySDK model
            accepts

        Returns:
            Generator object which iterates over combined inference result objects.
            This allows you directly using the result in `for` loops.
        """

    def predict(self, data):
        """Perform whole inference lifecycle on a single frame.

        Args:

            data (any): Inference input data. Input data type depends on the model.
            It should be compatible to that regular PySDK model accepts.

        Returns:
            Combined inference result object.
        """
        for result in self.predict_batch([data]):
            return result
        return None

    def __call__(self, data):
        """Perform whole inference lifecycle on a single frame.

        Args:

            data (any): Inference input data. Input data type depends on the model.
            It should be compatible to that regular PySDK model accepts.

        Returns:
            Combined inference result object.
        """
        return self.predict(data)


@dataclass
class _FrameInfo:
    """Class to hold frame info"""

    result1: dg.postprocessor.InferenceResults  # model 1 result
    sub_result: int  # sub-result index in model 1 result


class CompoundModelBase(ModelLike):
    """
    Compound model class which combines two models into one pipeline.
    """

    class NonBlockingQueue(queue.Queue):
        """
        Specialized non-blocking queue which acts as iterator.
        """

        def __iter__(self):
            while True:
                try:
                    value = self.get_nowait()
                    if value is None:
                        break  # `None` sentinel signals end of queue
                    yield value
                except queue.Empty:
                    yield None  # in case of empty queue, yield None

    def __init__(self, model1, model2):
        """
        Constructor.

        Args:
            model1: PySDK model to be used for the first step of the pipeline
            model2: PySDK model to be used for the second step of the pipeline
        """
        self.model1 = model1
        self.model2 = model2
        self.queue = CompoundModelBase.NonBlockingQueue()

    @abstractmethod
    def queue_result1(self, result1):
        """
        Process result of the first model and put it into the queue.
        To be implemented in derived classes.

        Args:
            result1: prediction result of the first model
        """

    @abstractmethod
    def transform_result2(self, result2):
        """
        Transform result of the second model.
        To be implemented in derived classes.

        Args:
            result2: combined prediction result of the second model

        Returns:
            Transformed result to be returned as compound model result
        """

    def predict_batch(self, data):
        """
        Perform whole inference lifecycle for all objects in given iterator object (for example, `list`).

        Args:
            data (iterator): Inference input data iterator object such as list or generator function.
            Each element returned by this iterator should be compatible to that regular PySDK model
            accepts

        Returns:
            Generator object which iterates over combined inference result objects.
            This allows you directly using the result in `for` loops.
        """

        # use non-blocking mode for nested model and regular mode for the first model
        self.model1.non_blocking_batch_predict = False
        self.model2.non_blocking_batch_predict = True

        # iterator over predictions of nested model
        model2_iter = self.model2.predict_batch(self.queue)

        for result1 in self.model1.predict_batch(data):
            # put result of the first model into the queue
            self.queue_result1(result1)

            # process all recognized license plates ready so far
            while result2 := next(model2_iter):
                if (transformed_result2 := self.transform_result2(result2)) is not None:
                    yield transformed_result2

        self.queue.put(None)  # signal end of queue to nested model
        self.model2.non_blocking_batch_predict = False  # restore blocking mode
        # process all remaining recognized license plates
        for result2 in model2_iter:
            if (transformed_result2 := self.transform_result2(result2)) is not None:
                yield transformed_result2


class CombiningCompoundModel(CompoundModelBase):
    """
    Compound model class which combines two models into one pipeline simple way:
    both models are executed in parallel on the same input data and results are combined.

    Restriction: both models should have the same result types.
    """

    def queue_result1(self, result1):
        """
        Process result of the first model and put it into the queue.

        This implementation puts the original image into the queue,
        supplying result as a frame info.

        Args:
            result1: prediction result of the first model
        """
        self.queue.put((result1.image, _FrameInfo(result1, -1)))

    def transform_result2(self, result2):
        """
        Transform result of the second model.

        This implementation appends result of the first model to the result of the second model.

        Args:
            result2: prediction result of the second model

        Returns:
            Combined result of both models. It's `info` property is the result of the first model.
        """
        result2.results.extend(result2.info.result1.results)
        return result2


class CroppingCompoundModel(CompoundModelBase):
    """
    Compound model class which crops original image according to results of the first model
    and then passes cropped images to the second model.
    It returns the result of the second model as a compound model result.

    Restriction: first model should be of object detection type.
    """

    def __init__(self, model1, model2, crop_extent=0):
        """
        Constructor.

        Args:
            model1: PySDK object detection model
            model2: PySDK classification model
            crop_extent: extent of cropping in percent of bbox size
        """

        super().__init__(model1, model2)
        self._crop_extent = crop_extent

    def queue_result1(self, result1):
        """
        Process result of the first model and put it into the queue.

        This implementation puts the original image into the queue,
        supplying result as a frame info.

        Args:
            result1: prediction result of the first model
        """
        if len(result1.results) == 0 or "bbox" not in result1.results[0]:
            # no bbox detected: put black image into the queue to keep things going
            self.queue.put(
                (np.zeros((2, 2, 3), dtype=np.uint8), _FrameInfo(result1, -1))
            )
        else:
            image_size = self.image_size(result1.image)
            for idx, obj in enumerate(result1.results):
                adj_bbox = self._adjust_bbox(obj["bbox"], image_size)
                if hasattr(result1.image, "crop"):
                    cropped_img = result1.image.crop(adj_bbox)
                else:
                    cropped_img = result1.image[
                        adj_bbox[1] : adj_bbox[3], adj_bbox[0] : adj_bbox[2]
                    ]
                self.queue.put((cropped_img, _FrameInfo(result1, idx)))

    def transform_result2(self, result2):
        """
        Transform result of the second model.

        This implementation appends result of the first model to the result of the second model.

        Args:
            result2: prediction result of the second model

        Returns:
            Result of second model. It's `info` property is the result of the first model.
        """
        return result2

    def image_size(self, image):
        """Get image size in (width, height) format"""
        if hasattr(image, "shape"):
            return (image.shape[1], image.shape[0])
        elif hasattr(image, "size"):
            return image.size
        else:
            raise Exception("Unsupported image type")

    def _adjust_bbox(self, bbox, image_size):
        """
        Inflate bbox coordinates to crop extent and adjust to image size

        Args:
            bbox: bbox coordinates in [x1, y1, x2, y2] format
            image_size: image size in (width, height) format
        """
        scale = self._crop_extent * 0.01 * 0.5
        maxval = [image_size[0], image_size[1], image_size[0], image_size[1]]
        dx = (bbox[2] - bbox[0]) * scale
        dy = (bbox[3] - bbox[1]) * scale
        adjust = [-dx, -dy, dx, dy]
        return [min(maxval[i], max(0, round(bbox[i] + adjust[i]))) for i in range(4)]


class CroppingAndClassifyingCompoundModel(CroppingCompoundModel):
    """
    Compound model class which crops original image according to results of the first model
    and then passes cropped images to the second classification model, which adjusts bbox labels
    according to classification results.
    It returns the result of the **first** model where bbox labels are patched with
    label results of the second model.

    Restriction: first model should be of object detection type,
    second model should be of classification type.
    """

    def __init__(self, model1, model2, crop_extent=0):
        """
        Constructor.

        Args:
            model1: PySDK object detection model
            model2: PySDK classification model
            crop_extent: extent of cropping in percent of bbox size
        """

        if model1.image_backend != model2.image_backend:
            raise Exception(
                f"Image backends of both models should be the same, but got {model1.image_backend} and {model2.image_backend}"
            )

        super().__init__(model1, model2, crop_extent)
        self._current_result = None

    def transform_result2(self, result2):
        """
        Transform result of the second model.

        This implementation adjusts bbox labels detected by the first model according to
        classification results of the second model.

        Args:
            result2: prediction result of the second model

        Returns:
            Result of the first model with patched labels according to
            classification results of the second model.
        """

        result1 = result2.info.result1
        idx = result2.info.sub_result

        if idx >= 0:
            # patch bbox label with recognized class label
            label = (
                result2.results[0]["label"] if self.model2.overlay_show_labels else ""
            )
            if self.model2.overlay_show_probabilities:
                score = result2.results[0]["score"]
                score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                result1.results[idx]["label"] = (
                    label + (": " if label else "") + score_str
                )
            else:
                result1.results[idx]["label"] = label

        # return result when frame changes
        ret = None
        if result1 is not self._current_result:
            if self._current_result is not None:
                ret = self._current_result
            self._current_result = result1
        return ret

    def predict_batch(self, data):
        """
        Perform whole inference lifecycle for all objects in given iterator object (for example, `list`).

        Args:
            data (iterator): Inference input data iterator object such as list or generator function.
            Each element returned by this iterator should be compatible to that regular PySDK model
            accepts

        Returns:
            Generator object which iterates over combined inference result objects.
            This allows you directly using the result in `for` loops.
        """
        for result in super().predict_batch(data):
            yield result
        if self._current_result is not None:
            yield self._current_result


class CroppingAndDetectingCompoundModel(CroppingCompoundModel):
    """
    Compound model class which crops original image according to results of the first model
    and then passes cropped images to the second detection model, which performs detections
    in each bbox and combines detection results from multiple bboxes from the same frame.
    It returns the combined results of the **second** model where bbox coordinates are translated
    to original image coordinates.

    Restriction: first model should be of object detection type
    (or pseudo object detection type like `RegionExtractionPseudoModel),
    second model should be of object detection type.
    """

    def __init__(
        self,
        model1,
        model2,
        *,
        crop_extent=0,
        add_model1_results=False,
        nms_options: Optional[NmsOptions] = None,
    ):
        """
        Constructor.

        Args:
            model1: PySDK object detection model
            model2: PySDK object detection model
            crop_extent: extent of cropping in percent of bbox size
            add_model1_results: True to add detections of model1 to the combined result
        """

        if model1.image_backend != model2.image_backend:
            raise Exception(
                f"Image backends of both models should be the same, but got {model1.image_backend} and {model2.image_backend}"
            )

        super().__init__(model1, model2, crop_extent)
        self._current_result: Optional[dg.postprocessor.InferenceResults] = None
        self._current_result1: Optional[dg.postprocessor.InferenceResults] = None
        self._add_model1_results = add_model1_results
        self._nms_options: Optional[NmsOptions] = nms_options

    def transform_result2(self, result2):
        """
        Transform result of the second model.

        This implementation combines results of the **second** model over all bboxes detected by the first model,
        translating bbox coordinates to original image coordinates.

        Args:
            result2: detection result of the second model

        Returns:
            Combined results of the **second** model over all bboxes detected by the first model,
            where bbox coordinates are translated to original image coordinates.
        """

        result1 = result2.info.result1
        idx = result2.info.sub_result

        if idx >= 0:
            # adjust bbox coordinates to original image coordinates
            x, y = result1.results[idx]["bbox"][:2]
            for r in result2._inference_results:
                r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()
            if self._add_model1_results:
                # prepend result from the first model to the combined result if requested
                result2._inference_results.insert(0, result1.results[idx])

        ret = None
        if result1 is self._current_result1:
            # frame continues: append second model results to the combined result
            if self._current_result is not None and idx >= 0:
                self._current_result._inference_results.extend(
                    result2._inference_results
                )

        else:
            # new frame comes
            if self._current_result is not None:
                # patch combined result image to be original image
                self._current_result._input_image = result1.image

                if self._nms_options is not None:
                    # apply NMS to combined result
                    nms(
                        self._current_result,
                        iou_threshold=self._nms_options.threshold,
                        use_iou=self._nms_options.use_iou,
                        merge_boxes=self._nms_options.merge_boxes,
                    )

                # return combined result of previous frame
                ret = self._current_result

            self._current_result = result2
            self._current_result1 = result1

        return ret

    def predict_batch(self, data):
        """
        Perform whole inference lifecycle for all objects in given iterator object (for example, `list`).

        Args:
            data (iterator): Inference input data iterator object such as list or generator function.
            Each element returned by this iterator should be compatible to that regular PySDK model
            accepts

        Returns:
            Generator object which iterates over combined inference result objects.
            This allows you directly using the result in `for` loops.
        """
        for result in super().predict_batch(data):
            yield result
        if self._current_result is not None:
            yield self._current_result


class RegionExtractionPseudoModel(ModelLike):
    """
    Pseudo model class which extracts regions from given image according to given ROI boxes.
    """

    def __init__(
        self,
        roi_list: Union[list, np.ndarray],
        model2: dg.model.Model,
        *,
        motion_detect: Optional[MotionDetectOptions] = None,
    ):
        """
        Constructor.

        Args:
            roi_list: list of ROI boxes in [x1, y1, x2, y2] format
            model2: model, which will be used as a second step of the compound model pipeline
            motion_detect: motion detection options.
                When None, motion detection is disabled.
                When enabled, ROI boxes where motion is not detected will be skipped.
        """
        self._roi_list = roi_list
        self._model2 = model2
        self._base_img: list = []  # base image for motion detection
        self._motion_detect = motion_detect

    # fallback all model-like attributes to the second model
    def __getattr__(self, attr):
        return getattr(self._model2, attr)

    def predict_batch(self, data):
        """
        Perform whole inference lifecycle for all objects in given iterator object (for example, `list`).

        Args:
            data (iterator): Inference input data iterator object such as list or generator function.
            Each element returned by this iterator should be compatible to that regular PySDK model
            accepts

        Returns:
            Generator object which iterates over combined inference result objects.
            This allows you directly using the result in `for` loops.
        """

        preprocessor = dg._preprocessor.create_image_preprocessor(
            self._model2.model_info,  # we do copy here to avoid modifying original model parameters
            image_backend=self._model2.image_backend,
            pad_method="",  # to disable resizing/padding
        )
        preprocessor.image_format = "RAW"  # to avoid unnecessary JPEG encoding

        all_rois = [True] * len(self._roi_list)

        for element in data:
            # extract frame and frame info from data
            if isinstance(element, tuple):
                # if data is tuple, we treat first element as frame data and second element as frame info
                frame, frame_info = element
            else:
                # otherwise we treat data as frame data and if it is string, we set frame info equal to frame data
                frame, frame_info = element, element if isinstance(element, str) else ""

            # do pre-processing
            preprocessed_data = preprocessor.forward(frame)

            image = preprocessed_data["image_input"]

            if self._motion_detect is not None:
                motion_img, base_img = detect_motion(
                    self._base_img[0] if self._base_img else None, image
                )
                self._base_img.append(base_img)
                if len(self._base_img) > int(self._motion_detect.look_back):
                    self._base_img.pop(0)

                if motion_img is None:
                    motion_detected = all_rois
                else:
                    motion_detected = [
                        cv2.countNonZero(motion_img[roi[1] : roi[3], roi[0] : roi[2]])
                        > self._motion_detect.threshold
                        * (roi[2] - roi[0])
                        * (roi[3] - roi[1])
                        for roi in self._roi_list
                    ]
            else:
                motion_detected = all_rois

            roi_list = [
                {"bbox": roi, "label": "ROI", "score": idx, "category_id": idx}
                for idx, roi in enumerate(self._roi_list)
                if motion_detected[idx]
            ]

            # generate pseudo inference results
            result = dg.postprocessor.DetectionResults(
                model_params=self._model2._model_parameters,
                input_image=image,
                model_image=image,
                inference_results=roi_list,
                draw_color=self._model2.overlay_color,
                line_width=self._model2.overlay_line_width,
                show_labels=self._model2.overlay_show_labels,
                show_probabilities=self._model2.overlay_show_probabilities,
                alpha=self._model2.overlay_alpha,
                font_scale=self._model2.overlay_font_scale,
                fill_color=self._model2.input_letterbox_fill_color,
                frame_info=frame_info,
                conversion=lambda x, y: (x, y),
                label_dictionary=self._model2.label_dictionary,
            )
            yield result
