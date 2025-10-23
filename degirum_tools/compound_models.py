# compound_models.py: compound model toolkit for PySDK samples
# Copyright DeGirum Corporation 2023
# All rights reserved
# Implements classes for multi-model aggregation.

"""
Compound Models Module Overview
======================

This module provides a toolkit for creating compound models using the DeGirum PySDK.

A compound model orchestrates multiple underlying models into a pipeline to enable complex inference scenarios. Common examples include:

- Detecting objects and then classifying each detected object.
- Running coarse detection first, then applying a refined detection model on specific regions.
- Combining outputs from multiple independent models into a unified inference result.

Compound models run in a single thread and are intended primarily for simple usage scenarios. Compound models still provide efficient batch prediction pipelining using batch_predict() in non-blocking mode.
For more performant applications requiring better scalability and more flexible connections, we recommend using Gizmos, which in multiple threads.

## Key Concepts

- **Model Composition**:
  Compound models sequentially (or concurrently) invoke multiple models. Typically, results from the first model (e.g., bounding boxes from detection) feed into subsequent models (classification or refined detection).

- **Pipeline Workflow**:
  A typical workflow involves:
    1. Run `model1` to identify regions of interest (ROIs).
    2. Crop these ROIs and run them through `model2`.
    3. Integrate or transform outputs from `model2` back into the original context.

- **Unified Model Interface**:
  All compound models follow the same interface as regular models in DeGirum SDK, providing `.predict()` for single frames and `.predict_batch()` for iterators of frames.

## Included Compound Models

- **CombiningCompoundModel**:
  Combines results from two models run concurrently on the same input.

- **CroppingCompoundModel**:
  Crops regions identified by `model1` and feeds them into `model2`.

- **CroppingAndClassifyingCompoundModel**:
  Specialized pipeline: object detection (`model1`) followed by classification (`model2`) of each detected object.

- **CroppingAndDetectingCompoundModel**:
  Pipeline for refined detection: initial coarse detection (`model1`) followed by detailed detection (`model2`) within each ROI.

- **RegionExtractionPseudoModel**:
  Extracts predefined regions of interest without actual inference, optionally filtering by motion detection.

## Basic Usage Examples

**Detection + Classification**:
```python
from degirum_tools import ModelSpec, remote_assets
from degirum_tools.compound_models import CroppingAndClassifyingCompoundModel

# Describe the individual models once
detector_spec = ModelSpec(
    model_name="<your_detection_model>",
    inference_host_address="@cloud",  # Can be '@cloud', host:port, or '@local'
    zoo_url="degirum/degirum",
)

classifier_spec = ModelSpec(
    model_name="<your_classification_model>",
    inference_host_address="@cloud",
    zoo_url="degirum/degirum",
)

with detector_spec.load_model() as detector, classifier_spec.load_model() as classifier:
    # Creating a compound model pipeline
    compound_model = CroppingAndClassifyingCompoundModel(detector, classifier)

    # Single frame inference using predict()
    print("Using predict():")
    single_result = compound_model(remote_assets.cat)
    print(single_result)

    # Batch inference using predict_batch()
    print("Using predict_batch():")
    for batch_result in compound_model.predict_batch(
        [remote_assets.cat, remote_assets.two_cats]
    ):
        print(batch_result)
```

**Detection + Detection**:
```python
from degirum_tools import ModelSpec, remote_assets
from degirum_tools.compound_models import CombiningCompoundModel

# Describe the detectors up front
detector1_spec = ModelSpec(
    model_name="<your_first_detection_model>",
    inference_host_address="@cloud",  # Can be '@cloud', host:port, or '@local'
    zoo_url="degirum/degirum",
)

detector2_spec = ModelSpec(
    model_name="<your_second_detection_model>",
    inference_host_address="@cloud",
    zoo_url="degirum/degirum",
)

with detector1_spec.load_model() as detector1, detector2_spec.load_model() as detector2:
    # Creating a compound model that merges results from both detectors
    compound_detector = CombiningCompoundModel(detector1, detector2)

    # Single frame inference using predict()
    print("Using predict():")
    single_result = compound_detector(remote_assets.cat)
    print(single_result.results)

    # Batch inference using predict_batch()
    print("Using predict_batch():")
    for batch_result in compound_detector.predict_batch(
        [remote_assets.cat, remote_assets.two_cats]
    ):
        print(batch_result.results)
```

See class-level documentation below for details on individual classes and additional configuration options.
"""

import queue
import cv2
import numpy as np
import degirum as dg
import inspect
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Optional, List
from .image_tools import detect_motion
from .math_support import nms, NmsBoxSelectionPolicy
from .result_analyzer_base import (
    ResultAnalyzerBase,
    image_overlay_substitute,
    clone_result,
)
from .crop_extent import CropExtentOptions, extend_bbox
from .image_tools import crop_image, image_size


class ModelLike(ABC):
    """
    A base class which provides a common interface for all models, similar to PySDK model class.

    When calling `predict_batch(data)`, each item in `data` can be:
         - A single frame (image/array/etc.), or
         - A 2-element tuple in the form `(frame, frame_info)`.

    The `frame_info` object (of any type) then appears in the final `InferenceResults.info`
    attribute, allowing you to carry custom metadata through the pipeline.
    """

    @abstractmethod
    def predict_batch(self, data):
        """
        Perform a whole inference lifecycle for all objects in the given iterator object (for example, `list`).

        Each item in `data` can be a single frame (any type acceptable to the model) or
        a 2-element tuple `(frame, frame_info)`. In the latter case, `frame_info` is
        carried through and placed in `InferenceResults.info` for that frame.

        Args:
            data (iterator):
                Inference input data iterator object such as a list or a generator function.
                Each element returned by this iterator should be compatible with what
                regular PySDK models accept.

        Returns:
            (Iterator[degirum.postprocessor.InferenceResults or None]):
                A generator or iterator over the inference result objects (or None in non-blocking mode).
                This allows you to use the result in `for` loops.
        """

    def predict(self, data):
        """
        Perform a whole inference lifecycle on a single frame.

        Args:
            data (any):
                Inference input data, typically an image or array, or a tuple `(frame, frame_info)`.

        Returns:
            (degirum.postprocessor.InferenceResults or None):
                The combined inference result object, or None if no result.
        """
        for result in self.predict_batch([data]):
            return result
        return None

    def __call__(self, data):
        """
        Perform a whole inference lifecycle on a single frame (callable alias to `predict()`).

        Args:
            data (any):
                Inference input data, typically an image or array, or a tuple `(frame, frame_info)`.

        Returns:
            (degirum.postprocessor.InferenceResults or None):
                The combined inference result object, or None if no result.
        """
        return self.predict(data)


class FrameInfo:
    """
    Class to hold frame info.

    By default, DeGirum PySDK allows you to pass any arbitrary object as 'frame info'
    alongside each frame in `predict_batch()`.

    Attributes:
        result1 (any):
            The result object produced by the first model in a compound pipeline.
            For instance, an `InferenceResults` object.
        sub_result (int):
            The index of a sub-result within `result1` (e.g., which bounding box led
            to this cropped image).
    """

    def __init__(self, result1, sub_result):
        self.result1 = result1  # model 1 result
        self.sub_result = sub_result  # sub-result index in model 1 result


class CompoundModelBase(ModelLike):
    """
    Compound model class which combines two models into one pipeline.

    One model is considered *primary* (model1), and the other is *nested* (model2).

    The primary model (`model1`) processes the input frames. Its results
    are then passed to the nested model (`model2`).
    """

    class NonBlockingQueue(queue.Queue):
        """
        Specialized non-blocking queue which acts as an iterator to feed data to the nested model.
        """

        def __iter__(self):
            """
            Yield items from the queue until a `None` sentinel is reached.

            Yields:
                (any or None):
                    The item from the queue, or `None` if the queue is empty.
            """
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
            model1 (ModelLike):
                Model to be used for the first step of the pipeline.
            model2 (ModelLike):
                Model to be used for the second step of the pipeline.
        """
        self.model1 = model1
        self.model2 = model2
        self.queue = CompoundModelBase.NonBlockingQueue()
        # Soft limit for the queue size
        self._queue_soft_limit = model1.frame_queue_depth
        self._analyzers: List[ResultAnalyzerBase] = []

    @abstractmethod
    def queue_result1(self, result1):
        """
        Process the result of the first model and put it into the queue.

        Args:
            result1 (InferenceResults):
                Prediction result of the first model.
        """

    @abstractmethod
    def transform_result2(self, result2):
        """
        Transform (or integrate) the result of the second model.

        Args:
            result2 (InferenceResults):
                Prediction result of the second model.

        Returns:
            (degirum.postprocessor.InferenceResults or None):
                Transformed/combined result to be returned by the compound model.
                If None, that means no result is produced at this iteration.
        """

    def predict_batch(self, data):
        """
        Perform a whole inference lifecycle for all objects in the given iterator object (for example, `list`).

        Works in a pipeline fashion:
            1. Pass input frames (or `(frame, frame_info)` tuples) to `model1`.
            2. Use `queue_result1(result1)` to feed `model2`.
            3. Collect `model2` results, transform them with `transform_result2(result2)`,
            4. Yield the final output.

        Args:
            data (iterator):
                Inference input data iterator object such as a list or a generator function.
                Each element returned should be compatible with model inference requirements.

        Returns:
            (Iterator[degirum.postprocessor.InferenceResults or None]):
                Generator object which iterates over the combined inference result objects
                (or None in non-blocking mode). This allows you to use the result in `for` loops.
        """
        # Use non-blocking mode for nested model and regular mode for the first model
        self.model2.non_blocking_batch_predict = True

        # Iterator over predictions of the nested model
        model2_iter = self.model2.predict_batch(self.queue)

        def result_apply_final_steps(transformed_result):
            """
            Apply analyzers to the transformed result, if any are attached.
            """
            if self._analyzers:
                for analyzer in self._analyzers:
                    analyzer.analyze(transformed_result)

                image_overlay_substitute(transformed_result, self._analyzers)

            return transformed_result

        # 1. Run model1 on input data
        for result1 in self.model1.predict_batch(data):
            if result1 is not None:
                self.queue_result1(result1)

            while True:
                # 2. Process all results available so far from model2
                no_results = True
                while result2 := next(model2_iter):
                    if (
                        transformed_result := self.transform_result2(result2)
                    ) is not None:
                        yield result_apply_final_steps(transformed_result)
                        no_results = False

                if no_results and self.model1.non_blocking_batch_predict:
                    # If no result has come through, yield None in non-blocking mode
                    yield None

                if self.queue.qsize() < self._queue_soft_limit:
                    # If under the queue soft limit, break to feed the next data to model1
                    break

        # 3. Signal end of queue to model2
        self.queue.put(None)
        self.model2.non_blocking_batch_predict = False  # restore blocking mode

        # 4. Process any remaining results in model2's iterator
        for result2 in model2_iter:
            if (transformed_result := self.transform_result2(result2)) is not None:
                yield result_apply_final_steps(transformed_result)

    @property
    def non_blocking_batch_predict(self):
        """
        Flag controlling whether `predict_batch()` operates in non-blocking mode
        for model1. In non-blocking mode, `predict_batch()` can yield `None`
        when no results are immediately available.

        Returns:
            (bool):
                True if non-blocking mode is enabled, False otherwise.
        """
        return self.model1.non_blocking_batch_predict

    @non_blocking_batch_predict.setter
    def non_blocking_batch_predict(self, val: bool):
        self.model1.non_blocking_batch_predict = val

    @property
    def _custom_postprocessor(self) -> Optional[type]:
        """
        Custom postprocessor is not supported for compound models.
        This property will always return None.

        Note:
            Attempting to set this property will raise an Exception,
            because compound models do not support custom postprocessors.

        Returns:
            (None):
                Always returns None for compound models.
        """
        return None

    @_custom_postprocessor.setter
    def _custom_postprocessor(self, val: type):
        raise Exception("Custom postprocessor is not supported for compound models")

    def attach_analyzers(
        self, analyzers: Union[ResultAnalyzerBase, List[ResultAnalyzerBase], None]
    ):
        """
        Attach analyzers to a model.

        Args:
            analyzers (Union[ResultAnalyzerBase, list[ResultAnalyzerBase], None]):
                A single analyzer, or a list of analyzer objects, or `None` to detach all analyzers.
        """
        self._analyzers = (
            []
            if analyzers is None
            else (analyzers if isinstance(analyzers, list) else [analyzers])
        )

    def __getattr__(self, attr):
        """
        Fallback for getters of model-like attributes to the primary model (model1).
        """
        return getattr(self.model1, attr)

    def __setattr__(self, key, value):
        """
        Intercepts attempts to set attributes. If the attribute already exists on the instance, the class, or
        is being set inside `__init__`, the attribute is set normally. Otherwise, the attribute assignment is
        delegated to the primary model (`model1`) if defined. This prevents adding new attributes outside
        of `__init__`.
        """
        if (
            key in self.__dict__
            or key in self.__class__.__dict__
            or (
                (cur_frame := inspect.currentframe()) is not None
                and (f_back := cur_frame.f_back) is not None
                and f_back.f_code.co_name == "__init__"
            )
        ):
            super().__setattr__(key, value)
        else:
            if "model1" in self.__dict__:
                setattr(self.model1, key, value)
            else:
                raise AttributeError(
                    f"Attempt to add new attribute '{key}' to {self.__class__}: it is prohibited for compound models to add new attributes outside of __init__"
                )


class CombiningCompoundModel(CompoundModelBase):
    """
    Compound model class which executes two models in parallel on the same input data
    and merges their results.

    Restriction: both models should produce the same type of inference results (e.g., both detection).
    """

    def queue_result1(self, result1):
        """
        Queues the original image from `result1` and a new `FrameInfo` instance
        that references `result1`. This `(frame, frame_info)` tuple is then read by `model2`.

        Args:
            result1 (InferenceResults):
                Inference result from model1. We extract `result1.image` as the frame,
                and create a `FrameInfo` so we know which `result1` this frame corresponds to.
        """
        self.queue.put((result1.image, FrameInfo(result1, -1)))

    def transform_result2(self, result2):
        """
        Merges results from `model2` into `result1` that was stored in `FrameInfo`.

        This implementation appends the second model's inference results
        to the first model's result list.

        Args:
            result2 (InferenceResults):
                Inference result of the second model, which has `info` attribute containing
                the `FrameInfo`.

        Returns:
            (degirum.postprocessor.InferenceResults):
                The merged inference results (model1 + model2).
        """
        result1 = result2.info.result1
        result1.results.extend(result2.results)
        return result1


class CroppingCompoundModel(CompoundModelBase):
    """
    Compound model class which crops the original image according to results of the first model
    and then passes these cropped images to the second model.

    Restriction: the first model should be of object detection type.
    """

    def __init__(
        self,
        model1,
        model2,
        crop_extent: float = 0.0,
        crop_extent_option: CropExtentOptions = CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
    ):
        """
        Constructor.

        Args:
            model1 (ModelLike):
                Object detection model that produces bounding boxes.
            model2 (ModelLike):
                Classification model that will process each cropped region.
            crop_extent (float):
                Extent of cropping (in percent of bbox size) to expand the bbox.
            crop_extent_option (CropExtentOptions):
                Method of applying extended crop to the input image for model2.
        """
        super().__init__(model1, model2)
        self._crop_extent = crop_extent
        self._crop_extent_option = crop_extent_option

    def queue_result1(self, result1):
        """
        Put the original image into the queue, along with bounding boxes from the first model.

        If no bounding boxes are detected, puts a small black image to keep the pipeline in sync.

        Args:
            result1 (InferenceResults):
                Prediction result of the first (object detection) model.
        """
        if len(result1.results) == 0 or "bbox" not in result1.results[0]:
            # No bbox detected: put a small black image into the queue to keep things going
            self.queue.put(
                (np.zeros((2, 2, 3), dtype=np.uint8), FrameInfo(result1, -1))
            )
        else:
            image_sz = image_size(result1.image)
            for idx, obj in enumerate(result1.results):
                adj_bbox = self._adjust_bbox(obj["bbox"], image_sz)
                # Patch bbox in the result with extended bbox
                obj["bbox"] = adj_bbox

                cropped_img = crop_image(result1.image, adj_bbox)
                self.queue.put((cropped_img, FrameInfo(result1, idx)))

    def _adjust_bbox(self, bbox, image_sz):
        """
        Inflate bbox coordinates to the crop extent according to the chosen approach
        and adjust to image size.

        Args:
            bbox (list[float]):
                Bbox coordinates in [x1, y1, x2, y2] format.
            image_sz (tuple[int, int]):
                Image size (width, height) of the original image.

        Returns:
            (list[float]):
                Adjusted bbox coordinates.
        """
        _, h, w, _ = self.model2.input_shape[0]
        return extend_bbox(
            bbox,
            self._crop_extent_option,
            self._crop_extent,
            w / h,
            image_sz,
        )


class CroppingAndClassifyingCompoundModel(CroppingCompoundModel):
    """
    Compound model class which:
        1. Runs an object detection (model1) to generate bounding boxes.
        2. Crops each bounding box from the original image.
        3. Runs a classification (model2) on each cropped image.
        4. Patches the original detection results with the classification labels.

    Restriction: first model must be object detection, second model must be classification.
    """

    def __init__(
        self,
        model1,
        model2,
        crop_extent: float = 0.0,
        crop_extent_option: CropExtentOptions = CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
    ):
        """
        Constructor.

        Args:
            model1 (ModelLike):
                An object detection model producing bounding boxes.
            model2 (ModelLike):
                A classification model to classify each cropped region.
            crop_extent (float):
                Extent of cropping (in percent of bbox size).
            crop_extent_option (CropExtentOptions):
                Specifies how to adjust the bounding box before cropping.
        """
        if model1.image_backend != model2.image_backend:
            raise Exception(
                f"Image backends of both models should be the same, but got {model1.image_backend} and {model2.image_backend}"
            )

        super().__init__(model1, model2, crop_extent, crop_extent_option)
        self._current_result: Optional[list] = None
        self._current_result1: Optional[dg.postprocessor.InferenceResults] = None

    def _finalize_current_result(self):
        """
        Finalize the current results by returning a copy of the first model results
        patched with classification labels.

        Returns:
            (degirum.postprocessor.InferenceResults or None):
                Final result for the previous frame, or None if there was no frame.
        """
        if self._current_result is not None and self._current_result1 is not None:
            ret = copy.copy(self._current_result1)
            ret._inference_results = self._current_result
            return ret
        return None

    def transform_result2(self, result2):
        """
        Transform (patch) the classification result into the original detection results.

        Args:
            result2 (InferenceResults):
                Classification result of model2.

        Returns:
            (degirum.postprocessor.InferenceResults or None):
                The detection result (from model1) patched with classification labels,
                or None if we haven't moved to a new frame yet.
        """
        result1 = result2.info.result1
        idx = result2.info.sub_result

        ret = None
        if result1 is not self._current_result1:
            # new frame comes; yield the finalized previous frame
            ret = self._finalize_current_result()
            self._current_result = copy.deepcopy(result1._inference_results)
            self._current_result1 = result1

        if idx >= 0 and self._current_result is not None:
            r = result2.results[0]
            label = r.get("label")
            if label is not None:
                self._current_result[idx]["label"] = label
            score = r.get("score")
            if score is not None:
                self._current_result[idx]["score"] = score
            category_id = r.get("category_id")
            if category_id is not None:
                self._current_result[idx]["category_id"] = category_id

        return ret

    def predict_batch(self, data):
        """
        Perform the full inference lifecycle for all objects in the given iterator (for example, `list`),
        but patch model1 bounding box labels with classification results from model2.

        Args:
            data (iterator):
                Iterator of input frames for model1.
                Each element returned by this iterator should be compatible with regular PySDK models.

        Returns:
            (Iterator[degirum.postprocessor.InferenceResults]):
                Yields the detection results with patched classification labels after each frame completes.
        """
        self._current_result = None
        self._current_result1 = None
        for result in super().predict_batch(data):
            yield result
        if self._current_result is not None:
            yield self._finalize_current_result()


@dataclass
class NmsOptions:
    """
    Options for non-maximum suppression (NMS) algorithm.

    Attributes:
        threshold (float):
            IoU or IoS threshold for box clustering (range [0..1]).
        use_iou (bool):
            If True, use IoU for box clustering, otherwise IoS.
        box_select (NmsBoxSelectionPolicy):
            Box selection policy (e.g., keep the box with the highest probability).
    """

    threshold: float
    use_iou: bool = True
    box_select: NmsBoxSelectionPolicy = NmsBoxSelectionPolicy.MOST_PROBABLE


class CroppingAndDetectingCompoundModel(CroppingCompoundModel):
    """
    Compound model class which:
        1. Uses an object detection model (model1) to generate bounding boxes (ROIs).
        2. Crops each bounding box from the original image.
        3. Uses another object detection model (model2) to further detect objects in each cropped region.
        4. Combines the results of the second model from all cropped regions, mapping coords back to the original image.

    Optionally, you can add model1 detections to the final result and/or apply NMS.

    When model1 results are added, each detection from model2 will have a `crop_index` field, indicating which bounding box from model1 it corresponds to.

    Restriction:
        First model should be object detection or pseudo-detection model like `RegionExtractionPseudoModel`, second model should be object detection.
    """

    def __init__(
        self,
        model1,
        model2,
        *,
        crop_extent: float = 0.0,
        crop_extent_option: CropExtentOptions = CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
        add_model1_results: bool = False,
        nms_options: Optional[NmsOptions] = None,
    ):
        """
        Constructor.

        Args:
            model1 (ModelLike):
                Object detection model (or pseudo-detection).
            model2 (ModelLike):
                Object detection model.
            crop_extent (float):
                Extent of cropping in percent of bbox size.
            crop_extent_option (CropExtentOptions):
                Method of applying extended crop to the input image for model2.
            add_model1_results (bool):
                If True, merges model1 detections into the final combined result.
                Each detection from model2 will have a `crop_index` field, indicating which bounding box from model1 it corresponds to.
            nms_options (Optional[NmsOptions]):
                If provided, applies non-maximum suppression (NMS) to the combined result.
        """
        if model1.image_backend != model2.image_backend:
            raise Exception(
                f"Image backends of both models should be the same, but got {model1.image_backend} and {model2.image_backend}"
            )

        super().__init__(model1, model2, crop_extent, crop_extent_option)
        self._current_result: Optional[list] = None
        self._current_result1: Optional[dg.postprocessor.InferenceResults] = None
        self._add_model1_results = add_model1_results
        self._nms_options: Optional[NmsOptions] = nms_options

    def _finalize_current_result(self):
        """
        Finalize the combined detection results from model2
        (optionally merged with model1 results). Apply NMS if requested.

        Returns:
            (degirum.postprocessor.InferenceResults or None):
                The final combined detection results for the previous frame, or None if no data.
        """
        if self._current_result is not None and self._current_result1 is not None:
            # preserve original result1
            result1 = self._current_result1._inference_results

            # we create final result based on the model1 result
            ret = copy.copy(self._current_result1)
            ret._inference_results = self._current_result

            if self._nms_options is not None:
                nms(
                    ret,
                    iou_threshold=self._nms_options.threshold,
                    use_iou=self._nms_options.use_iou,
                    box_select=self._nms_options.box_select,
                )

            if self._add_model1_results:
                # prepend model1 results
                ret._inference_results = result1 + ret._inference_results

            return ret
        return None

    def transform_result2(self, result2):
        """
        Combine detection results from model2 for each bbox from model1, translating
        coordinates back to the original image space.

        Args:
            result2 (InferenceResults):
                Detection result of the second model.

        Returns:
            (degirum.postprocessor.InferenceResults or None):
                The final detection results for the previous frame if a new frame started,
                or None otherwise.
        """
        result1 = result2.info.result1
        idx = result2.info.sub_result

        if idx >= 0:
            # Adjust bbox coordinates to the original image coordinate space
            result2 = clone_result(result2)
            x, y = result1.results[idx]["bbox"][:2]
            for r in result2._inference_results:
                if "bbox" in r:
                    r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()

                if "landmarks" in r:
                    for m in r["landmarks"]:
                        m["landmark"][0] += x
                        m["landmark"][1] += y

                if self._add_model1_results:
                    r["crop_index"] = idx  # mark which bbox this result belongs to

        ret = None
        if result1 is self._current_result1:
            # Frame continues: append second model results to the combined result
            if self._current_result is not None and idx >= 0:
                self._current_result.extend(result2._inference_results)
        else:
            # New frame comes: return combined result of previous frame
            ret = self._finalize_current_result()
            self._current_result = result2._inference_results
            self._current_result1 = result1

        return ret

    def predict_batch(self, data):
        """
        Perform the full inference lifecycle for all objects in the given iterator object (for example, `list`):

        1. model1 detects or extracts bounding boxes (ROIs).
        2. Each ROI is passed to model2 for detection.
        3. model2 results for each ROI are merged and mapped back to original coordinates.
        4. (Optional) NMS is applied and results from model1 can be included.

        Args:
            data (iterator):
                Iterator of input frames for model1.
                Each element returned by this iterator should be compatible with regular PySDK models.

        Returns:
            (Iterator[degirum.postprocessor.InferenceResults]):
                Generator object which iterates over final detection results with possibly merged bounding boxes,
                adjusted to original image coordinates.
        """
        self._current_result = None
        self._current_result1 = None
        for result in super().predict_batch(data):
            yield result
        if self._current_result is not None:
            yield self._finalize_current_result()


@dataclass
class MotionDetectOptions:
    """
    Options for motion detection algorithm.

    Attributes:
        threshold (float):
            Threshold for motion detection [0..1], representing
            fraction of changed pixels relative to frame size.
        look_back (int):
            Number of frames to look back to detect motion.
    """

    threshold: float
    look_back: int = 1


class RegionExtractionPseudoModel(ModelLike):
    """
    Pseudo-model class which extracts regions from a given image according to given ROI boxes.
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
            roi_list (Union[list, np.ndarray]):
                Can be:
                    - list of ROI boxes in `[x1, y1, x2, y2]` format,
                    - 2D NumPy array of shape (N, 4),
                    - 3D NumPy array of shape (K, M, 4), which will be flattened.
            model2 (dg.model.Model):
                The second model in the pipeline.
            motion_detect (Optional[MotionDetectOptions]):
                - When None, disabled motion detection.
                - When not None, applies motion detection before extracting ROI boxes. Boxes without motion are skipped.
        """
        if isinstance(roi_list, np.ndarray) and len(roi_list.shape) == 3:
            roi_list = roi_list.reshape(-1, roi_list.shape[-1])

        self._roi_list = roi_list
        self._model2 = model2
        self._base_img: list = []  # base image for motion detection
        self._motion_detect = motion_detect
        self._non_blocking_batch_predict = False
        self._custom_postprocessor: Optional[type] = None

    @property
    def non_blocking_batch_predict(self):
        """
        Controls non-blocking mode for `predict_batch()`.

        Returns:
            (bool):
                True if non-blocking mode is enabled; otherwise False.
        """
        return self._non_blocking_batch_predict

    @non_blocking_batch_predict.setter
    def non_blocking_batch_predict(self, val: bool):
        self._non_blocking_batch_predict = val

    @property
    def custom_postprocessor(self) -> Optional[type]:
        """
        Custom postprocessor class. Required for attaching analyzers to the pseudo-model.

        When set, this replaces the default postprocessor with a user-defined postprocessor.

        Returns:
            (Optional[type]):
                The user-defined postprocessor class, or None if not set.
        """
        return self._custom_postprocessor

    @custom_postprocessor.setter
    def custom_postprocessor(self, val: type):
        self._custom_postprocessor = val

    def __getattr__(self, attr):
        """
        Fallback for getters of model-like attributes to `model2`.
        """
        return getattr(self._model2, attr)

    def predict_batch(self, data):
        """
        Perform a pseudo-inference that outputs bounding boxes defined in `roi_list`.

        If motion detection is enabled, skip ROIs where motion is not detected.

        Args:
            data (iterator):
                Iterator over the input images or frames.
                Each element returned by this iterator should be compatible with regular PySDK models.

        Returns:
            (Iterator[degirum.postprocessor.InferenceResults or None]):
                Yields pseudo-inference results containing ROIs as bounding boxes,
                or yields None in non-blocking mode when no data is available.
        """
        preprocessor = dg._preprocessor.create_image_preprocessor(
            self._model2.model_info,
            image_backend=self._model2.image_backend,
            pad_method="",  # disable resizing/padding
        )
        preprocessor.image_format = "RAW"  # avoid unnecessary JPEG encoding

        all_rois = [True] * len(self._roi_list)

        for element in data:
            if element is None:
                if self._non_blocking_batch_predict:
                    yield None
                else:
                    raise Exception(
                        "Model misconfiguration: input data iterator returns None but non-blocking batch predict mode is not enabled"
                    )
                continue

            # Extract frame and info
            if isinstance(element, tuple):
                frame, frame_info = element
            else:
                frame, frame_info = element, element if isinstance(element, str) else ""

            # Do pre-processing
            preprocessed_data = preprocessor.forward(frame)
            image = preprocessed_data["image_input"]

            # Optionally detect motion
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
                {"bbox": roi, "label": f"ROI{idx}", "score": 1.0, "category_id": idx}
                for idx, roi in enumerate(self._roi_list)
                if motion_detected[idx]
            ]

            # Generate pseudo-inference results
            pp = (
                self._custom_postprocessor
                if self._custom_postprocessor
                else dg.postprocessor.DetectionResults
            )
            result = pp(
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
