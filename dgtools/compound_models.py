#
# compound_models.py: compound model toolkit for PySDK samples
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes for multi-model aggregation.
#

import queue
from abc import ABC, abstractmethod


class CompoundModelBase(ABC):
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
        self.queue.put((result1.image, result1))

    def transform_result2(self, result2):
        """
        Transform result of the second model.

        This implementation appends result of the first model to the result of the second model.

        Args:
            result2: prediction result of the second model

        Returns:
            Combined result of both models. It's `info` property is the result of the first model.
        """
        result2.results.extend(result2.info.results)
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
            self.queue.put((result1.image, result1))
        else:
            image_size = self.image_size(result1.image)
            for obj in result1.results:
                adj_bbox = self._adjust_bbox(obj["bbox"], image_size)
                if hasattr(result1.image, "crop"):
                    cropped_img = result1.image.crop(adj_bbox)
                else:
                    cropped_img = result1.image[
                        adj_bbox[1] : adj_bbox[3], adj_bbox[0] : adj_bbox[2]
                    ]
                self.queue.put((cropped_img, result1))

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

    def _find_bbox(self, results, image, orig_image):
        """
        Find index of bbox in the first model's result list, which size is equal to the size of given image

        Args:
            results: list of inference results of the first model
            image: image to find matching bbox
            orig_image: original image from which image was cropped
        """

        image_size = self.image_size(image)
        orig_image_size = self.image_size(orig_image)

        def cond(obj):
            adj_bbox = self._adjust_bbox(obj["bbox"], orig_image_size)
            return image_size == (
                round(adj_bbox[2]) - round(adj_bbox[0]),
                round(adj_bbox[3]) - round(adj_bbox[1]),
            )

        return next((i for i, obj in enumerate(results) if cond(obj)), -1)

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

        This implementation appends result of the first model to the result of the second model.

        Args:
            result2: prediction result of the second model

        Returns:
            Result of second model. It's `info` property is the result of the first model.
        """

        # patch bbox label with recognized class label
        idx = self._find_bbox(result2.info.results, result2.image, result2.info.image)
        if idx >= 0:
            result2.info.results[idx]["label"] = result2.results[0]["label"]

        # return result when frame changes
        ret = None
        if result2.info is not self._current_result:
            if self._current_result is not None:
                ret = self._current_result
        self._current_result = result2.info
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
