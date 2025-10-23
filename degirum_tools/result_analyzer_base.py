# result_analyzer_base.py: base class for result analyzers
# Copyright DeGirum Corporation 2023
# All rights reserved

"""
Result Analyzer Base Module Overview
====================================

This module provides a base class (`ResultAnalyzerBase`) for performing custom post-processing and
image annotation on DeGirum PySDK inference results. These analyzers can be used with
compound models, streaming gizmos, and regular models to add advanced data processing and annotation steps
to inference pipelines.

Key Concepts
------------

- **Analysis**:
  By overriding the `analyze()` method, child classes can read and augment the `InferenceResults`
  (e.g., by adding extra keys to the internal `results` list).

- **Annotation**:
  By overriding the `annotate()` method, child classes can draw additional overlays on the
  original input image (e.g., bounding boxes, text labels, or any custom markings).

- **Integration**:
  Analyzers can be attached to a model or a compound model via the `attach_analyzers()` method,
  so their analysis and annotation is automatically applied to each inference result.

Typical Usage Example
---------------------

1. Create a custom analyzer subclass:
   ```python
   from degirum_tools.result_analyzer_base import ResultAnalyzerBase

   class MyCustomAnalyzer(ResultAnalyzerBase):
       def analyze(self, result):
           # E.g., add custom fields to each detection
           for r in result.results:
               r["custom_info"] = "my_data"

       def annotate(self, result, image):
           # E.g., draw text or bounding boxes on the image
           # Return the annotated image
           return image
   ```
2. Attach it to a model or compound model:
   ```python
   model.attach_analyzers(MyCustomAnalyzer())
   ```
3. Run inference. Each `InferenceResults` object will be passed through your analyzer, optionally
   modifying the result data and providing a new overlay.
"""

import numpy as np
import copy
from abc import ABC, abstractmethod
from typing import List, Union


class ResultAnalyzerBase(ABC):
    """
    Base class for result analyzers which can modify or extend the content of inference results
    and optionally annotate images with new data.

    Subclasses should override:
      - `analyze(result)`: to augment or inspect the inference result.
      - `annotate(result, image)`: to draw additional overlays or text onto the provided image.
    """

    @abstractmethod
    def analyze(self, result):
        """
        Analyze and optionally modify a DeGirum PySDK inference result.

        This method should access and potentially modify `result.results` (the list of detections,
        classifications, or similar structures) to add any custom fields. These modifications
        will then appear in downstream processes or when the result is displayed/serialized.

        Args:
            result (degirum.postprocessor.InferenceResults):
                The inference result object to analyze. Subclasses can read and/or modify
                the internal `results` list or other properties.
        """

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Annotate an image with additional data derived from the analysis step.

        Called after `analyze()` has been invoked on `result`. This method is typically used to
        draw bounding boxes, text, or other graphical elements representing the analysis data.

        Args:
            result (degirum.postprocessor.InferenceResults):
                The (already analyzed) inference result object.
            image (numpy.ndarray):
                The original (or base) image to annotate.

        Returns:
            numpy.ndarray:
                The annotated image. By default, this base implementation returns the image
                unchanged. Subclasses should override to perform custom drawing.
        """
        return image

    def analyze_and_annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Helper method to perform both analysis and annotation in one step.

        1. Calls `self.analyze(result)`.
        2. Calls `self.annotate(result, image)`.

        Args:
            result (degirum.postprocessor.InferenceResults):
                The inference result object to process.
            image (numpy.ndarray):
                The image to annotate.

        Returns:
            numpy.ndarray:
                The annotated image after analysis.
        """
        self.analyze(result)
        return self.annotate(result, image)

    def finalize(self):
        """
        Perform any finalization or cleanup actions before the analyzer is discarded.

        This can be useful for analyzers that accumulate state (e.g., for multi-frame analysis).
        By default, this does nothing.
        """

    def __del__(self):
        """
        Called when the analyzer object is about to be destroyed.

        Invokes `finalize()` to ensure any open resources are cleaned up.
        """
        self.finalize()


def subclass_result_with_analyzers(
    result, analyzers: Union[ResultAnalyzerBase, List[ResultAnalyzerBase]]
) -> type:
    """
    Create a subclass of the given inference `result` class or object that includes the specified analyzers.

    This method dynamically defines a new class that inherits from the original result class and
    overrides the `image_overlay` property to apply the analyzers' `annotate()` methods.

    Args:
        result (degirum.postprocessor.InferenceResults):
            The inference result class or object to subclass.
        analyzers (List[ResultAnalyzerBase]):
            A list of analyzer objects to apply for annotation.

    Returns:
        (degirum.postprocessor.InferenceResults):
            A subclassed result object with the new `image_overlay` property.
    """

    base_class: type = result if isinstance(result, type) else type(result)

    # NOTE: this class name is also used in the attach_analyzers()
    class _DGAnalyzerResult(base_class):

        # class variable to store analyzer list
        _analyzers = analyzers if isinstance(analyzers, list) else [analyzers]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)  # call base class constructor
            # apply analyzers
            for analyzer in _DGAnalyzerResult._analyzers:
                analyzer.analyze(self)

        @property
        def image_overlay(self):
            image = super().image_overlay  # call base class property
            for analyzer in _DGAnalyzerResult._analyzers:
                image = analyzer.annotate(self, image)
            return image

    return _DGAnalyzerResult


def image_overlay_substitute(result, analyzers: List[ResultAnalyzerBase]):
    """
    Substitute the `image_overlay` property of the given inference `result` object
    so that future calls to `result.image_overlay` automatically apply the analyzers'
    `annotate()` methods.

    This method creates a new class that inherits from the original result class and
    overrides the `image_overlay` property. The new class does the following:
    1. Calls the base class's `image_overlay` property to get the image annotated by original result class.
    2. Applies each analyzer's `annotate()` method to the image, in order.

    Args:
        result (degirum.postprocessor.InferenceResults):
            The inference result whose `image_overlay` property will be replaced.
        analyzers (List[ResultAnalyzerBase]):
            A list of analyzer objects to apply for annotation.
    """

    # replace the class of the result object
    result.__class__ = subclass_result_with_analyzers(result, analyzers)


def clone_result(result):
    """
    Create a shallow clone of a DeGirum PySDK inference result object, duplicating the
    internal inference results list but reusing references to the original image.

    This is useful when you want to create a separate copy of the result for further
    modifications without altering the original.

    Args:
        result (degirum.postprocessor.InferenceResults):
            The inference result object to clone.

    Returns:
        (degirum.postprocessor.InferenceResults):
            A cloned result object with `result._inference_results` deep-copied.
    """
    clone = copy.copy(result)
    clone._inference_results = copy.deepcopy(result._inference_results)
    return clone
