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
from typing import List


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


def image_overlay_substitute(result, analyzers: List[ResultAnalyzerBase]):
    """
    Substitute the `image_overlay` property of the given inference `result` object
    so that future calls to `result.image_overlay` automatically apply the analyzers'
    `annotate()` methods.

    This method:
        1. Preserves the original `image_overlay` property as a private reference.
        2. Defines a new property that iterates over all analyzers and applies their `annotate()` methods in sequence.

    Args:
        result (degirum.postprocessor.InferenceResults):
            The inference result whose `image_overlay` property will be replaced.
        analyzers (List[ResultAnalyzerBase]):
            A list of analyzer objects to apply for annotation.
    """

    def _overlay_analyzer_annotations(self):
        """
        Replaces the default `image_overlay` logic with successive calls to each
        analyzer's `annotate()` method, so that the final overlay includes all
        custom annotations.
        """
        image = self._orig_image_overlay_analyzer_annotations
        for analyzer in self._analyzers:
            image = analyzer.annotate(self, image)
        return image

    # Dynamically create a subclass with a new property `image_overlay`.
    # Store the original property as `_orig_image_overlay_analyzer_annotations`
    # so we can reference it in `_overlay_analyzer_annotations`.
    result.__class__ = type(
        result.__class__.__name__ + "_overlay_analyzer_annotations",
        (result.__class__,),
        {
            "image_overlay": property(_overlay_analyzer_annotations),
            "_orig_image_overlay_analyzer_annotations": result.__class__.image_overlay,
        },
    )
    setattr(result, "_analyzers", analyzers)


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
