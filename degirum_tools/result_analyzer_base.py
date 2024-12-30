#
# result_analyzer_base.py: base class for result analyzers
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#

from abc import ABC, abstractmethod
import numpy as np
from typing import List


class ResultAnalyzerBase(ABC):
    """
    Base class for various result analyzers
    """

    @abstractmethod
    def analyze(self, result):
        """
        Analyze inference result and augmented it with the results of analysis

        Args:
            result: PySDK inference result to analyze. As a result of analysis,
            it can be augmented with additional keys added to result.results[] dictionaries.
        """

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Annotate image with new data obtained as a result of process() method call

        Args:
            result: PySDK inference result to use for annotation. It should be analyzed by process() method.
            image (np.ndarray): image to use as a canvas for annotation

        Returns:
            np.ndarray: annotated image

        Default implementation does nothing and returns the original image.
        """
        return image

    def analyze_and_annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Helper method:
        Analyze inference result, augmented it with the results of analysis, and annotate given
        image with new data obtained as a result of analysis.

        Args:
            result: PySDK model result object
            image (np.ndarray): image to display on

        Returns:
            np.ndarray: annotated image
        """
        self.analyze(result)
        return self.annotate(result, image)

    def finalize(self):
        """
        Perform finalization/cleanup actions
        """

    def __del__(self):
        self.finalize()


def image_overlay_substitute(result, analyzers: List[ResultAnalyzerBase]):
    """Substitutes the `image_overlay` property of the given `result` object with a new one
    that overlays the original image with the analyzer annotations.

    - result: PySDK model result object
    - analyzers: list of analyzers to apply to the result
    """

    def _overlay_analyzer_annotations(self):
        """Image overlay method with all analyzer annotations applied"""
        image = self._orig_image_overlay_analyzer_annotations
        for analyzer in self._analyzers:
            image = analyzer.annotate(self, image)
        return image

    # redefine `image_overlay` property to `_overlay_analyzer_annotations` function so
    # that it will be called instead of the original one to annotate the image with analyzer results;
    # preserve original `image_overlay` property as `_orig_image_overlay_analyzer_annotations` property;
    # assign analyzer list to `_analyzers` attribute
    result.__class__ = type(
        result.__class__.__name__ + "_overlay_analyzer_annotations",
        (result.__class__,),
        {
            "image_overlay": property(_overlay_analyzer_annotations),
            "_orig_image_overlay_analyzer_annotations": result.__class__.image_overlay,
        },
    )
    setattr(result, "_analyzers", analyzers)
