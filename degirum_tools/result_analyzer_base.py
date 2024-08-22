#
# result_analyzer_base.py: base class for result analyzers
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#

from abc import ABC, abstractmethod
import numpy as np


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
