#
# eval_support.py: model evaluation support
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#

"""
Model Evaluation Support Module Overview
======================================

This module provides base classes and utilities for model evaluation, including
performance metrics calculation, ground truth comparison, and evaluation result
reporting. It supports various evaluation scenarios and metrics.

Key Features:
    - **Base Evaluator Class**: Abstract base class for model evaluators
    - **YAML Configuration**: Support for evaluator configuration via YAML files
    - **Flexible Evaluation**: Support for different evaluation metrics and scenarios
    - **Result Reporting**: Standardized evaluation result reporting
    - **Ground Truth Integration**: Support for comparing model outputs with ground truth

Typical Usage:
    1. Create a custom evaluator by subclassing ModelEvaluatorBase
    2. Configure the evaluator using YAML or constructor parameters
    3. Run evaluation on test datasets
    4. Analyze and report evaluation results

Integration Notes:
    - Works with DeGirum PySDK models
    - Supports standard evaluation metrics
    - Handles various input formats
    - Provides extensible evaluation framework

Key Classes:
    - `ModelEvaluatorBase`: Base class for model evaluators

Configuration Options:
    - Model parameters
    - Evaluation metrics
    - Dataset paths
    - Ground truth format
"""

import yaml, io
import degirum as dg
from typing_extensions import Self
from typing import Union
from abc import ABC, abstractmethod


class ModelEvaluatorBase(ABC):
    """Base class for model evaluators.

    This abstract class initializes a model object, loads configuration
    parameters and defines the interface for performing evaluation.

    Args:
        model (dg.model.Model): Model instance to evaluate.
        **kwargs (Any): Arbitrary model or evaluator parameters. Keys matching the
            model attributes are applied directly to the model. Remaining keys
            are assigned to the evaluator instance if such attributes exist.

    Attributes:
        model (dg.model.Model): The model being evaluated.
    """

    def __init__(self, model: dg.model.Model, **kwargs):
        """Initialize the evaluator.

        Args:
            model (dg.model.Model): PySDK model object.
            **kwargs (Any): Arbitrary model or evaluator parameters. Keys must either
                match model attributes or attributes of ``ModelEvaluatorBase``.
        """

        #
        # evaluator base parameters:
        #

        # show progress bar
        self.show_progress: bool = False

        # assign kwargs to model or to self
        for k, v in kwargs.items():
            if hasattr(model, k):
                setattr(model, k, v)
            elif hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"EvalParameters: invalid parameter: {k}")

        self.model = model

    @classmethod
    def init_from_yaml(
        cls, model: dg.model.Model, config_yaml: Union[str, io.TextIOBase]
    ) -> Self:
        """Construct an evaluator from a YAML file.

        Args:
            model (dg.model.Model): PySDK model object.
            config_yaml (Union[str, io.TextIOBase]): Path or open stream with
                evaluator configuration in YAML format.

        Returns:
            Self: Instantiated evaluator object.
        """

        if isinstance(config_yaml, io.TextIOBase):
            args = yaml.load(config_yaml, Loader=yaml.FullLoader)
        else:
            with open(config_yaml) as f:
                args = yaml.load(f, Loader=yaml.FullLoader)
        return cls(model, **args)

    @abstractmethod
    def evaluate(
        self,
        image_folder_path: str,
        ground_truth_annotations_path: str,
        max_images: int = 0,
    ) -> list:
        """Evaluate the model on a dataset.

        Args:
            image_folder_path (str): Directory containing evaluation images.
            ground_truth_annotations_path (str): Path to the ground truth JSON
                file in COCO format.
            max_images (int, optional): Maximum number of images to process.
                ``0`` uses all images. Defaults to ``0``.

        Returns:
            Evaluation statistics (algorithm specific).
        """
