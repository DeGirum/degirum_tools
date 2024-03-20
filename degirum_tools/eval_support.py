#
# eval_support.py: model evaluation support
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#


import yaml, io
import degirum as dg
from typing_extensions import Self
from typing import Union
from abc import ABC, abstractmethod


class ModelEvaluatorBase(ABC):

    def __init__(self, model: dg.model.Model, **kwargs):
        """
        Constructor.
        Args:
            model: PySDK model object
            kwargs (dict): arbitrary set of model parameters and evaluation parameters;
                must be either valid names of model object properties
                or non-model parameters as defined in ModelEvaluatorBase constructor
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
        """
        Construct model evaluator object from a yaml file.

        Args:
            model: PySDK model object
            config_yaml: path or file stream of the YAML file that contains model parameters and evaluator parameters

        Returns:
            model evaluator object
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
        """
        Perform model evaluation on given dataset.

        Args:
            image_folder_path (str): Path to images
            ground_truth_annotations_path (str): Path to the ground truth JSON annotations file (COCO format)
            max_images (int): max number of images used for evaluation. 0: all images in `image_folder_path` are used.

        Returns the evaluation statistics (algorithm specific)
        """
