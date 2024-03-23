#
# regression_eval.py: evaluation toolkit for regression models used in PySDK samples
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#

import json, os, numpy as np, degirum as dg

from .eval_support import ModelEvaluatorBase
from .ui_support import Progress


class ImageRegressionModelEvaluator(ModelEvaluatorBase):
    def __init__(self, model: dg.model.Model, **kwargs):
        """
        Constructor.

        Args:
            model (Detection model): PySDK detection model object
            kwargs (dict): arbitrary set of PySDK model parameters and the following evaluation parameters:
                show_progress (bool): show progress bar
        """

        super().__init__(model, **kwargs)

    def evaluate(
        self,
        image_folder_path: str,
        ground_truth_annotations_path: str,
        max_images: int = 0,
    ) -> list:
        """Evaluation for the Regression model.

            Args:
                image_folder_path (str): Path to the image dataset.
                ground_truth_annotations_path (str): Path to the groundtruth json annotations.
                num_val_images (int): max number of images used for evaluation. 0: all images in image_folder_path is used.
                print_frequency (int): Number of image batches to be evaluated before printing num evaluated images

        Returns:
            - Tuple(float, float): the MAE and the MSE.
        """
        with open(ground_truth_annotations_path, 'r') as fi:
            anno = json.load(fi)
        img_names = [anno["images"][i]["file_name"] for i in range(len(anno["images"]))]
        gt = [anno["images"][i]["value"] for i in range(len(anno["images"]))]
        img_path = os.path.split(ground_truth_annotations_path)[0]
        img_paths = [img_path + "/" + im_n for im_n in img_names]

        if max_images > 0:
            img_paths = img_paths[0:max_images]
            gt = gt[0:max_images]

        pred = []

        with self.model:
            if self.show_progress:
                progress = Progress(len(img_paths))
            for _, predictions in enumerate(
                self.model.predict_batch(img_paths)
            ):
                if self.show_progress:
                    progress.step()
                pred.append(predictions.results[0]["score"])

        diff = np.subtract(np.array(gt), np.array(pred))
        mae = np.mean(np.abs(diff))
        mse = np.mean(np.multiply(diff, diff))

        metrics = [mae, mse]

        return [metrics]
