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

        if not model.output_postprocess_type == "Classification":
            raise Exception("Model loaded for evaluation is not a Regression Model")

        # base constructor assigns kwargs to model or to self
        super().__init__(model, **kwargs)

    @staticmethod
    def compute_metrics(gt: list, pred: list) -> tuple:
        """
        Compute the Mean Absolute Error (MAE) and the Mean Squared Error (MSE) between the ground truth and the predictions.

        Args:
            gt (list): List of ground truth values.
            pred (list): List of predicted values.

        Returns:
            - Tuple(float, float): the MAE and the MSE.
        """
        diff = np.subtract(np.array(gt), np.array(pred))
        mae = np.mean(np.abs(diff))
        mse = np.mean(np.multiply(diff, diff))

        return mae, mse

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
            - List(float, float): the MAE and the MSE.
        """
        with open(ground_truth_annotations_path, "r") as fi:
            anno = json.load(fi)
        img_names = [anno["images"][i]["file_name"] for i in range(len(anno["images"]))]
        gt = [anno["images"][i]["value"] for i in range(len(anno["images"]))]
        img_paths = [os.path.join(image_folder_path, im_n) for im_n in img_names]
        if max_images > 0:
            img_paths = img_paths[0:max_images]
            gt = gt[0:max_images]

        pred = []
        gt_list = []

        with self.model:
            if self.show_progress:
                progress = Progress(len(img_paths))
            for predictions, img_gt in zip(self.model.predict_batch(img_paths), gt):
                pred.append(predictions.results[0]["score"])
                gt_list.append(img_gt)
                # progress bar update
                if self.show_progress:
                    mae, mse = self.compute_metrics(gt_list, pred)
                    accuracy_str = f"MAE: {mae:.3f}, MSE: {mse:.3f}"
                    progress.step(message=accuracy_str)

        mae, mse = self.compute_metrics(gt, pred)

        return [[mae, mse]]
