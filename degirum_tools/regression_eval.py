#
# regression_eval.py: evaluation toolkit for regression models used in PySDK samples
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#

import yaml
import json
import os
import numpy as np


class ImageRegressionModelEvaluator:
    def __init__(
        self,
        dg_model,
        input_resize_method="bilinear",
        input_pad_method="crop-first",
        image_backend="opencv",
        input_img_fmt="JPEG",
        input_letterbox_fill_color=(114, 114, 114),
        input_numpy_colorspace="auto",
    ):
        """
        Constructor.
            This class evaluates the MAE and MSE for Image Regression models.

            Args:
                dg_model (Regression model): Regression model from the DeGirum model zoo.
                input_resize_method (str): Input Resize Method.
                input_pad_method (str): Input Pad Method.
                image_backend (str): Image Backend.
                input_img_fmt (str): InputImgFmt.
                input_letterbox_fill_color (tuple): the RGB color for padding used in letterbox
                input_numpy_colorspace (str): input colorspace: ("BGR" to match OpenCV image backend)
        """

        self.dg_model = dg_model
        self.dg_model.input_resize_method = input_resize_method
        self.dg_model.input_pad_method = input_pad_method
        self.dg_model.image_backend = image_backend
        self.dg_model.input_image_format = input_img_fmt
        self.dg_model.input_numpy_colorspace = input_numpy_colorspace
        self.dg_model.input_letterbox_fill_color = input_letterbox_fill_color

    @classmethod
    def init_from_yaml(cls, dg_model, config_yaml):
        """
        config_yaml (str) : Path of the yaml file that contains all the arguments.

        """
        with open(config_yaml) as f:
            args = yaml.load(f, Loader=yaml.FullLoader)

        return cls(
            dg_model=dg_model,
            input_resize_method=args["input_resize_method"],
            input_pad_method=args["input_pad_method"],
            image_backend=args["image_backend"],
            input_img_fmt=args["input_img_fmt"],
            input_letterbox_fill_color=tuple(args["input_letterbox_fill_color"]),
            input_numpy_colorspace=args["input_numpy_colorspace"],
        )

    def evaluate(
        self,
        image_folder_path: str,
        ground_truth_annotations_path: str,
        num_val_images: int = 0,
        print_frequency: int = 0,
    ):
        """Evaluation for the Regression model.

            Args:
                image_folder_path (str): Path to the image dataset.
                ground_truth_annotations_path (str): Path to the groundtruth json annotations.
                num_val_images (int): max number of images used for evaluation. 0: all images in image_folder_path is used.
                print_frequency (int): Number of image batches to be evaluated before printing num evaluated images

        Returns the MAE and MSE.
        """
        with open(ground_truth_annotations_path, 'r') as fi:
            anno = json.load(fi)
        img_names = [anno["images"][i]["file_name"] for i in range(len(anno["images"]))]
        gt = [anno["images"][i]["value"] for i in range(len(anno["images"]))]
        img_path = os.path.split(ground_truth_annotations_path)[0]
        img_paths = [img_path + "/" + imn for imn in img_names]

        if num_val_images > 0:
            img_paths = img_paths[0:num_val_images]
            gt = gt[0:num_val_images]

        pred = []

        with self.dg_model:
            for image_number, predictions in enumerate(
                self.dg_model.predict_batch(img_paths)
            ):
                if print_frequency > 0:
                    if image_number % print_frequency == print_frequency - 1:
                        print(image_number + 1)
                pred.append(predictions.results[0]["score"])

        diff = np.subtract(np.array(gt), np.array(pred))
        mae = np.mean(np.abs(diff))
        mse = np.mean(np.multiply(diff, diff))

        metrics = [mae, mse]

        return [metrics]
