import yaml, os
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List


class ImageClassificationModelEvaluator:
    def __init__(
        self,
        dg_model,
        foldermap,
        top_k,
        output_confidence_threshold=0.001,
        input_resize_method="bicubic",
        input_pad_method="crop-last",
        image_backend="opencv",
        input_img_fmt="JPEG",
    ):
        """
        Constructor.
            This class computes the Top-k Accuracy for Classification models.

            Args:
                dg_model (Detection model): Classification model from the Degirum model zoo.
                top_k (list) : List of `k` values in top-k, default:[1,5].
                foldermap (dict): The key represents integer (starting from 0) and values represent the class names (folder names) of the validation dataset.
                                 - For example : Gender Classification model - foldermap = {0: "0", 1: "1"}
                input_resize_method (str): Input Resize Method.
                input_pad_method (str): Input Pad Method.
                image_backend (str): Image Backend.
                input_img_fmt (str): InputImgFmt.
        """

        self.dg_model = dg_model
        self.foldermap = foldermap
        self.top_k = top_k
        if self.dg_model.output_postprocess_type == "Classification":
            self.dg_model.output_confidence_threshold = output_confidence_threshold
            self.dg_model.input_resize_method = input_resize_method
            self.dg_model.input_pad_method = input_pad_method
            self.dg_model.image_backend = image_backend
            self.dg_model.input_image_format = input_img_fmt
        else:
            raise Exception("Model loaded for evaluation is not a Classification Model")

    @classmethod
    def init_from_yaml(cls, dg_model, config_yaml):
        with open(config_yaml) as f:
            args = yaml.load(f, Loader=yaml.FullLoader)

        return cls(
            dg_model=dg_model,
            foldermap=args["foldermap"],
            top_k=args["top_k"],
            output_confidence_threshold=args["output_confidence_threshold"],
            input_resize_method=args["input_resize_method"],
            input_pad_method=args["input_pad_method"],
            image_backend=args["image_backend"],
            input_img_fmt=args["input_img_fmt"],
        )

    def evaluate(self, image_folder_path: str):
        accuracies: Dict[int, float] = {}
        total_images = 0
        total_correct_predictions: Dict[int, int] = {k: 0 for k in self.top_k}
        misclassified_examples: Dict[int, int] = {k: 0 for k in self.top_k}
        for category_folder in os.listdir(image_folder_path):
            image_dir_path = Path(image_folder_path + "/" + category_folder)
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            all_images = [
                str(image_path)
                for image_path in image_dir_path.glob("*")
                if image_path.suffix.lower() in image_extensions
            ]
            for predictions in tqdm(self.dg_model.predict_batch(all_images)):
                # Iterate over each top_k value
                for k in self.top_k:
                    # Sort predictions and get top-k results
                    sorted_predictions = sorted(
                        predictions.results, key=lambda x: x["score"], reverse=True
                    )[:k]
                    top_categories = [
                        pred["category_id"] for pred in sorted_predictions
                    ]
                    top_classes = [self.foldermap[int(top)] for top in top_categories]
                    # Check if ground truth is in top-k predictions
                    if category_folder in top_classes:
                        total_correct_predictions[k] += 1
                    else:
                        misclassified_examples[k] += 1
            total_images += len(all_images)

        # Calculate accuracy for each top_k
        for k in self.top_k:
            accuracies[k] = total_correct_predictions[k] / total_images
            print(
                f"Total misclassified examples for Top-{k}: {misclassified_examples[k]}"
            )
            print(f"Top-{k} Accuracy for classification model: {accuracies[k]}")
        return accuracies
