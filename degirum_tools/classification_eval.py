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
            foldermap=args.get("foldermap", None),
            top_k=args.get("top_k", [1, 5]),
            output_confidence_threshold=args.get("output_confidence_threshold", [1, 5]),
            input_resize_method=args["input_resize_method"],
            input_pad_method=args["input_pad_method"],
            image_backend=args["image_backend"],
            input_img_fmt=args["input_img_fmt"],
        )

    @staticmethod
    def default_foldermap(folder_list: List[str]) -> Dict[int, str]:
        return {i: folder for i, folder in enumerate(folder_list)}

    def evaluate(self, image_folder_path: str):
        folder_list = sorted(os.listdir(image_folder_path))
        if self.foldermap is None:
            self.foldermap = self.default_foldermap(folder_list)
        # initialize
        per_class_accuracies = [0 for _ in range(len(self.top_k))]
        total_correct_predictions = [[0 for _ in range(len(self.top_k))] for _ in range(len(self.foldermap))]
        total_images_in_folder = [0 for _ in range(len(self.foldermap))]
        for folder_idx, category_folder in enumerate(self.foldermap):
            image_dir_path = Path(image_folder_path) / category_folder            
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            all_images = [
                str(image_path)
                for image_path in image_dir_path.glob("*")
                if image_path.suffix.lower() in image_extensions
            ]
            pbar = tqdm(self.dg_model.predict_batch(all_images), total=len(all_images))
            print(f"Processing {category_folder} folder")
            for predictions in pbar:
                # Iterate over each top_k value
                for k_i, k in enumerate(self.top_k):
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
                        total_correct_predictions[k_i][folder_idx] += 1

                total_images_in_folder[folder_idx] += 1
                per_class_accuracies = [total_correct_predictions[k_i][folder_idx] / total_images_in_folder[folder_idx] for k_i, _ in enumerate(self.top_k)]
                accuracy_str = ", ".join([f"Top{k}: {per_class_accuracies[k_i] * 100}% " for k_i, k in enumerate(self.top_k)])

                pbar.set_postfix(accuracy_str)
            
        total_images = sum(total_images_in_folder)
        accuracies = [sum(total_correct_predictions[k_i]) / total_images for k_i, _ in enumerate(self.top_k)]
        accuracy_str = ", ".join([f"Top{k}: {accuracies[i] * 100}% " for i, k in enumerate(self.top_k)])
        print(accuracy_str)
        return accuracies, per_class_accuracies
