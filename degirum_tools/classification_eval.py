import yaml, os
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List


def model_prediction(model, validation_images_dir, label, foldermap, top_k=[1, 5]):
    image_dir_path = Path(validation_images_dir + "/" + label)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    all_images = [
        str(image_path)
        for image_path in image_dir_path.glob("*")
        if image_path.suffix.lower() in image_extensions
    ]
    val_data = []
    prediction: Dict[int, List[List[str]]] = {
        k: [] for k in top_k
    }  # Initialize prediction dictionary
    for img in tqdm(all_images, desc="Processing Images"):
        val_data.append(img)
        with model:
            res = model(img)
            sorted_predictions = sorted(
                res.results, key=lambda x: x["score"], reverse=True
            )
            for k in top_k:
                top_predictions = sorted_predictions[:k]  # get the top k predictions
                top_categories = [pred["category_id"] for pred in top_predictions]
                prediction[k].append(
                    [
                        foldermap[int(top_categories[i])]
                        for i in range(len(top_categories))
                    ]
                )
    return (prediction, val_data)


def identify_misclassified_examples(prediction, gndtruth, val_data):
    misclassified_images: Dict[int, List[List[str]]] = {
        k: [] for k in prediction.keys()
    }  # Initialize misclassified_images dictionary

    for k, preds in prediction.items():  # Iterate over each top-k prediction
        for i in range(len(preds)):  # Iterate over predictions for each image
            if (
                gndtruth[i] not in preds[i]
            ):  # Check if ground truth is in top-k predictions
                misclassified_images[k].append(
                    val_data[i]
                )  # Append misclassified image to the corresponding list
        print(
            f"Count of misclassified examples for Top-{k}: {len(misclassified_images[k])}\n"
        )


def calculate_accuracy(predictions, ground_truth, top_k):
    total_correct_predictions = {k: 0 for k in top_k}
    total_predictions = {k: 0 for k in top_k}

    for category_id, category_predictions in predictions.items():
        for k in top_k:
            if k in category_predictions:
                for i, pred in enumerate(category_predictions[k]):
                    if ground_truth[category_id][i] in pred[:k]:
                        total_correct_predictions[k] += 1
                    total_predictions[k] += 1

    accuracies = {k: total_correct_predictions[k] / total_predictions[k] for k in top_k}
    return accuracies


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
        predictions = {}
        gndtruth = []
        ground_truth = {}
        for category_folder in os.listdir(image_folder_path):
            prediction, val_data = model_prediction(
                self.dg_model,
                image_folder_path,
                category_folder,
                self.foldermap,
                self.top_k,
            )
            gndtruth = [category_folder for _ in range(len(val_data))]
            print(
                f"Class : {category_folder}, Count of Groundtruth labels : {len(gndtruth)}\n"
            )
            identify_misclassified_examples(prediction, gndtruth, val_data)

            predictions[category_folder] = prediction
            ground_truth[category_folder] = gndtruth

        top_k_accuracy = calculate_accuracy(predictions, ground_truth, self.top_k)

        for k, accuracy in top_k_accuracy.items():
            print(f"Top-{k} Accuracy for classification model: {accuracy}")
        return top_k_accuracy
