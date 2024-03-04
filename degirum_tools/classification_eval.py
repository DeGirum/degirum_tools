import yaml, os
from tqdm import tqdm
from pathlib import Path


def count_files_in_directory(directory_path):
    # Ensure the directory path exists
    if not os.path.exists(directory_path):
        print(f"The directory path '{directory_path}' does not exist.")
        return -1
    # List all files in the directory
    files = [
        f
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]
    # Count the number of files
    file_count = len(files)
    return file_count


def model_prediction(model, validation_images_dir, label, labelsmap, k=1):
    image_dir_path = Path(validation_images_dir + "/" + label)
    all_images = [str(image_path) for image_path in image_dir_path.glob("*")]
    val_data, prediction = [], []
    for img in tqdm(all_images, desc="Processing Images"):
        val_data.append(img)
        # Check if the current item is a file
        if os.path.isfile(img):
            with model:
                res = model(img)
                sorted_predictions = sorted(
                    res.results, key=lambda x: x["score"], reverse=True
                )
                top_predictions = sorted_predictions[:k]  ## get the top k predictions
                top_labels = [pred["label"] for pred in top_predictions]
                prediction.append(
                    [labelsmap[top_labels[i]] for i in range(len(top_labels))]
                )
    return (prediction, val_data)


def identify_misclassified_examples(prediction, gndtruth, val_data):
    misclassified_images = []
    for i in range(len(prediction)):
        if gndtruth[i] not in prediction[i]:
            misclassified_images.append(val_data[i])
    return misclassified_images


def top_k_accuracy(total_data, total_misclassified):
    correct_predictions = total_data - total_misclassified
    accuracy = correct_predictions / total_data
    return accuracy


class ImageClassificationModelEvaluator:
    def __init__(
        self,
        dg_model,
        classmap,
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
                k (int) : The value of `k` in top-k.
                classmap (dict): The key represents the actual labels (as specified in the model JSON file) and values represent the class names (folder names) of the validation dataset.
                                 - For example : Gender Classification model - classmap = {"Male": "male", "Female": "female"}
                input_resize_method (str): Input Resize Method.
                input_pad_method (str): Input Pad Method.
                image_backend (str): Image Backend.
                input_img_fmt (str): InputImgFmt.
        """

        self.dg_model = dg_model
        self.classmap = classmap
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
            classmap=args["classmap"],
            top_k=args["top_k"],
            output_confidence_threshold=args["output_confidence_threshold"],
            input_resize_method=args["input_resize_method"],
            input_pad_method=args["input_pad_method"],
            image_backend=args["image_backend"],
            input_img_fmt=args["input_img_fmt"],
        )

    def evaluate(
        self,
        image_folder_path: str,
        print_frequency: int = 0,  # print_frequency (int): Number of image batches to be evaluated at a time.
    ):
        total_data = 0
        total_misclassified = 0
        misclassified_count_dct = {}

        for label in os.listdir(image_folder_path):
            gndtruth = []
            data_per_class = count_files_in_directory(
                os.path.join(image_folder_path, label)
            )
            total_data += data_per_class
            gndtruth = [label for _ in range(data_per_class)]
            print(f"Class : {label}, Count of Groundtruth labels : {len(gndtruth)}")

            prediction, val_data = model_prediction(
                self.dg_model, image_folder_path, label, self.classmap, k=self.top_k
            )

            misclassified_images = identify_misclassified_examples(
                prediction, gndtruth, val_data
            )
            print(f"Count of misclassified examples : {len(misclassified_images)}\n")
            misclassified_count_dct[label] = len(misclassified_images)
            total_misclassified += len(misclassified_images)

        accuracy = top_k_accuracy(total_data, total_misclassified)
        print(f"Top-{self.top_k} Accuracy for classification model: {accuracy}")
