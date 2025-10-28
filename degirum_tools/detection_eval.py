#
# detection_eval.py: object detection models evaluator
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#

import json, os, degirum as dg, numpy as np
from typing import List, Optional
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode
from .math_support import xyxy2xywh
from .eval_support import ModelEvaluatorBase
from .ui_support import Progress, stdoutRedirector


class ObjectDetectionModelEvaluator(ModelEvaluatorBase):
    """
    This class evaluates the mAP for Object Detection models.
    """

    def __init__(self, model: dg.model.Model, **kwargs):
        """
        Constructor.

        Args:
            model (Detection model): PySDK detection model object
            kwargs (dict): arbitrary set of PySDK model parameters and the following evaluation parameters:
                show_progress (bool): show progress bar
                classmap (dict): dictionary which maps model category IDs to dataset category IDs
                pred_path (str): path to save the predictions as a JSON file of None if not required
        """

        #
        # detection evaluator parameters:
        #

        # dictionary which maps model category IDs to dataset category IDs
        self.classmap: Optional[dict] = None
        # path to save the predictions as a JSON file
        self.pred_path: Optional[str] = None

        allowed_model_types = [
            "Detection",
            "DetectionYolo",
            "DetectionYoloV8",
            "DetectionYoloV10",
            "DetectionYoloHailo",
            "PoseDetectionYoloV8",
            "SegmentationYoloV8",
        ]

        if (model.output_postprocess_type not in allowed_model_types
                and model.inference_results_type not in allowed_model_types):
            raise Exception("Model loaded for evaluation is not a Detection Model")

        self.is_segmentation_model: bool = model.output_postprocess_type in [
            "SegmentationYoloV8"
        ]

        # base constructor assigns kwargs to model or to self
        super().__init__(model, **kwargs)

    def evaluate(
        self,
        image_folder_path: str,
        ground_truth_annotations_path: str,
        max_images: int = 0,
    ) -> list:
        """
        Evaluation for the detection model.

        Args:
            image_folder_path (str): Path to images
            ground_truth_annotations_path (str): Path to the ground truth JSON annotations file (COCO format)
            max_images (int): max number of images used for evaluation. 0: all images in `image_folder_path` are used.

        Returns:
            the mAP statistics: [bbox_stats, kp_stats] for pose detection models and [bbox_stats] for non-pose models.
        """

        jdict: List[dict] = []
        with stdoutRedirector():
            anno = COCO(ground_truth_annotations_path)
        num_images = len(anno.dataset["images"])
        files_dict = anno.dataset["images"][0:num_images]
        path_list: List[str] = []
        img_id_list: List[str] = []
        for image_number in range(0, num_images):
            image_id = files_dict[image_number]["id"]
            path = os.path.join(
                image_folder_path, files_dict[image_number]["file_name"]
            )
            if os.path.exists(path):
                path_list.append(path)
                img_id_list.append(image_id)

        # sort the image ids
        sorted_indices = sorted(range(len(img_id_list)), key=lambda i: img_id_list[i])
        sorted_img_id_list = [img_id_list[i] for i in sorted_indices]
        sorted_path_list = [path_list[i] for i in sorted_indices]

        if max_images > 0:
            sorted_path_list = sorted_path_list[0:max_images]
            sorted_img_id_list = sorted_img_id_list[0:max_images]

        # run the model inference on the images
        with self.model:
            results_list = []
            if self.show_progress:
                progress = Progress(len(sorted_path_list))
            for image_id, predictions in zip(
                sorted_img_id_list, self.model.predict_batch(sorted_path_list)
            ):
                if self.show_progress:
                    progress.step()
                # segmentation model addition
                if self.is_segmentation_model:
                    image = predictions._input_image
                    image_shape = (
                        getattr(image, "shape", (-1, -1))
                        or (
                            getattr(image, "height", -1),
                            getattr(image, "width", -1),
                        )
                    )[:2]
                    if any([s == -1 for s in image_shape]):
                        raise Exception("Cannot retrieve image shape.")
                    for ridx in range(len(predictions.results)):
                        predictions.results[ridx]["segmentation"] = (
                            ObjectDetectionModelEvaluator._process_segmentation(
                                predictions.results[ridx]["mask"], image_shape
                            )
                        )
                        del predictions.results[ridx]["mask"]
                results_list.append(
                    {"image_id": image_id, "results": predictions.results}
                )
            # convert the predictions to COCO json format
            ObjectDetectionModelEvaluator._convert_results_coco_json(
                results_list, jdict, self.classmap
            )

        # save the predictions to a json file
        if self.pred_path:
            with open(self.pred_path, "w") as f:
                json.dump(jdict, f, indent=4)

        with stdoutRedirector():
            pred = anno.loadRes(jdict)

            stats = []
            # bounding box map calculation
            bbox_stats = ObjectDetectionModelEvaluator._evaluate_coco(
                anno, pred, mAP_type="bbox", img_id_list=sorted_img_id_list
            )
            stats.append(bbox_stats)
            # pose keypoint map calculation
            if ObjectDetectionModelEvaluator._is_pose_model(jdict[0]):
                kp_stats = ObjectDetectionModelEvaluator._evaluate_coco(
                    anno, pred, mAP_type="keypoints", img_id_list=sorted_img_id_list
                )
                stats.append(kp_stats)
            # instance segmentation map calculation
            if ObjectDetectionModelEvaluator._is_segmentation_model(jdict[0]):
                segm_stats = ObjectDetectionModelEvaluator._evaluate_coco(
                    anno, pred, mAP_type="segm", img_id_list=sorted_img_id_list
                )
                stats.append(segm_stats)
            return stats

    @staticmethod
    def _process_keypoints(keypoints_res: List[dict]) -> List[float]:
        """
        Convert PySDK keypoint results format to pycocotools keypoint format

        Args:
            keypoints_res: The keypoint results dictionary output from PySDK.

        Returns:
            keypoints: The list of keypoint results in pycocotools format.
        """
        keypoints: List[float] = []
        for ldmks in keypoints_res:
            kypts = ldmks["landmark"][:2]
            kypts_score = ldmks["score"]
            keypoints.extend(float(x) for x in kypts)
            keypoints.append(kypts_score)
        return keypoints

    @staticmethod
    def _run_length_encode(x):
        """Encode predicted masks as RLE for COCO evaluation."""
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    @staticmethod
    def _process_segmentation(
        segmentation_res: dict, image_shape: tuple[int, int]
    ) -> List[float]:
        """
        Convert PySDK segmentation results format to pycocotools segmentation format

        Args:
            segmentation_res: The segmentation results dictionary output from PySDK.
            image_shape: The dimensions of the image for the given segmentation results, as a tuple (height, width)

        Returns:
            The segmentation results in pycocotools format.
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        x_min = segmentation_res["x_min"]
        y_min = segmentation_res["y_min"]
        data = segmentation_res["data"]
        data_shape = data.shape
        mask[y_min : y_min + data_shape[0], x_min : x_min + data_shape[1]] = data
        return ObjectDetectionModelEvaluator._run_length_encode(mask)

    @staticmethod
    def _convert_results_coco_json(res_list, jdict, class_map=None):
        """Serialize YOLO predictions to COCO json format."""
        max_category_id = 0
        for dict_res in res_list:
            image_id = dict_res["image_id"]
            results = dict_res["results"]
            for result in results:
                if "bbox" not in result.keys():
                    continue
                box = xyxy2xywh(np.asarray(result["bbox"]).reshape(1, 4) * 1.0)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                box_list = box.reshape(-1).tolist()
                category_id = (
                    class_map[result["category_id"]]
                    if class_map
                    else result["category_id"]
                )
                # detection base result
                detected_elem = {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [np.round(x, 3) for x in box_list],
                    "score": np.round(result["score"], 5),
                }
                # pose model addition
                if "landmarks" in result:
                    detected_elem["keypoints"] = (
                        ObjectDetectionModelEvaluator._process_keypoints(
                            result["landmarks"]
                        )
                    )
                # segmentation model addition
                if "segmentation" in result:
                    detected_elem["segmentation"] = result["segmentation"]

                jdict.append(detected_elem)
                max_category_id = max(max_category_id, category_id)
        return max_category_id

    @staticmethod
    def _evaluate_coco(
        anno: COCO, pred: COCO, mAP_type: str = "bbox", img_id_list: List[str] = []
    ):
        """
        Evaluation process based on the ground truth COCO object and the prediction object

        Args:
            anno (COCO): COCO ground truth annotation object
            pred (COCO): COCO prediction object
            img_id_list (List): List of the image ids to evaluate on.

        Returns:
            the mAP statistics.
        """
        eval_obj = COCOeval(anno, pred, mAP_type)
        if img_id_list:
            eval_obj.params.imgIds = [id for id in img_id_list]  # image IDs to evaluate
        eval_obj.evaluate()
        eval_obj.accumulate()
        eval_obj.summarize()

        return eval_obj.stats

    @staticmethod
    def _is_pose_model(element: dict):
        """Check if the it is a PySDK pose model
            Args:
                element (dict): detection result dict

        Returns True if it is a pose model.
        """
        return True if "keypoints" in element else False

    @staticmethod
    def _is_segmentation_model(element: dict):
        """Check if the it is a PySDK segmentation model
            Args:
                element (dict): detection result dict

        Returns True if it is a segmentation model.
        """
        return (
            True
            if ("segmentation" in element and "keypoints" not in element)
            and type(element["segmentation"]) is dict
            else False
        )
