#
# tile_compound_models.py: tiling-based compound models for object detection
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements tiling-based compound models for efficient large image processing
#

"""
Tile Compound Models Module Overview
===================================

This module implements tiling-based compound models for object detection.
It provides pseudo-models that extract image tiles and combine results
from real detection models to efficiently process large images.

Key Features:
    - **Tile Extraction**: Generate local and global tiles
      with configurable overlap
    - **Two-Stage Processing**: Run detection on each tile then merge results
    - **Edge-Aware Fusion**: Optional fusion of detections near tile boundaries
    - **Motion Filtering**: Skip tiles without motion to reduce computation
    - **Result Management**: Translate box coordinates and apply NMS

Typical Usage:
    1. Create a ``TileExtractorPseudoModel`` with grid parameters
    2. Wrap it in ``TileModel`` or a derived class along with a detection model
    3. Iterate over ``predict_batch`` to obtain merged detection results

Integration Notes:
    - Designed for DeGirum PySDK models
    - Works with compound model utilities such as cropping and NMS
    - Supports customization of crop extent and overlap thresholds
    - Compatible with local and cloud inference backends

Key Classes:
    - ``TileExtractorPseudoModel``: Produces image tiles for downstream models
    - ``TileModel``: Runs detection on each tile and merges results
    - ``LocalGlobalTileModel``: Combines global and local tiles
      with size-based filtering
    - ``BoxFusionTileModel``: Fuses edge detections across tiles

Configuration Options:
    - Tile grid size and overlap
    - Motion detection thresholds
    - Edge fusion parameters
    - NMS settings for final results
"""

import copy
from typing import List, MutableSequence, Optional, Sequence, Tuple, Union

import cv2, numpy as np, degirum as dg

from .math_support import nms, edge_box_fusion
from .image_tools import image_size
from .result_analyzer_base import clone_result
from .compound_models import (
    CropExtentOptions,
    ModelLike,
    NmsOptions,
    MotionDetectOptions,
    CroppingAndDetectingCompoundModel,
    detect_motion,
)


class _TileInfo:
    """
    Class to hold tile information.
    """

    def __init__(self, cols: int, rows: int, overlap: float, id: int):
        self.cols = cols
        self.rows = rows
        self.overlap = overlap
        self.id = id


class TileExtractorPseudoModel(ModelLike):
    """Extracts a grid of (optionally-overlapping) image tiles.

    The class behaves like a DeGirum pseudo-model: instead of running
    inference it produces synthetic detection results whose bounding boxes
    correspond to tile coordinates. These results are then consumed by a
    second, real model in a two-stage compound pipeline.

    Args:
        cols (int): Number of columns in the tile grid.
        rows (int): Number of rows in the tile grid.
        overlap_percent (float): Desired overlap between neighbouring tiles,
            expressed as a fraction in [0, 1].
        model2 (dg.model.Model): The downstream model that will receive each tile.
        global_tile (bool, optional): If True, emit an additional tile that
            represents the entire image (label "GLOBAL"). Mutually
            exclusive with tile_mask and motion_detect. Defaults to False.
        tile_mask (list[int] | None, optional): Indices of tiles (0-based, row-major) to keep.
            Tiles not listed are skipped. Ignored when global_tile is True.
        motion_detect (MotionDetectOptions | None, optional): Enable per-tile motion filtering. Tiles in which less than
            threshold x area pixels changed during the last look_back frames are
            suppressed. Ignored when global_tile is True.

    Raises:
        AssertionError: If mutually-exclusive arguments are supplied.
    """

    def __init__(
        self,
        cols: int,
        rows: int,
        overlap_percent: float,
        model2: dg.model.Model,
        *,
        global_tile: bool = False,
        tile_mask: Optional[list] = None,
        motion_detect: Optional[MotionDetectOptions] = None,
    ):
        """Constructor.

        Args:
            cols (int): Number of columns to divide the image into.
            rows (int): Number of rows to divide the image into.
            overlap_percent (float): Desired overlap between neighbouring tiles.
            model2 (dg.model.Model): Model which will be used as a second step of the compound model pipeline.
            global_tile (bool, optional): Indicates whether the global (whole) image should also be sent to model2.
            tile_mask (list[int] | None, optional): Optional list of indices to keep during tile generation.
                Tile indices are counted starting from the top row to the bottom row, left to right.
            motion_detect (MotionDetectOptions | None, optional): Motion detection options.
                When None, motion detection is disabled.
                When enabled, ROI boxes where motion is not detected will be skipped.
        """

        self._cols = cols
        self._rows = rows
        self._overlap_percent = overlap_percent
        self._model2 = model2
        self._non_blocking_batch_predict = False
        self._global_tile = global_tile

        self._tile_mask = tile_mask
        self._base_img: list = []  # base image for motion detection
        self._motion_detect = motion_detect
        self._custom_postprocessor: Optional[type] = None

        if global_tile:
            assert (
                tile_mask is None
            ), "Tile masking not supported with global tile inference."
            assert (
                motion_detect is None
            ), "Motion detection not supported with global tile inference."

    @property
    def non_blocking_batch_predict(self):
        return self._non_blocking_batch_predict

    @non_blocking_batch_predict.setter
    def non_blocking_batch_predict(self, val: bool):
        self._non_blocking_batch_predict = val

    @property
    def custom_postprocessor(self) -> Optional[type]:
        return self._custom_postprocessor

    @custom_postprocessor.setter
    def custom_postprocessor(self, val: type):
        self._custom_postprocessor = val

    # fallback all getters of model-like attributes to the first model
    def __getattr__(self, attr):
        return getattr(self._model2, attr)

    def _calculate_tile_parameters(self) -> List[float]:

        _, h, w, _ = self._model2.input_shape[0]
        model_aspect_ratio = w / h

        tile_width = self._width / (
            self._cols - self._overlap_percent * (self._cols - 1)
        )
        tile_height = self._height / (
            self._rows - self._overlap_percent * (self._rows - 1)
        )

        expand_offsets = [0.0, 0.0]

        tile_aspect_ratio = tile_width / tile_height

        if tile_aspect_ratio < model_aspect_ratio:
            # Expand the width
            dim = tile_height * model_aspect_ratio
            expand_offsets[0] = dim - tile_width
        elif tile_aspect_ratio > model_aspect_ratio:
            # Expand the height
            dim = tile_width / model_aspect_ratio
            expand_offsets[1] = dim - tile_height

        if expand_offsets[0] > tile_width * 2:
            raise Exception(
                "Horizontal overlap is greater than 100%. Lower the amount of columns."
            )
        elif expand_offsets[1] > tile_height * 2:
            raise Exception(
                "Vertical overlap is greater than 100%. Lower the amount of rows."
            )

        return [tile_width, tile_height] + expand_offsets

    def _get_slice(self, row: int, col: int) -> List[int]:
        tile_width, tile_height, expand_col, expand_row = self._tile_params

        # Calculate tile bounding box
        tlx = col * tile_width - col * self._overlap_percent * tile_width
        brx = tlx + tile_width
        if col == self._cols - 1:
            brx = self._width
        tly = row * tile_height - row * self._overlap_percent * tile_height
        bry = tly + tile_height
        if row == self._rows - 1:
            bry = self._height

        # Expand tile to fit the aspect ratio.
        if expand_row:
            expand_row_half = expand_row / 2

            # If first row
            if row == 0:
                bry += expand_row
            # If last row
            elif row == self._rows - 1:
                tly -= expand_row
            else:
                tly -= expand_row_half
                bry += expand_row_half

        if expand_col:
            expand_col_half = expand_col / 2

            # If first col
            if col == 0:
                brx += expand_col
            # If last col
            elif col == self._cols - 1:
                tlx -= expand_col
            else:
                tlx -= expand_col_half
                brx += expand_col_half

        tile_bbox = list(map(round, [tlx, tly, brx, bry]))

        return tile_bbox

    def predict_batch(self, data):
        """
        Perform whole inference lifecycle for all objects in given iterator object (for example, `list`).

        Args:
            data (Iterable): Inference input data iterator object such as list or generator function.
                Each element returned by this iterator should be compatible to that regular PySDK model
                accepts.

        Yields:
            (DetectionResults): Combined inference result objects.
                This allows you directly using the result in `for` loops.
        """

        preprocessor = dg._preprocessor.create_image_preprocessor(
            self._model2.model_info,  # we do copy here to avoid modifying original model parameters
            image_backend=self._model2.image_backend,
            pad_method="",  # to disable resizing/padding
        )
        preprocessor.image_format = "RAW"  # to avoid unnecessary JPEG encoding

        for element in data:
            if element is None:
                if self._non_blocking_batch_predict:
                    yield None
                else:
                    raise Exception(
                        "Model misconfiguration: input data iterator returns None but non-blocking batch predict mode is not enabled"
                    )

            # extract frame and frame info from data
            if isinstance(element, tuple):
                # if data is tuple, we treat first element as frame data and second element as frame info
                frame, frame_info = element
            else:
                # otherwise we treat data as frame data and if it is string, we set frame info equal to frame data
                frame, frame_info = element, element if isinstance(element, str) else ""

            if frame is None:
                if self._non_blocking_batch_predict:
                    yield None
                    continue
                else:
                    raise Exception(
                        "Model misconfiguration: input data iterator returns None but non-blocking batch predict mode is not enabled"
                    )

            # do pre-processing
            preprocessed_data = preprocessor.forward(frame)
            image = preprocessed_data["image_input"]

            try:
                self._height = image.shape[0]
                self._width = image.shape[1]
            except AttributeError:
                self._height = image.size[1]
                self._width = image.size[0]

            self._tile_params = self._calculate_tile_parameters()

            tile_list = []

            for row in range(self._rows):
                for col in range(self._cols):
                    tile_list.append(self._get_slice(row, col))

            all_tiles = [True] * len(tile_list)

            if self._tile_mask is None:
                self._tile_mask = [i for i in range(self._rows * self._cols)]

            if self._motion_detect is not None:
                motion_img, base_img = detect_motion(
                    self._base_img[0] if self._base_img else None, image
                )
                self._base_img.append(base_img)
                if len(self._base_img) > int(self._motion_detect.look_back):
                    self._base_img.pop(0)

                if motion_img is None:
                    motion_detected = all_tiles
                else:
                    motion_detected = [
                        cv2.countNonZero(
                            motion_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                        )
                        > self._motion_detect.threshold
                        * (bbox[2] - bbox[0])
                        * (bbox[3] - bbox[1])
                        for bbox in tile_list
                    ]
            else:
                motion_detected = all_tiles

            tile_results = [
                {
                    "bbox": bbox,
                    "label": "LOCAL",
                    "score": 1.0,
                    "category_id": idx,
                    "tile_info": _TileInfo(
                        self._cols, self._rows, self._overlap_percent, idx
                    ),
                }
                for idx, bbox in enumerate(tile_list)
                if (motion_detected[idx] and idx in self._tile_mask)
            ]

            if self._global_tile:
                tile_results.append(
                    {
                        "bbox": [0, 0, self._width, self._height],
                        "label": "GLOBAL",
                        "score": 1.0,
                        "category_id": -999,
                    }
                )

            # generate pseudo inference results
            pp = (
                self._custom_postprocessor
                if self._custom_postprocessor
                else dg.postprocessor.DetectionResults
            )
            result = pp(
                input_image=image,
                model_image=image,
                inference_results=tile_results,
                draw_color=self._model2.overlay_color,
                line_width=self._model2.overlay_line_width,
                show_labels=self._model2.overlay_show_labels,
                show_probabilities=self._model2.overlay_show_probabilities,
                alpha=self._model2.overlay_alpha,
                font_scale=self._model2.overlay_font_scale,
                fill_color=self._model2.input_letterbox_fill_color,
                frame_info=frame_info,
                conversion=lambda x, y: (x, y),
                label_dictionary=self._model2.label_dictionary,
            )
            yield result


def _reverse_enumerate(seq: Sequence):
    return zip(range(len(seq) - 1, -1, -1), reversed(seq))


class _EdgeMixin:
    # Make sure that in accumulate_results, you increment _slice_id
    _slice_id = 0

    _top_edge: Tuple[int, int, int, int]
    _bot_edge: Tuple[int, int, int, int]
    _left_edge: Tuple[int, int, int, int]
    _right_edge: Tuple[int, int, int, int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments

    @staticmethod
    def _calculate_relevant_edges(slice_id: int, cols: int, rows: int) -> List[bool]:
        # Determine relevant edges of slice. [Top, Bot, Left, Right]
        # This ignores the edges that are the true boundaries of the image.
        relevant_edge = [True, True, True, True]

        relevant_edge[2] = False if (slice_id + 1) % cols == 1 else True
        relevant_edge[3] = False if (slice_id + 1) % cols == 0 else True
        relevant_edge[0] = False if slice_id < cols else True
        relevant_edge[1] = False if slice_id >= (rows - 1) * cols else True

        return relevant_edge

    def _calculate_overlapped_edges(
        self,
        box: MutableSequence[Union[int, float]],
        relevant_edges: Sequence[bool],
        compensation: float = 1.0,
    ) -> Tuple[bool, bool, bool, bool]:
        overlap_top, overlap_bot, overlap_left, overlap_right = (
            False,
            False,
            False,
            False,
        )

        # Compensation for rounding errors due to slicing.
        box = copy.copy(box)
        box[0] -= compensation
        box[1] -= compensation
        box[2] += compensation
        box[3] += compensation

        if relevant_edges[0]:
            overlap_top = _EdgeMixin._is_box_overlap(box, self._top_edge)
        if relevant_edges[1]:
            overlap_bot = _EdgeMixin._is_box_overlap(box, self._bot_edge)
        if relevant_edges[2]:
            overlap_left = _EdgeMixin._is_box_overlap(box, self._left_edge)
        if relevant_edges[3]:
            overlap_right = _EdgeMixin._is_box_overlap(box, self._right_edge)

        return overlap_top, overlap_bot, overlap_left, overlap_right

    @staticmethod
    def _is_box_overlap(
        box1: Sequence[Union[int, float]], box2: Sequence[Union[int, float]]
    ) -> bool:
        # Boxes are in tlbr format
        # Zero width/height rectangle check.
        if (
            box1[0] == box1[2]
            or box1[1] == box1[3]
            or box2[0] == box2[2]
            or box2[1] == box2[3]
        ):
            return False

        # Rectangles are to the left/right of each other.
        if box1[0] > box2[2] or box2[0] > box1[2]:
            return False

        # Rectangles are above/below each other. (signs inverted because y axis is inverted)
        if box1[3] < box2[1] or box2[3] < box1[1]:
            return False

        return True


class TileModel(CroppingAndDetectingCompoundModel):
    """Tiling wrapper that runs detection on every tile.

    This compound model wires a TileExtractorPseudoModel (model 1)
    to a normal detection model (model 2). Each tile is cropped, passed to
    model 2, and the resulting boxes are translated back to the original
    image coordinates. Optionally, detections from model 1 can be merged or
    deduplicated via NMS.

    Args:
        model1 (TileExtractorPseudoModel): The tile generator.
        model2 (dg.model.Model): A detection model compatible with model1's image backend.
        crop_extent (float, optional): Extra context (percent of box size) to include around
            every tile before passing it to model 2.
        crop_extent_option (CropExtentOptions, optional): How the extra context is applied.
        add_model1_results (bool, optional): If True, detections produced by model 1 are
            appended to the final result.
        nms_options (NmsOptions | None, optional): Non-maximum suppression settings performed on the merged result.

    Attributes:
        output_postprocess_type (str): Mirrors model2.output_postprocess_type so
            downstream tooling recognises this as a detection model.

    Raises:
        Exception: If the two models use different image back-ends.
    """

    # Subclass to maintain compatibility with the ObjectDetectionEvaluator, specific to pseudo-model case.
    def __init__(
        self,
        model1,
        model2,
        *,
        crop_extent=0,
        crop_extent_option=CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
        add_model1_results=False,
        nms_options: Optional[NmsOptions] = None,
    ):
        """Constructor.

        Args:
            model1 (TileExtractorPseudoModel): Tile extractor pseudo-model.
            model2 (dg.model.Model): PySDK object detection model.
            crop_extent (float): Extent of cropping in percent of bbox size.
            crop_extent_option (CropExtentOptions): Method of applying extending crop to the input image for model2.
            add_model1_results (bool): True to add detections of model1 to the combined result.
            nms_options (NmsOptions | None, optional): Non-maximum suppression (NMS) options.
        """

        if model1.image_backend != model2.image_backend:
            raise Exception(
                f"Image backends of both models should be the same, but got {model1.image_backend} and {model2.image_backend}"
            )

        super().__init__(
            model1,
            model2,
            crop_extent=crop_extent,
            crop_extent_option=crop_extent_option,
            add_model1_results=add_model1_results,
            nms_options=nms_options,
        )

        self.output_postprocess_type = self.model2.output_postprocess_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model2._release_runtime()


class LocalGlobalTileModel(TileModel):
    """Runs a fine-/coarse tiling strategy with size-based result fusion.

    Two kinds of tiles are produced:

    * Local: the regular grid (fine resolution)
    * Global: a single full-frame tile

    After detection:

    * Large objects (area >= large_object_threshold x image_area) are kept
      only from the global tile.
    * Small objects are kept only from the local tiles.

    Args:
        model1 (TileExtractorPseudoModel): Must be configured with global_tile=True.
        model2 (dg.model.Model): Detection model run on each tile.
        large_object_threshold (float, optional): Area ratio separating "large" from "small" objects.
            Defaults to 0.01.
        crop_extent (float, optional): Extra context (percent of box size) to include around
            every tile before passing it to model 2.
        crop_extent_option (CropExtentOptions, optional): How the extra context is applied.
        add_model1_results (bool, optional): If True, detections produced by model 1 are
            appended to the final result.
        nms_options (NmsOptions | None, optional): Non-maximum suppression settings performed on the merged result.

    Note:
        Unlike BoxFusionLocalGlobalTileModel, no edge fusion is applied;
        objects that overlap tile borders may be duplicated.
    """

    def __init__(
        self,
        model1,
        model2,
        large_object_threshold: float = 0.01,
        *,
        crop_extent=0,
        crop_extent_option=CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
        add_model1_results=False,
        nms_options: Optional[NmsOptions] = None,
    ):
        """Constructor.

        Args:
            model1 (TileExtractorPseudoModel): Tile extractor pseudo-model.
            model2 (dg.model.Model): PySDK object detection model.
            large_object_threshold (float): A threshold to determine if an object is considered large or not.
                This is relative to the area of the original image.
            crop_extent (float): Extent of cropping in percent of bbox size.
            crop_extent_option (CropExtentOptions): Method of applying extending crop to the input image for model2.
            add_model1_results (bool): True to add detections of model1 to the combined result.
            nms_options (NmsOptions | None, optional): Non-maximum suppression (NMS) options.
        """

        super().__init__(
            model1,
            model2,
            crop_extent=crop_extent,
            crop_extent_option=crop_extent_option,
            add_model1_results=add_model1_results,
            nms_options=nms_options,
        )

        self._large_obj_thr = large_object_threshold

    def transform_result2(self, result2):
        """Transform result of the **second model**.

        This implementation combines results of the **second model** over all bboxes detected by the first model,
        translating bbox coordinates to original image coordinates.

        Args:
            result2 (dg.postprocessor.DetectionResults): Detection result of the **second model**.

        Returns:
            (DetectionResults or None): Combined results of the **second model** over all bboxes detected by the first model,
                where bbox coordinates are translated to original image coordinates. Returns None if no new frame is available.
        """

        result1 = result2.info.result1
        idx = result2.info.sub_result

        # This presupposes the last element is the GLOBAL box.
        width, height = result1.results[-1]["bbox"][2:4]
        assert (
            result1.results[-1]["category_id"] == -999
        ), "Global tile does not exist, make sure the tile extractor is set to extract a global tile."

        def _is_large_object(box, thr):
            area = (box[2] - box[0]) * (box[3] - box[1])
            return area >= width * height * thr

        if idx >= 0:
            # adjust bbox coordinates to original image coordinates
            result2 = clone_result(result2)
            is_global = True if result1.results[idx]["label"] == "GLOBAL" else False
            overlap_percent = result1.results[0]["tile_info"].overlap

            x, y = result1.results[idx]["bbox"][:2]

            if is_global:
                for i, r in _reverse_enumerate(result2._inference_results):
                    if "bbox" not in r:
                        continue
                    r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()
                    if not _is_large_object(
                        r["bbox"], self._large_obj_thr * overlap_percent * 5
                    ):
                        del result2._inference_results[i]
            else:
                for i, r in _reverse_enumerate(result2._inference_results):
                    if "bbox" not in r:
                        continue
                    r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()
                    if _is_large_object(r["bbox"], self._large_obj_thr):
                        del result2._inference_results[i]

        ret = None
        if result1 is self._current_result1:
            # frame continues: append second model results to the combined result
            if self._current_result is not None and idx >= 0:
                self._current_result.extend(result2._inference_results)
        else:
            # new frame comes: return combined result of previous frame
            ret = self._finalize_current_result()
            self._current_result = copy.deepcopy(result2._inference_results)
            self._current_result1 = result1

        return ret


class BoxFusionTileModel(_EdgeMixin, TileModel):
    """TileModel with edge/overlap-aware bounding-box fusion.

    After model 2 has produced detections for every tile, boxes whose centres
    fall within the user-defined edge band are compared with neighbours.
    If the 1-D IoU in either axis exceeds fusion_threshold the boxes are
    merged (weighted-box fusion).

    Args:
        model1 (TileExtractorPseudoModel): Tile generator.
        model2 (dg.model.Model): Detection model.
        edge_threshold (float, optional): Size of the edge band expressed as a fraction of tile
            width/height. Defaults to 0.02.
        fusion_threshold (float, optional): 1-D IoU needed to fuse two edge boxes. Defaults to 0.8.
        crop_extent (float, optional): Extra context (percent of box size) to include around
            every tile before passing it to model 2.
        crop_extent_option (CropExtentOptions, optional): How the extra context is applied.
        add_model1_results (bool, optional): If True, detections produced by model 1 are
            appended to the final result.
        nms_options (NmsOptions | None, optional): Non-maximum suppression settings performed on the merged result.
    """

    def __init__(
        self,
        model1,
        model2,
        edge_threshold: float = 0.02,
        fusion_threshold: float = 0.8,
        *,
        crop_extent=0,
        crop_extent_option=CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
        add_model1_results=False,
        nms_options: Optional[NmsOptions] = None,
    ):
        """Constructor.

        Args:
            model1 (TileExtractorPseudoModel): Tile extractor pseudo-model.
            model2 (dg.model.Model): PySDK object detection model.
            edge_threshold (float, optional): A threshold to determine if an object is considered an edge detection
                or not. The edge_threshold determines the amount of space next to the tiles edges
                where if a detection overlaps this space it is considered an edge detection. This
                edge space is relative (a percent) of the width/height of a tile.
            fusion_threshold (float, optional): A threshold to determine whether or not to fuse two edge detections.
                This corresponds to the 1D-IoU of two boxes, of either dimension. If the boxes
                overlap in both dimensions and one of the dimension's 1D-IoU is greater than the
                fusion_threshold, the boxes are fused.
            crop_extent (float, optional): Extent of cropping in percent of bbox size.
            crop_extent_option (CropExtentOptions, optional): Method of applying extending crop to the input image for model2.
            add_model1_results (bool, optional): True to add detections of model1 to the combined result.
            nms_options (NmsOptions | None, optional): Non-maximum suppression (NMS) options.
        """

        super().__init__(
            model1,
            model2,
            crop_extent=crop_extent,
            crop_extent_option=crop_extent_option,
            add_model1_results=add_model1_results,
            nms_options=nms_options,
        )

        self._edge_thr = edge_threshold
        self._fusion_thr = fusion_threshold

    def _categorize(self, cols, rows, slice_id, dets: MutableSequence[dict]):
        # Separates central boxes from edge boxes.
        relevant_edges = _EdgeMixin._calculate_relevant_edges(slice_id, cols, rows)

        central_boxes = []
        edge_boxes = []

        for i, det in enumerate(dets):
            if "bbox" not in det:
                continue

            overlap_top, overlap_bot, overlap_left, overlap_right = (
                self._calculate_overlapped_edges(det["bbox"], relevant_edges)
            )

            if overlap_top:
                edge_boxes.append(i)  # edge_boxes.append(det)
            elif overlap_bot:
                edge_boxes.append(i)
            elif overlap_left:
                edge_boxes.append(i)
            elif overlap_right:
                edge_boxes.append(i)
            else:
                central_boxes.append(i)  # central_boxes.append(i)

        return central_boxes, edge_boxes

    def _finalize_current_result(self):
        if self._current_result is not None and self._current_result1 is not None:
            if self._add_model1_results:
                ret = clone_result(self._current_result1)
                ret._inference_results.extend(self._current_result)
            else:
                ret = copy.copy(self._current_result1)
                ret._inference_results = self._current_result

            height, width = image_size(ret.image)

            # Perform fusion of boxes labeled as edge detections.
            edge_boxes = []
            for i, r in _reverse_enumerate(ret._inference_results):
                if "wbf_info" in r:
                    # Normalize
                    r["wbf_info"] = np.divide(
                        r["wbf_info"], [width, height, width, height]
                    ).tolist()
                    edge_boxes.append(ret._inference_results.pop(i))

            edge_boxes = edge_box_fusion(edge_boxes, self._fusion_thr)

            for r in edge_boxes:
                r["label"] = self.model2.label_dictionary[r["category_id"]]
                r["bbox"] = np.multiply(
                    r["bbox"], [width, height, width, height]
                ).tolist()

            ret._inference_results.extend(edge_boxes)

            if self._nms_options is not None:
                # apply NMS to combined result
                nms(
                    ret,
                    iou_threshold=self._nms_options.threshold,
                    use_iou=self._nms_options.use_iou,
                    box_select=self._nms_options.box_select,
                )

            return ret
        return None

    def transform_result2(self, result2):
        """Transform result of the **second model**.

        This implementation combines results of the **second model** over all bboxes detected by the first model,
        translating bbox coordinates to original image coordinates.

        Args:
            result2 (dg.postprocessor.DetectionResults): Detection result of the **second model**.

        Returns:
            (DetectionResults or None): Combined results of the **second model** over all bboxes detected by the first model,
                where bbox coordinates are translated to original image coordinates. Returns None if no new frame is available.
        """

        result1 = result2.info.result1
        idx = result2.info.sub_result

        if idx >= 0:
            # this is probably not the best way to do it.
            # maybe make _categorize instead returns indices.
            x, y = result1.results[idx]["bbox"][:2]

            if result1.results[idx]["label"] != "GLOBAL":
                tile_info = result1.results[idx]["tile_info"]
                slice_id = tile_info.id
                cols = tile_info.cols
                rows = tile_info.rows

                # Generate edge boundaries
                try:
                    h, w, _ = result2.image.shape
                except AttributeError:
                    w, h = result2.image.size
                self._top_edge = (0, 0, w, int(self._edge_thr * h))
                self._bot_edge = (0, int(h - self._edge_thr * h), w, h)
                self._left_edge = (0, 0, int(w * self._edge_thr), h)
                self._right_edge = (int(w - w * self._edge_thr), 0, w, h)

                central_boxes, edge_boxes = self._categorize(
                    cols, rows, slice_id, result2._inference_results
                )

                # for r in edge_boxes:
                #     r['wbf_info'] = np.add(r["bbox"], [x, y, x, y]).tolist()

                # central_boxes.extend(edge_boxes)
                # result2._inference_results = central_boxes

                for i in edge_boxes:
                    result2._inference_results[i]["wbf_info"] = np.add(
                        result2._inference_results[i]["bbox"], [x, y, x, y]
                    ).tolist()

            # adjust bbox coordinates to original image coordinates
            result2 = clone_result(result2)
            for r in result2._inference_results:
                if "bbox" not in r:
                    continue
                r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()

        ret = None
        if result1 is self._current_result1:
            # frame continues: append second model results to the combined result
            if self._current_result is not None and idx >= 0:
                self._current_result.extend(result2._inference_results)

        else:
            # new frame comes: return combined result of previous frame
            ret = self._finalize_current_result()
            self._current_result = copy.deepcopy(result2._inference_results)
            self._current_result1 = result1

        return ret


class BoxFusionLocalGlobalTileModel(BoxFusionTileModel):
    """Local-/global-tiling plus edge fusion.

    Combines the size-based filtering of LocalGlobalTileModel with the
    edge-aware fusion of BoxFusionTileModel.

    Args:
        model1 (TileExtractorPseudoModel): Must output a global tile.
        model2 (dg.model.Model): Detection model.
        large_object_threshold (float, optional): Area ratio that classifies a detection as "large".
            Defaults to 0.01.
        edge_threshold (float, optional): Width of the edge band as a fraction of tile dimensions.
            Defaults to 0.02.
        fusion_threshold (float, optional): 1-D IoU used by the fusion logic. Defaults to 0.8.
        crop_extent (float, optional): Extra context (percent of box size) to include around
            every tile before passing it to model 2.
        crop_extent_option (CropExtentOptions, optional): How the extra context is applied.
        add_model1_results (bool, optional): If True, detections produced by model 1 are
            appended to the final result.
        nms_options (NmsOptions | None, optional): Non-maximum suppression settings performed on the merged result.
    """

    def __init__(
        self,
        model1,
        model2,
        large_object_threshold: float = 0.01,
        edge_threshold: float = 0.02,
        fusion_threshold: float = 0.8,
        *,
        crop_extent=0,
        crop_extent_option=CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
        add_model1_results=False,
        nms_options: Optional[NmsOptions] = None,
    ):
        """Constructor.

        Args:
            model1 (TileExtractorPseudoModel): Tile extractor pseudo-model.
            model2 (dg.model.Model): PySDK object detection model.
            large_object_threshold (float): A threshold to determine if an object is considered large or not.
                This is relative to the area of the original image.
            edge_threshold (float): A threshold to determine if an object is considered an edge detection
                or not. The edge_threshold determines the amount of space next to the tiles edges
                where if a detection overlaps this space it is considered an edge detection. This
                edge space is relative (a percent) of the width/height of a tile.
            fusion_threshold (float): A threshold to determine whether or not to fuse two edge detections.
                This corresponds to the 1D-IoU of two boxes, of either dimension. If the boxes
                overlap in both dimensions and one of the dimension's 1D-IoU is greater than the
                fusion_threshold, the boxes are fused.
            crop_extent (float): Extent of cropping in percent of bbox size.
            crop_extent_option (CropExtentOptions): Method of applying extending crop to the input image for model2.
            add_model1_results (bool): True to add detections of model1 to the combined result.
            nms_options (NmsOptions | None, optional): Non-maximum suppression (NMS) options.
        """

        super().__init__(
            model1,
            model2,
            edge_threshold,
            fusion_threshold,
            crop_extent=crop_extent,
            crop_extent_option=crop_extent_option,
            add_model1_results=add_model1_results,
            nms_options=nms_options,
        )

        self._large_obj_thr = large_object_threshold

    def transform_result2(self, result2):
        """Transform result of the **second model**.

        This implementation combines results of the **second model** over all bboxes detected by the first model,
        translating bbox coordinates to original image coordinates.

        Args:
            result2 (dg.postprocessor.DetectionResults): Detection result of the **second model**.

        Returns:
            (DetectionResults or None): Combined results of the **second model** over all bboxes detected by the first model,
                where bbox coordinates are translated to original image coordinates. Returns None if no new frame is available.
        """

        result1 = result2.info.result1
        idx = result2.info.sub_result

        # This presupposes the last element is the GLOBAL box.
        width, height = result1.results[-1]["bbox"][2:4]
        assert (
            result1.results[-1]["category_id"] == -999
        ), "Global tile does not exist, make sure the tile extractor is set to extract a global tile."

        def _is_large_object(box, thr):
            area = (box[2] - box[0]) * (box[3] - box[1])
            return area >= width * height * thr

        if idx >= 0:
            # adjust bbox coordinates to original image coordinates
            is_global = True if result1.results[idx]["label"] == "GLOBAL" else False
            overlap_percent = result1.results[0]["tile_info"].overlap

            x, y = result1.results[idx]["bbox"][:2]

            if is_global:
                for i, r in _reverse_enumerate(result2._inference_results):
                    if "bbox" not in r:
                        continue
                    r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()
                    if not _is_large_object(
                        r["bbox"], self._large_obj_thr * overlap_percent * 5
                    ):
                        del result2._inference_results[i]
            else:
                for i, r in _reverse_enumerate(result2._inference_results):
                    if "bbox" not in r:
                        continue
                    if _is_large_object(r["bbox"], self._large_obj_thr):
                        del result2._inference_results[i]

                return super().transform_result2(result2)

        ret = None
        if result1 is self._current_result1:
            # frame continues: append second model results to the combined result
            if self._current_result is not None and idx >= 0:
                self._current_result.extend(result2._inference_results)

        else:
            # new frame comes: return combined result of previous frame
            ret = self._finalize_current_result()
            self._current_result = result2._inference_results
            self._current_result1 = result1

        return ret
