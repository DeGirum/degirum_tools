import copy
from typing import List, MutableSequence, Optional, Sequence, Tuple, Union

import numpy as np

import degirum as dg
from degirum_tools import CroppingAndDetectingCompoundModel, CropExtentOptions, ModelLike, NmsOptions
from degirum_tools.math_support import nms, edge_box_fusion

class TileExtractorPseudoModel(ModelLike):
    """
    Pseudo model class which extracts regions from given image according to given ROI boxes.
    """

    def __init__(
        self,
        cols: int,
        rows: int,
        overlap_percent: float,
        model2: dg.model.Model,
        *,
        global_tile:bool = False,
        #tile_mask: Union[list, np.ndarray],
    ):
        """
        Constructor.

        Args:
            cols: Number of columns to divide the image into.
            rows: Number of rows to divide the image into.
            model2: model, which will be used as a second step of the compound model pipeline
            tile_mask: NOT IMPLEMENTED YET. A col x row list or nd.array of bools which indicates whether or not to process a tile.
            global_tile: Indicates whether the global (whole) image should also be sent to model2.
        """

        self._cols = cols
        self._rows = rows
        self._overlap_percent = overlap_percent
        self._model2 = model2
        self._non_blocking_batch_predict = False
        self._global_tile = global_tile
        self._aspect_aware = True # Right now have no intentions of ever disablint this. Probably should refactor out.

    @property
    def non_blocking_batch_predict(self):
        return self._non_blocking_batch_predict

    @non_blocking_batch_predict.setter
    def non_blocking_batch_predict(self, val: bool):
        self._non_blocking_batch_predict = val

    @property
    def image_backend(self) -> str:
        return self._model2.image_backend
    
    def _calculate_tile_parameters(self) -> List[float]:
        model_aspect_ratio = self._model2.model_info.InputW[0] / self._model2.model_info.InputH[0]
        
        tile_width  = self._width  / (self._cols - self._overlap_percent * (self._cols - 1))
        tile_height = self._height / (self._rows - self._overlap_percent * (self._rows - 1))

        expand_offsets = [0.0, 0.0]

        if self._aspect_aware:
            tile_aspect_ratio = tile_width / tile_height

            if tile_aspect_ratio < model_aspect_ratio:
                # Expand the width
                if model_aspect_ratio >= 1:
                    dim = tile_height * model_aspect_ratio
                else:
                    dim = tile_height / model_aspect_ratio
                
                expand_offsets[0] =  dim - tile_width
            elif tile_aspect_ratio > model_aspect_ratio:
                # Expand the height
                if model_aspect_ratio >= 1:
                    dim = tile_width / model_aspect_ratio
                else:
                    dim = tile_width * model_aspect_ratio

                expand_offsets[1] =  dim - tile_height

        if expand_offsets[0] > tile_width * 2:
            raise Exception('Horizontal overlap is greater than 100%. Lower the amount of columns.')
        elif expand_offsets[1] > tile_height * 2:
            raise Exception('Vertical overlap is greater than 100%. Lower the amount of rows.')

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
            data (iterator): Inference input data iterator object such as list or generator function.
            Each element returned by this iterator should be compatible to that regular PySDK model
            accepts

        Returns:
            Generator object which iterates over combined inference result objects.
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

            # do pre-processing
            preprocessed_data = preprocessor.forward(frame)
            image = preprocessed_data["image_input"]

            self._height = image.shape[0]
            self._width =  image.shape[1]
            self._tile_params = self._calculate_tile_parameters()

            tile_list = []

            for row in range(self._rows):
                for col in range(self._cols):
                    tile_list.append(self._get_slice(row, col))

            tile_list = [
                {"bbox": bbox, "label":f"LOCAL_{self._cols}x{self._rows}@{self._overlap_percent}_{idx}", "score": 1.0, "category_id": idx}
                for idx, bbox in enumerate(tile_list)
            ]

            if self._global_tile:
                tile_list.append({"bbox": [0, 0, self._width, self._height], "label":f"GLOBAL", "score": 1.0, "category_id": -999})

            # generate pseudo inference results
            result = dg.postprocessor.DetectionResults(
                model_params=self._model2._model_parameters,
                input_image=image,
                model_image=image,
                inference_results=tile_list,
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

def _reverse_enumerate(l: Sequence):
    return zip(range(len(l) - 1, -1, -1), reversed(l))

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
        relevant_edge = [True, True, True, True]

        relevant_edge[2] = False if (slice_id + 1) % cols == 1 else True
        relevant_edge[3] = False if (slice_id + 1) % cols == 0 else True
        relevant_edge[0] = False if slice_id < cols else True
        relevant_edge[1] = False if slice_id >= (rows - 1) * cols else True

        return relevant_edge
    
    def _calculate_overlapped_edges(self, 
                                    box: MutableSequence[Union[int, float]], 
                                    relevant_edges: Sequence[bool], 
                                    compensation: float=1.0) -> Tuple[bool, bool, bool, bool]:
        overlap_top, overlap_bot, overlap_left, overlap_right = False, False, False, False

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
    def _is_box_overlap(box1: Sequence[Union[int, float]], box2: Sequence[Union[int, float]])-> bool:
        # Boxes are in tlbr format
        # Zero width/height rectangle check.
        if (box1[0] == box1[2] or box1[1] == box1[3] or
            box2[0] == box2[2] or box2[1] == box2[3]):
            return False
        
        # Rectangles are to the left/right of each other.
        if (box1[0] > box2[2] or box2[0] > box1[2]):
            return False
        
        # Rectangles are above/below each other. (signs inverted because y axis is inverted)
        if (box1[3] < box2[1] or box2[3] < box1[1]):
            return False
        
        return True
        
TileModel = CroppingAndDetectingCompoundModel

class LocalGlobalTileModel(TileModel):
    def __init__(
        self,
        model1,
        model2,
        large_object_threshold,
        *,
        crop_extent=0,
        crop_extent_option=CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
        add_model1_results=False,
        nms_options: Optional[NmsOptions] = None,
    ):
        super().__init__(model1, 
                         model2, 
                         crop_extent=crop_extent, 
                         crop_extent_option=crop_extent_option, 
                         add_model1_results=add_model1_results, 
                         nms_options=nms_options)

        self._large_obj_thr = large_object_threshold

    def transform_result2(self, result2):
        """
        Transform result of the second model.

        This implementation combines results of the **second** model over all bboxes detected by the first model,
        translating bbox coordinates to original image coordinates.

        Args:
            result2: detection result of the second model

        Returns:
            Combined results of the **second** model over all bboxes detected by the first model,
            where bbox coordinates are translated to original image coordinates.
        """

        result1 = result2.info.result1
        idx = result2.info.sub_result

        # This presupposes the last element is the GLOBAL box.
        width, height = result1.results[-1]['bbox'][2:4]
        assert result1.results[-1]['category_id'] == -999, 'Global tile does not exist, make sure the tile extractor is set to extract a global tile.'

        def _is_large_object(box, thr):
            area = (box[2] - box[0]) * (box[3] - box[1])
            return area >= width * height * thr

        if idx >= 0:
            # adjust bbox coordinates to original image coordinates
            global_result = True if result1.results[idx]['label'].split('_')[0] == 'GLOBAL' else False
            overlap_percent = float(result1.results[0]['label'].split('_')[1].split('@')[1])

            x, y = result1.results[idx]["bbox"][:2]

            if global_result:
                for i, r in _reverse_enumerate(result2._inference_results):
                    if "bbox" not in r:
                        continue
                    r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()
                    if not _is_large_object(r['bbox'], self._large_obj_thr * overlap_percent * 5):
                        del result2._inference_results[i]
            else:
                for i, r in _reverse_enumerate(result2._inference_results):
                    if "bbox" not in r:
                        continue
                    r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()
                    if _is_large_object(r['bbox'], self._large_obj_thr):
                        del result2._inference_results[i]

            if self._add_model1_results:
                # prepend result from the first model to the combined result if requested
                result2._inference_results.insert(0, result1.results[idx])

        ret = None
        if result1 is self._current_result1:
            # frame continues: append second model results to the combined result
            if self._current_result is not None and idx >= 0:
                self._current_result._inference_results.extend(
                    result2._inference_results
                )

        else:
            # new frame comes: return combined result of previous frame
            ret = self._finalize_current_result(result1)
            self._current_result = result2
            self._current_result1 = result1

        return ret

class BoxFusionTileModel(_EdgeMixin, TileModel):
    def __init__(
        self,
        model1,
        model2,
        edge_threshold,
        fusion_threshold,
        *,
        crop_extent=0,
        crop_extent_option=CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
        add_model1_results=False,
        nms_options: Optional[NmsOptions] = None,
    ):
        super().__init__(model1,
                         model2,
                         crop_extent=crop_extent, 
                         crop_extent_option=crop_extent_option, 
                         add_model1_results=add_model1_results, 
                         nms_options=nms_options)
        
        self._edge_thr = edge_threshold
        self._fusion_thr = fusion_threshold

    def _categorize(self, cols, rows, slice_id, dets: MutableSequence[dict]):
        relevant_edges = _EdgeMixin._calculate_relevant_edges(slice_id, cols, rows)

        central_boxes = []
        edge_boxes = []

        for det in dets:
            if 'bbox' not in det:
                continue

            overlap_top, overlap_bot, overlap_left, overlap_right = self._calculate_overlapped_edges(det['bbox'], relevant_edges)

            if overlap_top:
                edge_boxes.append(det)
            elif overlap_bot:
                edge_boxes.append(det)
            elif overlap_left:
                edge_boxes.append(det)
            elif overlap_right:
                edge_boxes.append(det)
            else:
                central_boxes.append(det)

        return central_boxes, edge_boxes
    
    def _finalize_current_result(self, result1):
        if self._current_result is not None:
            # patch combined result image to be original image # IS THIS GOING TO BE AN ISSUE
            self._current_result._input_image = result1.image

            height, width, _ = result1.image.shape

            edge_boxes = []
            for i, r in _reverse_enumerate(self._current_result._inference_results):
                if "wbf_info" in r:
                    # Normalize
                    r["wbf_info"] = np.divide(r["wbf_info"], [width, height, width, height]).tolist()
                    edge_boxes.append(self._current_result._inference_results.pop(i)) #I'm not sure this works hopefully it does.

            edge_boxes = edge_box_fusion(edge_boxes, self._fusion_thr)

            for r in edge_boxes:
                r['label'] = self.model2.label_dictionary[r['category_id']]
                r["bbox"] = np.multiply(r["bbox"], [width, height, width, height]).tolist()
            
            self._current_result._inference_results.extend(edge_boxes)

            if self._nms_options is not None:
                # apply NMS to combined result
                nms(
                    self._current_result,
                    iou_threshold=self._nms_options.threshold,
                    use_iou=self._nms_options.use_iou,
                    box_select=self._nms_options.box_select,
                )
            return self._current_result
        return None
    
    def transform_result2(self, result2):
        """
        Transform result of the second model.

        This implementation combines results of the **second** model over all bboxes detected by the first model,
        translating bbox coordinates to original image coordinates.

        Args:
            result2: detection result of the second model

        Returns:
            Combined results of the **second** model over all bboxes detected by the first model,
            where bbox coordinates are translated to original image coordinates.
        """

        result1 = result2.info.result1
        idx = result2.info.sub_result

        if idx >= 0:
            # this is probably not the best way to do it. things i might change in the future
            # _categorize instead returns indices. Instead of storing edge_box flag in the dict, store the inference results separately?
            x, y = result1.results[idx]["bbox"][:2]

            slice_params = result1.results[idx]['label']

            if slice_params != "GLOBAL":
                slice_id = int(slice_params.split('_')[2])
                cols = int(slice_params.split('_')[1].split('@')[0].split('x')[0])
                rows = int(slice_params.split('_')[1].split('@')[0].split('x')[1])

                # Generate edge boundaries

                h, w, _ = result2.image.shape
                self._top_edge = (0, 0, w, int(self._edge_thr * h))
                self._bot_edge = (0, int(h - self._edge_thr * h), w, h)
                self._left_edge = (0, 0, int(w * self._edge_thr), h)
                self._right_edge = (int(w - w * self._edge_thr), 0, w, h)

                central_boxes, edge_boxes = self._categorize(cols, rows, slice_id, result2._inference_results)

                for r in edge_boxes:
                    r['wbf_info'] = np.add(r["bbox"], [x, y, x, y]).tolist() # probably not the best way to do this also

                central_boxes.extend(edge_boxes)
                result2._inference_results = central_boxes

            # adjust bbox coordinates to original image coordinates
            for r in result2._inference_results:
                if "bbox" not in r:
                    continue
                r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()
            if self._add_model1_results:
                # prepend result from the first model to the combined result if requested
                result2._inference_results.insert(0, result1.results[idx])

        ret = None
        if result1 is self._current_result1:
            # frame continues: append second model results to the combined result
            if self._current_result is not None and idx >= 0:
                self._current_result._inference_results.extend(
                    result2._inference_results
                )

        else:
            # new frame comes: return combined result of previous frame
            ret = self._finalize_current_result(result1)
            self._current_result = result2
            self._current_result1 = result1

        return ret


class BoxFusionLocalGlobalTileModel(BoxFusionTileModel):
    def __init__(
        self,
        model1,
        model2,
        large_object_threshold,
        edge_threshold,
        fusion_threshold,
        *,
        crop_extent=0,
        crop_extent_option=CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
        add_model1_results=False,
        nms_options: Optional[NmsOptions] = None,
    ):
        super().__init__(model1, 
                         model2, 
                         edge_threshold, 
                         fusion_threshold, 
                         crop_extent=crop_extent, 
                         crop_extent_option=crop_extent_option, 
                         add_model1_results=add_model1_results, 
                         nms_options=nms_options)

        self._large_obj_thr = large_object_threshold

    def transform_result2(self, result2):
        """
        Transform result of the second model.

        This implementation combines results of the **second** model over all bboxes detected by the first model,
        translating bbox coordinates to original image coordinates.

        Args:
            result2: detection result of the second model

        Returns:
            Combined results of the **second** model over all bboxes detected by the first model,
            where bbox coordinates are translated to original image coordinates.
        """

        result1 = result2.info.result1
        idx = result2.info.sub_result

        # This presupposes the last element is the GLOBAL box.
        width, height = result1.results[-1]['bbox'][2:4]
        assert result1.results[-1]['category_id'] == -999, 'Global tile does not exist, make sure the tile extractor is set to extract a global tile.'

        def _is_large_object(box, thr):
            area = (box[2] - box[0]) * (box[3] - box[1])
            return area >= width * height * thr

        if idx >= 0:
            # adjust bbox coordinates to original image coordinates
            global_result = True if result1.results[idx]['label'].split('_')[0] == 'GLOBAL' else False
            overlap_percent = float(result1.results[0]['label'].split('_')[1].split('@')[1])

            x, y = result1.results[idx]["bbox"][:2]

            if global_result:
                for i, r in _reverse_enumerate(result2._inference_results):
                    if "bbox" not in r:
                        continue
                    r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()
                    if not _is_large_object(r['bbox'], self._large_obj_thr * overlap_percent * 5):
                        del result2._inference_results[i]
            else:
                for i, r in _reverse_enumerate(result2._inference_results):
                    if "bbox" not in r:
                        continue
                    #r["bbox"] = np.add(r["bbox"], [x, y, x, y]).tolist()
                    if _is_large_object(r['bbox'], self._large_obj_thr):
                        del result2._inference_results[i]

                return super().transform_result2(result2)

            if self._add_model1_results:
                # prepend result from the first model to the combined result if requested
                result2._inference_results.insert(0, result1.results[idx])

        ret = None
        # Equivalent to accumulate_results?
        if result1 is self._current_result1:
            # frame continues: append second model results to the combined result
            if self._current_result is not None and idx >= 0:
                self._current_result._inference_results.extend(
                    result2._inference_results
                )

        else:
            # new frame comes: return combined result of previous frame
            ret = self._finalize_current_result(result1)
            self._current_result = result2
            self._current_result1 = result1

        return ret