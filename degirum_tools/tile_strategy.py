from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generator, Iterable, List, Sequence, Tuple, Union

import numpy as np
from degirum_tools.math_support import weighted_boxes_fusion
from degirum.aiclient import ModelParams


def _reverse_enumerate(l: Iterable):
    return zip(range(len(l) - 1, -1, -1), reversed(l))


class _TileType(Enum):
    BASIC_SLICE = 1
    GLOBAL = 2


class _EdgeMixin:
    # Make sure that in accumulate_results, you increment _slice_id
    _slice_id = 0

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
                                    box: Sequence[Union[int, float]], 
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
    
class BaseTileStrategy(ABC):
    @abstractmethod
    def _generate_tiles(self, image: np.ndarray):
        ''''''
    @abstractmethod
    def _accumulate_results(self, results: Sequence[dict], info: Any):
        ''''''
    @abstractmethod
    def _get_results(self):
        ''''''

    def _set_model_parameters(self, params: ModelParams):
        self._model_params = params

    def _set_label_dict(self, label_dict: dict):
        self._label_dict = label_dict

    @staticmethod
    def _translate_box_abs(tl: Sequence[Union[int, float]], 
                           box: Sequence[Union[int, float]], 
                           in_place: bool=False) -> Sequence[Union[int, float]]:
        """ Translates bounding box to original image based on the position of a slice's top left corner.
        Args:
            tl: Top left of a tile with format relative to original image (x, y).
            box: Box coordinates (tlbr) relative to a slice.
            in_place: modifies the original coordinates object if True, else returns a copy of the coordinates.
        """
        if not in_place:
            box = copy.copy(box)

        box[0] += tl[0]
        box[2] += tl[0]
        box[1] += tl[1]
        box[3] += tl[1]

        return box
    
    @staticmethod
    def _translate_box_scaled(scale: Sequence[float], 
                              box: Sequence[Union[int, float]], 
                              in_place: bool=False) -> Sequence[float]:
        """Translates bounding box to original image based on the scaling factor.
        Args:
            scale: A singular float for the scaling ratio if the image was scaled evenly, 
                   or a tuple containing the x scaling factor and y scaling factor respectively.
            box: Box coordinates (tlbr) relative to a slice.
            in_place: modifies the original coordinates object if True, else returns a copy of the coordinates.
        """
        if isinstance(scale, tuple):
            scale_x = scale[0]
            scale_y = scale[1]
        elif isinstance(scale, float):
            scale_x = scale
            scale_y = scale
        else:
            raise TypeError("Invalid scaling factor.")

        if not in_place:
            box = copy.copy(box)

        box[0] /= scale_x
        box[2] /= scale_x
        box[1] /= scale_y
        box[3] /= scale_y

        return box
    

class SimpleTiling(BaseTileStrategy):
    def __init__(self, cols: int, rows: int, overlap_percent: float):
        self._num_cols = cols
        self._num_rows = rows
        self._overlap_percent = overlap_percent
        self._results = []
        self._aspect_aware = True

    def _calculate_tile_parameters(self) -> List[float]:
        model_aspect_ratio = self._model_params.InputW[0] / self._model_params.InputH[0]
        
        tile_width  = self._width  / (self._num_cols - self._overlap_percent * (self._num_cols - 1))
        tile_height = self._height / (self._num_rows - self._overlap_percent * (self._num_rows - 1))

        expand_offsets = [0, 0]

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
    
    def _get_slice(self, row: int, col: int) -> Tuple[np.ndarray, Tuple[_TileType, List[int]]]:
        tile_width, tile_height, expand_col, expand_row = self._tile_params

        # Calculate tile bounding box
        tlx = col * tile_width - col * self._overlap_percent * tile_width
        brx = tlx + tile_width
        if col == (self._num_cols-1):
            brx = self._width
        tly = row * tile_height - row * self._overlap_percent * tile_height
        bry = tly + tile_height
        if row == self._num_rows - 1:
            bry = self._height

        # Expand tile to fit the aspect ratio.
        if expand_row:
            expand_row_half = expand_row / 2

            # If first row
            if row == 0:
                bry += expand_row
            # If last row
            elif row == self._num_rows - 1:
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
            elif col == self._num_cols - 1:
                tlx -= expand_col
            else:
                tlx -= expand_col_half
                brx += expand_col_half
        
        info = list(map(round, [tlx, tly, brx, bry]))

        return self._image[info[1]:info[3], info[0]:info[2]], (_TileType.BASIC_SLICE, info)
    
    def _generate_tiles(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, Tuple[_TileType, List]], None, None]:
        self._image = image
        h, w, _ = image.shape

        self._width = w
        self._height = h
        self._tile_params = self._calculate_tile_parameters()

        def source():
            for row in range(self._num_rows):
                for col in range(self._num_cols):
                    yield self._get_slice(row, col)
        
        return source()

    def _accumulate_results(self, results: Sequence[dict], info: Tuple[_TileType, List]):
        if info[0] == _TileType.BASIC_SLICE:
            for res in results:
                if res.get('bbox') is None:
                    continue
                
                SimpleTiling._translate_box_abs(info[1][0:2], res['bbox'], in_place=True)

            self._results.extend(results)
        
    def _get_results(self) -> List[dict]:
        results = copy.deepcopy(self._results)
        self._results.clear()
        return results
    
    def __str__(self) -> str:
        return '{}x{} tiles, {}% overlap.'.format(self._num_cols, self._num_rows, self._overlap_percent * 100)


class LocalGlobalTiling(SimpleTiling):
    def __init__(self, cols: int, rows: int, overlap_percent: float, large_obj_threshold: float):
        super().__init__(cols, rows, overlap_percent)
        self._large_obj_thr = large_obj_threshold

    def _generate_tiles(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, Tuple[_TileType, List]], None, None]:
        super_gen = super(LocalGlobalTiling, self)._generate_tiles(image)

        self._large_obj_size = self._large_obj_thr * self._width * self._height

        for tile in super_gen:
            yield tile

        yield image, (_TileType.GLOBAL, [0, 0])

    def _accumulate_results(self, results: Sequence[dict], info: Tuple[_TileType, List]):
        def _is_large_object(box):
            area = (box[2] - box[0]) * (box[3] - box[1])

            return area >= self._large_obj_size

        # This indicates the global image scope/large object detections.
        if info[0] == _TileType.GLOBAL:
            for i, res in _reverse_enumerate(results):
                if res.get('bbox') is None:
                    continue
                if not _is_large_object(res['bbox']):
                    del results[i]
        elif info[0] == _TileType.BASIC_SLICE:
            for i, res in _reverse_enumerate(results):
                if res.get('bbox') is None:
                    continue
                if _is_large_object(res['bbox']):
                    del results[i]
                else:
                    SimpleTiling._translate_box_abs(info[1][0:2], res['bbox'], in_place=True)
        
        self._results.extend(results)


# Performs WBF on edge boxes and NMS on central boxes.
class WBFSimpleTiling(_EdgeMixin, SimpleTiling):
    def __init__(self, cols: int, rows: int, overlap_percent: float, edge_thr: float=0.02, wbf_thr: float=0.8):
        super().__init__(cols, rows, overlap_percent)
        self._edge_thr = edge_thr
        self._wbf_thr = wbf_thr
        self._edge_results = []

    def _generate_tiles(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, Tuple[_TileType, List]], None, None]:
        super_gen = super(WBFSimpleTiling, self)._generate_tiles(image)

        w = self._tile_params[0] + self._tile_params[2]
        h = self._tile_params[1] + self._tile_params[3]
        self._top_edge = (0, 0, w, int(self._edge_thr * h))
        self._bot_edge = (0, int(h - self._edge_thr * h), w, h)
        self._left_edge = (0, 0, int(w * self._edge_thr), h)
        self._right_edge = (int(w - w * self._edge_thr), 0, w, h)

        for tile in super_gen:
            yield tile

    def _accumulate_results(self, results: Sequence[dict], info: Tuple[_TileType, List]):
        central_boxes, edge_boxes = self._categorize(results)

        for res in central_boxes:
            if res.get('bbox') is None:
                    continue
            SimpleTiling._translate_box_abs(info[1][0:2], res['bbox'], in_place=True)
        for res in edge_boxes:
            if res.get('bbox') is None:
                    continue
            SimpleTiling._translate_box_abs(info[1][0:2], res['bbox'], in_place=True)
            x1,y1,x2,y2 = res['bbox']
            # WBF requires normalized coordinates.
            res['wbf_info'] = [x1/self._width, y1/self._height, x2/self._width, y2/self._height]
        
        self._results.extend(central_boxes)
        self._edge_results.extend(edge_boxes)
        self._slice_id += 1
    
    def _get_results(self) -> List[dict]:
        edge_results = weighted_boxes_fusion([self._edge_results], iou_thr=self._wbf_thr)

        for det in edge_results:
            det['label'] = self._label_dict[det['category_id']]
            box = det['bbox']
            box[0] *= self._width
            box[1] *= self._height
            box[2] *= self._width
            box[3] *= self._height

        results = copy.deepcopy(self._results)
        results.extend(edge_results)

        self._results.clear()
        self._edge_results.clear()
        return results

    def _categorize(self, dets: List[dict]):
        slice_id = self._slice_id
        cols = self._num_cols
        rows = self._num_rows

        relevant_edges = _EdgeMixin._calculate_relevant_edges(slice_id, cols, rows)

        central_boxes = []
        edge_boxes = []

        for det in dets:
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
    

class WBFLocalGlobalTiling(WBFSimpleTiling):
    def __init__(self, cols: int, rows: int, overlap_percent: float, large_object_thr: float, edge_thr: float=0.02, wbf_thr: float=0.8):
        super().__init__(cols, rows, overlap_percent, edge_thr, wbf_thr)
        self._large_obj_thr = large_object_thr

    def _generate_tiles(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, Tuple[_TileType, List]], None, None]:
        super_gen = super(WBFLocalGlobalTiling, self)._generate_tiles(image)

        self._large_obj_size = self._large_obj_thr * image.shape[0] * image.shape[1]

        for tile in super_gen:
            yield tile

        yield image, (_TileType.GLOBAL, [0, 0])
    
    def _accumulate_results(self, results: Sequence[dict], info: Tuple[_TileType, List]):
        def _is_large_object(box, thr):
            area = (box[2] - box[0]) * (box[3] - box[1])

            return area >= self._width * self._height * thr

        if info[0] == _TileType.GLOBAL:
            for i, res in _reverse_enumerate(results):
                if res.get('bbox') is None:
                    continue
                if not _is_large_object(res['bbox'], self._large_obj_thr * self._overlap_percent * 5):
                    del results[i]
        elif info[0] == _TileType.BASIC_SLICE:
            for i, res in _reverse_enumerate(results):
                if res.get('bbox') is None:
                    continue
                if _is_large_object(res['bbox'], self._large_obj_thr):
                    del results[i]

            super(WBFLocalGlobalTiling, self)._accumulate_results(results, info)
            return
        
        self._results.extend(results)