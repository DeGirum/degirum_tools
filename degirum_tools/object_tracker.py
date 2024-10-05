#
# object_tracker.py: multi-object tracker
#
# Copyright DeGirum Corporation 2023
# All rights reserved
#
# Implements classes for multi-object tracking
#

# MIT License
#
# Copyright (c) 2022 Roboflow
# Copyright (c) 2021 Yifu Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import cv2, numpy as np, scipy.linalg
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from .math_support import box_iou_batch, AnchorPoint, get_anchor_coordinates
from .ui_support import put_text, deduce_text_color, color_complement, rgb_to_bgr
from .result_analyzer_base import ResultAnalyzerBase


FAKE_EMBED_CONST = 300.5


def _bb_1d_ioa(boxes: np.ndarray, ref_box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Function which calculates the 1-dimensional intersection over area
    of a set of boxes with respect to one box. The maximum and minimum
    1-dimensional IoA is calculated.

    Args:
        boxes (ndarray): A set of bounding boxes with dimensions N x 4
        ref_box (ndarray): A bounding box

    Returns:
        Tuple[ndarray, ndarray]: Returns a tuple of arrays of dimension N,
            the first array contains the larger IoAs and the second contains
            the smaller IoAs.
    '''
    xA = np.maximum(boxes[:, 0], ref_box[0])
    yA = np.maximum(boxes[:, 1], ref_box[1])
    xB = np.minimum(boxes[:, 2], ref_box[2])
    yB = np.minimum(boxes[:, 3], ref_box[3])

    # Mask out boxes with no overlap in one of the dimensions.
    inter_x, inter_y = np.maximum(xB - xA, 0), np.maximum(yB - yA, 0)
    mask = np.minimum(inter_x, inter_y) == 0
    inter_x[mask] = 0
    inter_y[mask] = 0

    # w_a, h_a = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    w_b, h_b = ref_box[2] - ref_box[0], ref_box[3] - ref_box[1]

    ioa_x, ioa_y = inter_x / w_b, inter_y / h_b

    return np.maximum(ioa_x, ioa_y), np.minimum(ioa_x, ioa_y)


def strack_1d_ioa(atracks: List[STrack], btracks: List[STrack]) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Function which calculates the batch 1-dimensional intersection over area
    of two sets of boxes. The maximum and minimum 1-dimensional IoAs are
    calculated.

    Args:
        atracks (List[ndarray]): A set of bounding boxes with dimensions N x 4
        btracks (List[ndarray]): A set of bounding boxes with dimensions M x 4

    Returns:
        Tuple[ndarray, ndarray]: Returns a tuple of arrays of dimension N x M,
            the first array contains the larger IoAs and the second contains
            the smaller IoAs.
    '''
    atlbrs = np.array([track.tlbr for track in atracks])
    btlbrs = np.array([track.tlbr for track in btracks])

    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.empty(0), np.empty(0)

    results_max = []
    results_min = []
    for box in btlbrs:
        ioa_max, ioa_min = _bb_1d_ioa(atlbrs, box)
        results_max.append(ioa_max)
        results_min.append(ioa_min)

    return np.vstack(results_max), np.vstack(results_min)


def strack_self_1d_ioa(tracks: List[STrack]) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Function which calculates the batch 1-dimensional intersection over area
    of one set of boxes with respect to itself. The maximum and minimum
    1-dimensional IoAs are calculated. Same box comparisons are zero-ed out.

    Args:
        atracks (List[ndarray]): A set of bounding boxes with dimensions N x 4
        btracks (List[ndarray]): A set of bounding boxes with dimensions M x 4

    Returns:
        Tuple[ndarray, ndarray]: Returns a tuple of arrays of dimension N x M,
            the first array contains the larger IoAs and the second contains
            the smaller IoAs.
    '''
    results_max, results_min = strack_1d_ioa(tracks, tracks)

    for i in range(len(tracks)):
        results_max[i][i] = 0
        results_min[i][i] = 0

    return results_max, results_min


def _iou_distance(
    atracks: List[Union[np.ndarray, STrack]],
    btracks: List[Union[np.ndarray, STrack]]
) -> np.ndarray:
    '''
    Function which calculates intersection over union distance between two
    sets of bounding boxes.

    Args:
        atracks (List[Union[np.ndarray, STrack]]): A set of bounding boxes
            with dimensions N x 4
        btracks (List[Union[np.ndarray, STrack]]): A set of bounding boxes
            with dimensions M x 4

    Returns:
        ndarray: Returns an array of dimension N x M that contains the IoU
            distances.
    '''
    if (isinstance(atracks[0], np.ndarray) and len(atracks) > 0) or (
        isinstance(btracks[0], np.ndarray) and len(btracks) > 0
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]  # type: ignore[union-attr]
        btlbrs = [track.tlbr for track in btracks]  # type: ignore[union-attr]

    _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if _ious.size != 0:
        _ious = box_iou_batch(np.asarray(atlbrs), np.asarray(btlbrs))
    cost_matrix = 1 - _ious

    return cost_matrix


def _box_diou_penalty_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    '''
    Calculates the batch penalty term for Distance-IoU between two sets
    of bounding boxes.

    Args:
        boxes_a (ndarray): A set of bounding boxes of size N x 4
        boxes_b (ndarray): A set of bounding boxes of size M x 4

    Returns:
        ndarray: Returns the DIoU penalty terms in an array of size N x M
    '''
    # box in tlbr format: [(x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1] = [x_center, y_center]
    a_centers = np.transpose([(boxes_a[:, 2] - boxes_a[:, 0]) / 2 + boxes_a[:, 0],
                              (boxes_a[:, 3] - boxes_a[:, 1]) / 2 + boxes_a[:, 1]])
    b_centers = np.transpose([(boxes_b[:, 2] - boxes_b[:, 0]) / 2 + boxes_b[:, 0],
                              (boxes_b[:, 3] - boxes_b[:, 1]) / 2 + boxes_b[:, 1]])

    center_distances_squared = (np.square(b_centers[np.newaxis, :, 0] - a_centers[:, np.newaxis, 0])
                                + np.square(b_centers[np.newaxis, :, 1] - a_centers[:, np.newaxis, 1]))

    # Minorly faster in 2x3, not sure how cdist scales compared to numpy solution.
    # center_distances_squared = cdist(a_centers, b_centers, 'sqeuclidean')

    minimum_bounding_w_squared = (np.square(np.maximum(boxes_b[np.newaxis, :, 2], boxes_a[:, np.newaxis, 2])
                                  - np.minimum(boxes_b[np.newaxis, :, 0], boxes_a[:, np.newaxis, 0])))

    minimum_bounding_h_squared = np.square(np.maximum(boxes_b[np.newaxis, :, 3], boxes_a[:, np.newaxis, 3])
                                           - np.minimum(boxes_b[np.newaxis, :, 1], boxes_a[:, np.newaxis, 1]))

    minimum_bounding_diag_squared = minimum_bounding_w_squared + minimum_bounding_h_squared

    return center_distances_squared / minimum_bounding_diag_squared


def _diou_distance(
    atracks: List[Union[np.ndarray, STrack]],
    btracks: List[Union[np.ndarray, STrack]]
) -> np.ndarray:
    '''
    Calculates the batch Distance-IoU between two sets of bounding boxes.

    Args:
        atracks (List[Union[np.ndarray, STrack]]): A set of bounding boxes
            of size N x 4
        btracks (List[Union[np.ndarray, STrack]]): A set of bounding boxes
            of size M x 4

    Returns:
        ndarray: Returns the DIoU distance in an array of size N x M
    '''
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]  # type: ignore[union-attr]
        btlbrs = [track.tlbr for track in btracks]  # type: ignore[union-attr]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size != 0:
        ious = box_iou_batch(np.asarray(atlbrs), np.asarray(btlbrs))

    diou_penalties = _box_diou_penalty_batch(np.asarray(atlbrs), np.asarray(btlbrs))
    dious = 1.0 - ious + diou_penalties
    cost_matrix = (2 - dious) / 2

    return cost_matrix


def _cosine_embedding_distance(
    atracks: Sequence[Union[np.ndarray, STrack]],
    btracks: Sequence[Union[np.ndarray, STrack]]
) -> np.ndarray:
    '''
    Calculates the batch cosine distance between two sets of features.
    Sets tracks with fake/dummy embeddings to the maximum distance.

    Args:
        atracks (List[STrack]): A set of N tracks with embeddings.
        btracks (List[STrack]): A set of M tracks with embeddings.

    Returns:
        ndarray: Returns the cosine distance in an array of size N x M
    '''
    is_strack = False

    if (
        not isinstance(atracks[0], np.ndarray)
        and not isinstance(btracks[0], np.ndarray)
    ):
        is_strack = True

        afeats = np.asarray([track.feature_module.comparable_feature() for track in atracks])  # type: ignore[union-attr]
        bfeats = np.asarray([track.feature_module.comparable_feature() for track in btracks])  # type: ignore[union-attr]

        afakes = np.asarray([track.feature_module.is_fake_embedding for track in atracks]).nonzero()  # type: ignore[union-attr]
        bfakes = np.asarray([track.feature_module.is_fake_embedding for track in btracks]).nonzero()  # type: ignore[union-attr]
    else:
        afeats = np.asarray(atracks)
        bfeats = np.asarray(btracks)

    if afeats.size == 0 or bfeats.size == 0:
        return np.zeros((len(afeats), len(bfeats)))

    a_normed = afeats / np.linalg.norm(afeats, axis=1)[:, np.newaxis]
    b_normed = bfeats / np.linalg.norm(bfeats, axis=1)[:, np.newaxis]

    cost_matrix = 1 - (a_normed @ b_normed.T)

    cost_matrix /= 2  # Divide by 2 so that the range is [0, 1]

    if is_strack:
        # Make fake embeddings max distance
        for idx in afakes:
            cost_matrix[idx, :] = 1.0

        for idx in bfakes:
            cost_matrix[:, idx] = 1.0

    return cost_matrix


class FeatureHistory(ABC):
    '''
    Abstract class for storing and calculating features to be used for
    reidentification. Has the ability to store a dummy/fake embedding
    for batched distance calculations.
    '''
    max_history_length = 40

    def __init__(self):
        self.current_feat: Optional[np.ndarray] = None
        self.historical_feats: deque[np.ndarray] = deque(maxlen=FeatureHistory.max_history_length)
        self.is_fake_embedding = False

    @classmethod
    def prepare_class(cls, max_history_length=40, **kwargs):
        '''
        prepare_class is used to store class variables needed for updating or
        calculating the features (e.g. weights, max deque length).

        Arguments should all be keyword arguments, with the **kwargs as the
        last argument. This is to ignore other keyword arguments passed in
        from ObjectTracker.
        '''
        FeatureHistory.max_history_length = max_history_length

    def set_fake_status(self, status):
        self.is_fake_embedding = status

    @abstractmethod
    def comparable_feature(self, *args, **kwargs) -> np.ndarray:
        '''
        comparable_feature is used to calculate the actual feature used in the
        distance calculation, e.g. an average from the history deque.
        '''
        ...

    @abstractmethod
    def update_features(self, feature: np.ndarray, track: Optional[STrack] = None):
        '''
        update_features is used to update historical_feats and current_feat.
        Information from the track itself can be used to conditionally update
        the feature.
        '''
        ...


class AverageFeatureHistory(FeatureHistory):
    '''
    Stores features for a maximum history length (counted in frames).
    The feature used for comparison is an average of these stored features.
    '''
    def comparable_feature(self, *args, **kwargs) -> np.ndarray:
        return np.average(self.historical_feats, axis=0)

    def update_features(self, feature: np.ndarray, track: Optional[STrack] = None):
        self.current_feat = feature
        self.historical_feats.append(feature)


class ExponentialFeatureHistory(FeatureHistory):
    '''
    This feature module calculates the exponential moving average of the
    features, updated frame by frame.
    '''
    weight = 0.9

    @classmethod
    def prepare_class(cls, exp_history_weight=0.9, **kwargs):
        '''
        Argument passing method.

        Args:
            exp_history_weight (float): The momentum term for exponential
                smoothing.
        '''
        ExponentialFeatureHistory.weight = exp_history_weight
        super().prepare_class(**kwargs)

    def comparable_feature(self, *args, **kwargs) -> np.ndarray:
        assert isinstance(self.current_feat, np.ndarray)
        return self.current_feat

    def update_features(self, feature: np.ndarray, track: Optional[STrack] = None):
        # BoT-SORT  eki = αek−1i + (1 − α)fki
        # traditionally formulated as eki = αfki + (1 - α)ek-1i
        if self.current_feat is None:
            self.current_feat = feature
        else:
            # self.current_feat = ExponentialFeatureHistory.weight * self.historical_feats[-1] + (1 - ExponentialFeatureHistory.weight) * feature
            self.current_feat = ExponentialFeatureHistory.weight * self.current_feat + (1 - ExponentialFeatureHistory.weight) * feature
            self.current_feat /= np.linalg.norm(self.current_feat)

        self.historical_feats.append(feature)


class FaceExponentialFeatureHistory(ExponentialFeatureHistory):
    '''
    This feature module calculates the conditional exponential moving average
    of the features for face recognition. Features are only updated if the
    embedding is not fake and the matching used the ReID distance instead of
    the IoU distance at least one time during the track's history.
    '''
    def update_features(self, feature: np.ndarray, track: Optional[STrack] = None):
        if track is None:  # Initiating a track.
            self.current_feat = feature
        elif self.current_feat is None:
            self.current_feat = feature
        elif isinstance(track, STrack):
            assert isinstance(track.feature_module, FeatureHistory)
            if (
                not track.feature_module.is_fake_embedding
                and track.data.get('reid_used') is True
            ):
                feature /= np.linalg.norm(feature)

                if self.current_feat is None:
                    self.current_feat = feature
                else:
                    self.current_feat = ExponentialFeatureHistory.weight * self.current_feat + (1 - ExponentialFeatureHistory.weight) * feature
                    self.current_feat /= np.linalg.norm(self.current_feat)
                self.historical_feats.append(feature)
            elif (not track.feature_module.is_fake_embedding
                  and track.data.get('reid_used') is False
                  and track.data.get('reid_used_once', False) is False):
                self.current_feat = feature


class DistanceWeightedAverageHistory(AverageFeatureHistory):
    '''
    This feature module calculates the distance weighted average of the
    features in a defined window. Distance weighting causes outliers to
    have smaller effect on the centroid.
    '''
    distance_function = _cosine_embedding_distance

    @classmethod
    def prepare_class(cls, distance_function=_cosine_embedding_distance, **kwargs):
        DistanceWeightedAverageHistory.distance_function = distance_function
        super().prepare_class(**kwargs)

    def comparable_feature(self, *args, **kwargs) -> np.ndarray:
        dists = DistanceWeightedAverageHistory.distance_function(self.historical_feats, self.historical_feats)
        # The average is maxlen - 1 because one of the distances is the comparison to itself which is zero.
        weights = 1.0 / (np.sum(dists, axis=0) / (DistanceWeightedAverageHistory.max_history_length - 1.0))
        return np.average(self.historical_feats, weights=weights, axis=0)


class MetricFuser(ABC):
    '''
    MetricFusers provide an interface to fuse and manipulate distance/cost
    matrices. These are namely the distance/cost matrices generated from
    the bounding box and/or embeddings.
    '''
    def __init__(self, **kwargs):
        self.prepare(**kwargs)

    def prepare(self, **kwargs):
        '''
        prepare is used to store parameters needed to mask/fuse different
        distance/cost matrices.

        Arguments should all be keyword arguments, with the **kwargs as the
        last argument. This is to ignore other keyword arguments passed in
        from ObjectTracker.
        '''
        pass

    @abstractmethod
    def fuse(
        self,
        track_list_a: List[STrack],
        track_list_b: List[STrack],
        bbox_distance_matrix: Optional[np.ndarray],
        reid_distance_matrix: Optional[np.ndarray]
    ) -> np.ndarray:
        '''
        The fuse method generates a singular cost matrix from one or two
        different distance/cost matrices. Information stored in the tracks
        may be used to conditionally mask/weight/fuse elements from one or
        both matrices.
        '''
        ...


class IdentityBboxFuser(MetricFuser):
    '''
    This fuser returns the identical distance matrix generated from the
    bounding box information.
    '''
    def fuse(
        self,
        track_list_a: List[STrack],
        track_list_b: List[STrack],
        bbox_distance_matrix: Optional[np.ndarray],
        reid_distance_matrix: Optional[np.ndarray],
    ) -> np.ndarray:
        assert isinstance(bbox_distance_matrix, np.ndarray)
        return bbox_distance_matrix


class GatedBBoxOnlyFuser(MetricFuser):
    '''
    This is a gated bounding box distance matrix fuser.
    It will mask out distances greater than the gating threshold.
    '''
    def prepare(self, gating_threshold: float = 0.7, **kwargs):
        '''
        Args:
            gating_threshold (float): The maximum distance allowed in the
                distance matrix.
        '''
        self.max_distance = gating_threshold

    def fuse(
        self,
        track_list_a: list[STrack],
        track_list_b: list[STrack],
        bbox_distance_matrix: Optional[np.ndarray],
        reid_distance_matrix: Optional[np.ndarray],
    ) -> np.ndarray:
        assert isinstance(bbox_distance_matrix, np.ndarray)

        if self.max_distance is not None:
            bbox_distance_matrix[bbox_distance_matrix > self.max_distance] = 1000

        return bbox_distance_matrix


class BBoxDetectionScoreFuser(MetricFuser):
    '''
    This is a detection score - bounding box distance matrix fuser.
    It will multiply the detection score with the corresponding distance.
    '''
    def fuse(self,
             track_list_a: List[STrack],
             track_list_b: List[STrack],
             bbox_distance_matrix: Optional[np.ndarray],
             reid_distance_matrix: Optional[np.ndarray]) -> np.ndarray:
        assert isinstance(bbox_distance_matrix, np.ndarray)
        if bbox_distance_matrix.size == 0:
            return bbox_distance_matrix

        iou_sim = 1 - bbox_distance_matrix
        det_scores = np.array([det.score for det in track_list_b])
        det_scores = np.expand_dims(det_scores, axis=0).repeat(
            bbox_distance_matrix.shape[0], axis=0
        )
        fuse_sim = iou_sim * det_scores
        fuse_cost = 1 - fuse_sim
        return fuse_cost


class WeightedBBoxReIDAndDetectionScoreFuser(BBoxDetectionScoreFuser):
    '''
    Common style (JDE/FairMOT) of bbox/reid fusion.
    '''
    def prepare(self,
                cost_matrix_weight: float = 0.666666,
                **kwargs):
        self.weight = cost_matrix_weight

    def fuse(self,
             track_list_a: List[STrack],
             track_list_b: List[STrack],
             bbox_distance_matrix: Optional[np.ndarray],
             reid_distance_matrix: Optional[np.ndarray]) -> np.ndarray:
        # matrices not optional, assert for mypy
        assert isinstance(bbox_distance_matrix, np.ndarray)
        assert isinstance(reid_distance_matrix, np.ndarray)

        cost_matrix = self.weight * bbox_distance_matrix + (1 - self.weight) * reid_distance_matrix
        return super().fuse(track_list_a, track_list_b, cost_matrix, reid_distance_matrix)


class MinimumWeightedBBoxReIDAndDetectionScoreFuser(BBoxDetectionScoreFuser):
    '''
    BoT-SORT style fusion. The feature distance is weighted by 1.0, and
    feature distances are masked if they are greater than a defined feature
    distance as well as if the bounding boxes do not overlap (proximity_thr).
    Bounding box and feature distance matrices are fused using a min operation.
    '''
    def prepare(self,
                cost_matrix_weight: float = 1.0,
                reid_thr: float = 0.25,
                proximity_thr: float = 0.5,
                **kwargs):
        '''
        Args:
            cost_matrix_weight (float): weight used to scale the feature
                distance matrix
            reid_thr (float): Maximum distance allowed for reid distance
            proximity_thr (float): Minimum overlap allowed for reid_distance
        '''
        self.reid_thr = reid_thr
        self.proximity_thr = proximity_thr
        self.weight = cost_matrix_weight

    def fuse(self,
             track_list_a: List[STrack],
             track_list_b: List[STrack],
             bbox_distance_matrix: Optional[np.ndarray],
             reid_distance_matrix: Optional[np.ndarray]) -> np.ndarray:
        # matrices not optional, assert for mypy
        assert isinstance(bbox_distance_matrix, np.ndarray)
        assert isinstance(reid_distance_matrix, np.ndarray)

        reid_distance_matrix = reid_distance_matrix * self.weight
        reid_distance_matrix[reid_distance_matrix > self.reid_thr] = 1
        reid_distance_matrix[bbox_distance_matrix > self.proximity_thr] = 1

        bbox_distance_matrix = super().fuse(track_list_a, track_list_b, bbox_distance_matrix, reid_distance_matrix)

        return np.minimum(bbox_distance_matrix, reid_distance_matrix)


class GatedDeepSortFuser(MetricFuser):
    '''
    DeepSORT style fuser, operates only on feature distance matrix.
    '''
    chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919
    }

    def prepare(
        self,
        max_distance: float = 0.2,
        gated_cost: float = np.inf,
        only_position: bool = False,
        **kwargs
    ):
        self.max_distance = max_distance
        self.gated_cost = gated_cost
        self.only_position = only_position

    def fuse(self,
             track_list_a: List[STrack],
             track_list_b: List[STrack],
             bbox_distance_matrix: Optional[np.ndarray],
             reid_distance_matrix: Optional[np.ndarray]) -> np.ndarray:
        # reid_distance_matrix is not optional. assert for mypy
        assert isinstance(reid_distance_matrix, np.ndarray)

        if len(track_list_a) == 0 or len(track_list_b) == 0:
            return reid_distance_matrix

        cost_matrix = reid_distance_matrix
        kalman_filter = track_list_a[0].kalman_filter
        assert isinstance(kalman_filter, _KalmanFilter)

        gating_dim = 2 if self.only_position else 4
        gating_threshold = GatedDeepSortFuser.chi2inv95[gating_dim]

        measurements = np.asarray([STrack.tlwh_to_xyah(track.tlwh) for track in track_list_b])
        for row, track in enumerate(track_list_a):
            assert isinstance(track.mean, np.ndarray)
            assert isinstance(track.covariance, np.ndarray)
            gating_distance = kalman_filter.gating_distance(
                track.mean, track.covariance, measurements, self.only_position
            )
            cost_matrix[row, gating_distance > gating_threshold] = self.gated_cost

        cost_matrix[cost_matrix > self.max_distance] = 1000
        return cost_matrix


class ConditionalFaceRecognitionFuser(BBoxDetectionScoreFuser):
    def prepare(self,
                appearance_thresh: float = 0.3,
                area_thresh: float = 0.00342935528,
                rot_thresh: float = 0.34,
                **kwargs):
        '''
        Args:
            appearance_thresh (float): Maximum feature distance.
            area_thresh (float): Minimum relative area (wrt. image size)
            rot_thresh (float): Maximum rotation heuristic threshold.
                The rotation heuristic is the ratio of the horizontal
                distance between the left eye/nose and the right eye/nose.
        '''
        self.appearance_thresh = appearance_thresh
        self.area_thresh = area_thresh
        self.rot_thresh = rot_thresh

    def fuse(self,
             track_list_a: list[STrack],
             track_list_b: list[STrack],
             bbox_distance_matrix: Optional[np.ndarray],
             reid_distance_matrix: Optional[np.ndarray]) -> np.ndarray:
        # Both distance matrices are not optional in this fuser.
        # Can't override types, so assertion is necessary for mypy.
        assert isinstance(bbox_distance_matrix, np.ndarray)
        assert isinstance(reid_distance_matrix, np.ndarray)

        fused_iou_distance = super().fuse(track_list_a, track_list_b, bbox_distance_matrix, None)

        num_tracks = len(track_list_a)
        num_dets = len(track_list_b)

        if num_tracks > 0 and num_dets > 0:
            dets_areas = np.array([track.data['relative_area'] for track in track_list_b])
            dets_rot_heuristics = np.array([track.data['rotate_heuristic'] for track in track_list_b])
            dets_overlap_statuses = np.array([track.data['overlap_status'] for track in track_list_b])
            tracks_reid_used_once = np.array([track.data.get('reid_used_once', False) for track in track_list_a])
            tracks_future_occlusion = np.array([track.data.get('future_occlusion', False) for track in track_list_a])

            # Broadcast to "zero-out" the column corresponding to a detection.
            # Do not use ReID if box is less than a relative area threshold (box too small).
            udets_areas_bcast = np.broadcast_to(dets_areas, (num_tracks, num_dets))
            reid_distance_matrix[udets_areas_bcast < self.area_thresh] = 1.0
            # Do not use ReID if box is above rotation heuristic threshold (over horizontal rotation.)
            urotate_heuristics_bcast = np.broadcast_to(dets_rot_heuristics, (num_tracks, num_dets))
            reid_distance_matrix[urotate_heuristics_bcast > self.rot_thresh] = 1.0
            # Do not use ReID if box is at the edge of the camera field.
            uoverlap_statuses_bcast = np.broadcast_to(dets_overlap_statuses, (num_tracks, num_dets))
            reid_distance_matrix[uoverlap_statuses_bcast == True] = 1.0  # noqa: E712
            # Do not use ReID if embedding distance is greater than a threshold.
            reid_distance_matrix[reid_distance_matrix > self.appearance_thresh] = 1.0

            # Weight tracks that have been ReIDed once by half.
            reided_once_mask = np.broadcast_to(tracks_reid_used_once[:, None], (num_tracks, num_dets))
            reided_once_mask = np.logical_and(reid_distance_matrix != 1.0, reided_once_mask)
            reid_distance_matrix[reided_once_mask] *= 0.5

            # Do not use IoU if detection is flagged as future occluded.
            curr_occ_mask = np.broadcast_to(tracks_future_occlusion[:, None], (num_tracks, num_dets))
            fused_iou_distance[curr_occ_mask == True] = 1.0  # noqa: E712

        dists = np.minimum(fused_iou_distance, reid_distance_matrix)
        reid_used = dists == reid_distance_matrix

        for i in range(num_dets):
            track_list_b[i].pre_match_idxdata['reid_used'] = reid_used[:, i]

        return dists


class FutureOcclusionMaskFuser(MetricFuser):
    def fuse(self,
             track_list_a: list[STrack],
             track_list_b: list[STrack],
             bbox_distance_matrix: Optional[np.ndarray],
             reid_distance_matrix: Optional[np.ndarray]) -> np.ndarray:
        # bbox_distance_matrix is not optional in this fuser.
        # assert so mypy does not throw a type error.
        assert isinstance(bbox_distance_matrix, np.ndarray)

        num_tracks = len(track_list_a)
        num_dets = len(track_list_b)

        if num_tracks > 0 and num_dets > 0:
            tracks_future_occlusion = np.array([track.data.get('future_occlusion', False) for track in track_list_a])
            curr_occ_mask = np.broadcast_to(tracks_future_occlusion[:, None], (num_tracks, num_dets))
            bbox_distance_matrix[curr_occ_mask == True] = 1.0  # noqa: E712

        return bbox_distance_matrix


class AssociationMetrics:
    '''
    A class that represents one association/matching stage of the
    tracker. There are three stages, the first (high confidence
    detections), the second (medium confidence detections), and
    the unmatched (inactive tracks).
    '''
    def __init__(
        self,
        distance_fuser: MetricFuser,
        bbox_distance_function: Optional[Callable] = None,
        reid_distance_function: Optional[Callable] = None,
    ):
        '''
        Args:
            distance_fuser (MetricFuser): An instance of a MetricFuser class.
            bbox_distance_function (Optional[Callable]): A distance function
                that operates on the bounding box coordinates.
            reid_distance_function (Optional[Callable]): A distance function
                that operates on the embeddings.
        '''
        self.use_reid = True if reid_distance_function else False
        self.bbox_distance_fn = bbox_distance_function
        self.reid_distance_fn = reid_distance_function
        self.distance_fuser = distance_fuser

    def compute_distance_matrix(
        self,
        track_list_a: List[STrack],
        track_list_b: List[STrack]
    ) -> np.ndarray:
        '''
        Computes the distance matrix (matrices) between two lists of tracks
        and generates a final matrix using a MetricFuser fuse operation.
        '''
        if self.bbox_distance_fn:
            bbox_dists = self.bbox_distance_fn(track_list_a, track_list_b)
        else:
            bbox_dists = None

        if self.reid_distance_fn:
            reid_dists = self.reid_distance_fn(track_list_a, track_list_b)
        else:
            reid_dists = None

        return self.distance_fuser.fuse(track_list_a, track_list_b, bbox_dists, reid_dists)

    def prepare(self, **kwargs):
        '''
        Method used to pass tracker plugin arguments to the MetricFuser.
        '''
        self.distance_fuser.prepare(**kwargs)


class MatchingMetrics:
    '''
    MathingMetrics store the first, second, and unmatched association stages.
    '''
    def __init__(self,
                 first_association_metrics: AssociationMetrics,
                 second_association_metrics: AssociationMetrics,
                 unconfirmed_metrics: AssociationMetrics
                 ):

        self.first = first_association_metrics
        self.second = second_association_metrics
        self.unconfirmed = unconfirmed_metrics

    def prepare(self, **kwargs):
        '''
        Method used to pass tracker plugin arguments to AssociationMetrics.
        '''
        self.first.prepare(**kwargs)
        self.second.prepare(**kwargs)
        self.unconfirmed.prepare(**kwargs)


class TrackRemover(ABC):
    '''
    The TrackRemover class provides the criteria necessary for removing
    tracks (tracks that are lost, i.e. do not pass the first or second
    association).
    '''
    def __init__(self, **kwargs):
        self.prepare(**kwargs)

    def prepare(self, **kwargs):
        '''
        Method used to pass tracker plugin arguments from ObjectTracker.
        Arguments should all be keyword arguments and the last argument
        should be **kwargs.
        '''
        pass

    @abstractmethod
    def remove(self, track_list: List[STrack], frame_id: int) -> List[STrack]:
        '''
        Takes a list of lost tracks and returns a list of tracks for
        removal.
        '''
        ...


class DefaultTrackRemover(TrackRemover):
    '''
    The DefaultTrackRemover removes tracks based on a minimum time in frames
    a track is lost.
    '''
    def prepare(self, track_buffer: int = 30, **kwargs):
        '''
        Args:
            track_buffer (int): Time in frames a track is allowed to be lost.
        '''
        self._max_time_lost = track_buffer

    def remove(self, track_list: List[STrack], frame_id: int) -> List[STrack]:
        removed_stracks = []

        for track in track_list:
            if frame_id - track.end_frame > self._max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        return removed_stracks


class OnlyIoUTrackRemover(TrackRemover):
    '''
    The OnlyIoUTrackRemover, removes tracks that have never been ReIDed.
    Tracks that have been ReIDed stay forever. Tracks that have only been
    tracked using IoU remain until a minimum amount of frames lost.
    '''
    def prepare(self, track_buffer=30, **kwargs):
        '''
        Args:
            track_buffer (int): Time in frames a track is allowed to be lost.
        '''
        self._max_time_lost = track_buffer

    def remove(self, track_list: List[STrack], frame_id):
        removed_stracks = []

        for track in track_list:
            if track.data.get('reid_used_once') is None:
                if frame_id - track.end_frame > self._max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)

        return removed_stracks


class PreStageProcessor(ABC):
    '''
    The PreStageProcessor is used to append additional information to the
    track, before the first or second association stage.
    '''
    def __init__(self, **kwargs):
        self.prepare(**kwargs)

    def prepare(self, **kwargs):
        pass

    @abstractmethod
    def process(self,
                tracks: List[STrack],
                detections: List[STrack]):
        ...


class FutureOcclusionProcessor(PreStageProcessor):
    '''
    The FutureOcclusionProcessor takes detections and marks them
    as future_occluded if they pass a threshold for their largest
    dimension intersection over area and a threshold for their
    smallest dimension intersection over area.
    '''
    def prepare(self,
                max_1d_ioa_thresh: float = 0.85,
                min_1d_ioa_thresh: float = 0.25,
                **kwargs):
        '''
        Args:
            max_1d_ioa_thresh (float): Minimum IoA for the dimension with the
                larger IoA
            min_1d_ioa_thresh (float): Minimum IoA for the dimension with the
                smaller IoA
        '''
        self.max_1d_ioa_thresh = max_1d_ioa_thresh
        self.min_1d_ioa_thresh = min_1d_ioa_thresh

    def process(self,
                tracks: List[STrack],
                detections: List[STrack]):
        if len(detections) > 0:
            ioa_max, ioa_min = strack_self_1d_ioa(detections)
            occ_max_mask = ioa_max > self.max_1d_ioa_thresh
            occ_min_mask = ioa_min > self.min_1d_ioa_thresh
            occ_mask = np.logical_and(occ_max_mask, occ_min_mask)
            future_occluded_dets = list(np.where(occ_mask == True)[1])  # noqa: E712

            for idx in range(len(detections)):
                if idx in future_occluded_dets:
                    detections[idx].pre_match_data['future_occlusion'] = True
                else:
                    detections[idx].pre_match_data['future_occlusion'] = False


class KnownTrackInitiator(ABC):
    '''
    KnownTrackInitiator is used to create tracks before any frame update.
    To be implemented.
    '''
    @abstractmethod
    def initiate_track(self):
        ...


def ReidUsedOnceHook(self, frame_id: int):
    '''
    The ReidUsedOnceHook is a hook used in track updates.
    The hook must define self as the first parameter to access
    track variables.
    '''
    if self.data.get('reid_used'):
        if self.data.get('reid_use_history') is None:
            self.data['reid_use_history'] = []

        self.data['reid_used_once'] = True
        self.data['reid_use_history'].append(frame_id)


class NewTrackInitiator(ABC):
    '''
    The NewTrackInitiator class is used to provide the criteria
    for new track initiation.
    '''
    def __init__(self, **kwargs):
        self.prepare(**kwargs)

    def prepare(self, **kwargs):
        pass

    @abstractmethod
    def initiate_track(self, detection: STrack) -> bool:
        ...


class ConfidenceNewTrackInitiator(NewTrackInitiator):
    '''
    The confidence new track initiator only initiates a new track
    if it is delta above the high track confidence threshold.
    '''
    def prepare(
        self,
        track_thresh: float = 0.6,
        new_track_delta: float = 0.1,
        **kwargs
    ):
        self.track_thresh = track_thresh
        self.delta = new_track_delta

    def initiate_track(self, detection: STrack) -> bool:
        if detection.data['score'] > self.track_thresh + self.delta:
            return True
        else:
            return False


class TrackerModules:
    def __init__(self,
                 matching_metrics: MatchingMetrics,
                 track_remover: Type[TrackRemover],
                 feature_history: Optional[Type[FeatureHistory]],
                 pre_assoc1_process: Optional[Type[PreStageProcessor]] = None,
                 pre_assoc2_process: Optional[Type[PreStageProcessor]] = None,
                 known_track_initiator: Optional[Type[KnownTrackInitiator]] = None,
                 new_track_hook: Optional[NewTrackInitiator] = None,
                 track_update_hook: Optional[Callable] = None,
                 results_key_blacklist: Optional[List] = None):
        self.matching_metrics = matching_metrics
        self.track_remover = track_remover
        self.feature_history = feature_history
        self.pre_assoc1_process = pre_assoc1_process
        self.pre_assoc2_process = pre_assoc2_process
        self.known_track_initiatior = known_track_initiator
        self.new_track_hook = new_track_hook
        self.track_update_hook = track_update_hook
        self.results_key_blacklist = results_key_blacklist


# DeepSORT inspired tracker plugin
gated_bbox_iou_associater = AssociationMetrics(GatedBBoxOnlyFuser(gating_threshold=0.7),
                                               bbox_distance_function=_iou_distance)
gated_deep_sort_associater = AssociationMetrics(GatedDeepSortFuser(),
                                                bbox_distance_function=_iou_distance,
                                                reid_distance_function=_cosine_embedding_distance)
deepsort_metrics = MatchingMetrics(gated_deep_sort_associater, gated_bbox_iou_associater, gated_deep_sort_associater)
deepsort_tracker = TrackerModules(matching_metrics=deepsort_metrics,
                                  track_remover=DefaultTrackRemover,
                                  feature_history=ExponentialFeatureHistory)

# ByteTrack tracker plugin
bbox_detscore_associater = AssociationMetrics(BBoxDetectionScoreFuser(), bbox_distance_function=_iou_distance)
identity_associater = AssociationMetrics(IdentityBboxFuser(), bbox_distance_function=_iou_distance)

bytetrack_matchmetrics = MatchingMetrics(bbox_detscore_associater, identity_associater, bbox_detscore_associater)
bytetrack_tracker = TrackerModules(matching_metrics=bytetrack_matchmetrics,
                                   track_remover=DefaultTrackRemover,
                                   feature_history=None)

# BoT-SORT tracker plugin
minweighted_bboxreid_detscore_associater = AssociationMetrics(MinimumWeightedBBoxReIDAndDetectionScoreFuser(),
                                                              bbox_distance_function=_iou_distance,
                                                              reid_distance_function=_cosine_embedding_distance)

botsortreid_matchmetrics = MatchingMetrics(minweighted_bboxreid_detscore_associater,
                                           identity_associater,
                                           minweighted_bboxreid_detscore_associater)
botsortreid_tracker = TrackerModules(matching_metrics=botsortreid_matchmetrics,
                                     track_remover=DefaultTrackRemover,
                                     feature_history=ExponentialFeatureHistory)

# DeGirum Face Recognition tracker plugin
cond_face_associater = AssociationMetrics(ConditionalFaceRecognitionFuser(),
                                          bbox_distance_function=_iou_distance,
                                          reid_distance_function=_cosine_embedding_distance)
future_occlusion_associater = AssociationMetrics(FutureOcclusionMaskFuser(),
                                                 bbox_distance_function=_iou_distance)

facerecognition_matchmetrics = MatchingMetrics(cond_face_associater, future_occlusion_associater, cond_face_associater)
facerecognition_tracker = TrackerModules(matching_metrics=facerecognition_matchmetrics,
                                         track_remover=OnlyIoUTrackRemover,
                                         feature_history=FaceExponentialFeatureHistory,
                                         pre_assoc1_process=FutureOcclusionProcessor,
                                         pre_assoc2_process=FutureOcclusionProcessor,
                                         track_update_hook=ReidUsedOnceHook)

available_tracker_plugins: Dict[str, TrackerModules] = {'bytetrack': bytetrack_tracker,
                                                        'botsort-reid': botsortreid_tracker,
                                                        'deepsort': deepsort_tracker,
                                                        'dg-facetrack': facerecognition_tracker}


class _KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from an unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h) with
                center position (x, y), aspect ratio a, and height h.

        Returns:
            Tuple[ndarray, ndarray]: Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object
                state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of
                the object state at the previous time step.

        Returns:
            Tuple[ndarray, ndarray]: Returns the mean vector and
                covariance matrix of the predicted state.
                Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            Tuple[ndarray, ndarray]: Returns the projected mean and
                covariance matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix
                of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 dimensional covariance matrices
                of the object states at the previous time step.

        Returns:
            Tuple[ndarray, ndarray]: Returns the mean vector and
                covariance matrix of the predicted state.
                Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + np.asarray(motion_cov)

        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4-dimensional measurement vector (x, y, a, h),
                where (x, y) is the center position, a the aspect ratio,
                and h the height of the bounding box.

        Returns:
            Tuple[ndarray, ndarray]: Returns the measurement-corrected
                state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        only_position: float = False
    ) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4-dimensional measurement vector (x, y, a, h),
                where (x, y) is the center position, a the aspect ratio,
                and h the height of the bounding box.
            only_position (Optional[bool]): If True, distance computation is done
            with respect to the bounding box center position only.

        Returns:
            ndarray: Returns an array of length N, where the i-th element contains the
                squared Mahalanobis distance between (mean, covariance) and
                `measurements[i]`.
        """

        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurement = measurement[:, :2]

        d = measurement - mean

        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)

        return squared_maha


class _TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@dataclass
class _IDCounter:
    """
    Track ID counter
    """

    _count: int = 0


class STrack:
    """
    Class which represents one object track.
    """

    track_update_call: Optional[Callable] = None

    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        obj_idx: int,
        id_counter: _IDCounter,
        feature_module: Optional[type[FeatureHistory]] = None,
        obj_data: Dict = {}
    ):
        self.id_counter = id_counter
        self.track_id = 0
        self.is_activated = False
        self.state = _TrackState.New

        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0

        # multi-camera
        self.location = (np.inf, np.inf)

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter: Optional[_KalmanFilter] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.score = score
        self.obj_idx = obj_idx
        self.tracklet_len = 0

        if feature_module:
            self.feature_module: Optional[FeatureHistory] = feature_module()
            if np.all(obj_data['embedding'] == FAKE_EMBED_CONST):
                self.feature_module.set_fake_status(True)

            self.feature_module.update_features(obj_data['embedding'])
            del obj_data['embedding']  # embeddings should only be stored in the feature_module
        else:
            self.feature_module = None

        # Data is a dictionary that contains track related data that can be used as criteria
        # in the various plugin modules.
        self.data = obj_data

        # pre_match_idxdata is indexable data which is transferred to data upon match.
        # The index of the matching track is used to index the pre_match_idxdata upon
        # storage. pre_match_data is data to be stored after matching. Use when data is
        # to be used for comparison in the next frame update.
        self.pre_match_idxdata: Dict[str, Any] = {}
        self.pre_match_data: Dict[str, Any] = {}

    @property
    def end_frame(self) -> int:
        return self.frame_id

    def next_id(self) -> int:
        self.id_counter._count += 1
        return self.id_counter._count

    def mark_lost(self):
        self.state = _TrackState.Lost

    def mark_removed(self):
        self.state = _TrackState.Removed

    @staticmethod
    def multi_predict(stracks, shared_kalman):
        if len(stracks) > 0:
            multi_mean = []
            multi_covariance = []
            for i, st in enumerate(stracks):
                multi_mean.append(st.mean.copy())
                multi_covariance.append(st.covariance)
                if st.state != _TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = shared_kalman.multi_predict(
                np.asarray(multi_mean), np.asarray(multi_covariance)
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: _KalmanFilter, frame_id: int):
        """Start a new tracklet

        Args:
            kalman_filter (_KalmanFilter): Kalman filter object to use
            frame_id (int): current frame id
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = _TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id: int, new_id: bool = False):
        """
        Reactivate lost track

        Args:
            new_track (STrack): new tracklet to reactivate from
            frame_id (int): current frame id
            new_id (bool): True assign new ID
        """

        if self.feature_module is not None:
            if new_track.feature_module is not None:
                if new_track.feature_module.current_feat is not None:
                    if not new_track.feature_module.is_fake_embedding:
                        self.feature_module.update_features(new_track.feature_module.comparable_feature(), self)
                        self.feature_module.set_fake_status(False)

        assert self.kalman_filter is not None
        assert self.mean is not None
        assert self.covariance is not None
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = _TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.obj_idx = new_track.obj_idx

        if self.track_update_call is not None:
            self.track_update_call(frame_id)

    def update(self, new_track, frame_id: int):
        """
        Update a matched track

        Args:
            new_track (STrack): new tracklet to update from
            frame_id (int): current frame id
        """

        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        if self.feature_module is not None:
            if new_track.feature_module is not None:
                if new_track.feature_module.current_feat is not None:
                    if not new_track.feature_module.is_fake_embedding:
                        self.feature_module.set_fake_status(False)
                        self.feature_module.update_features(new_track.feature_module.comparable_feature(), self)

        assert self.kalman_filter is not None
        assert self.mean is not None
        assert self.covariance is not None
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = _TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.obj_idx = new_track.obj_idx

        if self.track_update_call is not None:
            self.track_update_call(frame_id)

    @property
    def tlwh(self) -> np.ndarray:
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class _ByteTrack:
    """
    Multi-object tracking class.
    """

    def __init__(
        self,
        class_list: Optional[list] = None,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        tracker_plugin: Union[str, TrackerModules] = 'bytetrack',
        plugin_args: Optional[Dict] = None
    ):
        """
        Initialize the _ByteTrack object.

        Parameters:
            class_list (list, optional): list of classes to count; if None, all classes are counted
            track_thresh (float, optional): Detection confidence threshold for track activation.
            track_buffer (int, optional): Number of frames to buffer when a track is lost.
            match_thresh (float, optional): IOU threshold for matching tracks with detections.
            tracker_plugin: (Union[str, TrackerModules]): Choose from available trackers,
                bytetrack, botsort-reid, deepsort, or dg-facetrack, or supply a TrackerModules
                object for a custom two stage tracker.
            plugin_args (dict, optional): Additional arguments used for each tracker plugin type.
        """
        self._class_list = class_list
        self._track_thresh = track_thresh
        self._match_thresh = match_thresh
        self._frame_id = 0
        self._det_thresh = self._track_thresh + 0.1
        self._max_time_lost = track_buffer
        self._kalman_filter = _KalmanFilter()
        self._tracked_tracks: List[STrack] = []
        self._lost_tracks: List[STrack] = []
        self._removed_tracks: List[STrack] = []
        self._id_counter = _IDCounter()

        if plugin_args is None:
            plugin_args = {}

        plugin_args['track_thresh'] = track_thresh
        plugin_args['track_buffer'] = track_buffer
        plugin_args['match_thresh'] = match_thresh

        if isinstance(tracker_plugin, str):
            tracker_modules: TrackerModules = available_tracker_plugins[tracker_plugin]
        else:
            tracker_modules = tracker_plugin

        self.matching_module = tracker_modules.matching_metrics
        if plugin_args is None:
            plugin_args = {}

        self.matching_module.prepare(**plugin_args)

        self.track_remover = tracker_modules.track_remover(**plugin_args)

        self.feature_module = None
        if tracker_modules.feature_history is not None:
            self.feature_module = tracker_modules.feature_history
            tracker_modules.feature_history.prepare_class(**plugin_args)

        self.pre_assoc1_process = None
        self.pre_assoc2_process = None

        if tracker_modules.pre_assoc1_process is not None:
            self.pre_assoc1_process = tracker_modules.pre_assoc1_process(**plugin_args)
        if tracker_modules.pre_assoc2_process is not None:
            self.pre_assoc2_process = tracker_modules.pre_assoc2_process(**plugin_args)

        if tracker_modules.known_track_initiatior is not None:
            pass  # TODO: add this

        if tracker_modules.track_update_hook is not None:
            STrack.track_update_call = tracker_modules.track_update_hook

        if tracker_modules.results_key_blacklist is not None:
            self.results_key_blacklist = tracker_modules.results_key_blacklist
        else:
            self.results_key_blacklist = []

        if tracker_modules.new_track_hook is not None:
            self.new_track_hook: Optional[NewTrackInitiator] = tracker_modules.new_track_hook
        else:
            self.new_track_hook = None

    def update(self, result):
        """
        Updates the tracker with the provided detections and
            returns the updated detection results.

        Parameters:
            result: PySDK result object to update with. As a result of the update,
                it can be augmented with additional keys added to result.results[] dictionaries:
                "track_id" - unique track ID of the detected object
        """

        obj_indexes = np.arange(len(result.results))
        bboxes = np.array([obj["bbox"] for obj in result.results])
        scores = np.array([obj["score"] for obj in result.results])

        # apply class filtering
        if self._class_list is not None and len(scores) > 0:
            excluded_classes = np.array(
                [obj["label"] not in self._class_list for obj in result.results]
            )
            scores[excluded_classes] = 0

        self._frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = scores > self._track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self._track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets = bboxes[remain_inds]
        dets_second = bboxes[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        obj_indexes_keep = obj_indexes[remain_inds]
        obj_indexes_second = obj_indexes[inds_second]

        obj_data = [result.results[idx] for idx in obj_indexes_keep]
        obj_data_second = [result.results[idx] for idx in obj_indexes_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, i, self._id_counter, self.feature_module, d)
                for (tlbr, s, i, d) in zip(dets, scores_keep, obj_indexes_keep, obj_data)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self._tracked_tracks:
            track.obj_idx = -1  # clear object index in advance
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = _ByteTrack._join_tracks(tracked_stracks, self._lost_tracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self._kalman_filter)

        if self.pre_assoc1_process is not None:
            self.pre_assoc1_process.process(strack_pool, detections)

        dists = self.matching_module.first.compute_distance_matrix(strack_pool, detections)

        matches, u_track, u_detection = _ByteTrack._linear_assignment(
            dists, thresh=self._match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]

            for key in det.pre_match_idxdata:
                track.data[key] = det.pre_match_idxdata[key][itracked]
            for key in det.pre_match_data:
                track.data[key] = det.pre_match_data[key]

            det.pre_match_data = {}
            det.pre_match_idxdata = {}

            if track.state == _TrackState.Tracked:
                track.update(det, self._frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self._frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, i, self._id_counter, self.feature_module, d)
                for (tlbr, s, i, d) in zip(dets_second, scores_second, obj_indexes_second, obj_data_second)
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == _TrackState.Tracked
        ]

        if self.pre_assoc2_process is not None:
            self.pre_assoc2_process.process(r_tracked_stracks, detections_second)

        dists = self.matching_module.second.compute_distance_matrix(r_tracked_stracks, detections_second)

        matches, u_track, u_detection_second = _ByteTrack._linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]

            for key in det.pre_match_idxdata:
                track.data[key] = det.pre_match_idxdata[key][itracked]
            for key in det.pre_match_idxdata:
                track.data[key] = det.pre_match_data[key]
            det.pre_match_idxdata = {}
            det.pre_match_data = {}

            if track.state == _TrackState.Tracked:
                track.update(det, self._frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self._frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == _TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]

        dists = self.matching_module.unconfirmed.compute_distance_matrix(unconfirmed, detections)

        matches, u_unconfirmed, u_detection = _ByteTrack._linear_assignment(
            dists, thresh=0.7
        )

        for itracked, idet in matches:
            for key in detections[idet].pre_match_idxdata:
                unconfirmed[itracked].data[key] = detections[idet].pre_match_idxdata[key][itracked]
            for key in detections[idet].pre_match_data:
                unconfirmed[itracked].data[key] = detections[idet].pre_match_data[key]
            detections[idet].pre_match_idxdata = {}
            detections[idet].pre_match_data = {}

            unconfirmed[itracked].update(detections[idet], self._frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]

            if self.new_track_hook is not None:
                if not self.new_track_hook.initiate_track(track):
                    continue
            else:
                if track.score < self._det_thresh:
                    continue

            track.activate(self._kalman_filter, self._frame_id)
            activated_stracks.append(track)

        """ Step 5: Update state"""
        removed = self.track_remover.remove(self._lost_tracks, self._frame_id)
        removed_stracks.extend(removed)

        self._tracked_tracks = [
            t for t in self._tracked_tracks if t.state == _TrackState.Tracked
        ]
        self._tracked_tracks = _ByteTrack._join_tracks(
            self._tracked_tracks, activated_stracks
        )
        self._tracked_tracks = _ByteTrack._join_tracks(
            self._tracked_tracks, refind_stracks
        )
        self._lost_tracks = _ByteTrack._sub_tracks(
            self._lost_tracks, self._tracked_tracks
        )
        self._lost_tracks.extend(lost_stracks)
        self._lost_tracks = _ByteTrack._sub_tracks(
            self._lost_tracks, self._removed_tracks
        )
        self._removed_tracks.extend(removed_stracks)
        self._tracked_tracks, self._lost_tracks = _ByteTrack._remove_duplicate_tracks(
            self._tracked_tracks, self._lost_tracks
        )
        output_stracks = [track for track in self._tracked_tracks if track.is_activated]

        # update result
        for track in output_stracks:
            if track.obj_idx < 0:
                continue

            for key in track.data.keys():
                if key in self.results_key_blacklist:
                    continue

                result.results[track.obj_idx][key] = track.data[key]

            result.results[track.obj_idx]["bbox"] = track.tlbr.tolist()
            result.results[track.obj_idx]["track_id"] = track.track_id

        return result

    @staticmethod
    def _join_tracks(
        track_list_a: List[STrack], track_list_b: List[STrack]
    ) -> List[STrack]:
        """
        Joins two lists of tracks, ensuring that the resulting list does not
        contain tracks with duplicate track_id values.

        Parameters:
            track_list_a: First list of tracks (with track_id attribute).
            track_list_b: Second list of tracks (with track_id attribute).

        Returns:
            Combined list of tracks from track_list_a and track_list_b
                without duplicate track_id values.
        """
        seen_track_ids = set()
        result = []

        for track in track_list_a + track_list_b:
            if track.track_id not in seen_track_ids:
                seen_track_ids.add(track.track_id)
                result.append(track)

        return result

    @staticmethod
    def _sub_tracks(track_list_a: List, track_list_b: List) -> List:
        """
        Returns a list of tracks from track_list_a after removing any tracks
        that share the same track_id with tracks in track_list_b.

        Parameters:
            track_list_a: List of tracks (with track_id attribute).
            track_list_b: List of tracks (with track_id attribute) to
                be subtracted from track_list_a.
        Returns:
            List of remaining tracks from track_list_a after subtraction.
        """
        tracks = {track.track_id: track for track in track_list_a}
        track_ids_b = {track.track_id for track in track_list_b}

        for track_id in track_ids_b:
            tracks.pop(track_id, None)

        return list(tracks.values())

    @staticmethod
    def _remove_duplicate_tracks(tracks_a: List, tracks_b: List) -> Tuple[List, List]:
        pairwise_distance = _ByteTrack._iou_distance(tracks_a, tracks_b)
        matching_pairs = np.where(pairwise_distance < 0.15)

        duplicates_a, duplicates_b = set(), set()
        for track_index_a, track_index_b in zip(*matching_pairs):
            time_a = (
                tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame
            )
            time_b = (
                tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
            )
            if time_a > time_b:
                duplicates_b.add(track_index_b)
            else:
                duplicates_a.add(track_index_a)

        result_a = [
            track for index, track in enumerate(tracks_a) if index not in duplicates_a
        ]
        result_b = [
            track for index, track in enumerate(tracks_b) if index not in duplicates_b
        ]

        return result_a, result_b

    @staticmethod
    def _indices_to_matches(
        cost_matrix: np.ndarray, indices: np.ndarray, thresh: float
    ) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
        matched_cost = cost_matrix[tuple(zip(*indices))]
        matched_mask = matched_cost <= thresh

        matches = indices[matched_mask]
        unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
        unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
        return matches, unmatched_a, unmatched_b

    @staticmethod
    def _linear_assignment(
        cost_matrix: np.ndarray, thresh: float
    ) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])),
            )

        cost_matrix[cost_matrix > thresh] = thresh + 1e-4
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        indices = np.column_stack((row_ind, col_ind))

        return _ByteTrack._indices_to_matches(cost_matrix, indices, thresh)

    @staticmethod
    def _iou_distance(atracks: List, btracks: List) -> np.ndarray:
        if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
        ):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.tlbr for track in atracks]
            btlbrs = [track.tlbr for track in btracks]

        _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
        if _ious.size != 0:
            _ious = box_iou_batch(np.asarray(atlbrs), np.asarray(btlbrs))
        cost_matrix = 1 - _ious

        return cost_matrix

    @staticmethod
    def _fuse_score(cost_matrix: np.ndarray, detections: List) -> np.ndarray:
        if cost_matrix.size == 0:
            return cost_matrix
        iou_sim = 1 - cost_matrix
        det_scores = np.array([det.score for det in detections])
        det_scores = np.expand_dims(det_scores, axis=0).repeat(
            cost_matrix.shape[0], axis=0
        )
        fuse_sim = iou_sim * det_scores
        fuse_cost = 1 - fuse_sim
        return fuse_cost


class _Tracer:
    """
    Class to keep traces of tracked objects in video stream
    """

    def __init__(self, timeout_frames: int, trail_depth: int) -> None:
        """
        Constructor

        Args:
            timeout_frames (int): number of frames to keep inactive track
            trail_depth (int): number of frames in a trace to keep
        """
        self._timeout_count_dict: Dict[int, int] = {}
        self.active_trails: Dict[int, list] = {}
        self.trail_classes: Dict[int, str] = {}
        self._timeout_count_initial = timeout_frames
        self._trace_depth = trail_depth

    def update(self, result):
        """
        Update object traces with current frame result.

        Args:
            result: PySDK result object to update with.
                result.results[] dictionaries containing "track_id" and "bbox" keys will be used to update taces
        """

        # array of tracked object indexes and bboxes
        tracked_objects = np.array(
            [
                [idx, obj["track_id"]] + obj["bbox"]
                for idx, obj in enumerate(result.results)
                if "track_id" in obj
            ],
            dtype=np.int32,
        )

        if len(tracked_objects) > 0:
            # update active trails
            for idx, tid, x1, y1, x2, y2 in tracked_objects[:, 0:6]:
                bbox = np.array([x1, y1, x2, y2])
                trail = self.active_trails.get(tid, None)
                if trail is None:
                    trail = []
                    self.active_trails[tid] = trail
                    self.trail_classes[tid] = result.results[idx]["label"]
                trail.append(bbox)
                if len(trail) > self._trace_depth:
                    trail.pop(0)

                self._timeout_count_dict[tid] = self._timeout_count_initial

            inactive_set = set(self._timeout_count_dict.keys()) - set(
                tracked_objects[:, 1]
            )
        else:
            inactive_set = set(self._timeout_count_dict.keys())

        # remove inactive trails
        for tid in inactive_set:
            self._timeout_count_dict[tid] -= 1
            if self._timeout_count_dict[tid] == 0:
                del (
                    self._timeout_count_dict[tid],
                    self.active_trails[tid],
                    self.trail_classes[tid],
                )


class ObjectTracker(ResultAnalyzerBase):
    """
    Class to track objects in video stream.

    Analyzes the object detection `result` object passed to `analyze` method and, for each detected
    object in the `result.results[]` list, keeps its frame-to-frame track, assigning that track a
    unique track ID. Only objects belonging to the class list specified by the `class_list` constructor
    parameter are tracked.

    Updates each element of `result.results[]` list by adding the "track_id" key containing that unique
    track ID.

    If the `trail_depth` constructor parameter is not zero, also adds `trails` dictionary to the
    `result` object. This dictionary is keyed by track IDs and contains lists of (x1, y1, x2, y2)
    coordinates of object bounding boxes for every active trail.

    """

    def __init__(
        self,
        *,
        class_list: Optional[list] = None,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        anchor_point: AnchorPoint = AnchorPoint.BOTTOM_CENTER,
        trail_depth: int = 0,
        show_overlay: bool = True,
        annotation_color: Optional[tuple] = None,
        tracker_plugin: Union[str, TrackerModules] = 'bytetrack',
        plugin_args: Optional[Dict] = None
    ):
        """Constructor

        Args:
            class_list (list, optional): list of classes to count; if None, all classes are counted
            track_thresh (float, optional): Detection confidence threshold for track activation.
            track_buffer (int, optional): Number of frames to buffer when a track is lost.
            match_thresh (float, optional): IOU threshold for matching tracks with detections.
            anchor_point (AnchorPoint, optional): bbox anchor point to be used for showing object trails
            trail_depth (int, optional): number of frames in object trail to keep; 0 to disable tracing
            show_overlay: if True, annotate image; if False, send through original image
            annotation_color: Color to use for annotations, None to use complement to result overlay color
            tracker_plugin: (Union[str, TrackerModules]): Choose from available trackers,
                bytetrack, botsort-reid, deepsort, or dg-facetrack, or supply a TrackerModules
                object for a custom two stage tracker.
            plugin_args (dict, optional): Additional arguments used for each tracker plugin type.
        """
        self._tracker = _ByteTrack(class_list, track_thresh, track_buffer, match_thresh, tracker_plugin, plugin_args)
        self._anchor_point = anchor_point
        self._show_overlay = show_overlay
        self._annotation_color = annotation_color
        self._tracer: Optional[_Tracer] = (
            _Tracer(track_buffer, trail_depth) if trail_depth > 0 else None
        )

    def analyze(self, result):
        """
        Track object bounding boxes.
        Updates each element of `result.results[]` by adding the `track_id` key - unique track ID of the detected object
        If trail_depth is not zero, also adds `trails` dictionary to result object. This dictionary is keyed by track IDs
        and contains lists of (x1, y1, x2, y2) coordinates of object bounding boxes for every active trail.
        Also adds `trail_classes` dictionary to result object. This dictionary is keyed by track IDs and contains
        object class labels for every active trail.

        Args:
            result: PySDK model result object
        """
        self._tracker.update(result)
        if self._tracer is None:
            result.trails = {}
        else:
            self._tracer.update(result)
            result.trails = deepcopy(self._tracer.active_trails)
            result.trail_classes = deepcopy(self._tracer.trail_classes)

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """
        Display track IDs inside bounding boxes on a given image if tracing is disabled, trails computed using
        the specified bbox anchor point otherwise

        Args:
            result: PySDK result object to display (should be the same as used in analyze() method)
            image (np.ndarray): image to display on

        Returns:
            np.ndarray: annotated image
        """

        if not self._show_overlay:
            return image

        line_color = (
            color_complement(result.overlay_color)
            if self._annotation_color is None
            else self._annotation_color
        )
        text_color = deduce_text_color(line_color)

        if self._tracer is None:
            # if tracing is disabled, show track IDs inside bboxes
            for obj in result.results:
                if "track_id" in obj and "bbox" in obj:
                    track_id = str(obj["track_id"])
                    corner_pt = tuple(map(int, obj["bbox"][:2]))
                    put_text(
                        image,
                        str(track_id),
                        corner_pt,
                        font_color=text_color,
                        bg_color=line_color,
                        font_scale=result.overlay_font_scale,
                    )

        else:
            # if tracing is enabled, show trails

            all_trails = [
                get_anchor_coordinates(np.array(trail), self._anchor_point).astype(
                    np.int32
                )
                for _, trail in result.trails.items()
                if len(trail) > 1
            ]
            cv2.polylines(
                image,
                all_trails,
                False,
                rgb_to_bgr(line_color),
                result.overlay_line_width,
            )

            if result.overlay_show_labels:
                for tid, trail in result.trails.items():
                    if len(trail) > 1:
                        put_text(
                            image,
                            f"{result.trail_classes[tid]}: {tid}",
                            trail[-1],
                            font_color=text_color,
                            bg_color=line_color,
                            font_scale=result.overlay_font_scale,
                        )

        return image


class FaceObjectTracker(ObjectTracker):
    '''
    Custom annotator for dg-facetrack.
    Shows if the track has been ReIDed before and if the current decision
    was made using IoU or ReID.
    '''
    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        good_face_color = (0, 255, 0)
        bad_face_color = (255, 0, 0)

        good_face_color = color_complement(good_face_color)
        bad_face_color = color_complement(bad_face_color)

        good_face_text_color = deduce_text_color(good_face_color)
        bad_face_text_color = deduce_text_color(bad_face_color)

        for obj in result.results:
            if "track_id" in obj and "bbox" in obj:
                track_id = str(obj["track_id"])
                corner_pt = tuple(map(int, obj["bbox"][:2]))

                if "reid_used_once" in obj:
                    if obj["reid_used"]:
                        track_id = track_id + '_ReID'
                    else:
                        track_id = track_id + '_IoU'

                    put_text(
                        image,
                        track_id,
                        corner_pt,
                        font_color=good_face_color,
                        bg_color=good_face_text_color,
                        font_scale=result.overlay_font_scale,
                    )
                else:
                    track_id = '--' + track_id + '--'
                    put_text(
                        image,
                        track_id,
                        corner_pt,
                        font_color=bad_face_color,
                        bg_color=bad_face_text_color,
                        font_scale=result.overlay_font_scale,
                    )

        put_text(
            image,
            f'{(self._tracker._frame_id):04}',
            (0, 0),
            font_color=bad_face_color,
            bg_color=bad_face_text_color,
            font_scale=result.overlay_font_scale,
        )

        return image
