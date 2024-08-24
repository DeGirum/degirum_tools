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


import cv2, numpy as np, scipy.linalg
from enum import Enum
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from .math_support import box_iou_batch, AnchorPoint, get_anchor_coordinates
from .ui_support import put_text, deduce_text_color, color_complement
from .result_analyzer_base import ResultAnalyzerBase


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

    def __init__(
        self, tlwh: np.ndarray, score: float, obj_idx: int, id_counter: _IDCounter
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
    ):
        """
        Initialize the _ByteTrack object.

        Parameters:
            class_list (list, optional): list of classes to count; if None, all classes are counted
            track_thresh (float, optional): Detection confidence threshold for track activation.
            track_buffer (int, optional): Number of frames to buffer when a track is lost.
            match_thresh (float, optional): IOU threshold for matching tracks with detections.
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

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, i, self._id_counter)
                for (tlbr, s, i) in zip(dets, scores_keep, obj_indexes_keep)
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
        dists = _ByteTrack._iou_distance(strack_pool, detections)

        dists = _ByteTrack._fuse_score(dists, detections)
        matches, u_track, u_detection = _ByteTrack._linear_assignment(
            dists, thresh=self._match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
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
                STrack(STrack.tlbr_to_tlwh(tlbr), s, i, self._id_counter)
                for (tlbr, s, i) in zip(dets_second, scores_second, obj_indexes_second)
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == _TrackState.Tracked
        ]
        dists = _ByteTrack._iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = _ByteTrack._linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
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
        dists = _ByteTrack._iou_distance(unconfirmed, detections)

        dists = _ByteTrack._fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = _ByteTrack._linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self._frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self._det_thresh:
                continue
            track.activate(self._kalman_filter, self._frame_id)
            activated_stracks.append(track)

        """ Step 5: Update state"""
        for track in self._lost_tracks:
            if self._frame_id - track.end_frame > self._max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

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
        """
        self._tracker = _ByteTrack(class_list, track_thresh, track_buffer, match_thresh)
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
                image, all_trails, False, line_color, result.overlay_line_width
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
