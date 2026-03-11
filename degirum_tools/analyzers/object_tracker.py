#
# object_tracker.py: multi-object tracker
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements classes for multi-object tracking
#

# MIT License
#
# Copyright (c) 2022 Roboflow
# Copyright (c) 2021 Yifu Zhang
# Copyright (c) 2023 Jinkun Cao (OC-SORT)
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

"""
Object Tracker Analyzer Module Overview
====================================

Implements multi-object tracking with pluggable backends and per-track certainty scoring.

Supported tracker backends:
    - ``"bytetrack"`` -- `BYTETrack algorithm <https://github.com/ifzhang/ByteTrack>`_ (default)
    - ``"ocsort"`` -- `OC-SORT algorithm <https://github.com/noahcao/OC_SORT>`_ with
      Observation-Centric Re-Update (ORU) and Observation-Centric Momentum (OCM)

Per-track certainty fields added to each tracked detection:
    - ``track_confidence`` -- composite quality score blending detection score, match IoU, and track age
    - ``track_uncertainty`` -- motion uncertainty derived from Kalman filter covariance
    - ``track_occlusion_risk`` -- spatial overlap risk with other active tracks

Key Features:
    - **Persistent Object Identity**: Maintains consistent track IDs across frames
    - **Class Filtering**: Optionally tracks only specified object classes
    - **Track Lifecycle Management**: Handles track creation, updating, and removal
    - **Trail Visualization**: Records and displays object movement history
    - **Track Retention**: Configurable buffer for handling temporary object disappearances
    - **Visual Overlay**: Displays track IDs and optional trails on frames
    - **Integration Support**: Provides track IDs for downstream analyzers (e.g., zone counting, line crossing)

Typical Usage:
    1. Create an ``ObjectTracker`` instance with desired tracking parameters
    2. Process each frame's detection results through the tracker
    3. Access track IDs, certainty scores, and trails from the augmented results
    4. Optionally visualize tracking results using the annotate method
    5. Use track IDs in downstream analyzers for advanced analytics

Integration Notes:
    - Requires detection results with bounding boxes and confidence scores
    - Track IDs are added to detection results as ``track_id`` field
    - Trail information is stored in ``trails`` and ``trail_classes`` dictionaries
    - Works effectively with zone counting and line crossing analyzers
    - Supports both frame-based and time-based track retention

Key Classes:
    - ``STrack``: Internal class representing a single tracked object with state (ByteTrack)
    - ``ObjectTracker``: Main analyzer class that processes detections and maintains tracks

Configuration Options:
    - ``class_list``: Filter tracking to specific object classes
    - ``track_thresh``: Confidence threshold for initiating new tracks
    - ``track_buffer``: Frames to retain tracks after object disappearance
    - ``match_thresh``: IoU threshold for matching detections to existing tracks
    - ``trail_depth``: Number of recent positions to keep for trail visualization
    - ``show_overlay``: Enable/disable visual annotations
    - ``annotation_color``: Customize overlay appearance
"""

import cv2, numpy as np, scipy.linalg
from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from ..tools import (
    box_iou_batch,
    AnchorPoint,
    get_anchor_coordinates,
    put_text,
    deduce_text_color,
    color_complement,
    rgb_to_bgr,
)
from .result_analyzer_base import ResultAnalyzerBase


# ============================================================================
# Shared types and abstractions
# ============================================================================


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


@dataclass
class TrackOutput:
    """Output from a tracker backend for a single active track.

    Attributes:
        obj_idx: Index into the frame's detection list (-1 if prediction-only).
        track_id: Unique persistent track identifier.
        tlbr: Smoothed bounding box as [x1, y1, x2, y2].
        score: Detection confidence score for the most recent observation.
        tracklet_len: Number of consecutive frames this track has been active.
        covariance: Kalman filter covariance matrix (NxN) or None.
        match_iou: IoU between the track prediction and the matched detection (0 if new).
        match_step: Association stage that produced the match
            (1=high-score, 2=low-score, 3=unconfirmed, 0=newly created).
    """

    obj_idx: int
    track_id: int
    tlbr: np.ndarray
    score: float
    tracklet_len: int
    covariance: Optional[np.ndarray]
    match_iou: float
    match_step: int


class _TrackerBackend(ABC):
    """Abstract base class for tracker backends.

    Every backend receives pre-extracted detection arrays and returns a list of
    ``TrackOutput`` for active tracks. Class filtering is applied inside the backend.
    """

    @abstractmethod
    def update(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        class_list: Optional[list],
        labels: list,
    ) -> List[TrackOutput]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


# ============================================================================
# ByteTrack components
# ============================================================================


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


class STrack:
    """Represents a single tracked object in the multi-object tracking system.

    Each STrack holds the object's bounding box state, unique track identifier, detection confidence score,
    and tracking status (e.g., new, tracked, lost, removed). A Kalman filter is used internally to predict
    and update the object's state across frames.

    Tracks are created when new objects are detected, updated when detections are matched to existing tracks,
    and can be reactivated if a lost track matches a new detection. This class provides methods to manage
    the lifecycle of a track (activation, update, reactivation) and utility functions for bounding box format conversion.

    Attributes:
        track_id (int): Unique ID for this track.
        is_activated (bool): Whether the track has been activated (confirmed) at least once.
        state (_TrackState): Current state of the track (New, Tracked, Lost, or Removed).
        start_frame (int): Frame index when this track was first activated.
        frame_id (int): Frame index of the last update for this track (last seen frame).
        tracklet_len (int): Number of frames this track has been in the tracked state.
        score (float): Detection confidence score for the most recent observation of this track.
        obj_idx (int): Index of this object's detection in the frame's results list (used for internal bookkeeping).
    """

    def __init__(
        self, tlwh: np.ndarray, score: float, obj_idx: int, id_counter: _IDCounter
    ):
        """
        Constructor.

        Args:
            tlwh (np.ndarray): Initial bounding box in (x, y, w, h) format, where (x, y) is the top-left corner.
            score (float): Detection confidence score for this object.
            obj_idx (int): Index of this object's detection in the current frame's results list.
            id_counter (_IDCounter): Shared counter used to generate globally unique track_id values.
        """

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

        self.match_iou: float = 0.0
        self.match_step: int = 0

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
        """Activates this track with an initial detection.

        Initializes the track's state using the provided Kalman filter, assigns a new track ID,
        and sets the track status to "Tracked".

        Args:
            kalman_filter (_KalmanFilter): Kalman filter to associate with this track.
            frame_id (int): Frame index at which the track is initialized.
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
        """Reactivates a track that was previously lost, using a new detection.

        Updates the track's state with the new detection's information and sets the state to "Tracked".
        If new_id is True, a new track ID is assigned; otherwise, it retains the original ID.

        Args:
            new_track (STrack): New track (detection) to merge into this lost track.
            frame_id (int): Current frame index at which the track is reactivated.
            new_id (bool, optional): Whether to assign a new ID to the track. Defaults to False.
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
        """Updates this track with a new matched detection.

        Incorporates the detection's bounding box and score into this track's state, updates the
        Kalman filter prediction, and increments the track length. The track state is set to "Tracked".

        Args:
            new_track (STrack): The new detection track that matched this track.
            frame_id (int): Current frame index for the update.
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
        """Returns the track's current bounding box in (x, y, w, h) format.

        Returns:
            np.ndarray: Bounding box where (x, y) is the top-left corner.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Returns the track's bounding box in corner format (x_min, y_min, x_max, y_max).

        Returns:
            np.ndarray: Bounding box in (x_min, y_min, x_max, y_max) format.
        """
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Converts bounding box from (top-left x, y, width, height) to (center x, y, aspect ratio, height).

        Args:
            tlwh (np.ndarray): Bounding box in (x, y, w, h) format.

        Returns:
            np.ndarray: Bounding box in (center x, y, aspect ratio, height) format.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        """Converts bounding box from (top-left, bottom-right) to (top-left, width, height).

        Args:
            tlbr (np.ndarray): Bounding box in (x1, y1, x2, y2) format.

        Returns:
            np.ndarray: Bounding box in (x, y, w, h) format.
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class _ByteTrackBackend(_TrackerBackend):
    """
    Multi-object tracking class.
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        """
        Initialize the _ByteTrackBackend object.

        Parameters:
            track_thresh (float, optional): Detection confidence threshold for track activation.
            track_buffer (int, optional): Number of frames to buffer when a track is lost.
            match_thresh (float, optional): IOU threshold for matching tracks with detections.
        """
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

    def reset(self):
        """
        Resets the tracker to its initial state, clearing all existing tracks.
        """
        self._tracked_tracks.clear()
        self._lost_tracks.clear()
        self._removed_tracks.clear()

    def update(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        class_list: Optional[list],
        labels: list,
    ) -> List[TrackOutput]:
        """
        Updates the tracker with the provided detections and
            returns the updated detection results.

        Parameters:
            bboxes: (N, 4) array of detection bounding boxes [x1, y1, x2, y2].
            scores: (N,) array of detection confidence scores.
            class_list: list of classes to track; if None, all classes are tracked.
            labels: list of class labels for each detection.

        Returns:
            List[TrackOutput]: list of active track outputs for this frame.
        """
        obj_indexes = np.arange(len(scores))
        scores = scores.copy()

        # apply class filtering
        if class_list is not None and len(scores) > 0:
            excluded_classes = np.array(
                [label not in class_list for label in labels]
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
        tracked_stracks: List[STrack] = []
        for track in self._tracked_tracks:
            track.obj_idx = -1  # clear object index in advance
            track.match_iou = 0.0
            track.match_step = 0
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        # Predict the current location with KF
        strack_pool = _ByteTrackBackend._join_tracks(
            tracked_stracks, self._lost_tracks
        )
        STrack.multi_predict(strack_pool, self._kalman_filter)
        dists = _ByteTrackBackend._iou_distance(strack_pool, detections)

        # save raw IoU before fuse_score for match_iou recording
        raw_iou_sim = 1.0 - dists if dists.size > 0 else dists

        dists = _ByteTrackBackend._fuse_score(dists, detections)
        matches, u_track, u_detection = _ByteTrackBackend._linear_assignment(
            dists, thresh=self._match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            track.match_iou = float(raw_iou_sim[itracked, idet])
            track.match_step = 1
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
        dists2 = _ByteTrackBackend._iou_distance(r_tracked_stracks, detections_second)
        raw_iou_sim2 = 1.0 - dists2 if dists2.size > 0 else dists2
        matches, u_track, u_detection_second = _ByteTrackBackend._linear_assignment(
            dists2, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            track.match_iou = float(raw_iou_sim2[itracked, idet])
            track.match_step = 2
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
        dists3 = _ByteTrackBackend._iou_distance(unconfirmed, detections)
        raw_iou_sim3 = 1.0 - dists3 if dists3.size > 0 else dists3

        dists3 = _ByteTrackBackend._fuse_score(dists3, detections)
        matches, u_unconfirmed, u_detection = _ByteTrackBackend._linear_assignment(
            dists3, thresh=0.7
        )
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            track.match_iou = float(raw_iou_sim3[itracked, idet])
            track.match_step = 3
            track.update(detections[idet], self._frame_id)
            activated_stracks.append(track)
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
        self._tracked_tracks = _ByteTrackBackend._join_tracks(
            self._tracked_tracks, activated_stracks
        )
        self._tracked_tracks = _ByteTrackBackend._join_tracks(
            self._tracked_tracks, refind_stracks
        )
        self._lost_tracks = _ByteTrackBackend._sub_tracks(
            self._lost_tracks, self._tracked_tracks
        )
        self._lost_tracks.extend(lost_stracks)
        self._lost_tracks = _ByteTrackBackend._sub_tracks(
            self._lost_tracks, self._removed_tracks
        )
        self._removed_tracks = removed_stracks
        self._tracked_tracks, self._lost_tracks = (
            _ByteTrackBackend._remove_duplicate_tracks(
                self._tracked_tracks, self._lost_tracks
            )
        )

        # update result
        output_stracks = [
            track for track in self._tracked_tracks if track.is_activated
        ]

        outputs: List[TrackOutput] = []
        for track in output_stracks:
            if track.obj_idx < 0:
                continue
            outputs.append(
                TrackOutput(
                    obj_idx=track.obj_idx,
                    track_id=track.track_id,
                    tlbr=track.tlbr,
                    score=track.score,
                    tracklet_len=track.tracklet_len,
                    covariance=track.covariance,
                    match_iou=track.match_iou,
                    match_step=track.match_step,
                )
            )
        return outputs

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
        if not track_list_b:
            return track_list_a
        if not track_list_a:
            return track_list_b

        seen_track_ids = {track.track_id for track in track_list_a}
        result = list(track_list_a)

        for track in track_list_b:
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
        if not track_list_a or not track_list_b:
            return track_list_a

        track_ids_b = {track.track_id for track in track_list_b}
        return [track for track in track_list_a if track.track_id not in track_ids_b]

    @staticmethod
    def _remove_duplicate_tracks(tracks_a: List, tracks_b: List) -> Tuple[List, List]:
        pairwise_distance = _ByteTrackBackend._iou_distance(tracks_a, tracks_b)
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
            track
            for index, track in enumerate(tracks_a)
            if index not in duplicates_a
        ]
        result_b = [
            track
            for index, track in enumerate(tracks_b)
            if index not in duplicates_b
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

        return _ByteTrackBackend._indices_to_matches(cost_matrix, indices, thresh)

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


# ============================================================================
# OC-SORT components
# ============================================================================


def _bbox_to_z_ocsort(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] to [cx, cy, area, aspect_ratio] column vector."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r], dtype=np.float64).reshape((4, 1))


def _z_to_bbox_ocsort(x: np.ndarray) -> np.ndarray:
    """Convert state [cx, cy, area, aspect_ratio, ...] to [x1,y1,x2,y2]."""
    w = np.sqrt(max(float(x[2]) * float(x[3]), 0.0))
    h = float(x[2]) / (w + 1e-6)
    cx, cy = float(x[0]), float(x[1])
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


def _speed_direction(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute normalized velocity direction vector from bbox1 center to bbox2 center.
    Returns [dy, dx] (note: y-first to match OC-SORT convention)."""
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def _speed_direction_batch(
    dets: np.ndarray, tracks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise direction vectors from track positions to detection positions.

    Args:
        dets: (N, 4+) detection bboxes [x1,y1,x2,y2,...]
        tracks: (M, 4+) track/observation bboxes [x1,y1,x2,y2,...]

    Returns:
        (dy, dx) each of shape (M, N) -- normalized direction components.
    """
    tracks_exp = tracks[:, :4, np.newaxis]  # (M, 4, 1)
    CX1 = (dets[:, 0] + dets[:, 2]) / 2.0  # (N,)
    CY1 = (dets[:, 1] + dets[:, 3]) / 2.0
    CX2 = (tracks_exp[:, 0] + tracks_exp[:, 2]) / 2.0  # (M, 1)
    CY2 = (tracks_exp[:, 1] + tracks_exp[:, 3]) / 2.0
    dx = CX1 - CX2  # (M, N)
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    return dy / norm, dx / norm


def _k_previous_obs(
    observations: Dict[int, np.ndarray], cur_age: int, k: int
) -> np.ndarray:
    """Get the observation from k steps ago (or most recent if unavailable)."""
    if len(observations) == 0:
        return np.array([-1, -1, -1, -1, -1])
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


class _OCSortKalmanFilter:
    """Kalman filter for OC-SORT with 7-dim state [cx, cy, s, r, vcx, vcy, vs].

    s = area (w*h), r = aspect ratio (w/h).
    """

    def __init__(self):
        self._F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self._H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        self._R = np.eye(4, dtype=np.float64)
        self._R[2:, 2:] *= 10.0
        self._Q = np.eye(7, dtype=np.float64)
        self._Q[-1, -1] *= 0.01
        self._Q[4:, 4:] *= 0.01

    def initiate(
        self, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros((7, 1), dtype=np.float64)
        mean[:4] = measurement
        covariance = np.eye(7, dtype=np.float64)
        covariance[4:, 4:] *= 1000.0
        covariance *= 10.0
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (mean[6, 0] + mean[2, 0]) <= 0:
            mean[6, 0] *= 0.0
        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + self._Q
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean = self._H @ mean
        projected_cov = self._H @ covariance @ self._H.T + self._R

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._H.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


class _OCSortTrack:
    """Single tracked object for the OC-SORT backend."""

    def __init__(
        self,
        bbox: np.ndarray,
        score: float,
        obj_idx: int,
        id_counter: _IDCounter,
        delta_t: int = 3,
    ):
        self.id_counter = id_counter
        self.track_id = 0
        self.is_activated = False
        self.state = _TrackState.New

        self._kf = _OCSortKalmanFilter()
        self.mean, self.covariance = self._kf.initiate(_bbox_to_z_ocsort(bbox))

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.start_frame = 0
        self.frame_id = 0

        self.last_observation = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        self.observations: Dict[int, np.ndarray] = {}
        self.history_observations: List[np.ndarray] = []
        self.velocity: Optional[np.ndarray] = None
        self.delta_t = delta_t

        self.score = score
        self.obj_idx = obj_idx
        self.tracklet_len = 0
        self.match_iou: float = 0.0
        self.match_step: int = 0

    def next_id(self) -> int:
        self.id_counter._count += 1
        return self.id_counter._count

    def predict(self) -> np.ndarray:
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.tlbr

    def update_with_detection(
        self, bbox: np.ndarray, score: float, obj_idx: int, frame_id: int
    ):
        """Update track state with a matched detection."""
        if self.last_observation.sum() >= 0:
            previous_box = None
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if self.age - dt in self.observations:
                    previous_box = self.observations[self.age - dt]
                    break
            if previous_box is None:
                previous_box = self.last_observation
            self.velocity = _speed_direction(previous_box[:4], bbox)

        self.last_observation = np.concatenate([bbox, [score]])
        self.observations[self.age] = np.concatenate([bbox, [score]])
        self.history_observations.append(np.concatenate([bbox, [score]]))

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.tracklet_len += 1

        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, _bbox_to_z_ocsort(bbox)
        )

        self.score = score
        self.obj_idx = obj_idx
        self.frame_id = frame_id
        self.state = _TrackState.Tracked
        self.is_activated = True

    def update_no_detection(self):
        """Called when track is not matched to any detection this frame."""
        pass

    def activate(self, frame_id: int):
        self.track_id = self.next_id()
        self.state = _TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    @property
    def tlbr(self) -> np.ndarray:
        if self.last_observation.sum() >= 0:
            return self.last_observation[:4].copy()
        return _z_to_bbox_ocsort(self.mean.flatten())

    @property
    def predicted_tlbr(self) -> np.ndarray:
        return _z_to_bbox_ocsort(self.mean.flatten())

    def mark_lost(self):
        self.state = _TrackState.Lost

    def mark_removed(self):
        self.state = _TrackState.Removed


class _OCSortBackend(_TrackerBackend):
    """OC-SORT multi-object tracking backend.

    Implements Observation-Centric Re-Update (ORU), Observation-Centric Momentum (OCM),
    and optional BYTE-style low-score second association.
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.3,
        delta_t: int = 3,
        inertia: float = 0.2,
    ):
        self._track_thresh = track_thresh
        self._match_thresh = match_thresh
        self._max_age = track_buffer
        self._delta_t = delta_t
        self._inertia = inertia
        self._frame_count = 0
        self._trackers: List[_OCSortTrack] = []
        self._id_counter = _IDCounter()

    def reset(self):
        self._trackers.clear()
        self._frame_count = 0

    def update(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        class_list: Optional[list],
        labels: list,
    ) -> List[TrackOutput]:
        scores = scores.copy()

        if class_list is not None and len(scores) > 0:
            excluded = np.array([label not in class_list for label in labels])
            scores[excluded] = 0

        self._frame_count += 1
        dets_all = (
            np.column_stack([bboxes, scores]) if len(bboxes) > 0
            else np.empty((0, 5))
        )
        obj_indexes_all = np.arange(len(scores))

        # split high / low score detections
        remain_inds = scores > self._track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self._track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets = dets_all[remain_inds]
        dets_second = dets_all[inds_second]
        obj_indexes_keep = obj_indexes_all[remain_inds]
        obj_indexes_second = obj_indexes_all[inds_second]

        # predict all existing trackers
        trks = np.zeros((len(self._trackers), 4))
        to_del = []
        for t in range(len(self._trackers)):
            pos = self._trackers[t].predict()
            trks[t] = self._trackers[t].predicted_tlbr
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self._trackers.pop(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # gather velocities and k-previous observations for OCM
        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array([0.0, 0.0])
                for trk in self._trackers
            ]
        )
        k_observations = np.array(
            [
                _k_previous_obs(trk.observations, trk.age, self._delta_t)
                for trk in self._trackers
            ]
        ) if len(self._trackers) > 0 else np.empty((0, 5))

        # -- First association: high-score dets with OCM --
        matched, unmatched_dets, unmatched_trks = self._associate_ocm(
            dets, trks, self._match_thresh, velocities, k_observations, self._inertia
        )

        iou_for_match = np.array([])
        if len(dets) > 0 and len(trks) > 0:
            iou_for_match = box_iou_batch(
                np.asarray(dets[:, :4]), np.asarray(trks)
            )

        for m in matched:
            det_idx, trk_idx = int(m[0]), int(m[1])
            iou_val = float(iou_for_match[det_idx, trk_idx]) if iou_for_match.size > 0 else 0.0
            self._trackers[trk_idx].match_iou = iou_val
            self._trackers[trk_idx].match_step = 1
            self._trackers[trk_idx].update_with_detection(
                dets[det_idx, :4],
                float(dets[det_idx, 4]),
                int(obj_indexes_keep[det_idx]),
                self._frame_count,
            )

        # -- BYTE second association: low-score dets vs unmatched tracks --
        if len(dets_second) > 0 and len(unmatched_trks) > 0:
            u_trks_arr = trks[unmatched_trks] if len(trks) > 0 else np.empty((0, 4))
            iou_left = box_iou_batch(
                np.asarray(dets_second[:, :4]), np.asarray(u_trks_arr)
            )
            if iou_left.size > 0 and iou_left.max() > self._match_thresh:
                cost_left = 1.0 - iou_left
                cost_left[cost_left > (1.0 - self._match_thresh)] = (
                    1.0 - self._match_thresh + 1e-4
                )
                row_ind, col_ind = linear_sum_assignment(cost_left)
                to_remove_trk = []
                for r, c in zip(row_ind, col_ind):
                    if iou_left[r, c] < self._match_thresh:
                        continue
                    trk_idx = unmatched_trks[c]
                    self._trackers[trk_idx].match_iou = float(iou_left[r, c])
                    self._trackers[trk_idx].match_step = 2
                    self._trackers[trk_idx].update_with_detection(
                        dets_second[r, :4],
                        float(dets_second[r, 4]),
                        int(obj_indexes_second[r]),
                        self._frame_count,
                    )
                    to_remove_trk.append(c)
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array([unmatched_trks[c] for c in to_remove_trk])
                )

        # -- ORU: re-associate remaining unmatched dets with last observations --
        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            left_dets = dets[unmatched_dets]
            last_boxes = np.array(
                [self._trackers[t].last_observation[:4] for t in unmatched_trks]
            )
            valid_last = np.array(
                [self._trackers[t].last_observation.sum() >= 0 for t in unmatched_trks]
            )
            if valid_last.any():
                valid_indices = np.where(valid_last)[0]
                valid_last_boxes = last_boxes[valid_indices]
                iou_left = box_iou_batch(
                    np.asarray(left_dets[:, :4]), np.asarray(valid_last_boxes)
                )
                if iou_left.size > 0 and iou_left.max() > self._match_thresh:
                    cost_left = 1.0 - iou_left
                    cost_left[cost_left > (1.0 - self._match_thresh)] = (
                        1.0 - self._match_thresh + 1e-4
                    )
                    row_ind, col_ind = linear_sum_assignment(cost_left)
                    to_remove_det = []
                    to_remove_trk = []
                    for r, c in zip(row_ind, col_ind):
                        if iou_left[r, c] < self._match_thresh:
                            continue
                        det_i = unmatched_dets[r]
                        trk_i = unmatched_trks[valid_indices[c]]
                        self._trackers[trk_i].match_iou = float(iou_left[r, c])
                        self._trackers[trk_i].match_step = 3
                        self._trackers[trk_i].update_with_detection(
                            dets[det_i, :4],
                            float(dets[det_i, 4]),
                            int(obj_indexes_keep[det_i]),
                            self._frame_count,
                        )
                        to_remove_det.append(r)
                        to_remove_trk.append(valid_indices[c])
                    unmatched_dets = np.setdiff1d(
                        unmatched_dets,
                        np.array([unmatched_dets[r] for r in to_remove_det]),
                    )
                    unmatched_trks = np.setdiff1d(
                        unmatched_trks,
                        np.array([unmatched_trks[t] for t in to_remove_trk]),
                    )

        # mark unmatched trackers
        for t in unmatched_trks:
            self._trackers[t].update_no_detection()
            self._trackers[t].obj_idx = -1
            self._trackers[t].match_iou = 0.0
            self._trackers[t].match_step = 0

        # create new tracks for unmatched high-score detections
        for i in unmatched_dets:
            det_score = float(dets[i, 4])
            if det_score < self._track_thresh:
                continue
            trk = _OCSortTrack(
                dets[i, :4], det_score, int(obj_indexes_keep[i]),
                self._id_counter, self._delta_t,
            )
            trk.activate(self._frame_count)
            trk.last_observation = np.concatenate([dets[i, :4], [det_score]])
            trk.observations[trk.age] = trk.last_observation.copy()
            trk.history_observations.append(trk.last_observation.copy())
            self._trackers.append(trk)

        # collect outputs and remove dead tracks
        outputs: List[TrackOutput] = []
        trackers_to_keep = []
        for trk in self._trackers:
            if trk.time_since_update > self._max_age:
                continue
            trackers_to_keep.append(trk)

            cond_active = trk.time_since_update < 1
            cond_mature = (
                trk.hit_streak >= 1 or self._frame_count <= 1
            )
            if cond_active and cond_mature and trk.obj_idx >= 0:
                trk.is_activated = True
                outputs.append(
                    TrackOutput(
                        obj_idx=trk.obj_idx,
                        track_id=trk.track_id,
                        tlbr=trk.predicted_tlbr,
                        score=trk.score,
                        tracklet_len=trk.tracklet_len,
                        covariance=trk.covariance,
                        match_iou=trk.match_iou,
                        match_step=trk.match_step,
                    )
                )
        self._trackers = trackers_to_keep
        return outputs

    @staticmethod
    def _associate_ocm(
        dets: np.ndarray,
        trks: np.ndarray,
        iou_threshold: float,
        velocities: np.ndarray,
        k_observations: np.ndarray,
        inertia: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """First-round association with IoU + velocity direction consistency (OCM)."""
        if len(trks) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(dets)),
                np.empty((0,), dtype=int),
            )
        if len(dets) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.empty((0,), dtype=int),
                np.arange(len(trks)),
            )

        iou_matrix = box_iou_batch(np.asarray(dets[:, :4]), np.asarray(trks))

        # OCM: velocity direction consistency
        if k_observations.shape[0] > 0 and k_observations.shape[1] >= 4:
            valid_obs = k_observations[:, 0] >= 0
            if valid_obs.any():
                # _speed_direction_batch returns (M_trk, N_det); transpose to (N_det, M_trk)
                Y, X = _speed_direction_batch(dets[:, :4], k_observations[:, :4])
                inertia_Y = velocities[:, 0:1]  # (M, 1)
                inertia_X = velocities[:, 1:2]
                diff_angle_cos = np.clip(inertia_Y * Y + inertia_X * X, -1, 1)
                diff_angle = np.arccos(diff_angle_cos)
                diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

                vel_valid = np.linalg.norm(velocities, axis=1) > 1e-6
                diff_angle[~vel_valid] = 0
                diff_angle[~valid_obs] = 0
                diff_angle = diff_angle.T  # (M_trk, N_det) -> (N_det, M_trk)
            else:
                diff_angle = np.zeros_like(iou_matrix)
        else:
            diff_angle = np.zeros_like(iou_matrix)

        cost_matrix = 1.0 - (iou_matrix + inertia * diff_angle)
        gate = 1.0 - iou_threshold
        cost_matrix[cost_matrix > gate] = gate + 1e-4

        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(dets)),
                np.arange(len(trks)),
            )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        indices = np.column_stack((row_ind, col_ind))

        matched_cost = cost_matrix[row_ind, col_ind]
        matched_mask = matched_cost <= gate
        matches = indices[matched_mask]

        unmatched_dets = np.array(
            sorted(set(range(len(dets))) - set(matches[:, 0])) if len(matches) > 0
            else list(range(len(dets)))
        )
        unmatched_trks = np.array(
            sorted(set(range(len(trks))) - set(matches[:, 1])) if len(matches) > 0
            else list(range(len(trks)))
        )
        return matches, unmatched_dets, unmatched_trks


# ============================================================================
# Certainty scoring
# ============================================================================


class _CertaintyScorer:
    """Computes per-track certainty fields from TrackOutput data."""

    def __init__(
        self,
        confidence_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        age_saturate: int = 10,
        iou_risk_cap: float = 0.5,
    ):
        self._w_det, self._w_iou, self._w_age = confidence_weights
        self._age_saturate = float(max(age_saturate, 1))
        self._iou_risk_cap = max(iou_risk_cap, 1e-6)

    def compute(
        self, track_outputs: List[TrackOutput]
    ) -> List[Tuple[float, float, float]]:
        """Compute (track_confidence, track_uncertainty, track_occlusion_risk) per track."""
        n = len(track_outputs)
        if n == 0:
            return []

        tlbrs = np.array([to.tlbr for to in track_outputs])
        if n > 1:
            pairwise = box_iou_batch(tlbrs, tlbrs)
            np.fill_diagonal(pairwise, 0.0)
            max_ious = pairwise.max(axis=1)
        else:
            max_ious = np.zeros(n)

        results: List[Tuple[float, float, float]] = []
        for i, to in enumerate(track_outputs):
            s_det = to.score
            s_iou = to.match_iou * (0.7 if to.match_step == 2 else 1.0)
            s_age = min(to.tracklet_len / self._age_saturate, 1.0)
            conf = float(
                self._w_det * s_det + self._w_iou * s_iou + self._w_age * s_age
            )

            if to.covariance is not None:
                pos_cov = to.covariance[:2, :2]
                unc_raw = np.sqrt(max(np.trace(pos_cov), 0.0))
                bbox_h = max(float(to.tlbr[3] - to.tlbr[1]), 1.0)
                sigma_ref = max(bbox_h * 0.1, 5.0)
                unc = float(1.0 - np.exp(-unc_raw / sigma_ref))
            else:
                unc = 1.0

            occ = float(min(max_ious[i] / self._iou_risk_cap, 1.0))
            results.append((conf, unc, occ))

        return results


# ============================================================================
# Trail tracking
# ============================================================================


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

    def reset(self):
        """
        Reset all traces and timeouts.
        """
        self._timeout_count_dict.clear()
        self.active_trails.clear()
        self.trail_classes.clear()

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
            dtype=int,
        )

        if len(tracked_objects) > 0:
            # update active trails
            for row in tracked_objects:
                idx, tid, *bbox = map(int, row)
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


# ============================================================================
# Public API
# ============================================================================


class ObjectTracker(ResultAnalyzerBase):
    """Analyzer that tracks objects across frames in a video stream.

    This analyzer assigns persistent IDs to detected objects, allowing them to be tracked from frame to frame.
    It supports pluggable tracker backends (``"bytetrack"`` or ``"ocsort"``) and adds per-track certainty
    scores to each detection. Optionally, tracking can be restricted to specific object classes via the
    *class_list* parameter.

    After each call to ``analyze()``, the input result's detections are augmented with:
    - ``track_id`` -- persistent identity across frames
    - ``track_confidence`` -- composite quality score in [0, 1]
    - ``track_uncertainty`` -- motion uncertainty in [0, 1]
    - ``track_occlusion_risk`` -- spatial overlap risk in [0, 1]

    If a trail length is specified (non-zero *trail_depth*), the result will also contain
    ``trails`` and ``trail_classes`` dictionaries: ``trails`` maps each track ID to a list of recent bounding box
    coordinates (the object's trail), and ``trail_classes`` maps each track ID to the object's class label.
    These facilitate drawing object paths and labeling them.

    Functionality:
        - Unique ID assignment: Provides a unique ID for each object and maintains that ID across frames.
        - Class filtering: Ignores detections whose class is not in the specified *class_list*.
        - Track retention buffer: Continues to track objects for *track_buffer* frames after they disappear.
        - Trajectory history: Keeps a history of each object's movement up to *trail_depth* frames long.
        - Overlay support: Can overlay track IDs and trails on frames for visualization.

    Typical usage involves calling ``analyze()`` on each frame's detection results to update tracks, then
    ``annotate()`` to visualize or output the tracked results. For instance, in a video processing loop, use
    ``tracker.analyze(detections)`` followed by ``tracker.annotate(detections, frame)`` to maintain and display
    object tracks.
    """

    def __init__(
        self,
        *,
        tracker_type: str = "bytetrack",
        class_list: Optional[list] = None,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        reset_at_scene_cut: bool = False,
        anchor_point: AnchorPoint = AnchorPoint.BOTTOM_CENTER,
        trail_depth: int = 0,
        show_overlay: bool = True,
        annotation_color: Optional[tuple] = None,
        show_only_track_ids: bool = False,
        confidence_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        age_saturate: int = 10,
        iou_risk_cap: float = 0.5,
        delta_t: int = 3,
        inertia: float = 0.2,
    ):
        """Constructor.

        Args:
            tracker_type (str, optional): Tracker backend to use. ``"bytetrack"`` (default) or ``"ocsort"``.
            class_list (List[str], optional): List of object classes to track. If None, all detected classes are tracked.
            track_thresh (float, optional): Detection confidence threshold for initiating a new track.
            track_buffer (int, optional): Number of frames to keep a lost track before removing it.
            match_thresh (float, optional): Intersection-over-union (IoU) threshold for matching detections to existing tracks.
                Defaults to 0.8 for bytetrack, 0.3 is typical for ocsort.
            reset_at_scene_cut (bool, optional): If True, resets all tracks when a scene cut is detected.
                Requires the result to have a ``scene_cut`` attribute (set by SceneCutDetector).
                Use this to avoid tracking objects across scene transitions in videos with cuts or edits.
            anchor_point (AnchorPoint, optional): Anchor point on the bounding box used for trail visualization.
            trail_depth (int, optional): Number of recent positions to keep for each track's trail. Set 0 to disable trail tracking.
            show_overlay (bool, optional): If True, annotate the image; if False, return the original image.
            annotation_color (Tuple[int, int, int], optional): RGB tuple to use for annotations. If None, a contrasting color is chosen automatically.
            show_only_track_ids (bool, optional): If True, only track IDs are shown in the annotations. If False, trails and labels are also shown when available.
            confidence_weights (Tuple[float, float, float], optional): Weights (w_det, w_iou, w_age) for track_confidence computation.
            age_saturate (int, optional): Frames after which the age component of track_confidence saturates.
            iou_risk_cap (float, optional): IoU value at which track_occlusion_risk saturates to 1.0.
            delta_t (int, optional): (ocsort only) Observation lookback for velocity estimation.
            inertia (float, optional): (ocsort only) Weight of velocity direction consistency in association.
        """
        tracker_type = tracker_type.lower()
        if tracker_type == "bytetrack":
            self._backend: _TrackerBackend = _ByteTrackBackend(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
            )
        elif tracker_type == "ocsort":
            self._backend = _OCSortBackend(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                delta_t=delta_t,
                inertia=inertia,
            )
        else:
            raise ValueError(
                f"Unknown tracker_type '{tracker_type}'. Use 'bytetrack' or 'ocsort'."
            )

        self._class_list = class_list
        self._certainty = _CertaintyScorer(
            confidence_weights=confidence_weights,
            age_saturate=age_saturate,
            iou_risk_cap=iou_risk_cap,
        )
        self._anchor_point = anchor_point
        self._show_overlay = show_overlay
        self._annotation_color = annotation_color
        self._show_only_track_ids = show_only_track_ids
        # Reset tracker on scene cuts when enabled (requires SceneCutDetector in pipeline)
        self._reset_at_scene_cut = reset_at_scene_cut
        self._tracer: Optional[_Tracer] = (
            _Tracer(track_buffer, trail_depth) if trail_depth > 0 else None
        )

    def analyze(self, result):
        """Analyzes a detection result and maintains object tracks across frames.

        Matches the current frame's detections to existing tracks, assigns track IDs to each detection,
        computes per-track certainty scores, and updates or creates tracks as necessary. If trail_depth
        was set, this method also updates each track's trail of past positions.

        The input result is updated in-place. Each detection in result.results receives a "track_id"
        identifying its track, along with "track_confidence", "track_uncertainty", and
        "track_occlusion_risk" certainty fields. If trails are enabled, result.trails and
        result.trail_classes are updated to reflect the current active tracks.

        Args:
            result (InferenceResults): Model inference result for the current frame, containing
                detected object bounding boxes and classes.
        """
        # Reset tracker on scene cuts when enabled (requires SceneCutDetector in pipeline)
        if self._reset_at_scene_cut:
            # Reset tracker on scene cuts if enabled
            if not hasattr(result, "scene_cut"):
                raise AttributeError(
                    "reset_at_scene_cut is enabled but result does not have a 'scene_cut' attribute. "
                    "Please add SceneCutDetector before ObjectTracker in your analyzer pipeline."
                )
            if result.scene_cut:
                # Scene cut detected - reset all tracks to avoid tracking across scene boundaries
                self._backend.reset()
                if self._tracer is not None:
                    self._tracer.reset()

        # Extract detection arrays
        if len(result.results) > 0:
            bboxes = np.array([obj["bbox"] for obj in result.results])
            scores = np.array([obj["score"] for obj in result.results])
            labels_list = [obj.get("label", "") for obj in result.results]
        else:
            bboxes = np.empty((0, 4))
            scores = np.empty((0,))
            labels_list = []

        # Run tracker backend
        track_outputs = self._backend.update(
            bboxes, scores, self._class_list, labels_list
        )

        # Write track results back to result
        for to in track_outputs:
            if to.obj_idx < 0:
                continue
            result.results[to.obj_idx]["bbox"] = to.tlbr.tolist()
            result.results[to.obj_idx]["track_id"] = to.track_id

        # Compute and write certainty scores
        if track_outputs:
            certainties = self._certainty.compute(track_outputs)
            for to, (conf, unc, occ) in zip(track_outputs, certainties):
                if to.obj_idx < 0:
                    continue
                result.results[to.obj_idx]["track_confidence"] = conf
                result.results[to.obj_idx]["track_uncertainty"] = unc
                result.results[to.obj_idx]["track_occlusion_risk"] = occ

        # Trail tracking
        if self._tracer is None:
            result.trails = {}
        else:
            self._tracer.update(result)
            result.trails = deepcopy(self._tracer.active_trails)
            result.trail_classes = deepcopy(self._tracer.trail_classes)

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """Draws tracking annotations on an image.

        If trails are not being used, writes each object's track ID at its bounding box location.
        If trails are enabled, draws each object's trajectory and labels the end with the object's
        class name and track ID.

        Args:
            result (InferenceResults): The inference result that was previously analyzed.
            image (np.ndarray): The image (in BGR format) on which to draw the annotations.

        Returns:
            np.ndarray: The image with tracking annotations drawn.
        """

        if not self._show_overlay:
            return image

        line_color = (
            color_complement(result.overlay_color)
            if self._annotation_color is None
            else self._annotation_color
        )
        text_color = deduce_text_color(line_color)

        # when forced to or when tracing is disabled, show track IDs inside bboxes
        if self._tracer is None or self._show_only_track_ids:
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
            if result.overlay_line_width > 0:
                all_trails = [
                    get_anchor_coordinates(np.array(trail), self._anchor_point).astype(
                        int
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
