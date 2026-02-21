# scene_cut_detector.py: scene cut detection analyzer
#
# Copyright DeGirum Corporation 2026
# All rights reserved
# Implements analyzer class for detecting scene cuts in video streams.
#

#
# Copyright (C) 2014-2024 Brandon Castellano <http://www.bcastell.com>.
# PySceneDetect is licensed under the BSD 3-Clause License;
# visit the above page for details.
#


"""
Scene Cut Detector Analyzer Module Overview
====================================

This module provides an analyzer (`SceneCutDetector`) for detecting scene cuts in video
streams by comparing frame-to-frame differences using an adaptive thresholding approach.
The detector is based on the PySceneDetect adaptive algorithm, which calculates differences
in HSV color space and uses a rolling average to adapt to local changes.

Key Features:
    - **Adaptive Thresholding**: Uses rolling average of previous frames to adapt to gradual changes
    - **HSV Color Space**: Analyzes differences in hue, saturation, and luminance channels
    - **Fast Processing**: Supports luma-only mode and automatic frame resizing for performance
    - **Configurable Parameters**: Adjustable sensitivity, minimum scene length, and window size
    - **Real-time Detection**: Causal approach using only past frames for zero latency
    - **Scene Cut Flag**: Adds `scene_cut` boolean attribute to inference results

Typical Usage:
    1. Create a `SceneCutDetector` instance with desired parameters
    2. Attach it to a model or inference pipeline
    3. Process video frames through the analyzer
    4. Check `result.scene_cut` flag to detect scene transitions
    5. Use scene cut information for downstream processing or triggering actions

Put it before ObjectTracker in the analyzer pipeline to ensure cuts are detected before tracking is applied,
allowing you to reset object tracker on scene changes.

Integration Notes:
    - Works with any inference results that contain image data
    - Can be combined with other analyzers in a pipeline
    - Useful for video segmentation, activity detection, and content analysis
    - Maintains internal state to track frame history

Configuration Options:
    - `adaptive_threshold`: Sensitivity ratio for detecting cuts (higher = less sensitive)
    - `min_scene_len`: Minimum frames between detected cuts to avoid false positives
    - `window_width`: Number of previous frames for rolling average calculation
    - `min_content_val`: Minimum absolute change threshold for scene cuts
    - `luma_only`: Use only brightness changes for faster processing

Key Classes:
    - `SceneCutDetector`: Main analyzer class for scene cut detection
"""


import numpy as np
import cv2
from typing import Optional, Tuple, List
from .result_analyzer_base import ResultAnalyzerBase


class SceneCutDetector(ResultAnalyzerBase):
    """
    Analyzer for detecting scene cuts using adaptive thresholding.

    This analyzer examines consecutive frames and detects scene cuts when the frame-to-frame
    difference significantly exceeds the rolling average of recent frames. It uses HSV color
    space analysis to detect content changes while adapting to gradual variations like
    camera motion or lighting changes.

    The detector adds a `scene_cut` boolean attribute to each inference result indicating
    whether a scene cut was detected at that frame.

    Attributes:
        adaptive_threshold (float): Ratio threshold for scene cut detection.
        min_scene_len (int): Minimum frames between consecutive scene cuts.
        window_width (int): Number of previous frames used for rolling average.
        min_content_val (float): Minimum absolute content change threshold.
        luma_only (bool): Whether to use only luminance channel for comparison.
        resize_limit (int): Maximum image dimension to apply frame resizing to improve performance.
    """

    def __init__(
        self,
        *,
        adaptive_threshold: float = 3.0,
        min_scene_len: int = 15,
        window_width: int = 4,
        min_content_val: float = 15.0,
        luma_only: bool = False,
        resize_limit: int = 240,
    ):
        """
        Initialize the scene cut detector.

        Args:
            adaptive_threshold (float, optional): Ratio that the frame score must exceed
                relative to the average of surrounding frames to trigger a cut. Default 3.0
                means the frame must have 3x more change than its neighbors.
            min_scene_len (int, optional): Minimum number of frames between detected cuts.
                Default 15 frames.
            window_width (int, optional): Number of previous frames to use for computing
                the rolling average. Must be at least 1. Default 4.
            min_content_val (float, optional): Minimum absolute change threshold. Even if
                the adaptive ratio is exceeded, the absolute change must be at least this
                value. Default 15.0.
            luma_only (bool, optional): If True, only considers changes in luminance
                (brightness), ignoring color information for faster processing. Default False.
            resize_limit (int, optional): Maximum image dimension to apply frame resizing to improve
                performance. Frames larger than 1.5x this limit will be resized. Default 240.
        """
        super().__init__()

        if adaptive_threshold <= 1.0:
            raise ValueError(
                "adaptive_threshold must be greater than 1.0 for meaningful detection."
            )
        if min_scene_len < 1:
            raise ValueError("min_scene_len must be at least 1 frame.")
        if min_content_val <= 0:
            raise ValueError("min_content_val must be greater than 0.")
        if window_width < 1:
            raise ValueError("window_width must be at least 1.")
        if resize_limit < 16:
            raise ValueError("resize_limit must be at least 16 pixels.")

        self._adaptive_threshold = adaptive_threshold
        self._min_scene_len = min_scene_len
        self._min_content_val = min_content_val
        self._window_width = window_width
        self._luma_only = luma_only
        self._resize_limit = resize_limit

        # State variables
        self._last_hsv: Optional[
            Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]
        ] = None
        self._buffer: List[float] = []
        self._frame_count: int = 0
        self._last_cut_frame: int = (
            -min_scene_len
        )  # Allow cut on first frame if conditions met

    def analyze(self, result) -> None:
        """
        Analyze a frame and detect scene cuts using adaptive thresholding.

        This method processes the image from the inference result, calculates the
        frame-to-frame content difference, and sets `result.scene_cut` to True if
        a scene cut is detected, False otherwise.

        Uses a causal approach: compares the current frame score against the average of
        previous frame scores, enabling real-time detection with no latency.

        Args:
            result (degirum.postprocessor.InferenceResults): The inference result
                containing the image to analyze.

        Returns:
            None: Modifies `result` in place by adding the `scene_cut` attribute.
        """

        # Calculate content score (HSV difference from previous frame)
        content_score = self._calculate_hsv_diff(result.image)

        # Append current score to buffer
        self._buffer.append(content_score)
        self._frame_count += 1

        # Need at least window_width previous frames to compute average
        if len(self._buffer) <= self._window_width:
            result.scene_cut = False
            return

        # Keep only the frames we need (current + window_width previous)
        self._buffer = self._buffer[-(self._window_width + 1) :]

        # Calculate average of previous frames (exclude current)
        previous_scores = self._buffer[:-1]
        average_score = sum(previous_scores) / len(previous_scores)

        # Calculate adaptive ratio
        adaptive_ratio = (
            min(content_score / average_score, 255.0)
            if abs(average_score) >= 0.00001
            else 255.0
        )

        # Detect cut if both conditions are met
        threshold_met = (
            adaptive_ratio >= self._adaptive_threshold
            and content_score >= self._min_content_val
        )
        min_length_met = (
            self._frame_count - self._last_cut_frame
        ) >= self._min_scene_len

        if threshold_met and min_length_met:
            self._last_cut_frame = self._frame_count
            result.scene_cut = True
        else:
            result.scene_cut = False

    def _calculate_hsv_diff(self, frame_img: np.ndarray) -> float:
        """
        Calculate the HSV-based difference between current and previous frame.

        Args:
            frame_img (numpy.ndarray): Current frame in RGB format (as provided by
                inference results).

        Returns:
            float: Weighted average of differences in hue, saturation, and luma channels,
                or just luma difference if `luma_only` is True.
        """

        if not isinstance(frame_img, np.ndarray):
            # convert PIL image to numpy array if needed
            frame_img = np.array(frame_img)

        # Resize for performance
        height, width = frame_img.shape[:2]
        max_dim = max(height, width)
        if max_dim > 1.5 * self._resize_limit:
            scale = self._resize_limit / max_dim
            resize_width = int(width * scale)
            resize_height = int(height * scale)
            frame_img = cv2.resize(
                frame_img,
                (resize_width, resize_height),
                interpolation=cv2.INTER_AREA,
            )

        # Convert to HSV color space or grayscale
        if self._luma_only:
            lum = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            hue = sat = None
        else:
            hsv = cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV)
            hue, sat, lum = cv2.split(hsv)

        # First frame - store and return 0
        score = 0.0
        if self._last_hsv is not None:
            last_hue, last_sat, last_lum = self._last_hsv
            if self._luma_only:
                # Only calculate luma difference
                score = self._mean_pixel_diff(lum, last_lum)
            else:
                # Calculate weighted average of all three components
                # Type guards: when luma_only is False, hue/sat are not None
                assert hue is not None
                assert sat is not None
                assert last_hue is not None
                assert last_sat is not None
                hue_diff = self._mean_pixel_diff(hue, last_hue)
                sat_diff = self._mean_pixel_diff(sat, last_sat)
                lum_diff = self._mean_pixel_diff(lum, last_lum)
                score = (hue_diff + sat_diff + lum_diff) / 3.0

        # Store current frame for next comparison
        self._last_hsv = (hue, sat, lum)
        return score

    @staticmethod
    def _mean_pixel_diff(current: np.ndarray, previous: np.ndarray) -> float:
        """
        Calculate mean absolute difference between two single-channel images.

        Args:
            current (numpy.ndarray): Current frame channel (2D array).
            previous (numpy.ndarray): Previous frame channel (2D array).

        Returns:
            float: Mean absolute pixel difference.
        """
        # Convert to int32 to avoid overflow, compute absolute difference, then average
        num_pixels = float(current.shape[0] * current.shape[1])
        diff = np.sum(cv2.absdiff(current, previous))
        return float(diff / num_pixels)
