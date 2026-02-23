#
# test_scene_cut_detector.py: unit tests for scene cut detector analyzer
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# Implements unit tests to test scene cut detector analyzer
#

import numpy as np
import pytest
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
from degirum_tools import video_source, open_video_stream
from degirum_tools.analyzers.scene_cut_detector import SceneCutDetector


def test_scene_cut_detector_parameters():
    """Test SceneCutDetector parameter validation."""

    # Valid parameters
    detector = SceneCutDetector(
        adaptive_threshold=3.0,
        min_scene_len=10,
        window_width=5,
        min_content_val=20.0,
        luma_only=True,
        resize_limit=300,
    )
    assert detector is not None

    # Invalid adaptive_threshold
    with pytest.raises(ValueError, match="adaptive_threshold must be greater than 1.0"):
        SceneCutDetector(adaptive_threshold=0.5)

    # Invalid min_scene_len
    with pytest.raises(ValueError, match="min_scene_len must be at least 1"):
        SceneCutDetector(min_scene_len=0)

    # Invalid min_content_val
    with pytest.raises(ValueError, match="min_content_val must be greater than 0"):
        SceneCutDetector(min_content_val=-1.0)

    # Invalid window_width
    with pytest.raises(ValueError, match="window_width must be at least 1"):
        SceneCutDetector(window_width=0)

    # Invalid resize_limit
    with pytest.raises(ValueError, match="resize_limit must be at least 16"):
        SceneCutDetector(resize_limit=10)


@dataclass
class SceneCutTestCase:
    """Test case for scene cut detection.

    Attributes:
        video_path: Path to the MP4 video file
        ground_truth_path: Path to the .txt file with ground truth scene boundaries
    """

    video_path: Path
    ground_truth_path: Path


class MockResult:
    """Mock inference result for testing SceneCutDetector."""

    def __init__(self, image):
        self.image = image
        self.scene_cut = False  # Will be set by analyzer


def parse_ground_truth(gt_path: Path) -> Set[int]:
    """Parse ground truth file to extract scene cut frame numbers.

    Ground truth format: two columns, space-separated
        scene-start-frame    scene-end-frame

    Example:
        0	58
        59	138
        139	202

    Returns set of frame numbers where scene cuts occur (start of new scenes,
    excluding the first scene which starts at frame 0).

    Args:
        gt_path: Path to ground truth .txt file

    Returns:
        Set of frame numbers where scene cuts are detected
    """
    scene_cuts = set()
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                start_frame = int(parts[0])
                # Scene cuts occur at the start of new scenes (excluding frame 0)
                if start_frame > 0:
                    scene_cuts.add(start_frame)
    return scene_cuts


def execute_test_case(
    test_case: SceneCutTestCase, detector: SceneCutDetector
) -> tuple[Set[int], Set[int], float]:
    """Execute a scene cut detection test case.

    Opens the video file, processes each frame through the detector,
    and records frame numbers where scene cuts are detected.

    Args:
        test_case: Test case containing video and ground truth paths
        detector: Configured SceneCutDetector instance

    Returns:
        Tuple of (detected_cuts, ground_truth_cuts, fps) where:
            - detected_cuts: Set of frame numbers where cuts were detected
            - ground_truth_cuts: Set of frame numbers from ground truth
            - fps: Processing speed in frames per second
    """
    detected_cuts = set()
    frame_number = 0

    # Process video frames with timing
    start_time = time.time()
    with open_video_stream(str(test_case.video_path)) as stream:
        for frame in video_source(stream):
            # Create mock result with the frame
            result = MockResult(frame)

            # Analyze frame
            detector.analyze(result)

            # Record if scene cut detected
            if result.scene_cut:
                detected_cuts.add(frame_number)

            frame_number += 1

    elapsed_time = time.time() - start_time
    fps = frame_number / elapsed_time if elapsed_time > 0 else 0.0

    # Load ground truth
    ground_truth_cuts = parse_ground_truth(test_case.ground_truth_path)

    return detected_cuts, ground_truth_cuts, fps


def collect_test_cases() -> List[SceneCutTestCase]:
    """Collect all test cases from the scene_cuts directory.

    Finds all .mp4 files and their corresponding .txt ground truth files.

    Returns:
        List of SceneCutTestCase objects
    """
    test_dir = Path(__file__).parent / "images" / "scene_cuts"
    test_cases = []

    # Find all MP4 files
    for video_path in sorted(test_dir.glob("*.mp4")):
        # Find corresponding ground truth file
        gt_path = video_path.with_suffix(".txt")
        if gt_path.exists():
            test_cases.append(
                SceneCutTestCase(video_path=video_path, ground_truth_path=gt_path)
            )

    return test_cases


def test_scene_cut_detector_with_ground_truth():
    """Test SceneCutDetector against ground truth data.

    Processes each test video and compares detected scene cuts with ground truth.
    Uses default parameters for the detector.
    Iterates over all video test cases in the scene_cuts directory.
    """
    test_cases = collect_test_cases()

    print("")
    for test_case in test_cases:
        # Create detector with default parameters
        detector = SceneCutDetector()

        # Execute test case
        detected_cuts, ground_truth_cuts, fps = execute_test_case(test_case, detector)

        # Calculate metrics
        true_positives = detected_cuts & ground_truth_cuts
        false_positives = detected_cuts - ground_truth_cuts
        false_negatives = ground_truth_cuts - detected_cuts

        # Calculate precision and recall
        precision = len(true_positives) / len(detected_cuts) if detected_cuts else 0.0
        recall = (
            len(true_positives) / len(ground_truth_cuts) if ground_truth_cuts else 1.0
        )

        print(f"{test_case.video_path.name} processing speed: {fps:.1f} FPS")

        # Assert minimum performance thresholds
        # Allow some tolerance since scene cut detection is not perfect
        assert recall == 1, (
            f"Recall too low: {recall:.2%}. " f"Missed cuts: {sorted(false_negatives)}"
        )
        assert precision == 1, (
            f"Precision too low: {precision:.2%}. "
            f"False alarms: {sorted(false_positives)}"
        )


def generate_frames_with_cuts(
    cut_positions: List[int], total_frames: int, frame_size: tuple = (100, 100, 3)
) -> List[np.ndarray]:
    """Generate frames with scene cuts at specified positions.

    Each scene gets a different intensity value (50, 100, 150, 200, etc.).

    Args:
        cut_positions: List of frame numbers where cuts occur (sorted)
        total_frames: Total number of frames to generate
        frame_size: Size of each frame (height, width, channels)

    Returns:
        List of numpy arrays representing frames
    """
    frames = []
    scene_boundaries = [0] + sorted(cut_positions) + [total_frames]

    for scene_idx in range(len(scene_boundaries) - 1):
        start = scene_boundaries[scene_idx]
        end = scene_boundaries[scene_idx + 1]
        # Generate different intensity for each scene
        # Use values that are well-separated: 50, 100, 150, 200, 250, etc.
        intensity = 50 + (scene_idx * 50) % 206  # Stay within 50-255 range

        for _ in range(end - start):
            frame = np.ones(frame_size, dtype=np.uint8) * intensity
            frames.append(frame)

    return frames


def run_detector_test(
    detector: SceneCutDetector,
    cut_positions_for_generation: List[int],
    expected_cut_positions: List[int],
    total_frames: int = 50,
) -> None:
    """Run detector on generated frames and compare with expected cuts.

    Args:
        detector: SceneCutDetector instance to test
        cut_positions_for_generation: Where to place cuts in generated frames
        expected_cut_positions: Where we expect the detector to find cuts
        total_frames: Total number of frames to generate

    Raises:
        AssertionError: If detected cuts don't match expected cuts
    """
    # Generate frames
    frames = generate_frames_with_cuts(cut_positions_for_generation, total_frames)

    # Run detector
    detected_cuts = []
    for frame_num, frame in enumerate(frames):
        result = MockResult(frame)
        detector.analyze(result)
        if result.scene_cut:
            detected_cuts.append(frame_num)

    # Compare with expected
    assert detected_cuts == expected_cut_positions, (
        f"Expected cuts at {expected_cut_positions}, "
        f"but detected at {detected_cuts}"
    )


# Define test cases
@dataclass
class SyntheticTestCase:
    """Test case for synthetic frame testing.

    Attributes:
        name: Descriptive name for the test case
        detector_params: Parameters to pass to SceneCutDetector constructor
        cut_positions: Where to place cuts in generated frames
        expected_cuts: Where we expect detector to find cuts
        total_frames: Total number of frames to generate
    """

    name: str
    detector_params: dict
    cut_positions: List[int]
    expected_cuts: List[int]
    total_frames: int = 50


# Test cases covering various scenarios
SYNTHETIC_TEST_CASES = [
    SyntheticTestCase(
        name="basic_single_cut",
        detector_params={},  # Default parameters
        cut_positions=[15],  # Cut at frame 15
        expected_cuts=[15],  # Should detect at frame 15
        total_frames=30,
    ),
    SyntheticTestCase(
        name="multiple_cuts_with_spacing",
        detector_params={},
        cut_positions=[10, 25, 40],
        expected_cuts=[10, 25, 40],
        total_frames=50,
    ),
    SyntheticTestCase(
        name="min_scene_len_suppression",
        detector_params={
            "adaptive_threshold": 2.0,
            "min_scene_len": 10,
            "window_width": 3,
            "min_content_val": 10.0,
        },
        cut_positions=[8, 12, 25],  # Cuts at 8, 12 (too close), 25
        expected_cuts=[8, 25],  # Cut at 12 should be suppressed
        total_frames=35,
    ),
    SyntheticTestCase(
        name="min_scene_len_at_boundary",
        detector_params={
            "min_scene_len": 10,
            "adaptive_threshold": 2.0,
            "window_width": 3,
        },
        cut_positions=[8, 18, 28],  # Cuts exactly 10 frames apart
        expected_cuts=[8, 18, 28],  # All should be detected
        total_frames=40,
    ),
    SyntheticTestCase(
        name="early_cut_after_warmup",
        detector_params={"window_width": 4, "adaptive_threshold": 2.5},
        cut_positions=[5],  # Cut right after warmup period
        expected_cuts=[5],
        total_frames=20,
    ),
    SyntheticTestCase(
        name="luma_only_mode",
        detector_params={"luma_only": True, "adaptive_threshold": 2.0},
        cut_positions=[10, 25],
        expected_cuts=[10, 25],
        total_frames=35,
    ),
]


def test_scene_cut_detector_synthetic():
    """Test SceneCutDetector with synthetic generated frames.

    Uses the structured test framework to generate frames with cuts at
    specified positions and verify the detector finds them correctly.
    Iterates over all test cases defined in SYNTHETIC_TEST_CASES.
    """
    for test_case in SYNTHETIC_TEST_CASES:
        # Create detector with specified parameters
        detector = SceneCutDetector(**test_case.detector_params)

        # Run the test
        run_detector_test(
            detector=detector,
            cut_positions_for_generation=test_case.cut_positions,
            expected_cut_positions=test_case.expected_cuts,
            total_frames=test_case.total_frames,
        )
