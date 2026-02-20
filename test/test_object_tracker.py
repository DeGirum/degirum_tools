#
# test_object_tracker.py: unit tests for object tracker analyzer
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# Implements unit tests to test object tracker analyzer
#

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from copy import deepcopy


@dataclass
class TrajectorySpec:
    """Specification for object trajectory across frames.

    Attributes:
        start_frame: First frame where object appears (inclusive)
        stop_frame: Last frame where object appears (inclusive)
        start_bbox: Initial bounding box [x1, y1, x2, y2]
        velocity: Movement vector [dx, dy] per frame
        score: Detection confidence score
        label: Object class label
        expected_track_id: Expected track ID for this trajectory in output.
            If -1, detections from this trajectory should not be tracked (e.g., low score).
            Otherwise, this value is used as track_id in expected results.
    """

    start_frame: int
    stop_frame: int
    start_bbox: List[float]
    velocity: List[float]
    expected_track_id: int
    score: float = 1.0
    label: str = "object"
    active_immediately: bool = False

    def get_bbox_at_frame(self, frame_idx: int) -> Optional[List[float]]:
        """Calculate bbox position at given frame index.

        Args:
            frame_idx: Frame index to calculate bbox for

        Returns:
            Bounding box [x1, y1, x2, y2] or None if object not present in this frame
        """
        if frame_idx < self.start_frame or frame_idx > self.stop_frame:
            return None

        frames_elapsed = frame_idx - self.start_frame
        x1, y1, x2, y2 = self.start_bbox
        dx, dy = self.velocity

        return [
            x1 + dx * frames_elapsed,
            y1 + dy * frames_elapsed,
            x2 + dx * frames_elapsed,
            y2 + dy * frames_elapsed,
        ]


@dataclass
class MockResult:
    """Mock inference result for testing ObjectTracker.

    Attributes:
        results: List of detection dictionaries with bbox, score, label
    """

    results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TrackerTestCase:
    """Test case for ObjectTracker.

    Attributes:
        name: Descriptive name for the test case
        tracker: ObjectTracker instance configured for this test
        input_results: List of MockResult objects to feed to tracker
        expected_results: List of expected MockResult objects after processing
        bbox_tolerance: Maximum allowed bbox coordinate difference in pixels
    """

    name: str
    tracker: Any  # ObjectTracker instance
    input_results: List[MockResult]
    expected_results: List[MockResult]
    bbox_tolerance: float


def compute_adaptive_bbox_tolerance(
    trajectories: List[TrajectorySpec], relative_factor: float = 0.12
) -> float:
    """Compute adaptive bbox tolerance based on object sizes.

    This function computes a tolerance value that scales with the size of the
    tracked objects, making tests more robust to Kalman filter adjustments.

    Args:
        trajectories: List of trajectory specifications
        relative_factor: Fraction of bbox size to use as tolerance (default 12%)

    Returns:
        Tolerance value in pixels (minimum 1.0 pixel)
    """
    max_dimension = 0.0
    for traj in trajectories:
        x1, y1, x2, y2 = traj.start_bbox
        width = x2 - x1
        height = y2 - y1
        max_dimension = max(max_dimension, width, height)

    # Ensure minimum tolerance of 1.0 pixel
    return max(1.0, max_dimension * relative_factor)


def generate_tracker_test_case(
    name: str,
    *,
    num_frames: int,
    trajectories: List[TrajectorySpec],
    tracker_params: Optional[Dict[str, Any]] = None,
    scene_cuts: Optional[List[int]] = None,
) -> TrackerTestCase:
    """Generate a test case for ObjectTracker with specified trajectories.

    This function creates input results based on trajectory specifications and
    generates expected output results based on simple tracking rules:
    - First detection in each trajectory is not tracked (tracker needs second frame to confirm)
    - Subsequent detections get track_id from trajectory's expected_track_id
    - If expected_track_id is -1, detections are not tracked (rejected by tracker)
    - Scene cuts reset all tracking state

    Current implementation assumes:
    - Non-crossing trajectories
    - Linear motion only
    - Tracking starts from second detection of each object

    Args:
        num_frames: Total number of frames in the test sequence
        tracker_params: Optional dictionary of ObjectTracker constructor arguments
        trajectories: List of TrajectorySpec objects defining object movements.
            Each trajectory must specify expected_track_id:
            - Use positive integers (1, 2, 3...) for tracked objects
            - Use -1 for objects that should not be tracked
        name: Descriptive name for the test case
        scene_cuts: Optional list of frame indices where scene cuts occur.
            If None or empty, scene_cut attribute is not added to results.
            If provided, scene_cut attribute is added to all results (True for listed frames).

    Returns:
        TrackerTestCase with generated input and expected output results
    """
    from degirum_tools.analyzers.object_tracker import ObjectTracker

    if scene_cuts is None:
        scene_cuts = []

    # Generate input results
    input_results = []
    for frame_idx in range(num_frames):
        detections = []
        for traj in trajectories:
            bbox = traj.get_bbox_at_frame(frame_idx)
            if bbox is not None:
                detections.append(
                    {"bbox": bbox, "score": traj.score, "label": traj.label}
                )

        # Create result with or without scene_cut attribute
        if scene_cuts:
            # Add scene_cut attribute when scene_cuts list is provided
            result = MockResult(results=detections)
            setattr(result, "scene_cut", frame_idx in scene_cuts)
        else:
            # Don't add scene_cut attribute when scene_cuts is empty
            result = MockResult(results=detections)

        input_results.append(result)

    # Generate expected results by manually implementing tracking logic
    # This simulates what ObjectTracker should do without actually calling it
    if tracker_params is None:
        tracker_params = {}
    tracker = ObjectTracker(**tracker_params)

    # Check if trail tracking is enabled (simulating _Tracer initialization)
    trail_depth = tracker_params.get("trail_depth", 0)
    track_buffer = tracker_params.get("track_buffer", 30)
    enable_trailing = trail_depth > 0

    # Initialize tracer state if trails are enabled (simulating _Tracer internal state)
    if enable_trailing:
        active_trails: Dict[int, List[List[float]]] = {}
        trail_classes: Dict[int, int] = {}
        timeout_counters: Dict[int, int] = {}

    expected_results = []

    for frame_idx, inp_result in enumerate(input_results):
        # Create a deep copy to avoid modifying input
        exp_result = deepcopy(inp_result)

        res_idx = 0
        for traj in trajectories:
            if frame_idx >= traj.start_frame and frame_idx <= traj.stop_frame:
                if (
                    traj.active_immediately or frame_idx > traj.start_frame
                ) and traj.expected_track_id != -1:
                    # object is tracked from second detection onwards and only if expected_track_id is not -1
                    # special case: if active_immediately is True, it is tracked on first detection
                    exp_result.results[res_idx]["track_id"] = traj.expected_track_id

                res_idx += 1

        # Update trails if enabled (simulating _Tracer.update method)
        if enable_trailing:
            # Collect tracked objects from this frame (those with track_id)
            tracked_objects = [
                (obj["track_id"], obj["bbox"], obj["label"])
                for obj in exp_result.results
                if "track_id" in obj
            ]

            if tracked_objects:
                # Update active trails for tracked objects
                active_track_ids = set()
                for track_id, bbox, label in tracked_objects:
                    active_track_ids.add(track_id)

                    # Initialize trail if new
                    if track_id not in active_trails:
                        active_trails[track_id] = []
                        trail_classes[track_id] = label

                    # Add bbox to trail
                    active_trails[track_id].append(bbox)

                    # Limit trail depth
                    if len(active_trails[track_id]) > trail_depth:
                        active_trails[track_id].pop(0)

                    # Reset timeout
                    timeout_counters[track_id] = track_buffer

                # Handle inactive tracks
                inactive_track_ids = set(timeout_counters.keys()) - active_track_ids
            else:
                # No tracked objects this frame, all are inactive
                inactive_track_ids = set(timeout_counters.keys())

            # Decrement timeouts for inactive tracks and remove expired ones
            for track_id in inactive_track_ids:
                timeout_counters[track_id] -= 1
                if timeout_counters[track_id] == 0:
                    del active_trails[track_id]
                    del trail_classes[track_id]
                    del timeout_counters[track_id]

            # Add trails to expected result (deep copy to match ObjectTracker behavior)
            setattr(exp_result, "trails", deepcopy(active_trails))
            setattr(exp_result, "trail_classes", deepcopy(trail_classes))
        else:
            # If trails not enabled, set empty dict (match ObjectTracker behavior)
            setattr(exp_result, "trails", {})

        expected_results.append(exp_result)

    # Compute adaptive bbox tolerance based on object sizes
    bbox_tolerance = compute_adaptive_bbox_tolerance(trajectories)

    return TrackerTestCase(
        name=name,
        tracker=tracker,
        input_results=input_results,
        expected_results=expected_results,
        bbox_tolerance=bbox_tolerance,
    )


def run_tracker_test_case(test_case: TrackerTestCase) -> None:
    """Execute a tracker test case and verify results match expectations.

    This function runs the tracker on each input result, collects actual outputs,
    and compares them with expected results. Raises assertion errors if mismatches occur.

    Args:
        test_case: TrackerTestCase containing tracker, inputs, expected outputs,
            and bbox_tolerance (computed adaptively based on object sizes)

    Raises:
        AssertionError: If actual results don't match expected results
    """
    bbox_tolerance = test_case.bbox_tolerance
    actual_results = []

    # Run tracker on each input result
    for frame_idx, inp_result in enumerate(test_case.input_results):
        result = deepcopy(inp_result)
        test_case.tracker.analyze(result)
        actual_results.append(result)
        # print(frame_idx, result)

    # return

    # Compare actual vs expected results
    assert len(actual_results) == len(test_case.expected_results), (
        f"{test_case.name}: Number of results mismatch. "
        f"Expected {len(test_case.expected_results)}, got {len(actual_results)}"
    )

    for frame_idx, (actual, expected) in enumerate(
        zip(actual_results, test_case.expected_results)
    ):
        # Check number of detections
        assert len(actual.results) == len(expected.results), (
            f"{test_case.name} frame {frame_idx}: Detection count mismatch. "
            f"Expected {len(expected.results)}, got {len(actual.results)}"
        )

        # Check each detection
        for det_idx, (actual_det, expected_det) in enumerate(
            zip(actual.results, expected.results)
        ):
            # Check if track_id presence matches
            actual_has_track_id = "track_id" in actual_det
            expected_has_track_id = "track_id" in expected_det

            assert actual_has_track_id == expected_has_track_id, (
                f"{test_case.name} frame {frame_idx} detection {det_idx}: "
                f"track_id presence mismatch. Expected: {expected_has_track_id}, "
                f"Got: {actual_has_track_id}"
            )

            # If track_id exists, check it matches
            if expected_has_track_id:
                assert actual_det["track_id"] == expected_det["track_id"], (
                    f"{test_case.name} frame {frame_idx} detection {det_idx}: "
                    f"track_id mismatch. Expected: {expected_det['track_id']}, "
                    f"Got: {actual_det['track_id']}"
                )

            # Check bbox, score, label match
            # Bbox comparison with tolerance to account for Kalman filter adjustments
            actual_bbox = np.array(actual_det["bbox"])
            expected_bbox = np.array(expected_det["bbox"])
            bbox_diff = np.abs(actual_bbox - expected_bbox)
            max_diff = np.max(bbox_diff)

            assert max_diff <= bbox_tolerance, (
                f"{test_case.name} frame {frame_idx} detection {det_idx}: "
                f"bbox mismatch. Expected: {expected_det['bbox']}, "
                f"Got: {actual_det['bbox']}, "
                f"Max difference: {max_diff:.2f} pixels (tolerance: {bbox_tolerance})"
            )

            assert actual_det["score"] == expected_det["score"], (
                f"{test_case.name} frame {frame_idx} detection {det_idx}: "
                f"score mismatch"
            )
            assert actual_det["label"] == expected_det["label"], (
                f"{test_case.name} frame {frame_idx} detection {det_idx}: "
                f"label mismatch"
            )

        # Check scene_cut attribute if expected
        if hasattr(expected, "scene_cut"):
            assert hasattr(actual, "scene_cut"), (
                f"{test_case.name} frame {frame_idx}: "
                f"Expected scene_cut attribute but it's missing"
            )
            assert actual.scene_cut == expected.scene_cut, (
                f"{test_case.name} frame {frame_idx}: scene_cut mismatch. "
                f"Expected: {expected.scene_cut}, Got: {actual.scene_cut}"
            )

        # Check trails attribute if expected
        if hasattr(expected, "trails"):
            assert hasattr(actual, "trails"), (
                f"{test_case.name} frame {frame_idx}: "
                f"Expected trails attribute but it's missing"
            )
            assert set(actual.trails.keys()) == set(expected.trails.keys()), (
                f"{test_case.name} frame {frame_idx}: trails keys mismatch. "
                f"Expected track IDs: {set(expected.trails.keys())}, "
                f"Got track IDs: {set(actual.trails.keys())}"
            )

            # Check each trail
            for track_id in expected.trails:
                expected_trail = expected.trails[track_id]
                actual_trail = actual.trails[track_id]

                assert len(actual_trail) == len(expected_trail), (
                    f"{test_case.name} frame {frame_idx} track {track_id}: "
                    f"trail length mismatch. Expected: {len(expected_trail)}, "
                    f"Got: {len(actual_trail)}"
                )

                # Check each bbox in the trail with tolerance
                for bbox_idx, (actual_bbox, expected_bbox) in enumerate(
                    zip(actual_trail, expected_trail)
                ):
                    actual_bbox_arr = np.array(actual_bbox)
                    expected_bbox_arr = np.array(expected_bbox)
                    bbox_diff = np.abs(actual_bbox_arr - expected_bbox_arr)
                    max_diff = np.max(bbox_diff)

                    assert max_diff <= bbox_tolerance, (
                        f"{test_case.name} frame {frame_idx} track {track_id} "
                        f"trail position {bbox_idx}: bbox mismatch. "
                        f"Expected: {expected_bbox}, Got: {actual_bbox}, "
                        f"Max difference: {max_diff:.2f} pixels (tolerance: {bbox_tolerance})"
                    )

        # Check trail_classes attribute if expected
        if hasattr(expected, "trail_classes"):
            assert hasattr(actual, "trail_classes"), (
                f"{test_case.name} frame {frame_idx}: "
                f"Expected trail_classes attribute but it's missing"
            )
            assert actual.trail_classes == expected.trail_classes, (
                f"{test_case.name} frame {frame_idx}: trail_classes mismatch. "
                f"Expected: {expected.trail_classes}, Got: {actual.trail_classes}"
            )


# List of test cases for ObjectTracker
_test_cases = [
    # Test case 1: Single object with linear motion
    generate_tracker_test_case(
        "Single object linear motion",
        num_frames=10,
        trajectories=[
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[10.0, 10.0, 30.0, 30.0],
                velocity=[2.0, 1.0],  # Move right 2px, down 1px per frame
                expected_track_id=1,  # Expected to be tracked with ID 1
                active_immediately=True,  # Starts at frame 0
            )
        ],
    ),
    # Test case 2: Three concurrent trajectories
    generate_tracker_test_case(
        "Three concurrent objects",
        num_frames=15,
        trajectories=[
            TrajectorySpec(
                start_frame=0,
                stop_frame=14,
                start_bbox=[10.0, 10.0, 30.0, 30.0],
                velocity=[2.0, 0.5],
                expected_track_id=1,
                active_immediately=True,  # Starts at frame 0
            ),
            TrajectorySpec(
                start_frame=1,
                stop_frame=14,
                start_bbox=[50.0, 20.0, 70.0, 40.0],
                velocity=[1.0, 1.5],
                expected_track_id=2,
            ),
            TrajectorySpec(
                start_frame=2,
                stop_frame=14,
                start_bbox=[100.0, 50.0, 120.0, 70.0],
                velocity=[-1.0, 2.0],
                expected_track_id=3,
            ),
        ],
    ),
    # Test case 3: Three sequential trajectories with pauses
    generate_tracker_test_case(
        "Three sequential objects with pauses",
        num_frames=25,
        tracker_params={},
        trajectories=[
            # First object: frames 0-5
            TrajectorySpec(
                start_frame=0,
                stop_frame=5,
                start_bbox=[10.0, 10.0, 30.0, 30.0],
                velocity=[2.0, 1.0],
                expected_track_id=1,
                active_immediately=True,  # Starts at frame 0
            ),
            # Pause: frames 6-8 (no objects)
            # Second object: frames 9-14
            TrajectorySpec(
                start_frame=9,
                stop_frame=14,
                start_bbox=[50.0, 50.0, 70.0, 70.0],
                velocity=[1.5, 0.5],
                expected_track_id=2,
            ),
            # Pause: frames 15-17 (no objects)
            # Third object: frames 18-24
            TrajectorySpec(
                start_frame=18,
                stop_frame=24,
                start_bbox=[100.0, 30.0, 120.0, 50.0],
                velocity=[-1.0, 1.5],
                expected_track_id=3,
            ),
        ],
    ),
    # Test case 4: Testing track_thresh parameter
    generate_tracker_test_case(
        name="track_thresh filtering",
        num_frames=10,
        tracker_params={
            "track_thresh": 0.5
        },  # Only track objects with score >= 0.6 (track_thresh + 0.1)
        trajectories=[
            # High score - should be tracked
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[10.0, 10.0, 30.0, 30.0],
                velocity=[1.0, 0.5],
                score=0.9,
                expected_track_id=1,  # Should be tracked
                active_immediately=True,  # Starts at frame 0
            ),
            # Score above detection threshold - should be tracked
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[50.0, 20.0, 70.0, 40.0],
                velocity=[0.5, 1.0],
                score=0.65,  # Above det_thresh (0.6), should be tracked
                expected_track_id=2,  # Should be tracked
                active_immediately=True,  # Starts at frame 0
            ),
            # Score below detection threshold - should NOT be tracked
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[100.0, 30.0, 120.0, 50.0],
                velocity=[-0.5, 1.5],
                score=0.4,  # Below track_thresh (0.5), should NOT be tracked
                expected_track_id=-1,  # Should NOT be tracked
                active_immediately=True,  # Starts at frame 0 (but won't be tracked due to low score)
            ),
            # Very low score - should NOT be tracked
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[150.0, 40.0, 170.0, 60.0],
                velocity=[1.5, -0.5],
                score=0.1,
                expected_track_id=-1,  # Should NOT be tracked
                active_immediately=True,  # Starts at frame 0 (but won't be tracked due to low score)
            ),
        ],
    ),
    # Test case 5: Testing class_list parameter
    generate_tracker_test_case(
        name="class_list filtering",
        num_frames=10,
        tracker_params={
            "class_list": ["car", "bus"]
        },  # Only track "car" and "bus" classes
        trajectories=[
            # Car - should be tracked (in class_list)
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[10.0, 10.0, 30.0, 30.0],
                velocity=[1.0, 0.5],
                score=0.9,
                label="car",
                expected_track_id=1,  # Should be tracked
                active_immediately=True,  # Starts at frame 0
            ),
            # Truck - should NOT be tracked (not in class_list)
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[50.0, 20.0, 70.0, 40.0],
                velocity=[0.5, 1.0],
                score=0.85,
                label="truck",
                expected_track_id=-1,  # Should NOT be tracked
                active_immediately=True,  # Starts at frame 0 (but won't be tracked - wrong class)
            ),
            # Bus - should be tracked (in class_list)
            TrajectorySpec(
                start_frame=1,
                stop_frame=9,
                start_bbox=[100.0, 30.0, 130.0, 60.0],
                velocity=[-0.5, 1.5],
                score=0.8,
                label="bus",
                expected_track_id=2,  # Should be tracked
            ),
            # Motorcycle - should NOT be tracked (not in class_list)
            TrajectorySpec(
                start_frame=2,
                stop_frame=9,
                start_bbox=[150.0, 40.0, 170.0, 60.0],
                velocity=[1.5, -0.5],
                score=0.75,
                label="motorcycle",
                expected_track_id=-1,  # Should NOT be tracked
            ),
        ],
    ),
    # Test case 6: Testing reset_at_scene_cut parameter
    generate_tracker_test_case(
        name="reset_at_scene_cut behavior",
        num_frames=20,
        tracker_params={"reset_at_scene_cut": True},  # Reset tracker at scene cuts
        scene_cuts=[10],  # Scene cut occurs at frame 10
        trajectories=[
            # Continuous object trajectory split at scene cut frame
            # First part: frames 0-9, tracked as ID 1
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[10.0, 10.0, 30.0, 30.0],
                velocity=[2.0, 1.0],
                expected_track_id=1,  # Should be tracked with ID 1
                active_immediately=True,  # Starts at frame 0
            ),
            # Second part: frames 10-19, tracked as ID 2 after reset
            # Continuing from where first part ended to simulate continuous motion
            TrajectorySpec(
                start_frame=10,
                stop_frame=19,
                start_bbox=[30.0, 20.0, 50.0, 40.0],  # Position at frame 10
                velocity=[2.0, 1.0],  # Same velocity
                expected_track_id=2,  # Should be tracked with new ID 2 after reset
            ),
        ],
    ),
    # Test case 7: Testing track_buffer parameter - gap within buffer
    generate_tracker_test_case(
        name="track_buffer with gap within buffer",
        num_frames=15,
        tracker_params={
            "track_buffer": 5
        },  # Keep lost tracks for 5 frames before removing
        trajectories=[
            # First trajectory: frames 0-4 (object present)
            TrajectorySpec(
                start_frame=0,
                stop_frame=4,
                start_bbox=[10.0, 10.0, 30.0, 30.0],
                velocity=[2.0, 1.0],
                expected_track_id=1,  # Should be tracked with ID 1
                active_immediately=True,  # Starts at frame 0
            ),
            # Gap: frames 5-7 (3 frames, object not detected)
            # Second trajectory: frames 8-14 (object reappears)
            # Position continues from where first trajectory would have been at frame 8
            # Frame 4 position: [18, 14, 38, 34]
            # Frame 8 position if motion continued: [18+2*4, 14+1*4, 38+2*4, 34+1*4] = [26, 18, 46, 38]
            TrajectorySpec(
                start_frame=8,
                stop_frame=14,
                start_bbox=[26.0, 18.0, 46.0, 38.0],  # Continues from expected position
                velocity=[2.0, 1.0],  # Same velocity
                expected_track_id=1,  # Should maintain ID 1 (gap of 3 frames < buffer of 5)
                active_immediately=True,  # Reactivation of existing track
            ),
        ],
    ),
    # Test case 8: Testing track_buffer parameter - gap exceeds buffer
    generate_tracker_test_case(
        name="track_buffer with gap exceeding buffer",
        num_frames=20,
        tracker_params={
            "track_buffer": 3
        },  # Keep lost tracks for only 3 frames before removing
        trajectories=[
            # First trajectory: frames 1-5 (object present)
            TrajectorySpec(
                start_frame=1,
                stop_frame=5,
                start_bbox=[10.0, 10.0, 30.0, 30.0],
                velocity=[0, 0],
                expected_track_id=1,  # Should be tracked with ID 1
            ),
            # Gap: frames 6-10 (5 frames, object not detected)
            # With track_buffer=3, track is removed at frame 10 (=6+3+1, 3=track_buffer, +1 is decision pipeline delay)
            # Second trajectory: frames 11-19 (object reappears after long gap)
            TrajectorySpec(
                start_frame=11,
                stop_frame=19,
                start_bbox=[10.0, 10.0, 30.0, 30.0],  # Continues from expected position
                velocity=[2.0, 1.0],
                expected_track_id=2,  # Should get new ID 2
            ),
        ],
    ),
    # Test case 9: Testing match_thresh parameter - strict threshold
    generate_tracker_test_case(
        name="match_thresh strict (high threshold)",
        num_frames=20,
        tracker_params={"match_thresh": 0.5},  # Strict matching - requires high IoU
        trajectories=[
            # Object stationary for 10 frames to establish low-velocity model
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[10.0, 10.0, 60.0, 60.0],  # Larger bbox for higher tolerance
                velocity=[0, 0],  # no movement
                expected_track_id=1,
                active_immediately=True,
            ),
            # Object makes a moderate jump at frame 10
            # Kalman filter expects slow movement, actual jump causes moderate IoU
            # With strict match_thresh=0.6, match rejected → new track
            TrajectorySpec(
                start_frame=10,
                stop_frame=19,
                start_bbox=[20.0, 20.0, 70.0, 70.0],
                velocity=[1.0, 1.0],
                expected_track_id=2,  # New ID due to strict threshold
            ),
        ],
    ),
    # Test case 10: Testing match_thresh parameter - permissive threshold
    generate_tracker_test_case(
        name="match_thresh permissive (low threshold)",
        num_frames=20,
        tracker_params={"match_thresh": 0.9},  # Permissive matching - accepts lower IoU
        trajectories=[
            # Object stationary for 10 frames to establish low-velocity model
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[10.0, 10.0, 60.0, 60.0],  # Larger bbox for higher tolerance
                velocity=[0, 0],  # no movement
                expected_track_id=1,
                active_immediately=True,
            ),
            # Object makes a moderate jump at frame 10
            TrajectorySpec(
                start_frame=10,
                stop_frame=19,
                start_bbox=[20.0, 20.0, 70.0, 70.0],
                velocity=[0, 0],
                expected_track_id=1,  # Maintains ID 1 due to permissive threshold
                active_immediately=True,
            ),
        ],
    ),
    # Test case 11: Testing trail_depth parameter - trail tracking
    generate_tracker_test_case(
        name="trail_depth tracking",
        num_frames=15,
        tracker_params={
            "trail_depth": 5,  # Keep 5 frames of trail history
            "track_buffer": 3,  # Keep trails for 3 frames after object disappears
        },
        trajectories=[
            # Object moves in a straight line
            TrajectorySpec(
                start_frame=0,
                stop_frame=9,
                start_bbox=[10.0, 10.0, 30.0, 30.0],
                velocity=[3.0, 2.0],  # Move right 3px, down 2px per frame
                expected_track_id=1,
                active_immediately=True,
            ),
            # Second object with different trajectory
            TrajectorySpec(
                start_frame=5,
                stop_frame=14,
                start_bbox=[100.0, 100.0, 120.0, 120.0],
                velocity=[-2.0, 1.0],  # Move left 2px, down 1px per frame
                expected_track_id=2,
            ),
        ],
    ),
    # Test case 12: Testing trail_depth with object disappearing and reappearing (within track_buffer)
    generate_tracker_test_case(
        name="trail_depth with gap and trail persistence",
        num_frames=15,
        tracker_params={
            "trail_depth": 4,  # Keep 4 frames of trail history
            "track_buffer": 5,  # Keep trails for 5 frames after object disappears
        },
        trajectories=[
            # Object appears, then disappears for 2 frames, then reappears
            TrajectorySpec(
                start_frame=0,
                stop_frame=5,
                start_bbox=[10.0, 10.0, 30.0, 30.0],
                velocity=[2.0, 1.0],
                expected_track_id=1,
                active_immediately=True,
            ),
            # Same object reappears (within buffer, maintains ID)
            TrajectorySpec(
                start_frame=8,
                stop_frame=14,
                start_bbox=[
                    22.0,
                    16.0,
                    42.0,
                    36.0,
                ],  # Position after gap (continuation)
                velocity=[2.0, 1.0],
                expected_track_id=1,  # Should maintain ID 1
                active_immediately=True,  # Reactivated immediately
            ),
        ],
    ),
]


def test_object_tracker():
    """Test ObjectTracker with various scenarios."""
    for test_case in _test_cases:
        # print(f"\nRunning test case: {test_case.name}")
        run_tracker_test_case(test_case)
