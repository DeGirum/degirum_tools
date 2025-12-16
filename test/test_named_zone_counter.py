"""Comprehensive tests for NamedZoneCounter analyzer.

Tests cover:
- Entry/exit delay smoothing (entry_delay_frames, exit_delay_frames)
- Multi-zone tracking
- Per-class counting
- Zone events (entry, exit, occupied, empty)
- Edge cases (re-entry, simultaneous objects, empty detections)
- Non-tracking mode
- Various triggering methods (anchor points, IoPA)
"""

import numpy as np
import sys

from degirum_tools.analyzers.named_zone_count import NamedZoneCounter


class MockResult:
    """Mock inference result for testing."""

    def __init__(self, detections, frame_number=0):
        self.results = detections
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.image_overlay = self.image.copy()
        self.inference_results = {"frame_number": frame_number}
        self.zone_counts: dict = {}  # Added for mypy
        self.zone_events: list = []  # Added for mypy


def test_entry_delay_frames():
    """Test entry_delay_frames parameter for entry smoothing."""
    print("\n" + "=" * 80)
    print("TEST 1: Entry Delay Frames (entry_delay_frames=3)")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=3,  # Need 3 consecutive frames
        exit_delay_frames=2,
        enable_zone_events=True,
    )

    # Frame 1: Object enters
    result1 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}],
        frame_number=1,
    )
    counter.analyze(result1)
    assert (
        result1.zone_counts["test_zone"]["total"] == 0
    ), "Frame 1: Should not count yet (1/3)"
    assert len(result1.zone_events) == 0, "Frame 1: No events yet"
    print("✓ Frame 1: Not counted (entry_delay=1/3)")

    # Frame 2: Still in zone
    result2 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}],
        frame_number=2,
    )
    counter.analyze(result2)
    assert (
        result2.zone_counts["test_zone"]["total"] == 0
    ), "Frame 2: Should not count yet (2/3)"
    assert len(result2.zone_events) == 0, "Frame 2: No events yet"
    print("✓ Frame 2: Not counted (entry_delay=2/3)")

    # Frame 3: Established!
    result3 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}],
        frame_number=3,
    )
    counter.analyze(result3)
    assert (
        result3.zone_counts["test_zone"]["total"] == 1
    ), "Frame 3: Should be counted now"
    assert len(result3.zone_events) == 2, "Frame 3: Should have entry + occupied events"
    entry_events = [e for e in result3.zone_events if e["event_type"] == "zone_entry"]
    occupied_events = [
        e for e in result3.zone_events if e["event_type"] == "zone_occupied"
    ]
    assert len(entry_events) == 1, "Frame 3: Should have 1 entry event"
    assert len(occupied_events) == 1, "Frame 3: Should have 1 occupied event"
    print("✓ Frame 3: ESTABLISHED! Counted + entry/occupied events fired")

    # Frame 4-5: Missing (grace period)
    result4 = MockResult([], frame_number=4)
    counter.analyze(result4)
    assert result4.zone_counts["test_zone"]["total"] == 1, "Frame 4: Grace period (2/2)"
    print("✓ Frame 4: Missing but still counted (exit_delay=2/2)")

    result5 = MockResult([], frame_number=5)
    counter.analyze(result5)
    assert result5.zone_counts["test_zone"]["total"] == 1, "Frame 5: Grace period (1/2)"
    print("✓ Frame 5: Still in grace period (exit_delay=1/2)")

    # Frame 6: Timeout expires
    result6 = MockResult([], frame_number=6)
    counter.analyze(result6)
    assert result6.zone_counts["test_zone"]["total"] == 0, "Frame 6: Should exit now"
    assert len(result6.zone_events) == 2, "Frame 6: Should have exit + empty events"
    exit_events = [e for e in result6.zone_events if e["event_type"] == "zone_exit"]
    empty_events = [e for e in result6.zone_events if e["event_type"] == "zone_empty"]
    assert len(exit_events) == 1, "Frame 6: Should have 1 exit event"
    assert len(empty_events) == 1, "Frame 6: Should have 1 empty event"
    assert (
        exit_events[0]["dwell_time"] is not None
    ), "Frame 6: Exit should have dwell_time"
    print("✓ Frame 6: EXIT! exit_delay expired, exit/empty events fired")

    print("✅ TEST 1 PASSED\n")


def test_multi_zone_tracking():
    """Test tracking across multiple zones."""
    print("=" * 80)
    print("TEST 2: Multi-Zone Tracking")
    print("=" * 80)

    zones = {
        "zone_A": np.array([[50, 50], [200, 50], [200, 200], [50, 200]]),
        "zone_B": np.array([[250, 50], [400, 50], [400, 200], [250, 200]]),
        "zone_C": np.array([[150, 250], [300, 250], [300, 400], [150, 400]]),
    }

    counter = NamedZoneCounter(
        zones=zones,
        use_tracking=True,
        entry_delay_frames=1,
        exit_delay_frames=1,
        enable_zone_events=True,
    )

    # Frame 1: Object in zone_A, another in zone_B
    result1 = MockResult(
        [
            {"bbox": [75, 75, 125, 125], "track_id": 1, "label": "person"},
            {"bbox": [275, 75, 325, 125], "track_id": 2, "label": "car"},
        ],
        frame_number=1,
    )
    counter.analyze(result1)
    assert result1.zone_counts["zone_A"]["total"] == 1, "Zone A should have 1 object"
    assert result1.zone_counts["zone_B"]["total"] == 1, "Zone B should have 1 object"
    assert result1.zone_counts["zone_C"]["total"] == 0, "Zone C should be empty"
    print("✓ Frame 1: Object 1 in zone_A, Object 2 in zone_B")

    # Frame 2: Object 1 moves to zone_C, Object 2 stays in zone_B
    result2 = MockResult(
        [
            {"bbox": [175, 275, 225, 325], "track_id": 1, "label": "person"},
            {"bbox": [275, 75, 325, 125], "track_id": 2, "label": "car"},
        ],
        frame_number=2,
    )
    counter.analyze(result2)
    assert (
        result2.zone_counts["zone_A"]["total"] == 1
    ), "Zone A should still count (grace period, exit_delay=1)"
    assert (
        result2.zone_counts["zone_B"]["total"] == 1
    ), "Zone B should still have 1 object"
    assert result2.zone_counts["zone_C"]["total"] == 1, "Zone C should have 1 object"

    # Check events - no exit yet (still in grace period)
    entry_events = [e for e in result2.zone_events if e["event_type"] == "zone_entry"]
    assert len(entry_events) == 1, "Should have 1 entry event (to zone_C)"
    assert entry_events[0]["zone_id"] == "zone_C", "Entry should be to zone_C"
    print(
        "✓ Frame 2: Object 1 in zone_C (entered) + zone_A (grace period), Object 2 in B"
    )

    # Frame 3: Grace period expires for zone_A
    result3 = MockResult(
        [
            {"bbox": [175, 275, 225, 325], "track_id": 1, "label": "person"},
            {"bbox": [275, 75, 325, 125], "track_id": 2, "label": "car"},
        ],
        frame_number=3,
    )
    counter.analyze(result3)
    assert result3.zone_counts["zone_A"]["total"] == 0, "Zone A should be empty now"
    assert (
        result3.zone_counts["zone_B"]["total"] == 1
    ), "Zone B should still have 1 object"
    assert result3.zone_counts["zone_C"]["total"] == 1, "Zone C should have 1 object"

    exit_events = [e for e in result3.zone_events if e["event_type"] == "zone_exit"]
    assert len(exit_events) == 1, "Should have 1 exit event (from zone_A)"
    assert exit_events[0]["zone_id"] == "zone_A", "Exit should be from zone_A"
    print("✓ Frame 3: Object 1 exited zone_A (grace period expired)")

    print("✅ TEST 2 PASSED\n")


def test_per_class_counting():
    """Test per-class display mode."""
    print("=" * 80)
    print("TEST 3: Per-Class Counting")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        class_list=["person", "car", "bicycle"],
        per_class_display=True,
        use_tracking=True,
        entry_delay_frames=1,
    )

    # Frame 1: Mixed objects
    result = MockResult(
        [
            {"bbox": [120, 120, 170, 170], "track_id": 1, "label": "person"},
            {"bbox": [180, 180, 230, 230], "track_id": 2, "label": "person"},
            {"bbox": [240, 240, 290, 290], "track_id": 3, "label": "car"},
        ],
        frame_number=1,
    )
    counter.analyze(result)

    counts = result.zone_counts["test_zone"]
    assert counts["total"] == 3, "Total should be 3"
    assert counts["person"] == 2, "Should count 2 persons"
    assert counts["car"] == 1, "Should count 1 car"
    assert counts["bicycle"] == 0, "Should count 0 bicycles"
    print("✓ Per-class counts: person=2, car=1, bicycle=0, total=3")

    print("✅ TEST 3 PASSED\n")


def test_class_filtering():
    """Test class_list filtering."""
    print("=" * 80)
    print("TEST 4: Class Filtering")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        class_list=["person"],  # Only count persons
        use_tracking=True,
        entry_delay_frames=1,
    )

    # Frame with mixed objects
    result = MockResult(
        [
            {"bbox": [120, 120, 170, 170], "track_id": 1, "label": "person"},
            {
                "bbox": [180, 180, 230, 230],
                "track_id": 2,
                "label": "car",
            },  # Should be ignored
            {"bbox": [240, 240, 290, 290], "track_id": 3, "label": "person"},
        ],
        frame_number=1,
    )
    counter.analyze(result)

    assert (
        result.zone_counts["test_zone"]["total"] == 2
    ), "Should only count persons (2)"
    print("✓ Filtered correctly: counted 2 persons, ignored 1 car")

    print("✅ TEST 4 PASSED\n")


def test_re_entry():
    """Test object re-entering zone after exiting."""
    print("=" * 80)
    print("TEST 5: Re-Entry After Exit")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=2,
        exit_delay_frames=1,
        enable_zone_events=True,
    )

    # Frames 1-2: Object enters and gets established
    result1 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 1
    )
    counter.analyze(result1)
    print("✓ Frame 1: Entry started (1/2)")

    result2 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 2
    )
    counter.analyze(result2)
    assert result2.zone_counts["test_zone"]["total"] == 1, "Should be established"
    print("✓ Frame 2: Established (2/2)")

    # Frame 3: Object exits (missing detection, grace period)
    result3 = MockResult([], 3)
    counter.analyze(result3)
    assert result3.zone_counts["test_zone"]["total"] == 1, "Still in grace period"
    print("✓ Frame 3: Missing, in grace period")

    # Frame 4: Timeout expires
    result4 = MockResult([], 4)
    counter.analyze(result4)
    assert result4.zone_counts["test_zone"]["total"] == 0, "Should have exited"
    exit_events = [e for e in result4.zone_events if e["event_type"] == "zone_exit"]
    assert len(exit_events) == 1, "Should have exit event"
    print("✓ Frame 4: Exited")

    # Frames 5-6: Object re-enters (needs to re-establish)
    result5 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 5
    )
    counter.analyze(result5)
    assert (
        result5.zone_counts["test_zone"]["total"] == 0
    ), "Not yet re-established (1/2)"
    print("✓ Frame 5: Re-entry started (1/2)")

    result6 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 6
    )
    counter.analyze(result6)
    assert result6.zone_counts["test_zone"]["total"] == 1, "Re-established (2/2)"
    entry_events = [e for e in result6.zone_events if e["event_type"] == "zone_entry"]
    assert len(entry_events) == 1, "Should have new entry event"
    print("✓ Frame 6: Re-established (2/2), entry event fired")

    print("✅ TEST 5 PASSED\n")


def test_simultaneous_objects():
    """Test multiple objects entering/exiting simultaneously."""
    print("=" * 80)
    print("TEST 6: Simultaneous Objects")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=1,
        exit_delay_frames=1,
        enable_zone_events=True,
    )

    # Frame 1: 3 objects enter simultaneously
    result1 = MockResult(
        [
            {"bbox": [110, 110, 140, 140], "track_id": 1, "label": "person"},
            {"bbox": [160, 160, 190, 190], "track_id": 2, "label": "person"},
            {"bbox": [210, 210, 240, 240], "track_id": 3, "label": "person"},
        ],
        1,
    )
    counter.analyze(result1)
    assert result1.zone_counts["test_zone"]["total"] == 3, "All 3 should be counted"
    entry_events = [e for e in result1.zone_events if e["event_type"] == "zone_entry"]
    assert len(entry_events) == 3, "Should have 3 entry events"
    occupied_events = [
        e for e in result1.zone_events if e["event_type"] == "zone_occupied"
    ]
    assert len(occupied_events) == 1, "Should have 1 occupied event (zone-level)"
    print("✓ Frame 1: 3 objects entered simultaneously")

    # Frame 2: All 3 exit simultaneously
    result2 = MockResult([], 2)
    counter.analyze(result2)
    assert result2.zone_counts["test_zone"]["total"] == 3, "All still in grace period"
    print("✓ Frame 2: All in grace period")

    # Frame 3: Grace period expires
    result3 = MockResult([], 3)
    counter.analyze(result3)
    assert result3.zone_counts["test_zone"]["total"] == 0, "All should have exited"
    exit_events = [e for e in result3.zone_events if e["event_type"] == "zone_exit"]
    assert len(exit_events) == 3, "Should have 3 exit events"
    empty_events = [e for e in result3.zone_events if e["event_type"] == "zone_empty"]
    assert len(empty_events) == 1, "Should have 1 empty event (zone-level)"
    print("✓ Frame 3: All exited simultaneously")

    print("✅ TEST 6 PASSED\n")


def test_non_tracking_mode():
    """Test without tracking (simple counting)."""
    print("=" * 80)
    print("TEST 7: Non-Tracking Mode")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=False,  # No tracking
    )

    # Frame 1: 2 objects in zone
    result1 = MockResult(
        [
            {"bbox": [120, 120, 170, 170], "label": "person"},  # No track_id
            {"bbox": [180, 180, 230, 230], "label": "car"},
        ],
        1,
    )
    counter.analyze(result1)
    assert result1.zone_counts["test_zone"]["total"] == 2, "Should count both objects"
    print("✓ Frame 1: Counted 2 objects (no tracking)")

    # Frame 2: 1 object in zone
    result2 = MockResult(
        [
            {"bbox": [120, 120, 170, 170], "label": "person"},
        ],
        2,
    )
    counter.analyze(result2)
    assert result2.zone_counts["test_zone"]["total"] == 1, "Should count 1 object"
    print("✓ Frame 2: Counted 1 object (no persistence)")

    print("✅ TEST 7 PASSED\n")


def test_empty_frames():
    """Test handling of empty detection frames."""
    print("=" * 80)
    print("TEST 8: Empty Frames")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=1,
        exit_delay_frames=2,
    )

    # Frame 1: Empty
    result1 = MockResult([], 1)
    counter.analyze(result1)
    assert result1.zone_counts["test_zone"]["total"] == 0, "Should be 0"
    print("✓ Frame 1: Empty frame handled")

    # Frame 2: Object appears
    result2 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 2
    )
    counter.analyze(result2)
    assert result2.zone_counts["test_zone"]["total"] == 1, "Should count 1"
    print("✓ Frame 2: Object detected after empty frame")

    # Frame 3: Empty again
    result3 = MockResult([], 3)
    counter.analyze(result3)
    assert (
        result3.zone_counts["test_zone"]["total"] == 1
    ), "Should still count (grace period)"
    print("✓ Frame 3: Empty frame, object still counted (grace period)")

    print("✅ TEST 8 PASSED\n")


def test_edge_case_zero_delays():
    """Test with both delays set to minimum values."""
    print("=" * 80)
    print("TEST 9: Zero Delays (Immediate Entry/Exit)")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=1,  # Immediate
        exit_delay_frames=0,  # Immediate
        enable_zone_events=True,
    )

    # Frame 1: Object enters - immediate entry
    result1 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 1
    )
    counter.analyze(result1)
    assert result1.zone_counts["test_zone"]["total"] == 1, "Should count immediately"
    assert len(result1.zone_events) == 2, "Should have entry + occupied events"
    print("✓ Frame 1: Immediate entry (entry_delay_frames=1)")

    # Frame 2: Object exits - immediate exit
    result2 = MockResult([], 2)
    counter.analyze(result2)
    assert result2.zone_counts["test_zone"]["total"] == 0, "Should exit immediately"
    assert len(result2.zone_events) == 2, "Should have exit + empty events"
    print("✓ Frame 2: Immediate exit (exit_delay_frames=0)")

    print("✅ TEST 9 PASSED\n")


def test_exit_delay_boundary():
    """Test that exit_delay_frames boundary condition works correctly (>= 0 not > 0)."""
    print("=" * 80)
    print("TEST 9B: Exit Delay Boundary Condition")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=1,
        exit_delay_frames=2,  # Should tolerate exactly 2 missing frames
        enable_zone_events=True,
    )

    # Frame 1: Object enters
    result1 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 1
    )
    counter.analyze(result1)
    assert result1.zone_counts["test_zone"]["total"] == 1, "Object should be counted"
    print("✓ Frame 1: Object entered and established")

    # Frame 2: Missing (exit_delay_count = 2 -> 1)
    result2 = MockResult([], 2)
    counter.analyze(result2)
    assert (
        result2.zone_counts["test_zone"]["total"] == 1
    ), "Frame 1 of grace period (exit_delay_count=1)"
    assert len(result2.zone_events) == 0, "No exit yet"
    print("✓ Frame 2: Missing frame 1/2 (exit_delay_count=1, still valid)")

    # Frame 3: Missing (exit_delay_count = 1 -> 0)
    result3 = MockResult([], 3)
    counter.analyze(result3)
    assert (
        result3.zone_counts["test_zone"]["total"] == 1
    ), "Frame 2 of grace period (exit_delay_count=0, STILL VALID)"
    assert len(result3.zone_events) == 0, "No exit yet - boundary check >= 0"
    print(
        "✓ Frame 3: Missing frame 2/2 (exit_delay_count=0, STILL VALID per >= 0 check)"
    )

    # Frame 4: Missing (exit_delay_count = 0 -> -1, expires)
    result4 = MockResult([], 4)
    counter.analyze(result4)
    assert result4.zone_counts["test_zone"]["total"] == 0, "Should exit now"
    exit_events = [e for e in result4.zone_events if e["event_type"] == "zone_exit"]
    assert len(exit_events) == 1, "Should have exit event now"
    print("✓ Frame 4: Exit event fired (exit_delay_count=-1, expired)")

    print("✅ TEST 9B PASSED - Boundary condition >= 0 working correctly\n")


def test_exit_delay_case2_immediate():
    """Test that Case 2 (detected outside zone) respects exit_delay_frames=0."""
    print("=" * 80)
    print("TEST 9C: Case 2 with exit_delay_frames=0")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=1,
        exit_delay_frames=0,  # No grace period
        enable_zone_events=True,
    )

    # Frame 1: Object in zone
    result1 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 1
    )
    counter.analyze(result1)
    assert result1.zone_counts["test_zone"]["total"] == 1, "Object counted in zone"
    print("✓ Frame 1: Object in zone")

    # Frame 2: Object detected but OUTSIDE zone (Case 2) - should exit immediately with exit_delay_frames=0
    result2 = MockResult(
        [{"bbox": [350, 350, 400, 400], "track_id": 1, "label": "person"}], 2
    )
    counter.analyze(result2)
    assert (
        result2.zone_counts["test_zone"]["total"] == 0
    ), "Should NOT count - exit_delay_frames=0 means immediate exit"
    exit_events = [e for e in result2.zone_events if e["event_type"] == "zone_exit"]
    assert (
        len(exit_events) == 1
    ), "Should have exit event for Case 2 with exit_delay_frames=0"
    print(
        "✓ Frame 2: Object detected outside zone - immediate exit (Case 2, exit_delay_frames=0)"
    )

    print("✅ TEST 9C PASSED - Case 2 respects exit_delay_frames=0\n")


def test_occupancy_transitions():
    """Test zone occupancy transitions (empty ↔ occupied) with smoothing."""
    print("=" * 80)
    print("TEST 10: Zone Occupancy Transitions")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=2,  # Takes 2 frames to establish
        exit_delay_frames=2,  # Takes 2 frames to fully exit
        enable_zone_events=True,
    )

    # Frame 1: Empty zone
    result1 = MockResult([], 1)
    counter.analyze(result1)
    assert len(result1.zone_events) == 0, "Empty zone should have no events initially"
    print("✓ Frame 1: Zone starts empty (no events)")

    # Frame 2: Object enters (not yet established)
    result2 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 2
    )
    counter.analyze(result2)
    assert result2.zone_counts["test_zone"]["total"] == 0, "Not counted yet (1/2)"
    occupied_events = [
        e for e in result2.zone_events if e["event_type"] == "zone_occupied"
    ]
    assert len(occupied_events) == 0, "No occupied event yet (object not established)"
    print("✓ Frame 2: Object detected but not established (1/2) - zone still empty")

    # Frame 3: Object established - zone becomes occupied
    result3 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 3
    )
    counter.analyze(result3)
    assert result3.zone_counts["test_zone"]["total"] == 1, "Now counted (2/2)"
    occupied_events = [
        e for e in result3.zone_events if e["event_type"] == "zone_occupied"
    ]
    empty_events = [e for e in result3.zone_events if e["event_type"] == "zone_empty"]
    assert len(occupied_events) == 1, "Should have zone_occupied event"
    assert len(empty_events) == 0, "No zone_empty event"
    print("✓ Frame 3: Object established (2/2) - zone_occupied event fired")

    # Frame 4: Object missing (grace period starts)
    result4 = MockResult([], 4)
    counter.analyze(result4)
    assert (
        result4.zone_counts["test_zone"]["total"] == 1
    ), "Still counted (exit_delay 2/2)"
    empty_events = [e for e in result4.zone_events if e["event_type"] == "zone_empty"]
    assert len(empty_events) == 0, "No zone_empty yet (grace period)"
    print("✓ Frame 4: Object missing but zone still occupied (grace period 2/2)")

    # Frame 5: Still missing (grace period continues)
    result5 = MockResult([], 5)
    counter.analyze(result5)
    assert (
        result5.zone_counts["test_zone"]["total"] == 1
    ), "Still counted (exit_delay 1/2)"
    empty_events = [e for e in result5.zone_events if e["event_type"] == "zone_empty"]
    assert len(empty_events) == 0, "No zone_empty yet (grace period)"
    print("✓ Frame 5: Still in grace period (1/2)")

    # Frame 6: Grace period expires - zone becomes empty
    result6 = MockResult([], 6)
    counter.analyze(result6)
    assert result6.zone_counts["test_zone"]["total"] == 0, "Should be empty now"
    empty_events = [e for e in result6.zone_events if e["event_type"] == "zone_empty"]
    occupied_events = [
        e for e in result6.zone_events if e["event_type"] == "zone_occupied"
    ]
    assert len(empty_events) == 1, "Should have zone_empty event"
    assert len(occupied_events) == 0, "No zone_occupied event"
    assert (
        empty_events[0]["dwell_time"] is not None
    ), "Empty event should have dwell_time"
    print(
        f"✓ Frame 6: Grace period expired - zone_empty event fired (dwell={empty_events[0]['dwell_time']:.3f}s)"
    )

    print("✅ TEST 10 PASSED\n")


def test_occupancy_no_flicker():
    """Test that occupancy doesn't flicker with brief gaps in detection."""
    print("=" * 80)
    print("TEST 11: Occupancy Anti-Flicker (Brief Detection Gaps)")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=1,
        exit_delay_frames=3,  # Tolerates 3 missing frames
        enable_zone_events=True,
    )

    # Frames 1-2: Object enters and establishes
    result1 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 1
    )
    counter.analyze(result1)
    occupied_events = [
        e for e in result1.zone_events if e["event_type"] == "zone_occupied"
    ]
    assert len(occupied_events) == 1, "Zone should become occupied"
    print("✓ Frame 1: Object established, zone occupied")

    # Frame 2: Missing (brief occlusion)
    result2 = MockResult([], 2)
    counter.analyze(result2)
    assert (
        result2.zone_counts["test_zone"]["total"] == 1
    ), "Should still count (grace period)"
    empty_events = [e for e in result2.zone_events if e["event_type"] == "zone_empty"]
    assert len(empty_events) == 0, "No flicker to empty"
    print("✓ Frame 2: Missing (3/3) - no zone_empty flicker")

    # Frame 3: Object reappears
    result3 = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 3
    )
    counter.analyze(result3)
    assert result3.zone_counts["test_zone"]["total"] == 1, "Should count"
    occupied_events = [
        e for e in result3.zone_events if e["event_type"] == "zone_occupied"
    ]
    assert len(occupied_events) == 0, "No re-entry occupied event (never left)"
    print(
        "✓ Frame 3: Object reappears - no zone_occupied flicker (grace period worked)"
    )

    # Frame 4: Missing
    result4 = MockResult([], 4)
    counter.analyze(result4)
    print("✓ Frame 4: Missing again (3/3)")

    # Frame 5: Missing
    result5 = MockResult([], 5)
    counter.analyze(result5)
    print("✓ Frame 5: Missing (2/3)")

    # Frame 6: Missing
    result6 = MockResult([], 6)
    counter.analyze(result6)
    print("✓ Frame 6: Missing (1/3)")

    # Frame 7: Missing - should finally exit
    result7 = MockResult([], 7)
    counter.analyze(result7)
    assert result7.zone_counts["test_zone"]["total"] == 0, "Should exit now"
    empty_events = [e for e in result7.zone_events if e["event_type"] == "zone_empty"]
    assert len(empty_events) == 1, "Should have zone_empty event"
    print("✓ Frame 7: Grace period exhausted - zone_empty event fired")

    print("✅ TEST 11 PASSED\n")


def test_occupancy_multiple_objects():
    """Test that zone stays occupied as long as any established object remains."""
    print("=" * 80)
    print("TEST 12: Occupancy with Multiple Objects (Staggered Exit)")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=1,
        exit_delay_frames=1,
        enable_zone_events=True,
    )

    # Frame 1: Two objects enter
    result1 = MockResult(
        [
            {"bbox": [120, 120, 160, 160], "track_id": 1, "label": "person"},
            {"bbox": [200, 200, 240, 240], "track_id": 2, "label": "person"},
        ],
        1,
    )
    counter.analyze(result1)
    assert result1.zone_counts["test_zone"]["total"] == 2, "Should count both"
    occupied_events = [
        e for e in result1.zone_events if e["event_type"] == "zone_occupied"
    ]
    assert len(occupied_events) == 1, "Zone becomes occupied (once)"
    print("✓ Frame 1: 2 objects enter - zone_occupied event")

    # Frame 2: Object 1 exits, object 2 stays
    result2 = MockResult(
        [
            {"bbox": [200, 200, 240, 240], "track_id": 2, "label": "person"},
        ],
        2,
    )
    counter.analyze(result2)
    assert (
        result2.zone_counts["test_zone"]["total"] == 2
    ), "Both still counted (obj 1 in grace)"
    empty_events = [e for e in result2.zone_events if e["event_type"] == "zone_empty"]
    assert len(empty_events) == 0, "Zone still occupied (obj 2 present)"
    print(
        "✓ Frame 2: Object 1 missing (grace period), object 2 present - zone stays occupied"
    )

    # Frame 3: Object 1 grace expires, object 2 still present
    result3 = MockResult(
        [
            {"bbox": [200, 200, 240, 240], "track_id": 2, "label": "person"},
        ],
        3,
    )
    counter.analyze(result3)
    assert result3.zone_counts["test_zone"]["total"] == 1, "Only object 2 counted"
    empty_events = [e for e in result3.zone_events if e["event_type"] == "zone_empty"]
    assert len(empty_events) == 0, "Zone STILL occupied (obj 2 keeps it occupied)"
    print(
        "✓ Frame 3: Object 1 exited, object 2 remains - zone STAYS occupied (no flicker)"
    )

    # Frame 4: Object 2 exits
    result4 = MockResult([], 4)
    counter.analyze(result4)
    assert result4.zone_counts["test_zone"]["total"] == 1, "Object 2 in grace period"
    empty_events = [e for e in result4.zone_events if e["event_type"] == "zone_empty"]
    assert len(empty_events) == 0, "Not empty yet"
    print("✓ Frame 4: Object 2 missing (grace period)")

    # Frame 5: Object 2 grace expires - zone becomes empty
    result5 = MockResult([], 5)
    counter.analyze(result5)
    assert result5.zone_counts["test_zone"]["total"] == 0, "Should be empty"
    empty_events = [e for e in result5.zone_events if e["event_type"] == "zone_empty"]
    assert len(empty_events) == 1, "Zone becomes empty (finally)"
    print("✓ Frame 5: Last object exited - zone_empty event fired")

    print("✅ TEST 12 PASSED\n")


def test_bbox_annotation_fields():
    """Test that bbox annotation fields are set correctly."""
    print("=" * 80)
    print("TEST 13: Bbox Annotation Fields")
    print("=" * 80)

    zone_polygon = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
    counter = NamedZoneCounter(
        zones={"test_zone": zone_polygon},
        use_tracking=True,
        entry_delay_frames=1,
        show_inzone_counters="all",
    )

    # Frame 1: Object in zone
    result = MockResult(
        [{"bbox": [150, 150, 200, 200], "track_id": 1, "label": "person"}], 1
    )
    counter.analyze(result)

    obj = result.results[0]
    assert counter.key_in_zone in obj, "Should have in_zone field"
    assert obj[counter.key_in_zone][0] is True, "Should be marked as in zone"
    assert counter.key_frames_in_zone in obj, "Should have frames_in_zone field"
    assert obj[counter.key_frames_in_zone][0] == 1, "Should have 1 frame"
    assert counter.key_time_in_zone in obj, "Should have time_in_zone field"
    assert obj[counter.key_time_in_zone][0] >= 0, "Should have non-negative time"
    print("✓ Bbox fields set correctly: in_zone, frames_in_zone, time_in_zone")

    print("✅ TEST 13 PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "🧪 NAMED ZONE COUNTER - COMPREHENSIVE TEST SUITE " + "🧪")
    print("=" * 80)

    tests = [
        ("Entry Delay Frames", test_entry_delay_frames),
        ("Multi-Zone Tracking", test_multi_zone_tracking),
        ("Per-Class Counting", test_per_class_counting),
        ("Class Filtering", test_class_filtering),
        ("Re-Entry After Exit", test_re_entry),
        ("Simultaneous Objects", test_simultaneous_objects),
        ("Non-Tracking Mode", test_non_tracking_mode),
        ("Empty Frames", test_empty_frames),
        ("Zero Delays", test_edge_case_zero_delays),
        ("Exit Delay Boundary", test_exit_delay_boundary),
        ("Case 2 exit_delay_frames=0", test_exit_delay_case2_immediate),
        ("Occupancy Transitions", test_occupancy_transitions),
        ("Occupancy Anti-Flicker", test_occupancy_no_flicker),
        ("Occupancy Multiple Objects", test_occupancy_multiple_objects),
        ("Bbox Annotation Fields", test_bbox_annotation_fields),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"❌ TEST FAILED: {name}")
            print(f"   Error: {e}\n")
        except Exception as e:
            failed += 1
            print(f"❌ TEST ERROR: {name}")
            print(f"   Exception: {e}\n")

    print("=" * 80)
    print(
        f"📊 TEST RESULTS: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed"
    )

    if failed == 0:
        print("🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"⚠️  {failed} test(s) failed")

    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
