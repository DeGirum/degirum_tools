#
# zone_count.py: zone object counting with named zone support and clean architecture
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements ZoneCounter with separation of geometry and state concerns
#

"""
Named Zone Counter Analyzer Module Overview
============================================

This module provides an enhanced zone counting analyzer (`ZoneCounter`) with named zones
and clean separation of concerns. It builds upon the concepts from `ZoneCounter` but with
improved architecture and a more user-friendly API.

Key Features:
    - **Named Zones**: Use descriptive names instead of numeric indices
    - **Clean Architecture**: Separation of geometry (spatial) and state (temporal) logic
    - **Zone Events**: Track entry, exit, occupied, and empty events per zone
    - **All ZoneCounter Features**: Supports all triggering methods, tracking, timeouts, etc.
    - **Better Maintainability**: Smaller, focused components that are easier to test and extend

Architecture:
    The implementation separates concerns into focused components:

    - **_ZoneGeometry**: Handles spatial logic - polygon shapes, masks, and triggering
    - **_ZoneState**: Handles temporal logic - object tracking, timeouts, occupancy state
    - **ZoneCounter**: Orchestrates the components and generates events

Typical Usage:
    ```python
    from degirum_tools.analyzers import ZoneCounter

    counter = ZoneCounter(
        zones={
            "entrance": entrance_polygon,
            "parking_spot_1": spot1_polygon,
            "exit_area": exit_polygon,
        },
        use_tracking=True,
        timeout_frames=5,
        entry_delay_frames=3,
        enable_zone_events=True,
    )

    model.attach_analyzers(counter)
    result = model(frame)

    # Access named zone counts
    print(result.zone_counts)  # {"entrance": {...}, "parking_spot_1": {...}, ...}

    # Access zone events
    for event in result.zone_events:
        print(f"{event['event_type']} in {event['zone_id']} at {event['timestamp']}")
    ```

Zone Events:
    When `enable_zone_events=True`, the analyzer generates four types of events:

    - **zone_entry**: Track first enters a zone
    - **zone_exit**: Track exits a zone (after timeout)
    - **zone_occupied**: Zone transitions from empty to occupied
    - **zone_empty**: Zone transitions from occupied to empty

    Event structure:
    ```python
    {
        "event_type": str,              # Event type
        "zone_index": int,              # Numeric zone index (0-based)
        "zone_id": str,                 # Zone name/ID
        "timestamp": float,             # Unix timestamp
        "track_id": int | None,         # Track ID (entry/exit) or None (occupied/empty)
        "object_label": str | None,     # Object class (entry/exit) or None (occupied/empty)
        "dwell_time": float | None,     # Duration: in zone (exit), empty/occupied (transitions), None (entry)
        "frame_index": int | None,      # Frame index (if available)
    }
    ```

Integration Notes:
    - Requires detection results with bounding boxes
    - Zone events require `use_tracking=True`
    - Compatible with ObjectTracker analyzer upstream
    - Results structure matches ZoneCounter for easy migration

Configuration Options:
    - `zones`: Dictionary mapping zone names to polygons
    - `class_list`: Optional list of class labels to count
    - `triggering_position`: Anchor point or IoPA-based triggering
    - `timeout_frames`: Grace period for flickering/occlusion (exit smoothing)
    - `entry_delay_frames`: Consecutive frames required to establish presence (entry smoothing)
    - `enable_zone_events`: Generate zone-level events
    - `show_overlay`: Visual annotations
    - `per_class_display`: Show per-class counts

Smoothing Parameters:
    The analyzer provides symmetric smoothing for both entries and exits:

    - **timeout_frames** (exit smoothing): Objects must be absent for N consecutive frames
      before triggering exit events. Prevents false exits from brief occlusions or
      flickering detections. Default: 0 (immediate exit). Requires `use_tracking=True` if > 0.

    - **entry_delay_frames** (entry smoothing): Objects must be detected for N consecutive
      frames before triggering entry events. Prevents false entries from spurious or
      transient detections. Default: 1 (immediate entry). Requires `use_tracking=True` if > 1.
"""

import numpy as np
import cv2
import time
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from ..tools import (
    put_text,
    color_complement,
    deduce_text_color,
    rgb_to_bgr,
    CornerPosition,
    AnchorPoint,
    get_anchor_coordinates,
    xyxy2xywh,
    xywh2xyxy,
    tlbr2allcorners,
)
from .result_analyzer_base import ResultAnalyzerBase

__all__ = ["ZoneCounter", "NamedZoneCounter"]


@dataclass
class _ObjectState:
    """Tracks the state of an object within a zone.

    Attributes:
        exit_delay_count: Frames remaining before object is considered exited
        entry_delay_count: Frames accumulated to establish presence (for entry smoothing)
        object_label: Class label of the tracked object
        presence_count: Number of frames object has been in the zone
        entering_time: Timestamp when object first entered the zone
        is_established: Whether object presence is confirmed (entry delay threshold met)
    """

    exit_delay_count: int
    entry_delay_count: int
    object_label: str
    presence_count: int
    entering_time: float
    is_established: bool = False


class _ZoneGeometry:
    """Handles spatial geometry and zone triggering logic.

    This class is responsible for determining whether bounding boxes are inside
    a polygon zone based on geometric criteria (anchor points or IoPA).
    It is stateless and has no memory of previous frames.

    Attributes:
        polygon: Polygon vertices as (N,2) array
        mask: Binary mask for point-in-polygon tests
        polygon_area: Area of the polygon
        triggering_position: Anchor points for triggering (or None for IoPA)
        bounding_box_scale: Scale factor for bounding boxes
        iopa_threshold: IoPA threshold for zone membership
    """

    def __init__(
        self,
        polygon: np.ndarray,
        frame_resolution_wh: Tuple[int, int],
        triggering_position: Optional[Union[List[AnchorPoint], AnchorPoint]],
        bounding_box_scale: float = 1.0,
        iopa_threshold: float = 0.0,
    ):
        """Initialize zone geometry.

        Args:
            polygon: Polygon vertices
            frame_resolution_wh: Frame (width, height)
            triggering_position: Anchor point(s) or None for IoPA
            bounding_box_scale: Scale factor for bboxes (default 1.0)
            iopa_threshold: IoPA threshold (default 0.0)
        """
        self.width, self.height = frame_resolution_wh
        self.triggering_position = (
            triggering_position
            if triggering_position is None or isinstance(triggering_position, list)
            else [triggering_position]
        )
        self.bounding_box_scale = bounding_box_scale
        self.iopa_threshold = iopa_threshold

        # Create polygon mask and compute area
        self.polygon = polygon.astype(np.float32)
        self.mask = np.zeros((self.height + 1, self.width + 1))
        cv2.fillPoly(self.mask, [cv2.Mat(polygon.astype(int))], color=[1])
        self.polygon_area = cv2.contourArea(polygon.reshape(1, -1, 2))

    def trigger(self, bboxes: np.ndarray) -> np.ndarray:
        """Determine which bounding boxes are inside the zone.

        Uses either anchor point testing or IoPA (Intersection over Polygon Area)
        depending on configuration.

        Args:
            bboxes: Array of shape (N, 4) with boxes in (x0, y0, x1, y1) format

        Returns:
            Boolean array of length N indicating zone membership
        """
        if bboxes.size == 0:
            return np.array([], dtype=bool)

        # Clip bboxes to frame boundaries
        bboxes = bboxes.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, self.width)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, self.height)

        # Scale down bounding boxes if needed
        if self.bounding_box_scale < 1.0:
            bboxes_xywh = xyxy2xywh(bboxes).astype(float)
            bboxes_xywh[:, [2, 3]] *= self.bounding_box_scale
            bboxes = xywh2xyxy(bboxes_xywh)

        # Choose triggering method
        if self.triggering_position is None:
            # IoPA-based triggering
            return self._trigger_iopa(bboxes)
        else:
            # Anchor point-based triggering
            return self._trigger_anchor(bboxes)

    def _trigger_iopa(self, bboxes: np.ndarray) -> np.ndarray:
        """Trigger using Intersection over Polygon Area."""
        iopa = np.zeros(len(bboxes), dtype=float)
        bboxes_corners = tlbr2allcorners(bboxes).astype(np.float32)

        for i in range(len(bboxes_corners)):
            intersection_area = cv2.intersectConvexConvex(
                bboxes_corners[i].reshape(-1, 2), self.polygon
            )[0]
            iopa[i] = intersection_area / self.polygon_area

        return iopa > self.iopa_threshold

    def _trigger_anchor(self, bboxes: np.ndarray) -> np.ndarray:
        """Trigger using anchor points."""
        assert self.triggering_position is not None  # Type guard for mypy
        clipped_anchors = np.array(
            [
                np.ceil(get_anchor_coordinates(xyxy=bboxes, anchor=anchor)).astype(int)
                for anchor in self.triggering_position
            ]
        )

        # Check if any anchor point is inside the zone
        return np.logical_or.reduce(
            self.mask[clipped_anchors[:, :, 1], clipped_anchors[:, :, 0]]
        )


class _ZoneState:
    """Manages temporal state for a single zone.

    This class tracks object presence over time, handles timeouts for flickering,
    and maintains zone-level occupancy state for event generation.

    Attributes:
        object_states: Dictionary mapping track_id to _ObjectState
        was_occupied: Previous frame's occupancy state
    """

    def __init__(self, timeout_frames: int, entry_delay_frames: int):
        """Initialize zone state.

        Args:
            timeout_frames: Number of frames to tolerate absence before exit
            entry_delay_frames: Number of frames required to establish presence
        """
        self._timeout_frames = timeout_frames
        self._entry_delay_frames = entry_delay_frames
        self._object_states: Dict[int, _ObjectState] = {}
        self._was_occupied: bool = False
        self._state_change_time: Optional[float] = None  # Track last transition time

    def add_track(
        self, track_id: int, label: str, timestamp: float
    ) -> Tuple[bool, bool]:
        """Add or update a tracked object in the zone.

        Args:
            track_id: Unique object identifier
            label: Class label of the object
            timestamp: Current frame timestamp

        Returns:
            Tuple of (entry_event, is_fresh_entry):
                - entry_event: True if object just became established (entry delay threshold met)
                - is_fresh_entry: True if this is a brand new track (not a re-entry)
        """
        if track_id in self._object_states:
            # Object already tracked
            obj_state = self._object_states[track_id]

            # Check if this is a re-entry (object was in timeout/exit period)
            is_reentry = obj_state.exit_delay_count < self._timeout_frames

            # Reset exit delay
            obj_state.exit_delay_count = self._timeout_frames

            if is_reentry:
                # Object left and came back - reset presence tracking
                obj_state.presence_count = 1
                obj_state.entering_time = timestamp
                obj_state.entry_delay_count = 1
                obj_state.is_established = self._entry_delay_frames == 1
                return (
                    self._entry_delay_frames == 1
                ), False  # Entry event if immediate
            else:
                # Continuous presence - increment counter
                obj_state.presence_count += 1

                # Check if object just became established
                if not obj_state.is_established:
                    obj_state.entry_delay_count += 1
                    if obj_state.entry_delay_count >= self._entry_delay_frames:
                        obj_state.is_established = True
                        return True, False  # Entry event when entry delay threshold met

                return False, False  # No entry event for already-established tracks
        else:
            # New object entering zone for the first time
            self._object_states[track_id] = _ObjectState(
                exit_delay_count=self._timeout_frames,
                entry_delay_count=1,  # Start counting entry delay
                object_label=label,
                presence_count=1,
                entering_time=timestamp,
                is_established=(
                    self._entry_delay_frames == 1
                ),  # Immediately established if threshold is exactly 1 (immediate counting)
            )
            # Generate entry event only if immediately established
            return (self._entry_delay_frames == 1), True

    def update_exit_delay(self, track_id: int) -> bool:
        """Decrement exit delay for a track not currently detected.

        Args:
            track_id: Track ID to update

        Returns:
            True if track is still valid, False if exit delay expired
        """
        if track_id not in self._object_states:
            return False

        obj_state = self._object_states[track_id]
        obj_state.presence_count += 1
        obj_state.exit_delay_count -= 1

        return obj_state.exit_delay_count >= 0

    def remove_track(self, track_id: int) -> Optional[_ObjectState]:
        """Remove a track from the zone.

        Args:
            track_id: Track ID to remove

        Returns:
            The object state if it existed, None otherwise
        """
        return self._object_states.pop(track_id, None)

    def get_state(self, track_id: int) -> Optional[_ObjectState]:
        """Get the state for a track.

        Args:
            track_id: Track ID to query

        Returns:
            Object state if exists, None otherwise
        """
        return self._object_states.get(track_id)

    def get_tracked_ids(self) -> set:
        """Get all currently tracked IDs in this zone."""
        return set(self._object_states.keys())

    def should_exit(self, track_id: int) -> bool:
        """Check if a track should generate an exit event.

        Args:
            track_id: Track ID to check

        Returns:
            True if exit delay has expired (ready for exit)
        """
        obj_state = self._object_states.get(track_id)
        return obj_state is not None and obj_state.exit_delay_count == 0

    def update_occupancy(
        self, current_count: int, timestamp: float
    ) -> Tuple[bool, bool, Optional[float]]:
        """Update occupancy state and detect transitions.

        Args:
            current_count: Current number of objects in zone
            timestamp: Current timestamp

        Returns:
            Tuple of (occupied_event, empty_event, state_duration)
            state_duration is the time spent in the previous state (if transition occurred)
        """
        is_occupied = current_count > 0

        # Detect transitions
        occupied_event = not self._was_occupied and is_occupied
        empty_event = self._was_occupied and not is_occupied

        # Calculate duration of previous state if transitioning
        state_duration = None
        if (occupied_event or empty_event) and self._state_change_time is not None:
            state_duration = timestamp - self._state_change_time

        # Update state for next frame
        if occupied_event or empty_event:
            self._state_change_time = timestamp
        elif self._state_change_time is None:
            # First frame - initialize timestamp
            self._state_change_time = timestamp

        self._was_occupied = is_occupied

        return occupied_event, empty_event, state_duration


class ZoneCounter(ResultAnalyzerBase):
    """Analyzer that counts objects inside user-defined polygonal zones.

    This analyzer integrates with PySDK inference results to determine whether detected or
    tracked objects lie within user-defined polygon zones. It supports per-class counting,
    object tracking with separate entry/exit delays, zone events, and interactive editing.

    Backward compatible with old ZoneCounter API (list of zones) while supporting new
    dict-based named zones interface.

    Attributes:
        key_in_zone: Key for zone presence flags in result objects
        key_frames_in_zone: Key for frame counts in result objects
        key_time_in_zone: Key for time-in-zone in result objects
        key_zone_events: Key for zone events list in result objects
    """

    key_in_zone = "in_zone"
    key_frames_in_zone = "frames_in_zone"
    key_time_in_zone = "time_in_zone"
    key_zone_events = "zone_events"

    def __init__(
        self,
        zones: Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray, None] = None,
        *,
        count_polygons: Union[
            List[np.ndarray], np.ndarray, None
        ] = None,  # Old API alias
        class_list: Optional[List[str]] = None,
        per_class_display: bool = False,
        triggering_position: Optional[
            Union[List[AnchorPoint], AnchorPoint]
        ] = AnchorPoint.BOTTOM_CENTER,
        bounding_box_scale: float = 1.0,
        iopa_threshold: Union[float, List[float]] = 0.0,
        use_tracking: bool = False,
        timeout_frames: int = 0,
        entry_delay_frames: int = 1,
        enable_zone_events: bool = False,
        window_name: Optional[str] = None,
        show_overlay: bool = True,
        show_inzone_counters: Optional[str] = None,
        annotation_color: Optional[tuple] = None,
        annotation_line_width: Optional[int] = None,
    ):
        """Initialize ZoneCounter.

        Args:
            zones: Dict mapping zone names to polygons (new style) OR list of polygons (old style backward compat)
            class_list: List of class labels to count (None = all classes)
            per_class_display: Show per-class counts separately
            triggering_position: Anchor point(s) or None for IoPA
            bounding_box_scale: Scale factor for bboxes (greater than 0, up to 1)
            iopa_threshold: IoPA threshold (single value or list per zone)
            use_tracking: Enable object tracking
            timeout_frames: Number of consecutive missing frames tolerated before object exits zone (requires tracking if > 0)
            entry_delay_frames: Number of consecutive frames required to establish object presence before entry (requires tracking if > 1)
            enable_zone_events: Generate zone-level events (requires tracking)
            window_name: OpenCV window name for interactive polygon editing (None = disabled)
            show_overlay: Draw zone overlays
            show_inzone_counters: Show presence counters ('time', 'frames', 'all', None)
            annotation_color: RGB color for overlays (None = auto)
            annotation_line_width: Line width for overlays (None = auto)
        """
        # Handle backward compatibility: count_polygons (old) or zones (new)
        if count_polygons is not None:
            zones = count_polygons

        if zones is None:
            raise ValueError("Either 'zones' or 'count_polygons' must be specified")

        # Handle backward compatibility: zones can be dict (new) or list (old)
        if isinstance(zones, dict):
            self._zones_as_list = False
            self._zone_names = list(zones.keys())
            self._zone_polygons = [
                np.array(poly, dtype=np.int32) for poly in zones.values()
            ]
        else:
            # Old style: list of polygons - convert to dict with string indices
            self._zones_as_list = True
            if isinstance(zones, np.ndarray):
                zones = [zones]
            self._zone_polygons = [np.array(poly, dtype=np.int32) for poly in zones]
            self._zone_names = [str(i) for i in range(len(self._zone_polygons))]

        # Validate and store parameters
        self._class_list = [] if class_list is None else class_list
        self._per_class_display = per_class_display
        if class_list is None and per_class_display:
            raise ValueError(
                "class_list must be specified when per_class_display is True"
            )

        self._triggering_position = triggering_position
        self._bounding_box_scale = bounding_box_scale
        if not 0 < self._bounding_box_scale <= 1.0:
            raise ValueError("bounding_box_scale must be between 0 and 1")

        # Process iopa_threshold
        if isinstance(iopa_threshold, (int, float)):
            self._iopa_thresholds = [float(iopa_threshold)] * len(self._zone_names)
        else:
            if len(iopa_threshold) != len(self._zone_names):
                raise ValueError(
                    f"iopa_threshold list length must match number of zones ({len(self._zone_names)})"
                )
            self._iopa_thresholds = [float(t) for t in iopa_threshold]

        # Validate iopa_threshold values
        if self._triggering_position is None:
            for i, threshold in enumerate(self._iopa_thresholds):
                if threshold <= 0:
                    raise ValueError(
                        f"iopa_threshold[{i}] must be > 0 when triggering_position is None"
                    )

        # Tracking and delay parameters
        self._use_tracking = use_tracking
        self._timeout_frames = timeout_frames
        self._entry_delay_frames = entry_delay_frames

        if self._timeout_frames > 0 and not self._use_tracking:
            raise ValueError("timeout_frames > 0 requires use_tracking=True")
        if self._entry_delay_frames > 1 and not self._use_tracking:
            raise ValueError("entry_delay_frames > 1 requires use_tracking=True")

        self._enable_zone_events = enable_zone_events
        if self._enable_zone_events and not self._use_tracking:
            raise ValueError("enable_zone_events requires use_tracking=True")

        # Display options
        self._show_overlay = show_overlay
        self._show_inzone_counters = show_inzone_counters
        if self._show_inzone_counters not in [None, "time", "frames", "all"]:
            raise ValueError(
                "show_inzone_counters must be 'time', 'frames', 'all', or None"
            )
        if self._show_inzone_counters is not None and not self._show_overlay:
            raise ValueError("show_inzone_counters requires show_overlay=True")
        if self._show_inzone_counters is not None and not self._use_tracking:
            raise ValueError("show_inzone_counters requires use_tracking=True")

        self._annotation_color = annotation_color
        self._annotation_line_width = annotation_line_width

        # Interactive editing support
        self._win_name = window_name
        self._mouse_callback_installed = False
        self._gui_state: Optional[Dict] = None

        # Internal state (lazy initialized)
        self._geometries: Optional[List[_ZoneGeometry]] = None
        self._states: Optional[List[_ZoneState]] = None
        self._frame_wh: Optional[Tuple[int, int]] = None

    def analyze(self, result):
        """Analyze inference result and update zone counts and events.

        Args:
            result: Inference result to analyze
        """
        # Initialize result structures
        # Use list format (old style) if zones were specified as list, otherwise dict (new style)
        if self._zones_as_list:
            result.zone_counts = [
                dict.fromkeys(
                    (
                        self._class_list + ["total"]
                        if self._per_class_display
                        else ["total"]
                    ),
                    0,
                )
                for _ in self._zone_names
            ]
        else:
            result.zone_counts = {
                name: dict.fromkeys(
                    (
                        self._class_list + ["total"]
                        if self._per_class_display
                        else ["total"]
                    ),
                    0,
                )
                for name in self._zone_names
            }

        if self._enable_zone_events:
            result.zone_events = []

        # Lazy initialization
        self._lazy_init(result)
        if self._geometries is None:
            return

        # Filter objects
        filtered_objects = self._filter_objects(result)

        # Process each zone (even if no detections - need to handle missing tracks)
        bboxes = (
            np.array([obj["bbox"] for obj in filtered_objects])
            if filtered_objects
            else np.array([])
        )
        time_now = time.time()

        assert self._geometries is not None  # Type guard for mypy
        assert self._states is not None  # Type guard for mypy
        for zi, (zone_name, geom, state) in enumerate(
            zip(self._zone_names, self._geometries, self._states)
        ):
            # Get zone triggers
            triggers = geom.trigger(bboxes)

            # Process objects in this zone
            self._process_zone(
                result, zi, zone_name, triggers, filtered_objects, state, time_now
            )

    def _lazy_init(self, result):
        """Initialize geometry and state components on first call."""
        if self._geometries is None:
            self._frame_wh = (result.image.shape[1], result.image.shape[0])

            # Create geometry components
            self._geometries = [
                _ZoneGeometry(
                    polygon,
                    self._frame_wh,
                    self._triggering_position,
                    self._bounding_box_scale,
                    self._iopa_thresholds[i],
                )
                for i, polygon in enumerate(self._zone_polygons)
            ]

            # Create state components
            self._states = [
                _ZoneState(self._timeout_frames, self._entry_delay_frames)
                for _ in self._zone_names
            ]

            # Install mouse callback for interactive editing if window_name specified
            if not self._mouse_callback_installed and self._win_name is not None:
                self._install_mouse_callback()

    def _filter_objects(self, result) -> List[dict]:
        """Filter objects based on class list and tracking requirements.

        Returns only fresh detections - no synthetic objects from stale trails.
        Missing tracked objects are handled via exit_delay_frames in zone state.
        """

        def in_class_list(label):
            return (
                True
                if not self._class_list
                else False if label is None else label in self._class_list
            )

        # Filter detected objects (fresh detections only)
        filtered = [
            obj
            for obj in result.results
            if "bbox" in obj
            and in_class_list(obj.get("label"))
            and (not self._use_tracking or "track_id" in obj)
        ]

        # Initialize per-object zone fields
        nzones = len(self._zone_names)
        for obj in filtered:
            obj[self.key_in_zone] = [False] * nzones
            if self._use_tracking:
                obj[self.key_frames_in_zone] = [0] * nzones
                obj[self.key_time_in_zone] = [0.0] * nzones

        return filtered

    def _process_zone(
        self,
        result,
        zone_index: int,
        zone_name: str,
        triggers: np.ndarray,
        objects: List[dict],
        state: _ZoneState,
        timestamp: float,
    ):
        """Process a single zone - update counts, states, and generate events.

        Implements clean 3-case logic:
        1. Detected in zone (fresh bbox triggers)
        2. Detected outside zone (fresh bbox doesn't trigger, but has state)
        3. Missing detection (in state but not detected this frame)
        """
        # Access zone_counts with backward compatibility (list vs dict)
        zone_counts = (
            result.zone_counts[zone_index]
            if self._zones_as_list
            else result.zone_counts[zone_name]
        )

        # Collect track IDs
        detected_tids = (
            {obj["track_id"] for obj in objects} if self._use_tracking else set()
        )
        all_tracked_tids = state.get_tracked_ids() if self._use_tracking else set()
        missing_tids = all_tracked_tids - detected_tids

        # Case 1 & 2: Process fresh detections
        for obj, is_triggered in zip(objects, triggers):
            if is_triggered:
                # Case 1: Fresh detection IN zone
                self._handle_object_in_zone(
                    result, zone_index, zone_name, obj, state, zone_counts, timestamp
                )
            elif self._use_tracking:
                # Case 2: Fresh detection OUTSIDE zone (check if has state)
                self._handle_detected_outside_zone(
                    result, zone_index, zone_name, obj, state, zone_counts, timestamp
                )

        # Case 3: Process missing detections (tracked but not detected this frame)
        if self._use_tracking and missing_tids:
            self._handle_missing_detections(
                result,
                zone_index,
                zone_name,
                state,
                zone_counts,
                missing_tids,
                timestamp,
            )

        # Generate zone-level occupancy events
        if self._enable_zone_events:
            current_count = zone_counts["total"]
            self._generate_occupancy_events(
                result, zone_index, zone_name, state, current_count, timestamp
            )

    def _handle_object_in_zone(
        self,
        result,
        zone_index: int,
        zone_name: str,
        obj: dict,
        state: _ZoneState,
        zone_counts: dict,
        timestamp: float,
    ):
        """Handle an object that is currently in the zone."""
        # Mark object as in zone
        obj[self.key_in_zone][zone_index] = True

        if self._use_tracking:
            tid = obj["track_id"]
            entry_event, is_fresh = state.add_track(
                tid, obj.get("label", "unknown"), timestamp
            )

            # Get object state to check if established
            obj_state = state.get_state(tid)
            assert obj_state is not None  # Type guard for mypy

            # Count all established objects (zone_counts resets each frame)
            if obj_state.is_established:
                zone_counts["total"] += 1
                if self._per_class_display and obj.get("label"):
                    zone_counts[obj["label"]] += 1

            # Generate entry event when object becomes established
            if entry_event and self._enable_zone_events:
                self._generate_event(
                    result,
                    "zone_entry",
                    zone_index,
                    zone_name,
                    tid,
                    obj.get("label"),
                    timestamp,
                    None,  # No dwell_time for entry
                )

            # Update time-in-zone metrics
            assert obj_state is not None  # Type guard for mypy
            obj[self.key_frames_in_zone][zone_index] = obj_state.presence_count
            obj[self.key_time_in_zone][zone_index] = timestamp - obj_state.entering_time
        else:
            # Non-tracking mode: count all objects immediately
            zone_counts["total"] += 1
            if self._per_class_display and obj.get("label"):
                zone_counts[obj["label"]] += 1

    def _handle_detected_outside_zone(
        self,
        result,
        zone_index: int,
        zone_name: str,
        obj: dict,
        state: _ZoneState,
        zone_counts: dict,
        timestamp: float,
    ):
        """Handle fresh detection outside zone (Case 2: grace period).

        Object detected this frame but bbox doesn't trigger zone geometry.
        If it was previously in zone, apply grace period (timeout).
        """
        tid = obj["track_id"]
        obj_state = state.get_state(tid)

        if obj_state is not None:
            # Object was previously in zone - check if grace period configured
            if self._timeout_frames > 0:
                # Apply grace period
                still_valid = state.update_exit_delay(tid)

                if still_valid:
                    # Still within timeout - mark as in-zone
                    obj[self.key_in_zone][zone_index] = True

                    # Only count if established
                    if obj_state.is_established:
                        zone_counts["total"] += 1
                        if self._per_class_display and obj_state.object_label:
                            zone_counts[obj_state.object_label] += 1

                    obj[self.key_frames_in_zone][zone_index] = obj_state.presence_count
                    obj[self.key_time_in_zone][zone_index] = (
                        timestamp - obj_state.entering_time
                    )
                else:
                    # Timeout expired - generate exit event
                    if self._enable_zone_events and obj_state.is_established:
                        dwell_time = timestamp - obj_state.entering_time
                        self._generate_event(
                            result,
                            "zone_exit",
                            zone_index,
                            zone_name,
                            tid,
                            obj_state.object_label,
                            timestamp,
                            dwell_time,
                        )
                    state.remove_track(tid)
            else:
                # No timeout configured - immediate exit
                if self._enable_zone_events and obj_state.is_established:
                    dwell_time = timestamp - obj_state.entering_time
                    self._generate_event(
                        result,
                        "zone_exit",
                        zone_index,
                        zone_name,
                        tid,
                        obj_state.object_label,
                        timestamp,
                        dwell_time,
                    )
                state.remove_track(tid)

    def _handle_missing_detections(
        self,
        result,
        zone_index: int,
        zone_name: str,
        state: _ZoneState,
        zone_counts: dict,
        missing_tids: set,
        timestamp: float,
    ):
        """Handle missing detections (Case 3: tracked but not detected).

        These tracks are in zone state but have no detection this frame.
        They could be temporarily occluded or truly exited.
        """
        for tid in missing_tids:
            obj_state = state.get_state(tid)
            if obj_state is None:
                # Defensive check - should never happen since tid came from state.get_tracked_ids()
                continue

            if self._timeout_frames > 0:
                # Apply grace period - count missing objects during timeout
                still_valid = state.update_exit_delay(tid)

                if still_valid:
                    # Still in grace period - count established objects
                    if obj_state.is_established:
                        zone_counts["total"] += 1
                        if self._per_class_display and obj_state.object_label:
                            zone_counts[obj_state.object_label] += 1
                else:
                    # Timeout expired - exit event
                    if self._enable_zone_events and obj_state.is_established:
                        dwell_time = timestamp - obj_state.entering_time
                        self._generate_event(
                            result,
                            "zone_exit",
                            zone_index,
                            zone_name,
                            tid,
                            obj_state.object_label,
                            timestamp,
                            dwell_time,
                        )
                    state.remove_track(tid)
            else:
                # No timeout configured - immediate exit
                if self._enable_zone_events and obj_state.is_established:
                    dwell_time = timestamp - obj_state.entering_time
                    self._generate_event(
                        result,
                        "zone_exit",
                        zone_index,
                        zone_name,
                        tid,
                        obj_state.object_label,
                        timestamp,
                        dwell_time,
                    )
                state.remove_track(tid)

    def _generate_occupancy_events(
        self,
        result,
        zone_index: int,
        zone_name: str,
        state: _ZoneState,
        current_count: int,
        timestamp: float,
    ):
        """Generate zone-level occupancy transition events."""
        occupied_event, empty_event, state_duration = state.update_occupancy(
            current_count, timestamp
        )

        if occupied_event:
            # Zone became occupied - dwell_time is how long it was empty
            self._generate_event(
                result,
                "zone_occupied",
                zone_index,
                zone_name,
                None,
                None,
                timestamp,
                state_duration,
            )

        if empty_event:
            # Zone became empty - dwell_time is how long it was occupied
            self._generate_event(
                result,
                "zone_empty",
                zone_index,
                zone_name,
                None,
                None,
                timestamp,
                state_duration,
            )

    def _generate_event(
        self,
        result,
        event_type: str,
        zone_index: int,
        zone_id: str,
        track_id: Optional[int],
        object_label: Optional[str],
        timestamp: float,
        dwell_time: Optional[float] = None,
    ):
        """Generate and append an event to the result.

        All events use a fixed schema with consistent fields.
        dwell_time semantics:
        - zone_entry: None (no prior state)
        - zone_exit: Time track spent in zone
        - zone_occupied: Time zone was empty before occupation
        - zone_empty: Time zone was occupied before becoming empty
        """
        event = {
            "event_type": event_type,
            "zone_index": zone_index,
            "zone_id": zone_id,
            "timestamp": timestamp,
            "track_id": track_id,
            "object_label": object_label,
            "dwell_time": dwell_time,
            "frame_index": (
                result.frame_index if hasattr(result, "frame_index") else None
            ),
        }

        result.zone_events.append(event)

    @staticmethod
    def _mouse_callback(event: int, x: int, y: int, flags: int, self):
        """Mouse event callback for interactive polygon zone editing.

        Supports:
        - Left-click and drag to move entire polygon
        - Right-click and drag to move individual vertices

        Args:
            event: OpenCV mouse event code
            x: X coordinate of the mouse event
            y: Y coordinate of the mouse event
            flags: Additional event flags
            self: Instance of ZoneCounter
        """
        click_point = np.array((x, y))

        def zone_update():
            idx = self._gui_state["update"]
            if idx >= 0 and self._frame_wh is not None:
                if not np.array_equal(
                    self._geometries[idx].polygon, self._zone_polygons[idx]
                ):
                    self._geometries[idx] = _ZoneGeometry(
                        self._zone_polygons[idx],
                        self._frame_wh,
                        self._triggering_position,
                        self._bounding_box_scale,
                        self._iopa_thresholds[idx],
                    )

        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, polygon in enumerate(self._zone_polygons):
                if cv2.pointPolygonTest(polygon, (x, y), False) > 0:
                    self._gui_state["dragging"] = polygon
                    self._gui_state["offset"] = click_point
                    self._gui_state["update"] = idx
                    break
        if event == cv2.EVENT_RBUTTONDOWN:
            for idx, polygon in enumerate(self._zone_polygons):
                for pt in polygon:
                    if np.linalg.norm(pt - click_point) < 10:
                        self._gui_state["dragging"] = pt
                        self._gui_state["offset"] = click_point
                        self._gui_state["update"] = idx
                        break
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._gui_state.get("dragging") is not None:
                delta = click_point - self._gui_state["offset"]
                self._gui_state["dragging"] += delta
                self._gui_state["offset"] = click_point
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self._gui_state["dragging"] = None
            zone_update()
            self._gui_state["update"] = -1

    def _install_mouse_callback(self):
        """Install the OpenCV mouse callback on the attached window."""
        if self._win_name is not None:
            try:
                cv2.setMouseCallback(self._win_name, ZoneCounter._mouse_callback, self)
                self._gui_state = {"dragging": None, "update": -1}
                self._mouse_callback_installed = True
            except Exception:
                pass  # ignore errors

    def window_attach(self, win_name: str):
        """Attach an OpenCV window for interactive polygon editing.

        Args:
            win_name: Name of the OpenCV window to attach to
        """
        self._win_name = win_name
        if not self._mouse_callback_installed:
            self._install_mouse_callback()

    def annotate(self, result, image: np.ndarray) -> np.ndarray:
        """Draw zone overlays and counts on the image.

        Args:
            result: Inference result with zone data
            image: Image to annotate

        Returns:
            Annotated image
        """
        if not self._show_overlay or self._geometries is None:
            return image

        # Determine colors
        line_color = (
            color_complement(result.overlay_color)
            if self._annotation_color is None
            else self._annotation_color
        )
        text_color = deduce_text_color(line_color)
        line_width = (
            result.overlay_line_width
            if self._annotation_line_width is None
            else self._annotation_line_width
        )

        # Draw per-object zone presence counters
        if self._show_inzone_counters:
            self._annotate_objects(result, image, text_color, line_color)

        # Draw zone polygons and counts
        for zi, (zone_name, polygon) in enumerate(
            zip(self._zone_names, self._zone_polygons)
        ):
            # Draw polygon outline
            cv2.polylines(
                image,
                [cv2.Mat(polygon)],
                True,
                rgb_to_bgr(line_color),
                line_width,
            )

            # Draw zone label and counts (handle both list and dict formats)
            zone_counts_dict = (
                result.zone_counts[zi]
                if self._zones_as_list
                else result.zone_counts[zone_name]
            )

            if self._per_class_display:
                text = f"{zone_name}:"
                for class_name in self._class_list:
                    count = zone_counts_dict.get(class_name, 0)
                    text += f"\n {class_name}: {count}"
            else:
                count = zone_counts_dict.get("total", 0)
                text = f"{zone_name}: {count}"

            put_text(
                image,
                text,
                tuple(x + line_width for x in polygon[0]),
                font_color=text_color,
                bg_color=line_color,
                font_scale=result.overlay_font_scale,
            )

        return image

    def _annotate_objects(self, result, image: np.ndarray, text_color, bg_color):
        """Annotate individual objects with zone presence info."""
        for obj in result.results:
            if self.key_in_zone not in obj:
                continue

            text_lines = []
            for zi, in_zone in enumerate(obj[self.key_in_zone]):
                if not in_zone:
                    continue

                zone_name = self._zone_names[zi]
                frames = obj[self.key_frames_in_zone][zi]
                time_val = obj[self.key_time_in_zone][zi]

                if self._show_inzone_counters == "frames":
                    line = f"{zone_name}: {frames}#"
                elif self._show_inzone_counters == "time":
                    line = f"{zone_name}: {time_val:.1f}s"
                else:  # "all"
                    line = f"{zone_name}: {frames}#/{time_val:.1f}s"

                text_lines.append(line)

            if text_lines:
                text = "\n".join(text_lines)
                xy = np.array(obj["bbox"][:2]).astype(int) + result.overlay_line_width
                put_text(
                    image,
                    text,
                    tuple(xy),
                    corner_position=CornerPosition.TOP_LEFT,
                    font_color=text_color,
                    bg_color=bg_color,
                    font_scale=result.overlay_font_scale,
                )


# Backward compatibility alias
NamedZoneCounter = ZoneCounter
