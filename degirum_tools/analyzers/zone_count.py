#
# zone_count.py: zone object counting with clean architecture
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements ZoneCounter with separation of geometry and state concerns
#

"""
Zone Counter Analyzer Module Overview
=====================================

This module provides a zone counting analyzer (`ZoneCounter`) with support for both
traditional list-based zones and named zones (dictionary-based). It features clean
separation of concerns and a user-friendly API.

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
        timeout_frames=3,  # Hysteresis threshold
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

    - **zone_entry**: Track becomes established in zone (after timeout_frames + 1 consecutive in-zone detections)
    - **zone_exit**: Track exits zone (hysteresis counter reaches 0)
    - **zone_occupied**: Zone transitions from empty to occupied (when first track becomes established)
    - **zone_empty**: Zone transitions from occupied to empty (after all tracks exit)

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
    - `timeout_frames`: Hysteresis threshold for symmetric entry/exit smoothing
    - `enable_zone_events`: Generate zone-level events
    - `show_overlay`: Visual annotations
    - `per_class_display`: Show per-class counts

Hysteresis Smoothing:
    The analyzer uses symmetric hysteresis for stable zone presence detection:

    - **timeout_frames**: Controls the hysteresis threshold. Objects need `timeout_frames + 1`
      consecutive in-zone detections to become "established" (counted). The internal counter
      increments when in zone, decrements when outside or missing, and is clamped to
      [0, timeout_frames + 1]. Objects exit when the counter reaches 0.

    - **Symmetric behavior**: The same threshold applies for both entry and exit, providing
      smooth transitions without flickering. Brief departures don't reset the counter,
      and brief appearances don't immediately establish presence.

    - **Default**: 0 (immediate entry/exit on first detection). Requires `use_tracking=True` if > 0.
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

__all__ = ["ZoneCounter"]


@dataclass
class _ObjectState:
    """Tracks the state of an object within a zone.

    Attributes:
        accum_in_zone_count: Hysteresis counter for zone presence.
            Increments when object is in zone, decrements when outside/missing.
            Range: [0, timeout_frames + 1]. Object becomes established when
            counter reaches timeout_frames + 1, exits when counter reaches 0.
        object_label: Class label of the tracked object
        presence_count: Number of frames object has been established in zone
        established_time: Timestamp when object became established
        is_established: Whether object presence is confirmed (hysteresis threshold met)
    """

    accum_in_zone_count: int
    object_label: str
    presence_count: int
    established_time: Optional[float] = None
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

    This class tracks object presence over time using symmetric hysteresis,
    and maintains zone-level occupancy state for event generation.

    The hysteresis counter (accum_in_zone_count) increments when an object is
    detected in the zone and decrements when outside or missing. Objects become
    "established" when the counter reaches timeout_frames + 1, and exit when
    the counter reaches 0.

    Attributes:
        object_states: Dictionary mapping track_id to _ObjectState
        was_occupied: Previous frame's occupancy state
    """

    def __init__(self, timeout_frames: int):
        """Initialize zone state.

        Args:
            timeout_frames: Hysteresis threshold. Objects need timeout_frames + 1
                consecutive in-zone detections to become established, and
                timeout_frames + 1 consecutive out-of-zone/missing detections to exit.
        """
        self._timeout_frames = timeout_frames
        self._object_states: Dict[int, _ObjectState] = {}
        self._was_occupied: bool = False
        self._state_change_time: Optional[float] = None  # Track last transition time

    def add_or_update_track(self, track_id: int, label: str, timestamp: float) -> bool:
        """Add a new track or update an existing track detected in the zone.

        For new tracks, creates state and starts hysteresis counting.
        For existing tracks, increments the hysteresis counter (capped at timeout_frames + 1).
        When counter reaches the threshold, object becomes established.

        Args:
            track_id: Unique object identifier
            label: Class label of the object
            timestamp: Current frame timestamp

        Returns:
            True if object just became established (entry event)
        """
        max_count = self._timeout_frames + 1

        if track_id in self._object_states:
            # Existing tracked object - increment hysteresis counter
            obj_state = self._object_states[track_id]
            obj_state.accum_in_zone_count = min(
                obj_state.accum_in_zone_count + 1, max_count
            )

            # Check if object just became established
            if (
                not obj_state.is_established
                and obj_state.accum_in_zone_count == max_count
            ):
                obj_state.is_established = True
                obj_state.established_time = timestamp
                obj_state.presence_count = 1  # First frame as established
                return True  # Entry event

            # Only increment presence_count if already established
            if obj_state.is_established:
                obj_state.presence_count += 1

            return False  # No entry event
        else:
            # New object entering zone for the first time
            is_immediate = self._timeout_frames == 0
            self._object_states[track_id] = _ObjectState(
                accum_in_zone_count=1,
                object_label=label,
                presence_count=1 if is_immediate else 0,
                established_time=timestamp if is_immediate else None,
                is_established=is_immediate,  # Immediate if timeout_frames == 0
            )
            return is_immediate

    def decrement_counter(self, track_id: int) -> bool:
        """Decrement hysteresis counter for a track outside zone or missing.

        Args:
            track_id: Track ID to update

        Returns:
            True if track is still valid (counter > 0), False if counter reached 0
        """
        if track_id not in self._object_states:
            return False

        obj_state = self._object_states[track_id]
        # Only increment presence_count if established (tracks frames while established)
        if obj_state.is_established:
            obj_state.presence_count += 1
        obj_state.accum_in_zone_count -= 1

        return obj_state.accum_in_zone_count > 0

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
            True if hysteresis counter has reached 0 (ready for exit)
        """
        obj_state = self._object_states.get(track_id)
        return obj_state is not None and obj_state.accum_in_zone_count == 0

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
    object tracking with symmetric hysteresis (timeout_frames) for entry/exit smoothing,
    zone events, and interactive editing.

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
            timeout_frames: Hysteresis threshold for zone presence (requires tracking if > 0).
                Objects need timeout_frames + 1 consecutive in-zone detections to become
                established (counted), and the counter decrements when outside/missing.
                Objects exit when counter reaches 0.
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

        # Validate timeout_frames
        if not isinstance(timeout_frames, int) or timeout_frames < 0:
            raise ValueError("timeout_frames must be a non-negative integer (>= 0)")
        self._timeout_frames = timeout_frames

        if self._timeout_frames > 0 and not self._use_tracking:
            raise ValueError("timeout_frames > 0 requires use_tracking=True")

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

        if self._enable_zone_events and not hasattr(result, "zone_events"):
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
                _ZoneState(self._timeout_frames) for _ in self._zone_names
            ]

            # Install mouse callback for interactive editing if window_name specified
            if not self._mouse_callback_installed and self._win_name is not None:
                self._install_mouse_callback()

    def _filter_objects(self, result) -> List[dict]:
        """Filter objects based on class list and tracking requirements.

        Returns objects from `result.results` which have bounding boxes, match `class_list`,
        and (when tracking is enabled) provide a `track_id`.

        Missing tracked objects are handled via the hysteresis counter in zone state
        rather than by synthesizing detections in `result.results`.
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

    def _set_object_zone_attr(
        self,
        obj: dict,
        zone_index: int,
        obj_state: _ObjectState,
        timestamp: float,
    ):
        """Set zone presence fields on an object.

        Args:
            obj: Object dictionary to update
            zone_index: Index of the zone
            obj_state: Current state of the object
            timestamp: Current frame timestamp
        """
        obj[self.key_in_zone][zone_index] = True
        obj[self.key_frames_in_zone][zone_index] = obj_state.presence_count
        assert obj_state.established_time is not None
        obj[self.key_time_in_zone][zone_index] = timestamp - obj_state.established_time

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
        if self._use_tracking:
            tid = obj["track_id"]
            entry_event = state.add_or_update_track(
                tid, obj.get("label", "unknown"), timestamp
            )

            # Get object state to check if established
            obj_state = state.get_state(tid)
            assert obj_state is not None  # Type guard for mypy

            # Only mark in_zone, update metrics, and count when established
            if obj_state.is_established:
                self._set_object_zone_attr(obj, zone_index, obj_state, timestamp)
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
        else:
            # Non-tracking mode: count all objects immediately
            obj[self.key_in_zone][zone_index] = True
            zone_counts["total"] += 1
            if self._per_class_display and obj.get("label"):
                zone_counts[obj["label"]] += 1

    def _update_zone_state_for_absence(
        self,
        result,
        zone_index: int,
        zone_name: str,
        tid: int,
        obj_state: _ObjectState,
        state: _ZoneState,
        zone_counts: dict,
        timestamp: float,
    ) -> bool:
        """Update zone state when an object is absent (outside zone or missing).

        Decrements the hysteresis counter, updates zone counts if still valid and
        established, generates exit event and removes track if counter reaches 0.

        Args:
            result: Inference result for event generation
            zone_index: Index of the zone
            zone_name: Name of the zone
            tid: Track ID of the object
            obj_state: Current state of the object
            state: Zone state manager
            zone_counts: Dictionary to update counts
            timestamp: Current frame timestamp

        Returns:
            True if track is still valid (counter > 0), False if removed
        """
        still_valid = state.decrement_counter(tid)

        if still_valid:
            # Counter > 0, still tracked - count if established
            if obj_state.is_established:
                zone_counts["total"] += 1
                if self._per_class_display and obj_state.object_label:
                    zone_counts[obj_state.object_label] += 1
        else:
            # Counter reached 0 - generate exit event if established, then remove
            if self._enable_zone_events and obj_state.is_established:
                assert obj_state.established_time is not None
                dwell_time = timestamp - obj_state.established_time
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

        return still_valid

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
        """Handle fresh detection outside zone (hysteresis decrement).

        Object is detected this frame but its bbox does not trigger the zone geometry.
        If it was previously tracked in this zone, decrement the hysteresis counter.
        While the counter remains > 0, the object is still considered "in zone" for
        counting/event purposes (even if it is currently outside the polygon).
        """
        tid = obj["track_id"]
        obj_state = state.get_state(tid)

        if obj_state is not None:
            still_valid = self._update_zone_state_for_absence(
                result, zone_index, zone_name, tid, obj_state, state, zone_counts, timestamp
            )

            # Only set per-object fields if still valid AND established
            if still_valid and obj_state.is_established:
                self._set_object_zone_attr(obj, zone_index, obj_state, timestamp)

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
        """Handle missing detections (hysteresis decrement).

        These tracks are in zone state but have no detection this frame.
        Treated as outside zone - decrement hysteresis counter.
        """
        for tid in missing_tids:
            obj_state = state.get_state(tid)
            if obj_state is None:
                # Defensive check - should never happen since tid came from state.get_tracked_ids()
                continue

            self._update_zone_state_for_absence(
                result, zone_index, zone_name, tid, obj_state, state, zone_counts, timestamp
            )

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
            found = False
            for idx, polygon in enumerate(self._zone_polygons):
                for pt in polygon:
                    if np.linalg.norm(pt - click_point) < 10:
                        self._gui_state["dragging"] = pt
                        self._gui_state["offset"] = click_point
                        self._gui_state["update"] = idx
                        found = True
                        break
                if found:
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

            # Format zone name (add "Zone" prefix for list-based zones for backward compatibility)
            display_name = f"Zone {zone_name}" if self._zones_as_list else zone_name

            if self._per_class_display:
                text = f"{display_name}:"
                for class_name in self._class_list:
                    count = zone_counts_dict.get(class_name, 0)
                    text += f"\n {class_name}: {count}"
            else:
                count = zone_counts_dict.get("total", 0)
                text = f"{display_name}: {count}"

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
