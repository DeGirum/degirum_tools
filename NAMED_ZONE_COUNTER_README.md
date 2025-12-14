# NamedZoneCounter: Architecture and Design

## Overview

`NamedZoneCounter` is an enhanced zone counting analyzer that extends the capabilities of the original `ZoneCounter` with:
- **Named zones** - Use descriptive names instead of numeric indices
- **Zone-level events** - Track entry, exit, occupied, and empty transitions
- **Clean architecture** - Separation of spatial geometry and temporal state logic
- **Production-ready smoothing** - Timeout mechanism to filter noisy real-world data

## Architecture: Separation of Concerns

The implementation is structured around three focused components:

```
┌─────────────────────┐
│  NamedZoneCounter   │  Orchestration Layer
│  (analyze)          │  - Filters objects
└──────────┬──────────┘  - Coordinates processing
           │             - Generates events
           │
    ┌──────┴──────┐
    │             │
┌───▼────────┐ ┌──▼──────────┐
│ _ZoneGeometry│ │ _ZoneState  │
│ (spatial)    │ │ (temporal)  │
└──────────────┘ └─────────────┘
  Stateless        Stateful
  Pure geometry    Track history
```

### 1. _ZoneGeometry (Spatial Logic)

**Responsibility**: Determine if bounding boxes are inside a polygon zone

```python
class _ZoneGeometry:
    - polygon: Zone boundary
    - mask: Binary mask for point-in-polygon tests
    - trigger(bboxes): Returns boolean array of zone membership
```

**Key Features**:
- Stateless - no memory of previous frames
- Supports two triggering methods:
  - **Anchor points**: Test specific bbox points (center, bottom, etc.)
  - **IoPA**: Intersection over Polygon Area threshold
- Batch processing with numpy for efficiency

**Why separate?** Geometry is pure math - testable, reusable, no side effects.

### 2. _ZoneState (Temporal Logic)

**Responsibility**: Track object presence over time within a zone

```python
class _ZoneState:
    - _object_states: Dict[track_id → _ObjectState]
    - _was_occupied: Previous frame occupancy (for events)
    
    _ObjectState:
        - timeout_count: Frames until exit
        - object_label: Class name
        - presence_count: Total frames in zone
        - entering_time: Entry timestamp
```

**Key Features**:
- Manages track lifecycle: add, update timeout, remove
- Handles graceful exit with timeout mechanism
- Tracks occupancy transitions for zone-level events

**Why separate?** State management is complex - isolating it makes testing and debugging tractable.

### 3. NamedZoneCounter (Orchestration)

**Responsibility**: Coordinate geometry and state to produce results

```python
def analyze(result):
    1. Filter objects (class_list, tracking requirements)
    2. For each zone:
        a. Get triggers from geometry
        b. Process objects with state
        c. Generate events
```

**Why separate?** Orchestration logic changes independently from geometry/state algorithms.

## The Smoothing Mechanism

### Multi-Layer Noise Reduction

The system employs a three-stage filtering pipeline:

```
Real World (noisy detections/tracking)
    ↓
┌───────────────────────────────────┐
│ Layer 1: ObjectTracker            │  Maintains trails and track IDs
│ - Buffers recent detections       │  even when detector misses frames
│ - Preserves track_id continuity   │
└──────────────┬────────────────────┘
               ↓
┌───────────────────────────────────┐
│ Layer 2: NamedZoneCounter Timeout │  Grace period for brief absences
│ - timeout_frames parameter        │  (flicker, occlusion, edge cases)
│ - inactive_tids processing        │
└──────────────┬────────────────────┘
               ↓
┌───────────────────────────────────┐
│ Layer 3: Event Generation Logic   │  Only emit on sustained changes
│ - Entry: First appearance         │
│ - Exit: After timeout expires     │
│ - Occupied/Empty: State transitions│
└───────────────────────────────────┘
               ↓
Clean Events (actionable signals)
```

### Object Processing Categories

On each frame, detected objects fall into three categories:

```python
# Computed upfront for clarity and order-independence
detected_tids = {obj["track_id"] for obj in objects}
all_tracked_tids = state.get_tracked_ids()
inactive_tids = all_tracked_tids - detected_tids
```

**1. In Zone (is_triggered=True)**
```python
_handle_object_in_zone():
    - Mark object.in_zone[zi] = True
    - Increment zone counts
    - Add/update track in state
    - Generate entry event (if new)
    - Update time/frames metrics
```

**2. Missing from Zone (detected but not triggered)**
```python
_handle_object_missing():
    - Check if track exists in zone state
    - Decrement timeout_count
    - Keep counting during grace period
    - Generate exit event when timeout expires
```

**3. Inactive (not detected this frame)**
```python
_process_inactive_tracks(inactive_tids):
    - For each track only in state, not in detections
    - Decrement timeout_count
    - Continue counting during grace period
    - Generate exit event when timeout expires
    - Remove from state
```

### Why Three Categories?

- **In zone**: Active presence, reset timeout
- **Missing**: Object detected elsewhere in frame but not in this zone, use timeout grace
- **Inactive**: Track remembered by zone but completely absent from frame

This distinction is critical for **event quality**:
- Without timeout: Exit event on first absence → noise
- With timeout: Exit event only after sustained absence → signal

## Event Types

When `enable_zone_events=True`, four event types are generated:

### Track-Level Events (require tracking)

**Fixed Event Schema**: All events have the same structure for consistency and ease of use.

```python
{
    "event_type": str,           # Event type
    "zone_index": int,           # Numeric zone index (0-based)
    "zone_id": str,              # Zone name/ID
    "timestamp": float,          # Unix timestamp
    "track_id": int | None,      # Track ID (entry/exit) or None (occupied/empty)
    "object_label": str | None,  # Object class (entry/exit) or None (occupied/empty)
    "dwell_time": float | None,  # Duration context (see below)
    "frame_number": int | None,  # Frame index (if available)
}
```

**dwell_time Semantics** (context-dependent duration):
- `zone_entry`: `None` (no prior state)
- `zone_exit`: Time track spent IN the zone (seconds)
- `zone_occupied`: Time zone was EMPTY before occupation (seconds)
- `zone_empty`: Time zone was OCCUPIED before becoming empty (seconds)

### Track-Level Events (require tracking)

**zone_entry**: Object first enters zone
```python
{
    "event_type": "zone_entry",
    "zone_index": 0,
    "zone_id": "entrance",
    "timestamp": 1702512345.678,
    "track_id": 42,
    "object_label": "person",
    "dwell_time": None,      # No prior state
    "frame_number": 123
}
```

**zone_exit**: Object exits after timeout expires
```python
{
    "event_type": "zone_exit",
    "zone_index": 0,
    "zone_id": "entrance",
    "timestamp": 1702512350.234,
    "track_id": 42,
    "object_label": "person",
    "dwell_time": 4.556,     # Time track was in zone
    "frame_number": 253
}
```

### Zone-Level Events (occupancy transitions)

**zone_occupied**: Zone transitions from empty to occupied
```python
{
    "event_type": "zone_occupied",
    "zone_index": 1,
    "zone_id": "parking_spot",
    "timestamp": 1702512360.123,
    "track_id": None,        # No specific track
    "object_label": None,    # Could be any class
    "dwell_time": 8.5,       # Time zone was empty (vacancy duration)
    "frame_number": 456
}
```

**zone_empty**: Zone transitions from occupied to empty
```python
{
    "event_type": "zone_empty",
    "zone_index": 1,
    "zone_id": "parking_spot",
    "timestamp": 1702512400.789,
    "track_id": None,
    "object_label": None,
    "dwell_time": 40.666,    # Time zone was occupied (occupancy duration)
    "frame_number": 987
}
```


## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zones` | `Dict[str, np.ndarray]` | **Required** | Mapping of zone names to polygon vertices |
| `class_list` | `List[str]` | `None` | Classes to count (None = all) |
| `per_class_display` | `bool` | `False` | Show per-class counts separately |

### Triggering Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `triggering_position` | `AnchorPoint` or `List` | `BOTTOM_CENTER` | Anchor point(s) to test, or `None` for IoPA |
| `bounding_box_scale` | `float` | `1.0` | Scale down bboxes (0-1) before testing |
| `iopa_threshold` | `float` or `List[float]` | `0.0` | IoPA threshold (used when `triggering_position=None`) |

**Anchor Points**: `TOP_LEFT`, `TOP_CENTER`, `TOP_RIGHT`, `CENTER_LEFT`, `CENTER`, `CENTER_RIGHT`, `BOTTOM_LEFT`, `BOTTOM_CENTER`, `BOTTOM_RIGHT`

### Tracking and Events

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_tracking` | `bool` | `False` | Enable object tracking features |
| `timeout_frames` | `int` | `0` | Grace period frames (requires tracking) |
| `enable_zone_events` | `bool` | `False` | Generate zone events (requires tracking) |

### Display

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_overlay` | `bool` | `True` | Draw zone polygons and counts |
| `show_inzone_counters` | `str` | `None` | Show per-object counters: `"time"`, `"frames"`, `"all"`, or `None` |
| `annotation_color` | `tuple` | `None` | RGB color (None = auto) |
| `annotation_line_width` | `int` | `None` | Line width (None = auto) |

## Result Attributes

### zone_counts (Dict[str, Dict[str, int]])

Zone counts dictionary with zone names as keys:

```python
result.zone_counts = {
    "entrance": {"total": 5},
    "checkout": {"person": 3, "cart": 2, "total": 5},
}
```

### zone_events (List[Dict])

List of zone events (when `enable_zone_events=True`):

```python
result.zone_events = [
    {
        "event_type": "zone_entry",
        "zone_index": 0,
        "zone_id": "entrance",
        "timestamp": 1702512345.678,
        "track_id": 42,
        "object_label": "person",
        "frame_number": 123
    },
    # ... more events
]
```

### Per-Object Attributes

Each detection object gets additional attributes:

```python
obj["in_zone"]           # List[bool]: In each zone?
obj["frames_in_zone"]    # List[int]: Frame counts (if tracking)
obj["time_in_zone"]      # List[float]: Time in seconds (if tracking)
```

## Design Decisions and Rationale

### Why Separate ZoneCounter vs NamedZoneCounter?

**Backward compatibility**: Existing code using `ZoneCounter` continues to work unchanged.

**Clean slate**: NamedZoneCounter gets modern architecture without legacy constraints.

**Migration path**: Teams can migrate incrementally when ready.


## Summary

**NamedZoneCounter** provides production-ready zone analytics with:

1. **User-friendly API**: Named zones instead of indices
2. **Clean architecture**: Testable, maintainable components
3. **Event generation**: Track entry/exit and occupancy transitions
4. **Noise reduction**: Multi-layer smoothing for real-world reliability
5. **Backward compatible**: ZoneCounter remains unchanged

The timeout mechanism is the **key feature** that makes zone events viable for production use - without it, users would be overwhelmed by detection/tracking noise. The architecture's separation of concerns makes the system easier to understand, test, and extend.

---
