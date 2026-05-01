#
# gst_support.py: GStreamer pipeline builder for video sources
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements functions to build GStreamer pipelines for various video sources
#

"""
Simple GStreamer Pipeline Builder
Focus: Compatibility over optimization
"""

from __future__ import annotations

import os
import sys
import inspect
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import concurrent
from .. import logger_get
from .time_tools import Watchdog

if TYPE_CHECKING:
    from gi.repository import Gst


def setup_gst_environment(*plugin_dirs):
    """Set GST_PLUGIN_PATH before gi is imported, then import and return gi.

    GStreamer scans the plugin registry when the library first loads, so
    GST_PLUGIN_PATH must be in place before any gi import.

    To use custom Python plugins, place your plugin .py files under a
    subdirectory named `python` inside any directory passed here, and pass
    that parent directory as one of the arguments.

    Args:
        *plugin_dirs: Directory paths to prepend to GST_PLUGIN_PATH.
            Relative paths are resolved relative to the calling file.

    Returns:
        The imported gi module.
    """

    if plugin_dirs:
        caller_dir = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))

        abs_dirs = [
            os.path.join(caller_dir, d) if not os.path.isabs(d) else d
            for d in plugin_dirs
        ]

        existing = os.environ.get("GST_PLUGIN_PATH", "")
        all_dirs = abs_dirs + ([existing] if existing else [])
        if all_dirs:
            os.environ["GST_PLUGIN_PATH"] = ":".join(all_dirs)

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")

    from gi.repository import Gst

    Gst.init(None)

    return gi


class GstPipelineHandler:
    """Manages a single GStreamer pipeline lifecycle.

    Pass a fully-formed gst-launch pipeline string to the constructor;
    the pipeline starts immediately. Use `wait()` to block until it ends
    or `stop()` to tear it down early.

    A single GLib main loop shared across all instances is started
    automatically on first use.
    """

    _glib_loop = None
    _glib_loop_lock = threading.Lock()

    @classmethod
    def _ensure_main_loop(cls):
        from gi.repository import GLib

        with cls._glib_loop_lock:
            if cls._glib_loop is None or not cls._glib_loop.is_running():
                cls._glib_loop = GLib.MainLoop()
                threading.Thread(
                    target=cls._glib_loop.run,
                    daemon=True,
                    name="glib-main-loop",
                ).start()

    def __init__(self, pipeline_str, probe_element_name=""):
        from gi.repository import Gst

        self._Gst = Gst

        self._stopping = False
        self._pipeline = self._Gst.parse_launch(pipeline_str)
        self._future: concurrent.futures.Future = concurrent.futures.Future()
        self._watchdog = (
            Watchdog(time_limit=5.0, tps_threshold=0.0) if probe_element_name else None
        )

        if probe_element_name:
            element = self._pipeline.get_by_name(probe_element_name)
            if element is None:
                raise ValueError(
                    f"Element '{probe_element_name}' not found in pipeline"
                )
            pad = element.get_static_pad("src")
            if pad is None:
                raise ValueError(f"Element '{probe_element_name}' has no src pad")
            pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer)

        def on_bus_message(bus, message):
            mtype = message.type
            if mtype == Gst.MessageType.EOS:
                self.stop()
            elif mtype == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                if not self._stopping:
                    print(f"GStreamer error: {err.message} ({debug})", file=sys.stderr)
                self._pipeline.set_state(Gst.State.NULL)
                if not self._future.done():
                    self._future.set_exception(RuntimeError(f"{err.message} ({debug})"))

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", on_bus_message)

    def start(self):
        """Start the pipeline. Returns self for chaining."""
        self._ensure_main_loop()
        self._pipeline.set_state(self._Gst.State.PLAYING)
        return self

    def _on_buffer(self, pad, info):
        assert self._watchdog is not None
        self._watchdog.tick()
        return self._Gst.PadProbeReturn.OK

    def check(self):
        """Return current throughput.

        Returns:
            Tuple (is_active, fps).

        Raises:
            RuntimeError: If `probe_element_name` was not set at construction.
        """
        if self._watchdog is None:
            raise RuntimeError("check() requires probe_element_name to be set")
        return self._watchdog.check()

    def wait(self):
        """Block until the pipeline reaches EOS.

        Raises:
            RuntimeError: On pipeline error.
        """
        self._future.result()

    def stop(self):
        """Gracefully stop the pipeline."""
        self._stopping = True
        self._pipeline.set_state(self._Gst.State.NULL)
        self._pipeline.get_state(timeout=2 * self._Gst.SECOND)
        if not self._future.done():
            self._future.set_result(None)


# ---------------------------------------------------------------------------
# Base element class for custom GStreamer Python elements
# ---------------------------------------------------------------------------
class GstElementBase:
    """Mixin base for custom GStreamer Python elements.

    Because `gi` must not be imported before `setup_gst_environment` patches `GST_PLUGIN_PATH`,
    this class intentionally does **not** inherit from `Gst.Element`. Concrete plugin classes
    must use multiple inheritance so that `Gst.Element` appears in the MRO:

        class MyElement(GstElementBase, Gst.Element): ...

    **Inheritance order matters:** `GstElementBase` must be listed *before* `Gst.Element` so that
    cooperative `super()` calls resolve correctly and gst-python's GObject metaclass machinery works as expected.

    Subclasses must implement:

    * `get_metadata()` — return an `ElementMetadata` instance.
    * `get_pads()`     — return a list of `PadInfo` instances.
    * `_worker_func()` — worker thread body; runs until all sink queues are exhausted.

    `__init_subclass__` automatically builds `__gstmetadata__` and `__gsttemplates__` from those declarations
    so that gst-python can register the element.

    `__init__` creates all pads (SRC first so SINK pads can resolve `forward_to` by name), populates `self.sources`
    and `self.sinks`, then starts the worker thread.

    `do_set_state` closes all sink queues and joins the worker thread when the element transitions to NULL.
    """

    # ------------------------------------------------------------------
    # Nested types used in the public API
    # ------------------------------------------------------------------

    # Declared here so mypy knows these attributes exist on the class;
    # they are populated by __init_subclass__ at concrete-subclass creation time.
    __gstmetadata__: tuple
    __gsttemplates__: tuple

    class SinkPad:
        """Wraps a GStreamer sink pad with its buffer queue and frame metadata.

        Args:
            element: Parent `Gst.Element` that owns this pad.
            template_name: Name of the pad template (also used as the pad name).
            forward_to: Source pad to forward non-CAPS/non-EOS events to.
            forward_caps: Whether to forward CAPS events to `forward_to`.
            forward_eos: Whether to forward EOS events to `forward_to`.

        Attributes:
            pad: The underlying `Gst.Pad` added to the parent element.
            width: Frame width in pixels, populated on the first CAPS event.
            height: Frame height in pixels, populated on the first CAPS event.
            format: Pixel format string (e.g. `"NV12"`), populated on the first CAPS event.
            stride: Per-plane byte strides read from `GstVideoMeta` on the first
                buffer. `None` until the first buffer arrives, or if the buffer
                carries no `GstVideoMeta`.
            queue: Buffer queue fed by the chain function. Iterate over it in the
                worker thread; yields `Gst.Buffer` items and terminates on EOS.
        """

        def __init__(
            self,
            element: Gst.Element,
            template_name: str,
            forward_to: Optional[Gst.Pad] = None,
            *,
            forward_caps: bool = True,
            forward_eos: bool = True,
        ):
            from gi.repository import Gst

            self._Gst = Gst
            self.pad = Gst.Pad.new_from_template(
                element.get_pad_template(template_name), template_name
            )
            self.pad.set_chain_function_full(self._chain)
            self.pad.set_event_function_full(self._event)
            element.add_pad(self.pad)

            self._forward_to = forward_to
            self._forward_caps = forward_caps
            self._forward_eos = forward_eos

            self.width: Optional[int] = None
            self.height: Optional[int] = None
            self.format: Optional[str] = None
            self.stride: Optional[List[int]] = None

            # Holds Gst.Buffer items; None is the stop sentinel.
            from .. import streams

            self.queue = streams.Stream(10, True)

        def is_initialized(self) -> bool:
            return self.width is not None and self.height is not None

        def _chain(self, pad: Gst.Pad, parent, buf: Gst.Buffer) -> Gst.FlowReturn:
            if self.stride is None:
                from gi.repository import GstVideo

                meta = GstVideo.buffer_get_video_meta(buf)
                if meta is not None:
                    self.stride = list(meta.stride)
            self.queue.put(buf)
            return self._Gst.FlowReturn.OK

        def _event(self, pad: Gst.Pad, parent, event: Gst.Event) -> bool:
            if event.type == self._Gst.EventType.CAPS:
                caps = event.parse_caps()
                s = caps.get_structure(0)
                _, self.width = s.get_int("width")
                _, self.height = s.get_int("height")
                self.format = s.get_string("format")
                if self._forward_caps and self._forward_to:
                    return self._forward_to.push_event(event)

            elif event.type == self._Gst.EventType.EOS:
                self.queue.put(None)  # sentinel
                if self._forward_eos and self._forward_to:
                    return self._forward_to.push_event(event)

            elif self._forward_to:
                return self._forward_to.push_event(event)

            return True

    class PadDirection(Enum):
        """Direction of a pad template declared in `get_pads`."""

        SINK = 1
        SRC = 2

    @dataclass
    class ElementMetadata:
        """GStreamer element metadata returned by `get_metadata`.

        Attributes:
            longname: Human-readable element name.
            klass: Element classification (e.g. "Filter/Video").
            description: Short description of what the element does.
            author: Author / organization string.
        """

        longname: str
        klass: str
        description: str
        author: str

    @dataclass
    class PadInfo:
        """Descriptor for a single pad template returned by `get_pads`.

        `PadPresence.ALWAYS` is assumed for all pads. Use `PadDirection` instead of Gst types.

        Attributes:
            name: Pad name (used as both template name and pad name).
            direction: `PadDirection.SINK` or `PadDirection.SRC`.
            caps: Gst pad capabilities string (e.g. `"video/x-raw,format=NV12"`).
            forward_to: Name of the SRC pad to forward events to. `None` disables forwarding entirely.
            forward_caps: Whether to forward CAPS events to `forward_to`.
            forward_eos: Whether to forward EOS events to `forward_to`.
        """

        name: str
        direction: GstElementBase.PadDirection
        caps: str
        # SINK-pad forwarding controls
        forward_to: Optional[str] = None
        forward_caps: bool = True
        forward_eos: bool = True

    # ------------------------------------------------------------------
    # Subclass hook: build __gstmetadata__ / __gsttemplates__ automatically
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Only act when both classmethods are defined directly on this class,
        # so intermediate abstract layers are skipped gracefully.
        if "get_metadata" not in cls.__dict__ or "get_pads" not in cls.__dict__:
            return

        from gi.repository import Gst

        meta = cls.get_metadata()
        cls.__gstmetadata__ = (
            meta.longname,
            meta.klass,
            meta.description,
            meta.author,
        )

        templates = []
        for pad in cls.get_pads():
            gst_direction = (
                Gst.PadDirection.SINK
                if pad.direction == GstElementBase.PadDirection.SINK
                else Gst.PadDirection.SRC
            )
            templates.append(
                Gst.PadTemplate.new(
                    pad.name,
                    gst_direction,
                    Gst.PadPresence.ALWAYS,
                    Gst.Caps.from_string(pad.caps),
                )
            )
        cls.__gsttemplates__ = tuple(templates)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()

        from gi.repository import Gst

        self._Gst = Gst
        #: Dict of SRC pad name → Gst.Pad
        self.sources: Dict[str, Gst.Pad] = {}
        #: Dict of SINK pad name → GstElementBase.SinkPad
        self.sinks: Dict[str, GstElementBase.SinkPad] = {}

        pad_infos = self.get_pads()

        # Create SRC pads first so SINK pads can resolve forward_to by name.
        for info in pad_infos:
            if info.direction == GstElementBase.PadDirection.SRC:
                pad = Gst.Pad.new_from_template(
                    self.get_pad_template(info.name), info.name  # type: ignore[attr-defined]
                )
                self.add_pad(pad)  # type: ignore[attr-defined]
                self.sources[info.name] = pad

        # Create SINK pads.
        for info in pad_infos:
            if info.direction == GstElementBase.PadDirection.SINK:
                forward_pad = (
                    self.sources.get(info.forward_to)
                    if info.forward_to is not None
                    else None
                )
                sink = GstElementBase.SinkPad(
                    self,
                    info.name,
                    forward_to=forward_pad,
                    forward_caps=info.forward_caps,
                    forward_eos=info.forward_eos,
                )
                self.sinks[info.name] = sink

        # Start worker thread.
        self._worker = threading.Thread(
            target=self._worker_func,
            name=f"{type(self).__name__}-worker",
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def get_metadata(cls) -> GstElementBase.ElementMetadata:
        """Return element metadata.

        Returns:
            An `ElementMetadata` instance.

        Raises:
            NotImplementedError: Must be overridden in every concrete subclass.
        """
        raise NotImplementedError(f"{cls.__name__} must implement get_metadata()")

    @classmethod
    def get_pads(cls) -> List[GstElementBase.PadInfo]:
        """Return the list of pad descriptors.

        Returns:
            A list of `PadInfo` instances.

        Raises:
            NotImplementedError: Must be overridden in every concrete subclass.
        """
        raise NotImplementedError(f"{cls.__name__} must implement get_pads()")

    def _worker_func(self) -> None:
        """Worker thread body.

        Iterates over `self.sinks` queues, performs processing, and pushes
        results to the appropriate pad in `self.sources`. Returns when all
        sink queues have yielded their `None` sentinel.

        Raises:
            NotImplementedError: Must be overridden in every concrete subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _worker_func()"
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def do_set_state(self, state: Gst.State) -> Gst.StateChangeReturn:
        """Override: close sink queues and join worker thread on NULL transition."""
        if state == self._Gst.State.NULL:
            for sink in self.sinks.values():
                sink.queue.close()
            self._worker.join(timeout=2.0)
        return super().do_set_state(state)  # type: ignore[misc]


def _run_command(
    cmd: list, timeout: int = 5, check_for_lingering_process: bool = False
) -> Tuple[str, str, int]:
    """Run command and return stdout, stderr, returncode"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        # Check for lingering processes if requested
        if check_for_lingering_process and cmd:
            process_name = cmd[0]  # Use the first part of the command as process name
            lingering = subprocess.run(
                ["pgrep", process_name], capture_output=True, text=True
            )
            if lingering.stdout.strip():
                logger_get().warning(f"{process_name} process still running!")
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1


def _detect_camera_type(device_index: int) -> str:
    """
    Detect camera type using multiple methods
    Returns: 'rpi_csi', 'usb', or 'unknown'
    """
    device_path = f"/dev/video{device_index}"
    if not os.path.exists(device_path):
        raise FileNotFoundError(f"Camera device {device_path} not found")
    # Method 1: Check v4l2-ctl device info
    stdout, stderr, ret = _run_command(
        ["v4l2-ctl", "-d", device_path, "--info"], check_for_lingering_process=True
    )
    if ret == 0:
        info_lower = stdout.lower()
        # Raspberry Pi indicators (more comprehensive list)
        rpi_indicators = [
            "unicam",
            "csi",
            "rp1-cfe",
            "rp1_cfe",
            "bcm2835",
            "mmal",
            "raspberry",
            "rpi",
            "broadcom",
            "brcm",
            "vc4",
        ]
        for indicator in rpi_indicators:
            if indicator in info_lower:
                logger_get().info(f"Found RPi indicator '{indicator}' in device info")
                return "rpi_csi"
    else:
        logger_get().warning(f"v4l2-ctl failed for {device_path}: {stderr}")
    # Method 2: Check device path in /sys (fallback)
    try:
        device_name = Path(device_path).name  # e.g., 'video19'
        sys_path = f"/sys/class/video4linux/{device_name}/device"
        if os.path.exists(sys_path):
            # Read the real path
            real_path = os.readlink(sys_path).lower()
            # Check for RPi-specific paths
            rpi_path_indicators = [
                "platform/axi:csi",
                "platform/soc/csi",
                "platform/rp1",
                "bcm2835",
                "unicam",
                "cfe",
            ]
            for indicator in rpi_path_indicators:
                if indicator in real_path:
                    logger_get().info(
                        f"Found RPi path indicator '{indicator}' in sys path"
                    )
                    return "rpi_csi"
    except Exception as e:
        logger_get().debug(f"Sys path check failed: {e}")
    try:
        stdout, stderr, ret = _run_command(
            ["v4l2-ctl", "-d", device_path, "--list-formats"]
        )
        if ret == 0:
            formats_lower = stdout.lower()
            # logger.info(f"v4l2-ctl formats for {device_path}: {stdout}")
            #  RPi cameras often have specific format patterns
            if any(
                indicator in formats_lower for indicator in ["bayer", "rggb", "grbg"]
            ):
                logger_get().info("Found raw Bayer format - likely RPi CSI camera")
                return "rpi_csi"
    except Exception as e:
        logger_get().debug(f"Format check failed: {e}")
    # Method 4: Check high device numbers (RPi cameras often get high numbers)
    if device_index >= 10:
        logger_get().info(f"High device number {device_index} - likely RPi camera")
        # But still need confirmation from other methods, so this is just a hint
    # Everything else is treated as USB/webcam
    logger_get().info(
        f"No RPi indicators found - assuming USB/Webcam for {device_path}"
    )
    return "usb"


def _check_element_exists(element_name: str) -> bool:
    """Check if a GStreamer element exists"""
    stdout, stderr, ret = _run_command(["gst-inspect-1.0", element_name])
    return ret == 0


def _detect_platform() -> str:
    """Simple platform detection"""
    try:
        # Check device tree for ARM devices
        model_path = Path("/proc/device-tree/model")
        if model_path.exists():
            model = model_path.read_bytes().rstrip(b"\0\n").decode()
            if "Raspberry Pi" in model:
                return "raspberrypi"
            elif "Jetson" in model or "NVIDIA" in model:
                return "jetson"
    except Exception:
        pass
    # Check CPU info
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(errors="ignore")
        if "GenuineIntel" in cpuinfo:
            return "intel"
        elif "AuthenticAMD" in cpuinfo:
            return "amd"
    except Exception:
        pass
    return "generic"


def build_gst_pipeline(source):
    """Build a GStreamer pipeline string for various video sources.

    Automatically detects the source type and constructs an appropriate
    GStreamer pipeline string. Supports camera devices, RTSP streams,
    video files, and custom GStreamer pipeline strings.

    Args:
        source: Video source specification. One of:

            - `int`: Camera device index (e.g. 0, 1).
            - `str` of digits: Camera device index as string (e.g. "0").
            - `str` starting with `rtsp://`: RTSP stream URL.
            - `str` (file path): Path to a video file.
            - `str` (pipeline): Custom GStreamer pipeline string.

    Returns:
        GStreamer pipeline string.

    Raises:
        ValueError: If the source type is unknown or the file is not found.
        FileNotFoundError: If the camera device path does not exist.

    Examples:
        >>> build_gst_pipeline(0)
        'v4l2src device=/dev/video0 ! videoscale ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink'

        >>> build_gst_pipeline("rtsp://example.com/stream")
        'rtspsrc location="rtsp://example.com/stream" latency=0 protocols=tcp ! decodebin ! videoconvert ! videoscale ! appsink name=sink'

        >>> build_gst_pipeline("/path/to/video.mp4")
        'filesrc location="/path/to/video.mp4" ! decodebin ! videoconvert ! videoscale ! video/x-raw, format=BGR ! appsink name=sink'

        >>> build_gst_pipeline("v4l2src ! videoconvert ! appsink")
        'v4l2src ! videoconvert ! appsink'
    """
    platform = _detect_platform()
    format = "BGR"  # Default format for OpenCV compatibility

    # ==================== CUSTOM GSTREAMER PIPELINE ====================
    # 4. if source is string and contains GStreamer elements, treat as custom pipeline
    if isinstance(source, str) and _is_gstreamer_pipeline(source):
        logger_get().info(f"Using custom GStreamer pipeline: {source}")
        return source

    # ==================== CAMERA SOURCE ====================
    # 1. if source is int or str but has digit, convert it to int
    if isinstance(source, int):
        device_index = source
    elif isinstance(source, str) and source.isdigit():
        device_index = int(source)
    else:
        # Not a camera source, skip to other checks
        device_index = None
    if device_index is not None:
        device = _detect_camera_type(device_index)
        logger_get().info(f"Detected platform: {platform}, camera type: {device}")
        if device == "rpi_csi" and platform == "raspberrypi":
            # Raspberry Pi CSI Camera
            if _check_element_exists("libcamerasrc"):
                logger_get().info("Using libcamerasrc for RPi CSI camera")
                return f"libcamerasrc ! videoconvert ! video/x-raw,format={format} ! appsink name=sink"
            else:
                logger_get().warning("libcamerasrc not available, trying v4l2src")
        # USB/Webcam (or RPi fallback)
        return f"v4l2src device=/dev/video{device_index} ! videoscale ! videoconvert ! video/x-raw,format={format} ! appsink name=sink"

    # ==================== RTSP SOURCE ====================
    # 3. if source is str and starts with rtsp
    elif isinstance(source, str) and source.lower().startswith("rtsp://"):
        logger_get().info(f"Building RTSP pipeline for: {source}")
        # application/x-rtp,media=video selects only the video RTP pad from rtspsrc,
        # preventing decodebin from receiving the audio stream and emitting an
        # unlinked audio pad that would propagate a not-linked error upstream.
        return (
            f'rtspsrc location="{source}" latency=0 protocols=tcp ! '
            f"application/x-rtp,media=video ! "
            f"decodebin ! videoconvert ! videoscale ! "
            f"video/x-raw,format={format} ! appsink name=sink"
        )

    # ==================== FILE SOURCE ====================
    # 2. if source is str (and not RTSP, not digits, not custom pipeline)
    elif isinstance(source, str) and os.path.exists(source):
        logger_get().info(f"Building file pipeline for: {source}")
        # Always use decodebin for maximum compatibility
        return (
            f'filesrc location="{source}" ! '
            f"decodebin ! videoconvert ! videoscale ! "
            f"video/x-raw, format={format} ! "
            f"appsink name=sink"
        )
    else:
        raise ValueError(f"Unknown source type or file not found: {source}")


def _is_gstreamer_pipeline(source: str) -> bool:
    """
    Detect if the source string is a custom GStreamer pipeline.
    Args:
        source: String to check
    Returns:
        True if the string appears to be a GStreamer pipeline
    """
    # Check for common GStreamer elements and patterns
    gst_indicators = [
        "!",  # Pipeline separator
        "src",  # Source elements
        "sink",  # Sink elements
        "videoconvert",  # Common video element
        "appsink",  # Common sink for applications
        "v4l2src",  # Video source
        "filesrc",  # File source
        "rtspsrc",  # RTSP source
        "decodebin",  # Decoder
        "videoscale",  # Video scaler
        "video/x-raw",  # Video format
        "audio/x-raw",  # Audio format
    ]
    # Must contain pipeline separator and at least one GStreamer element
    has_pipeline_sep = "!" in source
    has_gst_element = any(indicator in source.lower() for indicator in gst_indicators)
    # Additional check: should not look like a simple file path or URL
    is_url = source.startswith(("http://", "https://", "rtsp://", "rtmp://"))
    is_absolute_path = source.startswith("/")
    is_relative_path = source.startswith("./") or source.startswith("../")
    has_dot = "." in source
    has_one_dot = len(source.split(".")) == 2
    is_simple_path = (
        not is_url
        and not is_absolute_path
        and not is_relative_path
        and has_dot
        and has_one_dot
    )
    return has_pipeline_sep and has_gst_element and not is_simple_path
