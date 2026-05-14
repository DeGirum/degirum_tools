#
# video_capture.py: GStreamer-based video capture class
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# Implements VideoCaptureGst: a cv2.VideoCapture-compatible class backed by GStreamer

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


class VideoCaptureGst:
    """GStreamer-based video capture class that mimics cv2.VideoCapture interface."""

    _gst_initialized = False

    @staticmethod
    def _ensure_gst():
        """Import and initialize GStreamer if not already done.

        Raises:
            ImportError: If GStreamer Python bindings are not available.
        """
        if VideoCaptureGst._gst_initialized:
            return
        try:
            from . import setup_gst_environment

            setup_gst_environment()
            VideoCaptureGst._gst_initialized = True
        except Exception as e:
            raise ImportError("GStreamer Python bindings (gi) not available") from e

    def __init__(self, pipeline_str):
        """Initialize GStreamer pipeline from string.

        Args:
            pipeline_str: GStreamer pipeline string, must include an appsink named "sink".
        """
        self._ensure_gst()

        from gi.repository import Gst
        from .pipeline_handler import GstPipelineHandler

        self._Gst = Gst

        try:
            self._handler = GstPipelineHandler(
                pipeline_str,
                appsink_names=["sink"],
                appsink_queue_maxsize=5,
                appsink_queue_drop=True,
                appsink_async_preroll=True,
            )
        except ValueError as e:
            raise Exception(
                f"Invalid GStreamer pipeline (no appsink): {pipeline_str}"
            ) from e
        except Exception as e:
            raise Exception(f"Invalid GStreamer pipeline: {pipeline_str}") from e

        self._handler.start(wait_timeout_s=15.0)

        self._appsink = self._handler.appsinks["sink"]

        self._running = True
        self._initialized = False
        self._frame_format = None
        self._frame_width: Optional[int] = None
        self._frame_height: Optional[int] = None
        self._frame_channels: Optional[int] = None
        self._conversion_func = None

    def _get_format_info(
        self, format_str: str, width: int, height: int
    ) -> tuple[int, int]:
        """Get channel count and expected buffer size for a given format.
        Args:
            format_str: GStreamer format string (e.g., 'BGR', 'RGB', 'I420')
            width: Frame width
            height: Frame height
        Returns:
            (channels, expected_size): Number of channels and expected buffer size
        """
        if not format_str:
            # Default to 3 channels if format is unknown
            return 3, width * height * 3
        # Common format mappings
        format_info = {
            # 3-channel formats
            "BGR": (3, width * height * 3),
            "RGB": (3, width * height * 3),
            "BGRx": (4, width * height * 4),
            "RGBx": (4, width * height * 4),
            "BGRA": (4, width * height * 4),
            "RGBA": (4, width * height * 4),
            # Grayscale
            "GRAY8": (1, width * height),
            "GRAY16_LE": (1, width * height * 2),
            "GRAY16_BE": (1, width * height * 2),
            # YUV formats (planar)
            "I420": (1, width * height * 3 // 2),  # 4:2:0 planar
            "YV12": (1, width * height * 3 // 2),  # 4:2:0 planar
            "NV12": (1, width * height * 3 // 2),  # 4:2:0 semi-planar
            "NV21": (1, width * height * 3 // 2),  # 4:2:0 semi-planar
            # Other common formats
            "YUY2": (2, width * height * 2),  # 4:2:2 packed
            "UYVY": (2, width * height * 2),  # 4:2:2 packed
        }
        return format_info.get(format_str, (3, width * height * 3))

    def _initialize_frame_processing(self, sample):
        """Initialize frame processing parameters from the first frame."""
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        self._frame_width = structure.get_value("width")
        self._frame_height = structure.get_value("height")
        format_str = structure.get_string("format")
        self._frame_format = format_str
        # Calculate format info once
        self._frame_channels, _ = self._get_format_info(
            format_str or "", self._frame_width, self._frame_height
        )
        # Determine conversion function once
        self._conversion_func = self._get_conversion_function(format_str)
        self._initialized = True

    def _get_conversion_function(self, format_str):
        """Get the appropriate conversion function for the format."""
        if format_str in ["RGB", "RGBx"]:
            return lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif format_str in ["I420", "YV12"]:
            return lambda frame: cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        elif format_str == "NV12":
            return lambda frame: cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
        elif format_str == "NV21":
            return lambda frame: cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV21)
        elif format_str in ["RGBA", "RGBx"]:
            return lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif format_str in ["YUY2"]:
            return lambda frame: cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        elif format_str in ["UYVY"]:
            return lambda frame: cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)
        elif format_str in ["BGRA"]:
            return lambda frame: frame[:, :, :3]  # Remove alpha channel
        else:
            # No conversion needed for BGR, BGRx, or unknown formats
            return None

    def _convert_frame(self, frame):
        """Apply the pre-determined conversion to the frame."""
        if self._conversion_func:
            return self._conversion_func(frame)
        return frame

    def read(self):
        """Read a frame from the GStreamer pipeline.

        Returns:
            (bool, np.ndarray): Success flag and frame data
        """
        if not self._running:
            return False, None
        sample = self._appsink.queue.get()
        if sample is None:
            self._running = False
            return False, None

        # Initialize frame processing on first frame
        if not self._initialized:
            self._initialize_frame_processing(sample)

        # Ensure frame dimensions are properly initialized
        if (
            self._frame_width is None
            or self._frame_height is None
            or self._frame_channels is None
        ):
            raise RuntimeError("Frame dimensions not properly initialized")

        buf = sample.get_buffer()

        success, mapinfo = buf.map(self._Gst.MapFlags.READ)
        if not success:
            return False, None

        try:
            # Use cached format info - much faster!
            if self._frame_format in ("NV12", "NV21", "I420", "YV12"):
                # YUV planar/semi-planar: buffer is H*W*1.5 bytes
                frame: np.ndarray = np.ndarray(
                    (self._frame_height * 3 // 2, self._frame_width),
                    buffer=mapinfo.data,
                    dtype=np.uint8,
                )
            elif self._frame_channels == 1:
                frame = np.ndarray(
                    (self._frame_height, self._frame_width),
                    buffer=mapinfo.data,
                    dtype=np.uint8,
                )
            elif self._frame_channels == 3:
                frame = np.ndarray(
                    (self._frame_height, self._frame_width, 3),
                    buffer=mapinfo.data,
                    dtype=np.uint8,
                )
            elif self._frame_channels == 4:
                frame = np.ndarray(
                    (self._frame_height, self._frame_width, 4),
                    buffer=mapinfo.data,
                    dtype=np.uint8,
                )
            else:
                frame = np.ndarray(
                    (self._frame_height, self._frame_width, self._frame_channels),
                    buffer=mapinfo.data,
                    dtype=np.uint8,
                )
            # Apply pre-determined conversion
            frame = self._convert_frame(frame)
            return True, frame
        finally:
            buf.unmap(mapinfo)

    def get(self, prop: int):
        """Get capture properties (mimics cv2.VideoCapture.get).

        Args:
            prop: OpenCV property constant (e.g., cv2.CAP_PROP_FRAME_WIDTH)

        Returns:
            Property value or None if not available
        """

        caps = self._appsink.sink_pad.get_current_caps()
        if not caps:
            return None

        structure = caps.get_structure(0)

        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return structure.get_value("width")
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return structure.get_value("height")
        elif prop == cv2.CAP_PROP_FPS:
            framerate = structure.get_fraction("framerate")
            if framerate:
                return framerate.value_numerator / framerate.value_denominator
            return None
        elif prop == cv2.CAP_PROP_FRAME_COUNT:
            duration = self._handler.pipeline.query_duration(self._Gst.Format.TIME)
            if duration[0]:
                fps = self.get(cv2.CAP_PROP_FPS) or 30.0
                return int((duration[1] / self._Gst.SECOND) * fps)
            return 0
        return None

    def set(self, prop: int, value: float) -> bool:
        """Set capture properties (mimics cv2.VideoCapture.set).

        Note: GStreamer pipelines are typically configured at creation time.
        Most properties cannot be changed dynamically after the pipeline is created.
        This method provides limited support for interface compatibility.

        Args:
            prop: OpenCV property constant (e.g., cv2.CAP_PROP_FPS)
            value: Property value to set

        Returns:
            bool: True if property was acknowledged, False if not supported
        """
        # GStreamer pipelines are configured at creation time via the pipeline string
        # Most properties (like FPS) are set in the pipeline and cannot be changed dynamically
        # For FPS override, it's typically handled in the pipeline string or metadata
        if prop == cv2.CAP_PROP_FPS:
            # FPS is set in the pipeline configuration, so we can't change it here
            # Return True to indicate "acknowledged" but don't actually change anything
            # The actual FPS comes from the pipeline configuration
            return True
        # For other properties, return False (not supported)
        return False

    def isOpened(self):
        """Check if the capture is opened.

        Returns:
            bool: True if pipeline is running
        """
        return self._running

    def release(self):
        """Release the GStreamer pipeline."""
        if self._running:
            self._handler.stop()
            self._running = False
