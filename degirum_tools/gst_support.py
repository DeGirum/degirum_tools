#!/usr/bin/env python3
"""
Simple GStreamer Pipeline Builder
Focus: Compatibility over optimization
"""

import os
import subprocess
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd: list, timeout: int = 5) -> Tuple[str, str, int]:
    """Run command and return stdout, stderr, returncode"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        # After running, check for lingering v4l2-ctl processes
        lingering = subprocess.run(["pgrep", "v4l2-ctl"], capture_output=True, text=True)
        if lingering.stdout.strip():
            print("Warning: v4l2-ctl process still running!")
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1


def detect_camera_type(device_index: int) -> str:
    """
    Detect camera type using multiple methods
    Returns: 'rpi_csi', 'usb', or 'unknown'
    """
    device_path = f"/dev/video{device_index}"
    if not os.path.exists(device_path):
        raise FileNotFoundError(f"Camera device {device_path} not found")
    # Method 1: Check v4l2-ctl device info
    stdout, stderr, ret = run_command(["v4l2-ctl", "-d", device_path, "--info"])
    if ret == 0:
        info_lower = stdout.lower()
        # Raspberry Pi indicators (more comprehensive list)
        rpi_indicators = [
            'unicam', 'csi', 'rp1-cfe', 'rp1_cfe', 'bcm2835', 'mmal',
            'raspberry', 'rpi', 'broadcom', 'brcm', 'vc4'
        ]
        for indicator in rpi_indicators:
            if indicator in info_lower:
                logger.info(f"Found RPi indicator '{indicator}' in device info")
                return "rpi_csi"
    else:
        logger.warning(f"v4l2-ctl failed for {device_path}: {stderr}")
    # Method 2: Check device path in /sys (fallback)
    try:
        device_name = Path(device_path).name  # e.g., 'video19'
        sys_path = f"/sys/class/video4linux/{device_name}/device"
        if os.path.exists(sys_path):
            # Read the real path
            real_path = os.readlink(sys_path).lower()
            # Check for RPi-specific paths
            rpi_path_indicators = ['platform/axi:csi', 'platform/soc/csi', 'platform/rp1', 'bcm2835', 'unicam', 'cfe']
            for indicator in rpi_path_indicators:
                if indicator in real_path:
                    logger.info(f"Found RPi path indicator '{indicator}' in sys path")
                    return "rpi_csi"
    except Exception as e:
        logger.debug(f"Sys path check failed: {e}")
    try:
        stdout, stderr, ret = run_command(["v4l2-ctl", "-d", device_path, "--list-formats"])
        if ret == 0:
            formats_lower = stdout.lower()
            # logger.info(f"v4l2-ctl formats for {device_path}: {stdout}")
            #  RPi cameras often have specific format patterns
            if any(indicator in formats_lower for indicator in ['bayer', 'rggb', 'grbg']):
                logger.info("Found raw Bayer format - likely RPi CSI camera")
                return "rpi_csi"
    except Exception as e:
        logger.debug(f"Format check failed: {e}")
    # Method 4: Check high device numbers (RPi cameras often get high numbers)
    if device_index >= 10:
        logger.info(f"High device number {device_index} - likely RPi camera")
        # But still need confirmation from other methods, so this is just a hint
    # Everything else is treated as USB/webcam
    logger.info(f"No RPi indicators found - assuming USB/Webcam for {device_path}")
    return "usb"


def check_element_exists(element_name: str) -> bool:
    """Check if a GStreamer element exists"""
    stdout, stderr, ret = run_command(["gst-inspect-1.0", element_name])
    return ret == 0


def detect_platform() -> str:
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
    platform = detect_platform()
    format = "BGR"  # Default format for OpenCV compatibility

    # ==================== CUSTOM GSTREAMER PIPELINE ====================
    # 4. if source is string and contains GStreamer elements, treat as custom pipeline
    if isinstance(source, str) and _is_gstreamer_pipeline(source):
        logger.info(f"Using custom GStreamer pipeline: {source}")
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
        device = detect_camera_type(device_index)
        print(f"Detected platform: {platform}, camera type: {device}")
        if device == "rpi_csi" and platform == "raspberrypi":
            # Raspberry Pi CSI Camera
            if check_element_exists("libcamerasrc"):
                logger.info("Using libcamerasrc for RPi CSI camera")
                return "libcamerasrc ! videoconvert ! video/x-raw,format={format} ! appsink name=sink"
            else:
                logger.warning("libcamerasrc not available, trying v4l2src")
        # USB/Webcam (or RPi fallback)
        return f"v4l2src device=/dev/video{device_index} ! videoscale ! videoconvert ! video/x-raw,format={format} ! appsink name=sink"

    # ==================== RTSP SOURCE ====================
    # 3. if source is str and starts with rtsp
    elif isinstance(source, str) and source.lower().startswith("rtsp://"):
        logger.info(f"Building RTSP pipeline for: {source}")
        # Use decodebin for automatic format handling
        return (
            f'rtspsrc location="{source}" latency=0 protocols=tcp ! '
            f'decodebin ! videoconvert ! videoscale ! '
            f'appsink name=sink'
        )

    # ==================== FILE SOURCE ====================
    # 2. if source is str (and not RTSP, not digits, not custom pipeline)
    elif isinstance(source, str) and os.path.exists(source):
        logger.info(f"Building file pipeline for: {source}")
        # Always use decodebin for maximum compatibility
        return (
            f'filesrc location="{source}" ! '
            f'decodebin ! videoconvert ! videoscale ! '
            f'video/x-raw, format={format} ! '
            f'appsink name=sink'
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
        '!',  # Pipeline separator
        'src',  # Source elements
        'sink',  # Sink elements
        'videoconvert',  # Common video element
        'appsink',  # Common sink for applications
        'v4l2src',  # Video source
        'filesrc',  # File source
        'rtspsrc',  # RTSP source
        'decodebin',  # Decoder
        'videoscale',  # Video scaler
        'video/x-raw',  # Video format
        'audio/x-raw',  # Audio format
    ]
    # Must contain pipeline separator and at least one GStreamer element
    has_pipeline_sep = '!' in source
    has_gst_element = any(indicator in source.lower() for indicator in gst_indicators)
    # Additional check: should not look like a simple file path or URL
    is_simple_path = (not source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')) and not source.startswith('/') and not source.startswith('./') and not source.startswith('../') and '.' in source and len(source.split('.')) == 2)
    return has_pipeline_sep and has_gst_element and not is_simple_path
