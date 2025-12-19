#
# test_gstreamer.py: unit tests for GStreamer integration
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements unit tests to test GStreamer pipeline building and integration
#

import pytest
import os
import sys
from pathlib import Path

# Ensure we use the local development version, not the installed one
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_gstreamer():
    """Test GStreamer integration: imports, initialization, pipeline building, and fallback"""

    # Test 1: Test that gi module is available
    try:
        import gi
    except ImportError:
        pytest.skip("gi module not available")
    assert hasattr(gi, "require_version")

    # Test 2: Test that GStreamer is available
    try:
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError):
        pytest.skip("Gst namespace not available")
    assert hasattr(Gst, "init")

    # Test 3: Test GStreamer initialization and version
    if not Gst.is_initialized():
        Gst.init(None)
    version = Gst.version_string()
    assert isinstance(version, str)
    assert version.count('.') >= 2

    # Test 4: Test GStreamer pipeline builder for file sources
    from degirum_tools.tools.gst_support import build_gst_pipeline

    # Create a dummy file for testing
    test_file = "test_video.mp4"
    Path(test_file).touch()

    try:
        pipeline = build_gst_pipeline(test_file)
        assert isinstance(pipeline, str)
        assert "filesrc" in pipeline
        assert f'location="{test_file}"' in pipeline
        assert "decodebin" in pipeline
        assert "appsink" in pipeline
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

    # Test 5: Test that GStreamer falls back to OpenCV when needed
    from degirum_tools.tools.video_support import open_video_stream

    # Test with invalid source - should raise an exception
    with pytest.raises(Exception) as exc_info:
        with open_video_stream("nonexistent_file.mp4", use_gstreamer=True) as stream:
            ret, frame = stream.read()

    # Verify the error message contains expected text
    error_msg = str(exc_info.value)
    assert "Error opening" in error_msg or "not found" in error_msg or "Unknown source type" in error_msg
