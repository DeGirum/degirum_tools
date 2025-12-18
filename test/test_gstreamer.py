import pytest
import os
import sys
from pathlib import Path

# Ensure we use the local development version, not the installed one
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_import_gi():
    """Test that gi module is available"""
    try:
        import gi
    except ImportError:
        pytest.skip("gi module not available")
    assert hasattr(gi, "require_version")


def test_import_gst():
    """Test that GStreamer is available"""
    import gi
    try:
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError):
        pytest.skip("Gst namespace not available")
    assert hasattr(Gst, "init")


def test_gst_init_and_version():
    """Test GStreamer initialization and version"""
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    if not Gst.is_initialized():
        Gst.init(None)
    version = Gst.version_string()
    assert isinstance(version, str)
    assert version.count('.') >= 2


def test_gst_pipeline_builder_file():
    """Test GStreamer pipeline builder for file sources"""
    # Import from local development version
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


def test_gst_pipeline_fallback():
    """Test that GStreamer falls back to OpenCV when needed"""
    # Import from local development version
    from degirum_tools.tools.video_support import open_video_stream

    # Test with invalid source - should fall back to OpenCV
    try:
        with open_video_stream("nonexistent_file.mp4", use_gstreamer=True) as stream:
            # Should still work (fallback to OpenCV)
            ret, frame = stream.read()
            assert isinstance(ret, bool)
    except Exception as e:
        # Expected to fail with nonexistent file
        assert "Error opening" in str(e) or "not found" in str(e) or "Unknown source type" in str(e)
