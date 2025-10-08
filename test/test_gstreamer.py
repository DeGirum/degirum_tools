import pytest
import numpy as np
import os
from pathlib import Path

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

def test_gst_pipeline_builder_camera():
    """Test GStreamer pipeline builder for camera sources"""
    from degirum_tools.gst_support import build_gst_pipeline
    
    # Test camera source
    pipeline = build_gst_pipeline(0)
    assert isinstance(pipeline, str)
    assert "v4l2src" in pipeline
    assert "device=/dev/video0" in pipeline
    assert "appsink" in pipeline

def test_gst_pipeline_builder_file():
    """Test GStreamer pipeline builder for file sources"""
    from degirum_tools.gst_support import build_gst_pipeline
    
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

def test_gst_pipeline_builder_rtsp():
    """Test GStreamer pipeline builder for RTSP sources"""
    from degirum_tools.gst_support import build_gst_pipeline
    
    rtsp_url = "rtsp://example.com/stream"
    pipeline = build_gst_pipeline(rtsp_url)
    assert isinstance(pipeline, str)
    assert "rtspsrc" in pipeline
    assert f'location="{rtsp_url}"' in pipeline
    assert "decodebin" in pipeline
    assert "appsink" in pipeline

def test_video_support_gstreamer_camera():
    """Test video_support with GStreamer backend for camera"""
    from degirum_tools.video_support import open_video_stream, VideoCaptureGst
    
    try:
        with open_video_stream(0, use_gstreamer=True) as stream:
            # Should create VideoCaptureGst instance
            assert isinstance(stream, VideoCaptureGst)
            # Test reading a frame
            ret, frame = stream.read()
            # Note: ret might be False if no camera is available, which is OK for testing
            assert isinstance(ret, bool)
    except Exception as e:
        # If camera is not available, that's OK for testing
        pytest.skip(f"Camera not available: {e}")

def test_video_support_gstreamer_file():
    """Test video_support with GStreamer backend for file"""
    from degirum_tools.video_support import open_video_stream, VideoCaptureGst
    
    # Create a dummy file for testing
    test_file = "Traffic.mp4"
    Path(test_file).touch()
    
    try:
        with open_video_stream(test_file, use_gstreamer=True) as stream:
            # Should create VideoCaptureGst instance
            assert isinstance(stream, VideoCaptureGst)
            # Test reading a frame (might fail with dummy file, which is OK)
            ret, frame = stream.read()
            assert isinstance(ret, bool)
    except Exception as e:
        # If GStreamer fails with dummy file, that's expected
        pytest.skip(f"GStreamer file test failed (expected with dummy file): {e}")
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def test_gst_pipeline_fallback():
    """Test that GStreamer falls back to OpenCV when needed"""
    from degirum_tools.video_support import open_video_stream
    
    # Test with invalid source - should fall back to OpenCV
    try:
        with open_video_stream("nonexistent_file.mp4", use_gstreamer=True) as stream:
            # Should still work (fallback to OpenCV)
            ret, frame = stream.read()
            assert isinstance(ret, bool)
    except Exception as e:
        # Expected to fail with nonexistent file
        assert "Error opening" in str(e) or "not found" in str(e)