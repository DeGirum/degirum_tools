from degirum_tools.video_support import open_video_stream


def test_gstreamer_pipeline():
    # Test with a sample GStreamer pipeline
    gst_pipeline = "Traffic.mp4"
    model = None  # Replace with a mock or actual model if needed
    try:
        with open_video_stream(gst_pipeline, source_type='gstream', model=model) as stream:
            assert stream.isOpened(), "Failed to open GStreamer pipeline"
            ret, frame = stream.read()
            assert ret, "Failed to read a frame from GStreamer pipeline"
            assert frame is not None, "Frame is None from GStreamer pipeline"
    except ImportError:
        print("GStreamer not available, skipping test")


def test_invalid_gstreamer_pipeline():
    invalid_pipeline = "invalid_pipeline ! appsink"
    model = None  # Replace with a mock or actual model if needed
    try:
        with open_video_stream(invalid_pipeline, source_type='gstream', model=model):
            pass  # This should not execute if the pipeline is invalid
    except Exception as e:
        # Match specific error messages
        assert "Invalid GStreamer pipeline" in str(e) or "file not found" in str(e), (
            f"Unexpected error message: {e}"
        )
    else:
        assert False, "Did not raise error for invalid GStreamer pipeline"


if __name__ == "__main__":
    print("Running tests...")
    test_gstreamer_pipeline()
    test_invalid_gstreamer_pipeline()
    print("All tests completed.")
