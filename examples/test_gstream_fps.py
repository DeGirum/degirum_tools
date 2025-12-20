import os
from degirum_tools import FPSMeter
from degirum_tools.tools.video_support import open_video_stream, video_source

# =============================================================================
# CONFIGURATION - Change these values as needed
# =============================================================================

# Video source: can be camera index (0, 1, 2...) or video file path
# For gstreamer pipeline, use a string with the full pipeline
# Use absolute path or path relative to script location
SOURCE = os.path.join(os.path.dirname(__file__), 'TrafficHD.mp4')  # Video file in same directory as script
# SOURCE = 0  # Camera index
# SOURCE = "v4l2src device=/dev/video0 ! videoscale ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink"  # GStreamer pipeline
# Example gstreamer pipeline:
# SOURCE = "v4l2src device=/dev/video0 ! videoscale ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink"

# =============================================================================

print(f"Starting video stream from source: {SOURCE}")
#print("Using GStreamer backend")
print("Press Ctrl+C to stop\n")

# Initialize FPS meter
fps = FPSMeter()

try:
    # Open video stream with GStreamer backend
    with open_video_stream(SOURCE, use_gstreamer=True) as stream:
        # Iterate over frames
        for frame in video_source(stream):
            # Record frame and get current FPS
            current_fps = fps.record()
            
            # Print FPS (overwrite previous line for cleaner output)
            print(f"\rFPS: {current_fps:.2f}", end="", flush=True)

except KeyboardInterrupt:
    print("\n\nStopped by user")
except Exception as e:
    print(f"\nError: {e}")

