import degirum as dg
import degirum_tools

# =============================================================================
# CONFIGURATION - Change these values as needed
# =============================================================================

# Video source: can be camera index (0, 1, 2...) or video file path
SOURCE = 0  # Change to "Traffic.mp4" for video file, or 0, 1, 2... for camera
# SOURCE = "v4l2src device=/dev/video0 ! videoscale ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink"
# SOURCE = "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink name=sink"
# SOURCE = "v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480 ! videoconvert ! appsink name=sink"
# SOURCE = "v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480 ! jpegdec ! videoconvert ! appsink name=sink"
# SOURCE = "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink name=sink"
# SOURCE="v4l2src device=/dev/video0 ! video/x-raw,format=YUY2 ! videoconvert ! tee name=t ! queue ! videoscale ! video/x-raw,width=320,height=240 ! appsink name=sink"
# SOURCE="v4l2src device=/dev/video0 ! capsfilter caps="video/x-raw,format=YUY2,width=640,height=480,framerate=30/1" ! videoconvert ! appsink name=sink sync=false max-buffers=1 drop=true"
# SOURCE="v4l2src device=/dev/video0 ! video/x-raw,format=YUY2 ! videoconvert ! queue max-size-buffers=5 leaky=downstream ! videoscale ! video/x-raw,width=512,height=384 ! appsink name=sink"
# SOURCE = "v4l2src device=/dev/video0 ! video/x-raw,format=YUY2 ! videoconvert ! queue max-size-buffers=5 leaky=downstream ! videoscale ! video/x-raw,width=512,height=384 ! appsink name=sink"
# Model configuration
MODEL_NAME = 'yolov8n_relu6_coco--640x640_quant_openvino_multidevice_1'
TOKEN = "dg_9FX1BGergbVpB8YfhHhhGJEfagbL339ffPBKk"

# =============================================================================

print("Loading model...")
model = dg.load_model(
    model_name=MODEL_NAME,
    inference_host_address='@cloud',
    zoo_url='degirum/intel',
    token=TOKEN,
)

print(f"Starting inference on source: {SOURCE}")
print("Press Ctrl+C to stop\n")

try:
    for inference_result in degirum_tools.predict_stream(model, SOURCE, "gstream"):
        # Print detection results
        print(f"Detections: {len(inference_result.results)}")
        for i, detection in enumerate(inference_result.results):
            # Handle different detection result formats
            if hasattr(detection, 'label') and hasattr(detection, 'score'):
                print(f"  {i + 1}. {detection.label} (confidence: {detection.score:.2f})")
            elif isinstance(detection, dict):
                label = detection.get('label', 'unknown')
                score = detection.get('score', 0.0)
                print(f"  {i + 1}. {label} (confidence: {score:.2f})")
            else:
                print(f"  {i + 1}. {detection}")
        print("-" * 40)

except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Error: {e}")
