# Multi-Stream Multi-Model AI Inference Simulator

This project provides advanced scripts to simulate processing multiple video streams with multiple AI models in parallel using the DeGirum tools streaming framework. It combines the concepts from the multi-RTSP simulator and the multi-camera/multi-model notebook to create a comprehensive testing and benchmarking solution.

## Overview

The simulator allows you to:
- Process multiple video streams simultaneously
- Run multiple AI models on each stream
- Capture detailed FPS metrics for each stream
- Visualize results in real-time
- Benchmark performance across different configurations

## Files

### 1. `multi_stream_multi_model_simulator.py` (Recommended)
- **Purpose**: Main simulator using video files for easy testing
- **Features**: 
  - Uses sample video files from DeGirum examples
  - No external dependencies (no RTSP servers needed)
  - Real-time FPS monitoring per stream
  - Multi-window display with AI overlays
  - Comprehensive performance reporting

### 2. `multi_rtsp_multi_model_simulator.py`
- **Purpose**: Advanced simulator using RTSP streams
- **Features**:
  - Generates RTSP server launcher script
  - Supports real RTSP stream simulation
  - Same multi-model inference capabilities
  - Requires GStreamer for RTSP servers

### 3. `multi_rtsp_simulator.py` (Original)
- **Purpose**: Basic multi-RTSP stream simulator
- **Features**: Simple parallel processing without streaming framework

## Quick Start

### Using Video Files (Recommended)

```bash
# Run 3 streams with 2 models, process 300 frames each
python multi_stream_multi_model_simulator.py 3 2

# Run 5 streams with 3 models, process 200 frames each
python multi_stream_multi_model_simulator.py 5 3 200
```

### Using RTSP Streams (Advanced)

```bash
# Run 3 streams with 2 models
python multi_rtsp_multi_model_simulator.py 3 2

# The script will generate and optionally start RTSP servers
```

## Architecture

The simulator implements a sophisticated streaming pipeline:

```
Video Sources → Preprocessors → AI Detectors → Result Combiner → FPS Monitors → Display
     ↓              ↓              ↓              ↓              ↓           ↓
   Stream 1     Resize/Prep    Model 1-4      Combine All    Monitor FPS  Show Results
   Stream 2     Resize/Prep    Model 1-4      Combine All    Monitor FPS  Show Results
   Stream N     Resize/Prep    Model 1-4      Combine All    Monitor FPS  Show Results
```

### Key Components

1. **VideoSourceGizmo**: Reads video streams (files or RTSP)
2. **AiPreprocessGizmo**: Resizes and preprocesses frames for AI models
3. **AiSimpleGizmo**: Runs AI inference with multi-input support
4. **AiResultCombiningGizmo**: Combines results from multiple models
5. **FPSMonitorGizmo**: Custom gizmo for detailed FPS tracking
6. **VideoDisplayGizmo**: Multi-window display with AI overlays

## Supported Models

The simulator uses compatible YOLO models with 512x512 input size:

1. `yolo_v5s_person_det--512x512_quant_n2x_orca1_1` - Person detection
2. `yolo_v5s_face_det--512x512_quant_n2x_orca1_1` - Face detection
3. `yolo_v5n_car_det--512x512_quant_n2x_orca1_1` - Car detection
4. `yolo_v5s_hand_det--512x512_quant_n2x_orca1_1` - Hand detection

## Performance Metrics

The simulator captures comprehensive performance data:

### Per Stream Metrics
- **Frame Count**: Total frames processed
- **Duration**: Processing time
- **Average FPS**: Overall frame rate
- **Peak FPS**: Maximum achieved frame rate

### Combined Metrics
- **Total Duration**: Overall execution time
- **Successful Streams**: Number of working streams
- **Total Frames**: Combined frame count
- **Combined FPS**: Sum of all stream FPS
- **Average FPS per Stream**: Efficiency metric

## Configuration Options

### Hardware Location
```python
hw_location = "@cloud"  # DeGirum Cloud Platform
# hw_location = "@local"  # Local machine
# hw_location = "192.168.1.100"  # AI Server IP
```

### Model Zoo
```python
model_zoo_url = "degirum/public"  # Public model zoo
# model_zoo_url = "degirum/hailo"  # Hailo models
```

### Video Sources
For video file simulator:
- Sample videos from DeGirum examples
- Supports up to 5 concurrent streams
- Automatic URL resolution

For RTSP simulator:
- Custom RTSP URLs
- Configurable port ranges
- GStreamer-based server generation

## Usage Examples

### Basic Testing
```bash
# Test with 2 streams and 1 model
python multi_stream_multi_model_simulator.py 2 1

# Test with 3 streams and 2 models
python multi_stream_multi_model_simulator.py 3 2
```

### Performance Benchmarking
```bash
# Benchmark with 5 streams and 4 models
python multi_stream_multi_model_simulator.py 5 4 500

# Short test for quick validation
python multi_stream_multi_model_simulator.py 2 2 100
```

### RTSP Testing
```bash
# Start RTSP servers and run inference
python multi_rtsp_multi_model_simulator.py 3 2 8554 300
```

## Output Example

```
================================================================================
MULTI STREAM MULTI-MODEL SIMULATOR
Streams: 3
Models: 2
Max Frames per Stream: 300
================================================================================

Using models: ['yolo_v5s_person_det--512x512_quant_n2x_orca1_1', 'yolo_v5s_face_det--512x512_quant_n2x_orca1_1']
Video sources: ['https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/WalkingPeople.mp4', ...]

Loading AI models...
✓ All models loaded successfully

Creating 3 video sources...
Creating 3 preprocessors...
Creating 2 AI detectors...
Creating result combiner...
Creating 3 FPS monitors...
Creating display...
Connecting pipeline...

Starting composition...

================================================================================
RESULTS SUMMARY
================================================================================

Stream 0 (WalkingPeople.mp4):
  Frames: 300
  Duration: 45.23s
  Average FPS: 6.63
  Peak FPS: 8.12

Stream 1 (Traffic.mp4):
  Frames: 300
  Duration: 42.18s
  Average FPS: 7.11
  Peak FPS: 8.45

Stream 2 (WalkingPeople.mp4):
  Frames: 300
  Duration: 44.91s
  Average FPS: 6.68
  Peak FPS: 8.23

================================================================================
COMBINED SUMMARY
================================================================================
Total Duration: 45.23 seconds
Successful Streams: 3/3
Total Frames Processed: 900
Combined FPS: 20.42
Average FPS per Stream: 6.81
Efficiency: 6.81 FPS per stream

Test completed successfully!
```

## Requirements

### Software Dependencies
- Python 3.7+
- degirum-tools
- degirum PySDK
- OpenCV
- NumPy

### For RTSP Simulator
- GStreamer 1.0
- gst-launch-1.0 command available

### Authentication
- DeGirum Cloud API token (for cloud inference)
- Configured in `env.ini` or environment variables

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check internet connection for cloud models
   - Verify API token configuration
   - Ensure model names are correct

2. **Video Source Errors**
   - Check video file URLs are accessible
   - For RTSP: ensure servers are running
   - Verify network connectivity

3. **Performance Issues**
   - Reduce number of streams/models
   - Use local inference instead of cloud
   - Check system resources

4. **Display Issues**
   - Ensure X11 forwarding (for remote execution)
   - Check OpenCV display support
   - Use headless mode if needed

### Debug Mode
Add debug prints by modifying the script:
```python
# Add to any section for debugging
print(f"Debug: {variable}")
```

## Advanced Usage

### Custom Models
Modify the model list in the script:
```python
model_names = [
    "your_custom_model--512x512_quant_n2x_orca1_1",
    # Add more models...
]
```

### Custom Video Sources
For video file simulator:
```python
video_sources = [
    "path/to/your/video1.mp4",
    "path/to/your/video2.mp4",
    # Add more sources...
]
```

### Custom FPS Monitoring
Extend the FPSMonitorGizmo class for additional metrics:
```python
class CustomFPSMonitorGizmo(FPSMonitorGizmo):
    def run(self):
        # Add custom monitoring logic
        super().run()
```

## Contributing

To extend the simulator:

1. **Add New Gizmos**: Create custom gizmos for specific functionality
2. **Support More Models**: Add model compatibility checks
3. **Enhanced Metrics**: Extend FPS monitoring capabilities
4. **New Video Sources**: Add support for different stream types

## License

This project follows the same license as the DeGirum tools package. 