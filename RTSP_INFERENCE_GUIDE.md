# RTSP Multi-Model Inference Guide

This guide explains how to use the separate RTSP server and inference scripts to test parallel inference on multiple RTSP streams.

## Overview

The solution consists of two separate scripts:
1. **`start_rtsp_servers.py`** - Starts RTSP servers
2. **`rtsp_multi_model_inference.py`** - Runs inference on RTSP streams

## Quick Start

### Step 1: Start RTSP Servers

```bash
# Start 3 RTSP servers on default ports (8554-8556)
python start_rtsp_servers.py 3

# Start 5 RTSP servers on custom ports (9000-9004)
python start_rtsp_servers.py 5 9000
```

The servers will start and show:
```
============================================================
RTSP SERVER LAUNCHER
============================================================
Starting 3 RTSP servers on ports 8554 - 8556
Press Ctrl+C to stop all servers...

Starting 3 RTSP servers...
Started RTSP server on port 8554
Started RTSP server on port 8555
Started RTSP server on port 8556

Started 3 RTSP servers.
RTSP URLs:
  rtsp://localhost:8554/video0
  rtsp://localhost:8555/video0
  rtsp://localhost:8556/video0

Now you can run: python rtsp_multi_model_inference.py <num_streams> <num_models>
```

### Step 2: Run Inference

In a new terminal, run the inference script:

```bash
# Run inference on 3 streams with 2 models
python rtsp_multi_model_inference.py 3 2

# Run inference on 5 streams with 3 models, custom port range
python rtsp_multi_model_inference.py 5 3 9000

# Run inference with custom frame limit
python rtsp_multi_model_inference.py 3 2 8554 200
```

## Architecture

The inference script uses the exact same logic as the notebook:

```
RTSP Sources → Preprocessors → AI Detectors → Result Combiner → FPS Monitors → Display
     ↓              ↓              ↓              ↓              ↓           ↓
   Stream 1     Resize/Prep    Model 1-4      Combine All    Monitor FPS  Show Results
   Stream 2     Resize/Prep    Model 1-4      Combine All    Monitor FPS  Show Results
   Stream N     Resize/Prep    Model 1-4      Combine All    Monitor FPS  Show Results
```

## Supported Models

The inference script uses compatible YOLO models with 512x512 input size:

1. `yolo_v5s_person_det--512x512_quant_n2x_orca1_1` - Person detection
2. `yolo_v5s_face_det--512x512_quant_n2x_orca1_1` - Face detection
3. `yolo_v5n_car_det--512x512_quant_n2x_orca1_1` - Car detection
4. `yolo_v5s_hand_det--512x512_quant_n2x_orca1_1` - Hand detection

## Usage Examples

### Basic Testing
```bash
# Terminal 1: Start servers
python start_rtsp_servers.py 2

# Terminal 2: Run inference
python rtsp_multi_model_inference.py 2 1
```

### Performance Testing
```bash
# Terminal 1: Start servers
python start_rtsp_servers.py 5 9000

# Terminal 2: Run inference
python rtsp_multi_model_inference.py 5 3 9000 500
```

### Custom Configuration
```bash
# Terminal 1: Start servers on custom ports
python start_rtsp_servers.py 3 10000

# Terminal 2: Run inference with custom parameters
python rtsp_multi_model_inference.py 3 2 10000 300
```

## Output Example

```
================================================================================
RTSP MULTI-MODEL INFERENCE
Streams: 3
Models: 2
Port Range: 8554 - 8556
Max Frames per Stream: 300
================================================================================

Using models: ['yolo_v5s_person_det--512x512_quant_n2x_orca1_1', 'yolo_v5s_face_det--512x512_quant_n2x_orca1_1']
RTSP URLs: ['rtsp://localhost:8554/video0', 'rtsp://localhost:8555/video0', 'rtsp://localhost:8556/video0']

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

Stream 0 (Port 8554):
  Frames: 300
  Duration: 45.23s
  Average FPS: 6.63
  Peak FPS: 8.12

Stream 1 (Port 8555):
  Frames: 300
  Duration: 42.18s
  Average FPS: 7.11
  Peak FPS: 8.45

Stream 2 (Port 8556):
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
- GStreamer 1.0 (for RTSP servers)

### Authentication
- DeGirum Cloud API token (for cloud inference)
- Configured in `env.ini` or environment variables

## Troubleshooting

### RTSP Server Issues
1. **GStreamer not found**: Install GStreamer 1.0
2. **Port already in use**: Use different base port
3. **Permission denied**: Run with appropriate permissions

### Inference Issues
1. **Connection refused**: Ensure RTSP servers are running
2. **Model loading errors**: Check internet connection and API token
3. **Performance issues**: Reduce number of streams/models

### Common Commands
```bash
# Check if GStreamer is installed
gst-launch-1.0 --version

# Check if ports are in use
netstat -tuln | grep 8554

# Kill processes on specific ports
sudo lsof -ti:8554 | xargs kill -9
```

## Advanced Usage

### Custom RTSP URLs
You can modify the inference script to use custom RTSP URLs:

```python
# In rtsp_multi_model_inference.py, change this line:
rtsp_urls = [
    "rtsp://your-server:8554/stream1",
    "rtsp://your-server:8555/stream2",
    "rtsp://your-server:8556/stream3",
]
```

### Different Models
Modify the model list in the inference script:

```python
model_names = [
    "your_custom_model--512x512_quant_n2x_orca1_1",
    # Add more models...
]
```

### Hardware Configuration
Change the inference location:

```python
hw_location = "@local"  # Local inference
# hw_location = "@cloud"  # Cloud inference
# hw_location = "192.168.1.100"  # AI Server
```

## Workflow

1. **Start RTSP servers** in one terminal
2. **Run inference** in another terminal
3. **Monitor results** in real-time
4. **Stop servers** when done (Ctrl+C)

This approach keeps the RTSP server creation separate from the inference logic, making it easier to test with different RTSP sources and configurations. 