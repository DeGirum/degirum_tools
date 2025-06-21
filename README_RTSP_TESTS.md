# RTSP GStreamer Debugging Test Suite

This test suite helps isolate and debug GStreamer RTSP integration issues in the degirum_tools library. It provides multiple test scripts to identify the root cause of intermittent failures.

## Test Scripts Overview

### 1. `rtsp_gstreamer_test.py` - Comprehensive GStreamer Test
A comprehensive test suite that checks various aspects of GStreamer RTSP handling:
- GStreamer availability and initialization
- Simple RTSP pipeline connectivity
- RTSP with appsink frame capture
- Connection stability with multiple iterations
- Different output formats (BGR, RGB, I420, NV12)
- Different latency settings (0ms, 100ms, 200ms, 500ms, 1000ms)

**Usage:**
```bash
python rtsp_gstreamer_test.py <rtsp_url> [test_duration_seconds]
```

**Example:**
```bash
python rtsp_gstreamer_test.py rtsp://admin:admin123@192.168.0.194:554 30
```

### 2. `degirum_rtsp_test.py` - DeGirum-Specific Test
This script specifically mimics the degirum_tools RTSP implementation to isolate issues:
- Uses the exact same pipeline string as degirum_tools
- Mimics the VideoCaptureGst class behavior
- Tests multiple iterations to check for intermittent failures
- Provides detailed timing and error analysis

**Usage:**
```bash
python degirum_rtsp_test.py <rtsp_url> [iterations]
```

**Example:**
```bash
python degirum_rtsp_test.py rtsp://admin:admin123@192.168.0.194:554 10
```

### 3. `rtsp_comparison_test.py` - OpenCV vs GStreamer Comparison
Compares OpenCV and GStreamer RTSP handling to identify differences:
- Tests both OpenCV and GStreamer with the same RTSP stream
- Compares connection times, frame rates, and error rates
- Helps identify if the issue is GStreamer-specific or general RTSP problem

**Usage:**
```bash
python rtsp_comparison_test.py <rtsp_url> [test_duration]
```

**Example:**
```bash
python rtsp_comparison_test.py rtsp://admin:admin123@192.168.0.194:554 30
```

## Output Files

Each test script generates:

1. **Log file** (e.g., `rtsp_gstreamer_test.log`) - Detailed debug information
2. **Results file** (e.g., `rtsp_test_results.json`) - Structured test results

## Interpreting Results

### Success Indicators
- **Connection established**: Pipeline starts successfully
- **Frames captured**: At least some frames are received
- **Stable performance**: Consistent frame rates across iterations

### Failure Patterns to Look For

1. **Connection Failures**
   - Pipeline fails to start
   - Appsink not found
   - State transition failures

2. **Frame Capture Issues**
   - No samples received from appsink
   - Buffer mapping failures
   - Caps (capabilities) issues

3. **Intermittent Failures**
   - Some iterations succeed, others fail
   - Varying connection times
   - Inconsistent frame rates

4. **Resource Issues**
   - Memory leaks
   - Pipeline cleanup failures
   - GStreamer initialization problems

## Common Root Causes

### 1. Network/RTSP Server Issues
- **Symptoms**: Both OpenCV and GStreamer fail
- **Solutions**: Check network connectivity, RTSP server status, authentication

### 2. GStreamer Configuration Issues
- **Symptoms**: OpenCV works, GStreamer fails
- **Solutions**: Check GStreamer installation, missing plugins, pipeline syntax

### 3. Resource Management Issues
- **Symptoms**: Works initially, fails after multiple iterations
- **Solutions**: Check for memory leaks, proper cleanup, resource limits

### 4. Timing/Latency Issues
- **Symptoms**: Intermittent failures, varying performance
- **Solutions**: Adjust latency settings, buffer sizes, connection timeouts

## Debugging Workflow

1. **Start with comparison test** to see if it's a GStreamer-specific issue
2. **Run comprehensive test** to identify specific failure points
3. **Use degirum-specific test** to isolate issues in the exact implementation
4. **Analyze logs** for patterns and error messages
5. **Test with different parameters** (latency, formats, resolutions)

## Example Analysis

```bash
# Run comparison test first
python rtsp_comparison_test.py rtsp://admin:admin123@192.168.0.194:554 30

# If GStreamer fails but OpenCV works, run comprehensive test
python rtsp_gstreamer_test.py rtsp://admin:admin123@192.168.0.194:554 30

# If issues persist, run degirum-specific test
python degirum_rtsp_test.py rtsp://admin:admin123@192.168.0.194:554 10
```

## Troubleshooting Tips

1. **Check GStreamer installation**:
   ```bash
   gst-launch-1.0 --version
   gst-inspect-1.0 rtspsrc
   ```

2. **Test basic RTSP connectivity**:
   ```bash
   gst-launch-1.0 rtspsrc location=rtsp://your_url latency=0 ! fakesink
   ```

3. **Monitor system resources**:
   ```bash
   top -p $(pgrep python)
   ```

4. **Check for GStreamer errors**:
   ```bash
   export GST_DEBUG=3
   python your_test_script.py
   ```

## Next Steps

After running these tests:

1. **Identify the failure pattern** from the logs and results
2. **Check the specific error messages** in the log files
3. **Compare with OpenCV behavior** to isolate GStreamer issues
4. **Test with different RTSP streams** to see if it's stream-specific
5. **Modify the degirum_tools implementation** based on findings

The test results will help pinpoint whether the issue is:
- Network/RTSP server related
- GStreamer configuration related
- Resource management related
- Implementation specific to degirum_tools 