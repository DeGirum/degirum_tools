#!/usr/bin/env python3
"""
RTSP GStreamer Test Script
==========================

This script provides a comprehensive test suite for debugging GStreamer RTSP integration issues.
It includes multiple test scenarios with detailed logging to help identify root causes of failures.

Usage:
    python rtsp_gstreamer_test.py [rtsp_url] [test_duration_seconds]

Example:
    python rtsp_gstreamer_test.py rtsp://admin:admin123@192.168.0.194:554 30
"""

import sys
import time
import logging
import traceback
import threading
from typing import Optional, Dict, Any
import numpy as np

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rtsp_gstreamer_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GStreamerRTSPTester:
    """Comprehensive RTSP GStreamer testing class"""
    
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.test_results = {}
        self.frame_count = 0
        self.error_count = 0
        self.start_time = None
        
    def test_gstreamer_availability(self) -> bool:
        """Test if GStreamer is available and working"""
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst, GLib
            Gst.init(None)
            logger.info("✓ GStreamer is available and initialized")
            return True
        except ImportError as e:
            logger.error(f"✗ GStreamer not available: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ GStreamer initialization failed: {e}")
            return False
    
    def test_simple_rtsp_pipeline(self) -> bool:
        """Test basic RTSP pipeline without appsink"""
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst, GLib
            
            # Simple pipeline to test RTSP connection
            pipeline_str = f"rtspsrc location={self.rtsp_url} latency=0 ! fakesink"
            logger.info(f"Testing simple pipeline: {pipeline_str}")
            
            pipeline = Gst.parse_launch(pipeline_str)
            pipeline.set_state(Gst.State.PLAYING)
            
            # Wait for pipeline to start
            state_change_result = pipeline.get_state(10 * Gst.SECOND)
            if state_change_result[1] != Gst.State.PLAYING:
                logger.error(f"✗ Pipeline failed to start. State: {state_change_result[1]}")
                pipeline.set_state(Gst.State.NULL)
                return False
            
            logger.info("✓ Simple RTSP pipeline started successfully")
            
            # Let it run for a few seconds
            time.sleep(3)
            
            pipeline.set_state(Gst.State.NULL)
            return True
            
        except Exception as e:
            logger.error(f"✗ Simple RTSP pipeline test failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def test_rtsp_with_appsink(self, width: int = 640, height: int = 480, duration: int = 10) -> bool:
        """Test RTSP pipeline with appsink for frame capture"""
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst, GLib
            
            # RTSP pipeline with appsink
            pipeline_str = (
                f"rtspsrc location={self.rtsp_url} latency=0 ! "
                f"decodebin ! videoconvert ! videoscale ! "
                f"video/x-raw,width={width},height={height},format=BGR ! "
                f"appsink name=sink emit-signals=true"
            )
            
            logger.info(f"Testing RTSP pipeline with appsink: {pipeline_str}")
            
            pipeline = Gst.parse_launch(pipeline_str)
            appsink = pipeline.get_by_name("sink")
            
            if not appsink:
                logger.error("✗ Appsink not found in pipeline")
                return False
            
            pipeline.set_state(Gst.State.PLAYING)
            
            # Wait for pipeline to start
            state_change_result = pipeline.get_state(10 * Gst.SECOND)
            if state_change_result[1] != Gst.State.PLAYING:
                logger.error(f"✗ Pipeline failed to start. State: {state_change_result[1]}")
                pipeline.set_state(Gst.State.NULL)
                return False
            
            logger.info("✓ RTSP pipeline with appsink started successfully")
            
            # Capture frames for specified duration
            start_time = time.time()
            frame_count = 0
            error_count = 0
            
            while time.time() - start_time < duration:
                try:
                    sample = appsink.emit("pull-sample")
                    if sample:
                        buf = sample.get_buffer()
                        caps = sample.get_caps()
                        if caps:
                            structure = caps.get_structure(0)
                            width_actual = structure.get_value("width")
                            height_actual = structure.get_value("height")
                            
                            success, mapinfo = buf.map(Gst.MapFlags.READ)
                            if success:
                                frame = np.ndarray((height_actual, width_actual, 3), 
                                                  buffer=mapinfo.data, dtype=np.uint8)
                                frame_count += 1
                                buf.unmap(mapinfo)
                                
                                if frame_count % 30 == 0:  # Log every 30 frames
                                    logger.info(f"Captured frame {frame_count}: {width_actual}x{height_actual}")
                            else:
                                error_count += 1
                                logger.warning(f"Failed to map buffer for frame {frame_count}")
                    else:
                        error_count += 1
                        logger.warning(f"No sample received for frame {frame_count}")
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error capturing frame {frame_count}: {e}")
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
            
            logger.info(f"✓ Captured {frame_count} frames with {error_count} errors over {duration} seconds")
            
            pipeline.set_state(Gst.State.NULL)
            return frame_count > 0
            
        except Exception as e:
            logger.error(f"✗ RTSP pipeline with appsink test failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def test_rtsp_connection_stability(self, num_iterations: int = 5, duration_per_iteration: int = 5) -> bool:
        """Test RTSP connection stability with multiple connect/disconnect cycles"""
        logger.info(f"Testing RTSP connection stability with {num_iterations} iterations")
        
        successful_iterations = 0
        
        for i in range(num_iterations):
            logger.info(f"Starting iteration {i+1}/{num_iterations}")
            
            try:
                if self.test_rtsp_with_appsink(duration=duration_per_iteration):
                    successful_iterations += 1
                    logger.info(f"✓ Iteration {i+1} successful")
                else:
                    logger.error(f"✗ Iteration {i+1} failed")
                
                # Wait between iterations
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"✗ Iteration {i+1} failed with exception: {e}")
                logger.error(traceback.format_exc())
        
        success_rate = successful_iterations / num_iterations
        logger.info(f"Connection stability test: {successful_iterations}/{num_iterations} successful ({success_rate:.1%})")
        
        return success_rate >= 0.8  # 80% success rate threshold
    
    def test_rtsp_with_different_formats(self) -> Dict[str, bool]:
        """Test RTSP with different output formats"""
        formats = {
            "BGR": "video/x-raw,format=BGR",
            "RGB": "video/x-raw,format=RGB", 
            "I420": "video/x-raw,format=I420",
            "NV12": "video/x-raw,format=NV12"
        }
        
        results = {}
        
        for format_name, format_caps in formats.items():
            logger.info(f"Testing RTSP with {format_name} format")
            
            try:
                import gi
                gi.require_version("Gst", "1.0")
                from gi.repository import Gst, GLib
                
                pipeline_str = (
                    f"rtspsrc location={self.rtsp_url} latency=0 ! "
                    f"decodebin ! videoconvert ! videoscale ! "
                    f"{format_caps},width=640,height=480 ! "
                    f"appsink name=sink emit-signals=true"
                )
                
                pipeline = Gst.parse_launch(pipeline_str)
                appsink = pipeline.get_by_name("sink")
                
                pipeline.set_state(Gst.State.PLAYING)
                state_change_result = pipeline.get_state(5 * Gst.SECOND)
                
                if state_change_result[1] == Gst.State.PLAYING:
                    # Try to capture one frame
                    sample = appsink.emit("pull-sample")
                    if sample:
                        results[format_name] = True
                        logger.info(f"✓ {format_name} format works")
                    else:
                        results[format_name] = False
                        logger.warning(f"✗ {format_name} format: no sample received")
                else:
                    results[format_name] = False
                    logger.error(f"✗ {format_name} format: pipeline failed to start")
                
                pipeline.set_state(Gst.State.NULL)
                
            except Exception as e:
                results[format_name] = False
                logger.error(f"✗ {format_name} format test failed: {e}")
        
        return results
    
    def test_rtsp_latency_settings(self) -> Dict[str, bool]:
        """Test RTSP with different latency settings"""
        latency_settings = [0, 100, 200, 500, 1000]  # milliseconds
        results = {}
        
        for latency in latency_settings:
            logger.info(f"Testing RTSP with latency={latency}ms")
            
            try:
                import gi
                gi.require_version("Gst", "1.0")
                from gi.repository import Gst, GLib
                
                pipeline_str = (
                    f"rtspsrc location={self.rtsp_url} latency={latency} ! "
                    f"decodebin ! videoconvert ! videoscale ! "
                    f"video/x-raw,width=640,height=480,format=BGR ! "
                    f"appsink name=sink emit-signals=true"
                )
                
                pipeline = Gst.parse_launch(pipeline_str)
                appsink = pipeline.get_by_name("sink")
                
                pipeline.set_state(Gst.State.PLAYING)
                state_change_result = pipeline.get_state(5 * Gst.SECOND)
                
                if state_change_result[1] == Gst.State.PLAYING:
                    # Try to capture a few frames
                    frame_count = 0
                    start_time = time.time()
                    
                    while frame_count < 10 and time.time() - start_time < 3:
                        sample = appsink.emit("pull-sample")
                        if sample:
                            frame_count += 1
                        time.sleep(0.1)
                    
                    results[f"latency_{latency}ms"] = frame_count > 0
                    logger.info(f"✓ Latency {latency}ms: captured {frame_count} frames")
                else:
                    results[f"latency_{latency}ms"] = False
                    logger.error(f"✗ Latency {latency}ms: pipeline failed to start")
                
                pipeline.set_state(Gst.State.NULL)
                
            except Exception as e:
                results[f"latency_{latency}ms"] = False
                logger.error(f"✗ Latency {latency}ms test failed: {e}")
        
        return results
    
    def run_comprehensive_test(self, test_duration: int = 30) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        logger.info("=" * 60)
        logger.info("Starting Comprehensive RTSP GStreamer Test")
        logger.info("=" * 60)
        
        results = {
            "rtsp_url": self.rtsp_url,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {}
        }
        
        # Test 1: GStreamer availability
        logger.info("\n1. Testing GStreamer availability...")
        results["tests"]["gstreamer_available"] = self.test_gstreamer_availability()
        
        if not results["tests"]["gstreamer_available"]:
            logger.error("GStreamer not available. Stopping tests.")
            return results
        
        # Test 2: Simple RTSP pipeline
        logger.info("\n2. Testing simple RTSP pipeline...")
        results["tests"]["simple_rtsp_pipeline"] = self.test_simple_rtsp_pipeline()
        
        # Test 3: RTSP with appsink
        logger.info("\n3. Testing RTSP with appsink...")
        results["tests"]["rtsp_with_appsink"] = self.test_rtsp_with_appsink(duration=test_duration)
        
        # Test 4: Connection stability
        logger.info("\n4. Testing connection stability...")
        results["tests"]["connection_stability"] = self.test_rtsp_connection_stability()
        
        # Test 5: Different formats
        logger.info("\n5. Testing different output formats...")
        results["tests"]["format_compatibility"] = self.test_rtsp_with_different_formats()
        
        # Test 6: Latency settings
        logger.info("\n6. Testing different latency settings...")
        results["tests"]["latency_settings"] = self.test_rtsp_latency_settings()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        
        for test_name, test_result in results["tests"].items():
            if isinstance(test_result, dict):
                success_count = sum(1 for v in test_result.values() if v)
                total_count = len(test_result)
                logger.info(f"{test_name}: {success_count}/{total_count} passed")
            else:
                status = "✓ PASS" if test_result else "✗ FAIL"
                logger.info(f"{test_name}: {status}")
        
        return results

def main():
    """Main function to run RTSP GStreamer tests"""
    if len(sys.argv) < 2:
        print("Usage: python rtsp_gstreamer_test.py <rtsp_url> [test_duration_seconds]")
        print("Example: python rtsp_gstreamer_test.py rtsp://admin:admin123@192.168.0.194:554 30")
        sys.exit(1)
    
    rtsp_url = sys.argv[1]
    test_duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    print(f"Starting RTSP GStreamer test with URL: {rtsp_url}")
    print(f"Test duration: {test_duration} seconds")
    print(f"Log file: rtsp_gstreamer_test.log")
    
    tester = GStreamerRTSPTester(rtsp_url)
    results = tester.run_comprehensive_test(test_duration)
    
    # Save results to file
    import json
    with open('rtsp_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nTest results saved to: rtsp_test_results.json")
    print("Check rtsp_gstreamer_test.log for detailed logs")

if __name__ == "__main__":
    main()