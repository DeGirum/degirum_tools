#!/usr/bin/env python3
"""
RTSP Performance Benchmarking Script
====================================

This script provides comprehensive performance benchmarking for RTSP streams:
1. Single RTSP stream CPU usage measurement
2. Maximum parallel RTSP streams without model inference
3. Maximum RTSP streams with YOLOv8n inference
4. Maximum RTSP streams with pipeline models (face detection + gender classification)

Usage:
    python rtsp_performance_benchmark.py [test_duration_seconds]

Example:
    python rtsp_performance_benchmark.py 60
"""

import sys
import time
import logging
import traceback
import threading
import psutil
import numpy as np
import gc
import signal
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import cv2
import degirum as dg
import degirum_tools
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rtsp_performance_benchmark.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_rtsp_urls(num_streams=20, base_port=8554):
    """Generate RTSP URLs with different ports to avoid port-level bottlenecks"""
    return [f"rtsp://localhost:{base_port + i}/video0" for i in range(num_streams)]

@dataclass
class BenchmarkResult:
    """Class to hold benchmark results"""
    test_name: str
    max_streams: int
    cpu_usage: float
    memory_usage: float
    fps_per_stream: float
    total_fps: float
    error_count: int
    duration: float
    details: Dict[str, Any]

class PerformanceMonitor:
    """Monitors system performance during benchmarks"""
    
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.start_time = None
        self.monitoring = False
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.cpu_samples = []
        self.memory_samples = []
        self.start_time = time.time()
        self.monitoring = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_samples.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_samples.append(memory.percent)
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
            
            time.sleep(1)
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return results"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        
        duration = time.time() - self.start_time if self.start_time else 0
        
        if self.cpu_samples:
            avg_cpu = np.mean(self.cpu_samples)
            max_cpu = np.max(self.cpu_samples)
        else:
            avg_cpu = max_cpu = 0
            
        if self.memory_samples:
            avg_memory = np.mean(self.memory_samples)
            max_memory = np.max(self.memory_samples)
        else:
            avg_memory = max_memory = 0
        
        return {
            'duration': duration,
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'avg_memory': avg_memory,
            'max_memory': max_memory,
            'cpu_samples': self.cpu_samples.copy(),
            'memory_samples': self.memory_samples.copy()
        }

class RTSPPerformanceBenchmark:
    """Main benchmarking class using degirum_tools integration"""
    
    def __init__(self, num_streams=20, test_duration=60, base_port=8554):
        self.rtsp_urls = get_rtsp_urls(num_streams, base_port)
        self.test_duration = test_duration
        self.results = []
        
        # Initialize models (will be loaded when needed)
        self.yolo_model = None
        self.pipeline_model = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self._cleanup()
        sys.exit(0)
        
    def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear model references
            if hasattr(self, 'yolo_model') and self.yolo_model:
                del self.yolo_model
                self.yolo_model = None
                
            if hasattr(self, 'pipeline_model') and self.pipeline_model:
                del self.pipeline_model
                self.pipeline_model = None
                
            # Force another garbage collection
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def benchmark_single_stream_cpu(self) -> BenchmarkResult:
        """Benchmark 1: Single RTSP stream CPU usage using degirum_tools"""
        logger.info("=== Benchmark 1: Single RTSP Stream CPU Usage ===")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        frame_count = 0
        error_count = 0
        start_time = time.time()
        
        try:
            # Use degirum_tools.predict_stream with a dummy model that just returns frames
            # Create a simple pass-through model
            class PassThroughModel:
                def predict(self, frame):
                    return type('Result', (), {
                        'image_overlay': frame,
                        'info': {'frame_id': 0}
                    })()
                
                def predict_batch(self, frames):
                    # Required by degirum_tools.predict_stream
                    for frame in frames:
                        yield self.predict(frame)
            
            dummy_model = PassThroughModel()
            
            # Use the first RTSP URL from our multi-RTSP server
            rtsp_url = self.rtsp_urls[0] if self.rtsp_urls else self.rtsp_urls[0]
            
            # Use degirum_tools.predict_stream with source_type="gstream"
            inference_results = degirum_tools.predict_stream(
                dummy_model, 
                rtsp_url, 
                source_type="gstream"
            )
            
            # Process frames for test duration
            for inference_result in inference_results:
                if time.time() - start_time >= self.test_duration:
                    break
                    
                if inference_result and hasattr(inference_result, 'image_overlay'):
                    frame_count += 1
                else:
                    error_count += 1
                    
                # Small delay to prevent busy waiting
                time.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Error in single stream benchmark: {e}")
            error_count += 1
        
        # Stop monitoring and collect results
        perf_stats = monitor.stop_monitoring()
        actual_duration = time.time() - start_time
        
        # Calculate FPS
        total_fps = frame_count / actual_duration if actual_duration > 0 else 0
        
        result = BenchmarkResult(
            test_name="Single Stream CPU",
            max_streams=1,
            cpu_usage=perf_stats['avg_cpu'],
            memory_usage=perf_stats['avg_memory'],
            fps_per_stream=total_fps,
            total_fps=total_fps,
            error_count=error_count,
            duration=actual_duration,
            details={
                'max_cpu': perf_stats['max_cpu'],
                'max_memory': perf_stats['max_memory'],
                'frame_count': frame_count,
                'cpu_samples': perf_stats['cpu_samples'],
                'memory_samples': perf_stats['memory_samples']
            }
        )
        
        logger.info(f"Single stream results: CPU={result.cpu_usage:.1f}%, "
                   f"Memory={result.memory_usage:.1f}%, FPS={result.fps_per_stream:.1f}")
        
        return result
    
    def benchmark_max_streams_no_inference(self) -> BenchmarkResult:
        """Benchmark 2: Maximum parallel RTSP streams without inference"""
        logger.info("=== Benchmark 2: Maximum Parallel RTSP Streams (No Inference) ===")
        
        max_streams = 0
        best_result = None
        
        # Test with increasing number of streams
        for num_streams in [1, 2, 3, 5, 10, 15, 20]:
            logger.info(f"Testing {num_streams} streams without inference...")
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            frame_counts = [0] * num_streams
            error_counts = [0] * num_streams
            start_time = time.time()
            
            try:
                # Create multiple dummy models for parallel processing
                dummy_models = []
                for i in range(num_streams):
                    class PassThroughModel:
                        def __init__(self, stream_id):
                            self.stream_id = stream_id
                        def predict(self, frame):
                            return type('Result', (), {
                                'image_overlay': frame,
                                'info': {'frame_id': 0, 'stream_id': self.stream_id}
                            })()
                        
                        def predict_batch(self, frames):
                            # Required by degirum_tools.predict_stream
                            for frame in frames:
                                yield self.predict(frame)
                    dummy_models.append(PassThroughModel(i))
                
                # Process streams in parallel using threads
                threads = []
                running = True
                
                def process_stream(stream_id, model):
                    nonlocal frame_counts, error_counts
                    try:
                        # Use different RTSP URL for each stream
                        rtsp_url = self.rtsp_urls[stream_id % len(self.rtsp_urls)]
                        inference_results = degirum_tools.predict_stream(
                            model, 
                            rtsp_url, 
                            source_type="gstream"
                        )
                        
                        for inference_result in inference_results:
                            if not running or time.time() - start_time >= self.test_duration:
                                break
                                
                            if inference_result and hasattr(inference_result, 'image_overlay'):
                                frame_counts[stream_id] += 1
                            else:
                                error_counts[stream_id] += 1
                                
                            time.sleep(0.001)
                            
                    except Exception as e:
                        logger.warning(f"Stream {stream_id} error: {e}")
                        error_counts[stream_id] += 1
                
                # Start threads
                for i in range(num_streams):
                    thread = threading.Thread(
                        target=process_stream, 
                        args=(i, dummy_models[i]),
                        daemon=True
                    )
                    thread.start()
                    threads.append(thread)
                
                # Wait for test duration
                time.sleep(self.test_duration)
                running = False
                
                # Wait for threads to finish
                for thread in threads:
                    thread.join(timeout=5)
                
            except Exception as e:
                logger.error(f"Error testing {num_streams} streams: {e}")
                running = False
            
            # Collect results
            perf_stats = monitor.stop_monitoring()
            actual_duration = time.time() - start_time
            
            total_frames = sum(frame_counts)
            total_errors = sum(error_counts)
            
            # Calculate FPS before viability check
            total_fps = total_frames / actual_duration if actual_duration > 0 else 0
            fps_per_stream = total_fps / num_streams
            
            # Check if configuration is viable
            if (total_frames > 0 and 
                perf_stats['avg_cpu'] < 90 and 
                perf_stats['avg_memory'] < 90 and
                fps_per_stream >= 10):  # Minimum 10 FPS per stream requirement
                
                best_result = BenchmarkResult(
                    test_name="Max Streams No Inference",
                    max_streams=num_streams,
                    cpu_usage=perf_stats['avg_cpu'],
                    memory_usage=perf_stats['avg_memory'],
                    fps_per_stream=fps_per_stream,
                    total_fps=total_fps,
                    error_count=total_errors,
                    duration=actual_duration,
                    details={
                        'max_cpu': perf_stats['max_cpu'],
                        'max_memory': perf_stats['max_memory'],
                        'frame_counts': frame_counts,
                        'error_counts': error_counts
                    }
                )
                
                max_streams = num_streams
                logger.info(f"✓ {num_streams} streams working: CPU={perf_stats['avg_cpu']:.1f}%, "
                           f"FPS per stream={fps_per_stream:.1f}")
            else:
                logger.info(f"✗ {num_streams} streams not viable: CPU={perf_stats['avg_cpu']:.1f}%, "
                           f"Frames={total_frames}, FPS per stream={fps_per_stream:.1f} (need >=10)")
                break
        
        if best_result:
            logger.info(f"Max streams without inference: {best_result.max_streams}, "
                       f"CPU={best_result.cpu_usage:.1f}%, FPS per stream={best_result.fps_per_stream:.1f}")
        
        return best_result or BenchmarkResult(
            test_name="Max Streams No Inference",
            max_streams=0,
            cpu_usage=0,
            memory_usage=0,
            fps_per_stream=0,
            total_fps=0,
            error_count=0,
            duration=0,
            details={'error': 'No working configuration found'}
        )
    
    def load_yolo_model(self):
        """Load YOLOv8n model"""
        if self.yolo_model is None:
            logger.info("Loading YOLOv8n model...")
            try:
                inference_host_address = "@local"
                zoo_url = 'degirum/hailo'
                
                
                self.yolo_model = dg.load_model(
                    model_name="yolov8n_coco--640x640_quant_hailort_hailo8_1",
                    inference_host_address=inference_host_address,
                    zoo_url=zoo_url,
                    token=degirum_tools.get_token(),
                    overlay_color=[(255,255,0),(0,255,0)]
                )
                logger.info("YOLOv8n model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8n model: {e}")
                raise
    
    def benchmark_yolo_inference(self) -> BenchmarkResult:
        """Benchmark 3: Maximum RTSP streams with YOLOv8n inference"""
        logger.info("=== Benchmark 3: Maximum RTSP Streams with YOLOv8n Inference ===")
        
        try:
            self.load_yolo_model()
        except Exception as e:
            return BenchmarkResult(
                test_name="YOLOv8n Inference",
                max_streams=0,
                cpu_usage=0,
                memory_usage=0,
                fps_per_stream=0,
                total_fps=0,
                error_count=0,
                duration=0,
                details={'error': f'Failed to load model: {e}'}
            )
        
        max_streams = 0
        best_result = None
        
        # Test with fewer streams due to inference overhead - STRICT MONITORING
        for num_streams in [1, 2, 3, 4, 5]:
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING {num_streams} STREAMS WITH YOLOv8n INFERENCE")
            logger.info(f"{'='*60}")
            
            # Force garbage collection before each test
            gc.collect()
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            inference_counts = [0] * num_streams
            error_counts = [0] * num_streams
            start_time = time.time()
            
            try:
                # Process streams in parallel using threads
                threads = []
                running = True
                
                def process_stream_with_inference(stream_id):
                    nonlocal inference_counts, error_counts
                    try:
                        logger.info(f"Starting stream {stream_id} with YOLOv8n inference...")
                        
                        # Use different RTSP URL for each stream
                        rtsp_url = self.rtsp_urls[stream_id % len(self.rtsp_urls)]
                        inference_results = degirum_tools.predict_stream(
                            self.yolo_model, 
                            rtsp_url, 
                            source_type="gstream"
                        )
                        
                        last_response_time = time.time()
                        for inference_result in inference_results:
                            if not running or time.time() - start_time >= self.test_duration:
                                break
                                
                            current_time = time.time()
                            # Check if we've waited more than 5 seconds for a response
                            if current_time - last_response_time > 5.0:
                                logger.warning(f"Stream {stream_id} timeout: no response for 5+ seconds")
                                error_counts[stream_id] += 1
                                break  # Stop this stream due to timeout
                                
                            if inference_result and hasattr(inference_result, 'image_overlay'):
                                inference_counts[stream_id] += 1
                                last_response_time = current_time  # Reset timeout counter
                            else:
                                error_counts[stream_id] += 1
                                
                            time.sleep(0.001)
                            
                    except Exception as e:
                        logger.error(f"Stream {stream_id} inference error: {e}")
                        logger.error(traceback.format_exc())
                        error_counts[stream_id] += 1
                
                # Start threads
                for i in range(num_streams):
                    thread = threading.Thread(
                        target=process_stream_with_inference, 
                        args=(i,),
                        daemon=True
                    )
                    thread.start()
                    threads.append(thread)
                    logger.info(f"Started thread {i} for stream {i}")
                
                # Wait for test duration with progress updates
                test_start = time.time()
                while time.time() - test_start < self.test_duration:
                    elapsed = time.time() - test_start
                    remaining = self.test_duration - elapsed
                    
                    # Print progress every 10 seconds
                    if int(elapsed) % 10 == 0 and elapsed > 0:
                        current_inferences = sum(inference_counts)
                        current_errors = sum(error_counts)
                        logger.info(f"Progress: {elapsed:.0f}s/{self.test_duration}s - "
                                  f"Inferences: {current_inferences}, Errors: {current_errors}")
                    
                    time.sleep(1)
                
                running = False
                logger.info("Test duration completed, stopping threads...")
                
                # Wait for threads to finish
                for i, thread in enumerate(threads):
                    thread.join(timeout=5)
                    if thread.is_alive():
                        logger.warning(f"Thread {i} did not finish within timeout")
                
            except Exception as e:
                logger.error(f"Error testing {num_streams} streams with inference: {e}")
                logger.error(traceback.format_exc())
                running = False
            
            # Collect results
            perf_stats = monitor.stop_monitoring()
            actual_duration = time.time() - start_time
            
            total_inferences = sum(inference_counts)
            total_errors = sum(error_counts)
            
            # Calculate FPS before viability check
            inference_fps = total_inferences / actual_duration if actual_duration > 0 else 0
            fps_per_stream = inference_fps / num_streams
            
            # Print detailed results for this test
            logger.info(f"\n{'='*60}")
            logger.info(f"RESULTS FOR {num_streams} STREAMS WITH YOLOv8n:")
            logger.info(f"{'='*60}")
            logger.info(f"CPU Usage: {perf_stats['avg_cpu']:.1f}% (max: {perf_stats['max_cpu']:.1f}%)")
            logger.info(f"Memory Usage: {perf_stats['avg_memory']:.1f}% (max: {perf_stats['max_memory']:.1f}%)")
            logger.info(f"Total Inferences: {total_inferences}")
            logger.info(f"Total Errors: {total_errors}")
            logger.info(f"Inference FPS: {inference_fps:.1f}")
            logger.info(f"FPS per Stream: {fps_per_stream:.1f}")
            logger.info(f"Test Duration: {actual_duration:.1f}s")
            
            # Check if configuration is viable
            if (total_inferences > 0 and 
                perf_stats['avg_cpu'] < 90 and 
                perf_stats['avg_memory'] < 90 and
                fps_per_stream >= 5):  # Reduced minimum FPS requirement for inference
                
                best_result = BenchmarkResult(
                    test_name="YOLOv8n Inference",
                    max_streams=num_streams,
                    cpu_usage=perf_stats['avg_cpu'],
                    memory_usage=perf_stats['avg_memory'],
                    fps_per_stream=fps_per_stream,
                    total_fps=inference_fps,
                    error_count=total_errors,
                    duration=actual_duration,
                    details={
                        'inference_counts': inference_counts,
                        'error_counts': error_counts,
                        'max_cpu': perf_stats['max_cpu'],
                        'max_memory': perf_stats['max_memory']
                    }
                )
                
                max_streams = num_streams
                logger.info(f"✓ {num_streams} streams with YOLOv8n WORKING")
            else:
                logger.info(f"✗ {num_streams} streams with YOLOv8n NOT VIABLE")
                if total_inferences == 0:
                    logger.info("  Reason: No successful inferences")
                if perf_stats['avg_cpu'] >= 90:
                    logger.info(f"  Reason: CPU usage too high ({perf_stats['avg_cpu']:.1f}%)")
                if perf_stats['avg_memory'] >= 90:
                    logger.info(f"  Reason: Memory usage too high ({perf_stats['avg_memory']:.1f}%)")
                if fps_per_stream < 5:
                    logger.info(f"  Reason: FPS per stream too low ({fps_per_stream:.1f})")
                break
            
            # Force cleanup between tests
            gc.collect()
            time.sleep(2)  # Give system time to stabilize
        
        if best_result:
            logger.info(f"\nMax streams with YOLOv8n: {best_result.max_streams}, "
                       f"Inference FPS={best_result.total_fps:.1f}")
        
        return best_result or BenchmarkResult(
            test_name="YOLOv8n Inference",
            max_streams=0,
            cpu_usage=0,
            memory_usage=0,
            fps_per_stream=0,
            total_fps=0,
            error_count=0,
            duration=0,
            details={'error': 'No viable configuration found'}
        )
    
    def load_pipeline_model(self):
        """Load pipeline model (face detection + gender classification)"""
        if self.pipeline_model is None:
            logger.info("Loading pipeline model...")
            try:
                inference_host_address = "@local"
                zoo_url = 'degirum/hailo'
                
                # Load individual models
                face_det_model = dg.load_model(
                    model_name="yolov8n_relu6_face--640x640_quant_hailort_hailo8_1",
                    inference_host_address=inference_host_address,
                    zoo_url='degirum/models_hailort',
                    token=degirum_tools.get_token(),
                    overlay_color=[(255,255,0),(0,255,0)]
                )
                
                gender_cls_model = dg.load_model(
                    model_name="yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8_1",
                    inference_host_address=inference_host_address,
                    zoo_url=zoo_url,
                    token=degirum_tools.get_token(),
                )
                
                # Create compound model
                self.pipeline_model = degirum_tools.CroppingAndClassifyingCompoundModel(
                    face_det_model, 
                    gender_cls_model, 
                    30.0
                )
                
                logger.info("Pipeline model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load pipeline model: {e}")
                raise
    
    def benchmark_pipeline_inference(self) -> BenchmarkResult:
        """Benchmark 4: Maximum RTSP streams with pipeline inference"""
        logger.info("=== Benchmark 4: Maximum RTSP Streams with Pipeline Inference ===")
        
        try:
            self.load_pipeline_model()
        except Exception as e:
            return BenchmarkResult(
                test_name="Pipeline Inference",
                max_streams=0,
                cpu_usage=0,
                memory_usage=0,
                fps_per_stream=0,
                total_fps=0,
                error_count=0,
                duration=0,
                details={'error': f'Failed to load model: {e}'}
            )
        
        max_streams = 0
        best_result = None
        
        # Test with even fewer streams due to pipeline overhead
        for num_streams in [1, 2, 3]:
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING {num_streams} STREAMS WITH PIPELINE INFERENCE")
            logger.info(f"{'='*60}")
            
            # Force garbage collection before each test
            gc.collect()
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            inference_counts = [0] * num_streams
            error_counts = [0] * num_streams
            start_time = time.time()
            
            try:
                # Process streams in parallel using threads
                threads = []
                running = True
                
                def process_stream_with_pipeline(stream_id):
                    nonlocal inference_counts, error_counts
                    try:
                        logger.info(f"Starting stream {stream_id} with pipeline inference...")
                        
                        # Use different RTSP URL for each stream
                        rtsp_url = self.rtsp_urls[stream_id % len(self.rtsp_urls)]
                        inference_results = degirum_tools.predict_stream(
                            self.pipeline_model, 
                            rtsp_url, 
                            source_type="gstream"
                        )
                        
                        last_response_time = time.time()
                        for inference_result in inference_results:
                            if not running or time.time() - start_time >= self.test_duration:
                                break
                                
                            current_time = time.time()
                            # Check if we've waited more than 5 seconds for a response
                            if current_time - last_response_time > 5.0:
                                logger.warning(f"Stream {stream_id} pipeline timeout: no response for 5+ seconds")
                                error_counts[stream_id] += 1
                                break  # Stop this stream due to timeout
                                
                            if inference_result and hasattr(inference_result, 'image_overlay'):
                                inference_counts[stream_id] += 1
                                last_response_time = current_time  # Reset timeout counter
                            else:
                                error_counts[stream_id] += 1
                                
                            time.sleep(0.001)
                            
                    except Exception as e:
                        logger.error(f"Stream {stream_id} pipeline error: {e}")
                        logger.error(traceback.format_exc())
                        error_counts[stream_id] += 1
                
                # Start threads
                for i in range(num_streams):
                    thread = threading.Thread(
                        target=process_stream_with_pipeline, 
                        args=(i,),
                        daemon=True
                    )
                    thread.start()
                    threads.append(thread)
                    logger.info(f"Started thread {i} for pipeline stream {i}")
                
                # Wait for test duration with progress updates
                test_start = time.time()
                while time.time() - test_start < self.test_duration:
                    elapsed = time.time() - test_start
                    remaining = self.test_duration - elapsed
                    
                    # Print progress every 10 seconds
                    if int(elapsed) % 10 == 0 and elapsed > 0:
                        current_inferences = sum(inference_counts)
                        current_errors = sum(error_counts)
                        logger.info(f"Progress: {elapsed:.0f}s/{self.test_duration}s - "
                                  f"Inferences: {current_inferences}, Errors: {current_errors}")
                    
                    time.sleep(1)
                
                running = False
                logger.info("Test duration completed, stopping threads...")
                
                # Wait for threads to finish
                for i, thread in enumerate(threads):
                    thread.join(timeout=5)
                    if thread.is_alive():
                        logger.warning(f"Thread {i} did not finish within timeout")
                
            except Exception as e:
                logger.error(f"Error testing {num_streams} streams with pipeline: {e}")
                logger.error(traceback.format_exc())
                running = False
            
            # Collect results
            perf_stats = monitor.stop_monitoring()
            actual_duration = time.time() - start_time
            
            total_inferences = sum(inference_counts)
            total_errors = sum(error_counts)
            
            # Calculate FPS before viability check
            inference_fps = total_inferences / actual_duration if actual_duration > 0 else 0
            fps_per_stream = inference_fps / num_streams
            
            # Print detailed results for this test
            logger.info(f"\n{'='*60}")
            logger.info(f"RESULTS FOR {num_streams} STREAMS WITH PIPELINE:")
            logger.info(f"{'='*60}")
            logger.info(f"CPU Usage: {perf_stats['avg_cpu']:.1f}% (max: {perf_stats['max_cpu']:.1f}%)")
            logger.info(f"Memory Usage: {perf_stats['avg_memory']:.1f}% (max: {perf_stats['max_memory']:.1f}%)")
            logger.info(f"Total Inferences: {total_inferences}")
            logger.info(f"Total Errors: {total_errors}")
            logger.info(f"Inference FPS: {inference_fps:.1f}")
            logger.info(f"FPS per Stream: {fps_per_stream:.1f}")
            logger.info(f"Test Duration: {actual_duration:.1f}s")
            
            # Check if configuration is viable
            if (total_inferences > 0 and 
                perf_stats['avg_cpu'] < 90 and 
                perf_stats['avg_memory'] < 90 and
                fps_per_stream >= 3):  # Even lower minimum FPS for pipeline
                
                best_result = BenchmarkResult(
                    test_name="Pipeline Inference",
                    max_streams=num_streams,
                    cpu_usage=perf_stats['avg_cpu'],
                    memory_usage=perf_stats['avg_memory'],
                    fps_per_stream=fps_per_stream,
                    total_fps=inference_fps,
                    error_count=total_errors,
                    duration=actual_duration,
                    details={
                        'inference_counts': inference_counts,
                        'error_counts': error_counts,
                        'max_cpu': perf_stats['max_cpu'],
                        'max_memory': perf_stats['max_memory']
                    }
                )
                
                max_streams = num_streams
                logger.info(f"✓ {num_streams} streams with pipeline WORKING")
            else:
                logger.info(f"✗ {num_streams} streams with pipeline NOT VIABLE")
                if total_inferences == 0:
                    logger.info("  Reason: No successful inferences")
                if perf_stats['avg_cpu'] >= 90:
                    logger.info(f"  Reason: CPU usage too high ({perf_stats['avg_cpu']:.1f}%)")
                if perf_stats['avg_memory'] >= 90:
                    logger.info(f"  Reason: Memory usage too high ({perf_stats['avg_memory']:.1f}%)")
                if fps_per_stream < 3:
                    logger.info(f"  Reason: FPS per stream too low ({fps_per_stream:.1f})")
                break
            
            # Force cleanup between tests
            gc.collect()
            time.sleep(2)  # Give system time to stabilize
        
        if best_result:
            logger.info(f"\nMax streams with pipeline: {best_result.max_streams}, "
                       f"Pipeline FPS={best_result.total_fps:.1f}")
        
        return best_result or BenchmarkResult(
            test_name="Pipeline Inference",
            max_streams=0,
            cpu_usage=0,
            memory_usage=0,
            fps_per_stream=0,
            total_fps=0,
            error_count=0,
            duration=0,
            details={'error': 'No viable configuration found'}
        )
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks"""
        logger.info(f"Starting RTSP performance benchmarks for: {self.rtsp_urls[0]}")
        logger.info(f"Test duration: {self.test_duration} seconds")
        
        results = []
        
        try:
            # Benchmark 1: Single stream CPU usage
            result1 = self.benchmark_single_stream_cpu()
            results.append(result1)
            
            # Benchmark 2: Max streams without inference
            result2 = self.benchmark_max_streams_no_inference()
            results.append(result2)
            
            time.sleep(5)  # Give some time to stabilize before next benchmarks

            # Benchmark 3: Max streams with YOLOv8n
            result3 = self.benchmark_yolo_inference()
            results.append(result3)

            time.sleep(5)  # Give some time to stabilize before next benchmarks
            
            # Benchmark 4: Max streams with pipeline
            result4 = self.benchmark_pipeline_inference()
            results.append(result4)
            
        finally:
            self._cleanup()
        
        self.results = results
        return results
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("RTSP PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        for result in self.results:
            print(f"\n{result.test_name}:")
            print(f"  Max Streams: {result.max_streams}")
            print(f"  CPU Usage: {result.cpu_usage:.1f}%")
            print(f"  Memory Usage: {result.memory_usage:.1f}%")
            print(f"  FPS per Stream: {result.fps_per_stream:.1f}")
            print(f"  Total FPS: {result.total_fps:.1f}")
            print(f"  Error Count: {result.error_count}")
            print(f"  Duration: {result.duration:.1f}s")
            
            if 'error' in result.details:
                print(f"  Error: {result.details['error']}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        
        # Find best configurations
        single_stream = next((r for r in self.results if r.test_name == "Single Stream CPU"), None)
        max_no_inference = next((r for r in self.results if r.test_name == "Max Streams No Inference"), None)
        max_yolo = next((r for r in self.results if r.test_name == "YOLOv8n Inference"), None)
        max_pipeline = next((r for r in self.results if r.test_name == "Pipeline Inference"), None)
        
        if single_stream and single_stream.max_streams > 0:
            print(f"• Single RTSP stream uses {single_stream.cpu_usage:.1f}% CPU")
        
        if max_no_inference and max_no_inference.max_streams > 0:
            print(f"• Maximum RTSP streams without inference: {max_no_inference.max_streams}")
            print(f"• CPU usage per stream: {max_no_inference.cpu_usage / max_no_inference.max_streams:.1f}%")
        
        if max_yolo and max_yolo.max_streams > 0:
            print(f"• Maximum RTSP streams with YOLOv8n: {max_yolo.max_streams}")
            print(f"• Inference FPS: {max_yolo.total_fps:.1f}")
        
        if max_pipeline and max_pipeline.max_streams > 0:
            print(f"• Maximum RTSP streams with pipeline: {max_pipeline.max_streams}")
            print(f"• Pipeline FPS: {max_pipeline.total_fps:.1f}")
        
        print("\n" + "="*80)

def main():
    """Main function"""
    # Default values - no need for command line arguments since we generate RTSP URLs
    rtsp_urls = get_rtsp_urls(num_streams=20, base_port=8554)
    test_duration = 60  # Default 60 seconds
    
    # Check if command line arguments are provided (for backward compatibility)
    if len(sys.argv) > 1:
        test_duration = int(sys.argv[1])
    
    try:
        # Create and run benchmarks
        benchmark = RTSPPerformanceBenchmark(num_streams=20, test_duration=test_duration, base_port=8554)
        results = benchmark.run_all_benchmarks()
        
        if not results:
            logger.error("No benchmark results obtained. Exiting.")
            sys.exit(1)
        
        # Print summary
        benchmark.print_summary()
        
        # Save results to file
        import json
        from datetime import datetime
        
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'rtsp_urls': rtsp_urls,
            'test_duration': test_duration,
            'results': [
                {
                    'test_name': r.test_name,
                    'max_streams': r.max_streams,
                    'cpu_usage': r.cpu_usage,
                    'memory_usage': r.memory_usage,
                    'fps_per_stream': r.fps_per_stream,
                    'total_fps': r.total_fps,
                    'error_count': r.error_count,
                    'duration': r.duration,
                    'details': r.details
                }
                for r in results
            ]
        }
        
        with open('rtsp_benchmark_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info("Benchmark results saved to rtsp_benchmark_results.json")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 