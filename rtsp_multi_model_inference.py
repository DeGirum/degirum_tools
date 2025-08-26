#!/usr/bin/env python3
"""
RTSP Multi-Model Inference
==========================

Script to run parallel inference on multiple RTSP streams using the notebook logic.
This script ingests RTSP streams and runs multiple AI models in parallel to test FPS.
No RTSP server creation - assumes servers are already running.
"""

import time
import os
import sys
import threading
from pathlib import Path
import degirum as dg
import degirum_tools
from degirum_tools import streams as dgstreams

class FPSMonitorGizmo(dgstreams.Gizmo):
    """Custom gizmo to monitor FPS for each stream"""
    
    def __init__(self, stream_id, max_frames=300):
        super().__init__([(10, False)])  # 10 frame buffer, no dropping
        self.stream_id = stream_id
        self.max_frames = max_frames
        self.frame_count = 0
        self.start_time = None
        self.fps_data = []
        
    def get_tags(self) -> list:
        return [self.name, "fps_monitor"]
        
    def run(self):
        """Run gizmo"""
        self.start_time = time.time()
        
        for data in self.get_input(0):
            if self._abort:
                break
                
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            if elapsed > 0:
                current_fps = self.frame_count / elapsed
                self.fps_data.append({
                    'frame': self.frame_count,
                    'fps': current_fps,
                    'timestamp': current_time
                })
            
            # Pass through the data
            self.send_result(data)
            
            if self.frame_count >= self.max_frames:
                break
        
        # Calculate final statistics
        if self.start_time and self.frame_count > 0:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time
            print(f"[Stream {self.stream_id}] Final Stats:")
            print(f"  Frames: {self.frame_count}")
            print(f"  Total Time: {total_time:.2f}s")
            print(f"  Average FPS: {avg_fps:.2f}")
            if self.fps_data:
                peak_fps = max([d['fps'] for d in self.fps_data])
                print(f"  Peak FPS: {peak_fps:.2f}")

def run_rtsp_multi_model_inference(num_streams, num_models, base_port=8554, max_frames=300):
    """Run multi-RTSP multi-model inference pipeline using notebook logic"""
    
    print("="*80)
    print(f"RTSP MULTI-MODEL INFERENCE")
    print(f"Streams: {num_streams}")
    print(f"Models: {num_models}")
    print(f"Port Range: {base_port} - {base_port + num_streams - 1}")
    print(f"Max Frames per Stream: {max_frames}")
    print("="*80)
    
    # Configuration (same as notebook)
    hw_location = "@cloud"  # or "@local" for local inference
    model_zoo_url = "degirum/public"
    
    # Model names (all with same input size for compatibility)
    model_names = [
        "yolo_v5s_person_det--512x512_quant_n2x_orca1_1",
        "yolo_v5s_face_det--512x512_quant_n2x_orca1_1", 
        "yolo_v5n_car_det--512x512_quant_n2x_orca1_1",
        "yolo_v5s_hand_det--512x512_quant_n2x_orca1_1",
    ]
    
    # Use only the requested number of models
    model_names = model_names[:num_models]
    
    # RTSP stream URLs
    rtsp_urls = [f"rtsp://localhost:{base_port + i}/video0" for i in range(num_streams)]
    
    print(f"\nUsing models: {model_names}")
    print(f"RTSP URLs: {rtsp_urls}")
    
    try:
        # Create PySDK AI model objects (same as notebook)
        print("\nLoading AI models...")
        models = [
            dg.load_model(
                model_name=model_name,
                inference_host_address=hw_location,
                zoo_url=model_zoo_url,
                token=degirum_tools.get_token(),
                overlay_line_width=2,
            )
            for model_name in model_names
        ]
        
        # Check that all models have the same input configuration (same as notebook)
        assert all(
            type(model._preprocessor) == type(models[0]._preprocessor)
            and model.model_info.InputH == models[0].model_info.InputH
            and model.model_info.InputW == models[0].model_info.InputW
            for model in models[1:]
        ), "All models must have the same input configuration"
        
        print("✓ All models loaded successfully")
        
        # Create video source gizmos (same as notebook)
        print(f"\nCreating {num_streams} video sources...")
        sources = [
            dgstreams.VideoSourceGizmo(src, stop_composition_on_end=True)
            for src in rtsp_urls
        ]
        
        # Create image resizer gizmos, one per video source (same as notebook)
        print(f"Creating {num_streams} preprocessors...")
        resizers = [dgstreams.AiPreprocessGizmo(models[0]) for _ in rtsp_urls]
        
        # Create multi-input detector gizmos, one per model (same as notebook)
        print(f"Creating {num_models} AI detectors...")
        detectors = [
            dgstreams.AiSimpleGizmo(model, inp_cnt=len(rtsp_urls)) 
            for model in models
        ]
        
        # Create result combiner gizmo (same as notebook)
        print("Creating result combiner...")
        combiner = dgstreams.AiResultCombiningGizmo(len(models))
        
        # Create FPS monitors for each stream
        print(f"Creating {num_streams} FPS monitors...")
        fps_monitors = [
            FPSMonitorGizmo(i, max_frames) for i in range(num_streams)
        ]
        
        # Create multi-window video display gizmo (same as notebook)
        print("Creating display...")
        win_captions = [f"Stream #{i}: RTSP {base_port + i}" for i in range(num_streams)]
        display = dgstreams.VideoDisplayGizmo(
            win_captions, show_ai_overlay=True, show_fps=True, multiplex=True
        )
        
        # Connect all gizmos in the pipeline (same as notebook)
        print("Connecting pipeline...")
        pipeline = (
            # Each source is connected to corresponding resizer
            (source >> resizer for source, resizer in zip(sources, resizers)),
            
            # Each resizer is connected to every detector
            (
                resizer >> detector[ri]
                for detector in detectors
                for ri, resizer in enumerate(resizers)
            ),
            
            # Each detector is connected to result combiner
            (detector >> combiner[di] for di, detector in enumerate(detectors)),
            
            # Result combiner is connected to FPS monitors
            (combiner >> fps_monitor for fps_monitor in fps_monitors),
            
            # FPS monitors are connected to display
            (fps_monitor >> display for fps_monitor in fps_monitors),
        )
        
        # Create and start composition (same as notebook)
        print("\nStarting composition...")
        start_time = time.time()
        
        composition = dgstreams.Composition(*pipeline)
        composition.start()
        
        # Wait for completion
        composition.wait()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Collect results
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        total_fps = 0
        total_frames = 0
        successful_streams = 0
        
        for i, fps_monitor in enumerate(fps_monitors):
            if fps_monitor.frame_count > 0:
                successful_streams += 1
                total_frames += fps_monitor.frame_count
                
                if fps_monitor.start_time:
                    elapsed = fps_monitor.elapsed_s
                    avg_fps = fps_monitor.frame_count / elapsed if elapsed > 0 else 0
                    total_fps += avg_fps
                    
                    print(f"\nStream {i} (Port {base_port + i}):")
                    print(f"  Frames: {fps_monitor.frame_count}")
                    print(f"  Duration: {elapsed:.2f}s")
                    print(f"  Average FPS: {avg_fps:.2f}")
                    if fps_monitor.fps_data:
                        peak_fps = max([d['fps'] for d in fps_monitor.fps_data])
                        print(f"  Peak FPS: {peak_fps:.2f}")
            else:
                print(f"\nStream {i} (Port {base_port + i}): ❌ No frames processed")
        
        # Summary
        print("\n" + "="*80)
        print("COMBINED SUMMARY")
        print("="*80)
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Successful Streams: {successful_streams}/{num_streams}")
        print(f"Total Frames Processed: {total_frames}")
        print(f"Combined FPS: {total_fps:.2f}")
        
        if successful_streams > 0:
            avg_fps_per_stream = total_fps / successful_streams
            print(f"Average FPS per Stream: {avg_fps_per_stream:.2f}")
            print(f"Efficiency: {avg_fps_per_stream:.2f} FPS per stream")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python rtsp_multi_model_inference.py <num_streams> <num_models> [base_port] [max_frames]")
        print("Example: python rtsp_multi_model_inference.py 3 2")
        print("Example: python rtsp_multi_model_inference.py 5 3 9000 200")
        print("\nNote: RTSP servers must be running on the specified ports before running this script.")
        sys.exit(1)
    
    num_streams = int(sys.argv[1])
    num_models = int(sys.argv[2])
    base_port = int(sys.argv[3]) if len(sys.argv) > 3 else 8554
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    
    if num_streams <= 0 or num_models <= 0:
        print("Error: Number of streams and models must be positive")
        sys.exit(1)
    
    if num_models > 4:
        print("Error: Maximum 4 models supported (due to input size compatibility)")
        sys.exit(1)
    
    if num_streams > 10:
        print("Warning: Running more than 10 streams may cause performance issues")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Check if RTSP servers are expected to be running
    print("="*60)
    print("RTSP MULTI-MODEL INFERENCE")
    print("="*60)
    print(f"Expected RTSP servers on ports: {base_port} - {base_port + num_streams - 1}")
    print("Make sure RTSP servers are running before proceeding.")
    
    response = input("\nAre RTSP servers running? (y/N): ")
    if response.lower() != 'y':
        print("Please start RTSP servers first, then run this script again.")
        sys.exit(1)
    
    try:
        success = run_rtsp_multi_model_inference(num_streams, num_models, base_port, max_frames)
        
        if not success:
            print("\n❌ Test failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 