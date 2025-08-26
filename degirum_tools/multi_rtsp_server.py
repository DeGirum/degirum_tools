#!/usr/bin/env python3
import gi
import sys
import os
from gi.repository import Gst, GstRtspServer, GObject, GLib

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")

Gst.init(None)

class FileRTSPMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.set_shared(True)

    def do_create_element(self, url):
        pipeline = (
            f"( filesrc location={self.video_path} ! decodebin ! "
            f"x264enc tune=zerolatency bitrate=512 speed-preset=ultrafast ! rtph264pay name=pay0 pt=96 )"
        )
        return Gst.parse_launch(pipeline)

class MultiStreamRTSPServer:
    def __init__(self, video_dir, base_port=8554, num_servers=20):
        self.video_dir = video_dir
        self.base_port = base_port
        self.num_servers = num_servers
        self.servers = []
        
        video_files = sorted([
            f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mkv', '.avi'))
        ])

        if not video_files:
            print("No video files found in:", video_dir)
            sys.exit(1)

        # Create multiple servers on different ports
        for i in range(num_servers):
            port = base_port + i
            video_file = video_files[i % len(video_files)]  # Cycle through video files
            path = os.path.join(video_dir, video_file)
            
            server = GstRtspServer.RTSPServer()
            server.set_service(str(port))
            mount_points = server.get_mount_points()
            
            factory = FileRTSPMediaFactory(path)
            mount_points.add_factory("/video0", factory)
            server.attach(None)
            
            self.servers.append(server)
            print(f"Exposed: rtsp://localhost:{port}/video0 from {path}")

if __name__ == "__main__":
    video_dir = "videos"
    base_port = 8554
    num_servers = 20
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        num_servers = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        base_port = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        video_dir = sys.argv[3]
    
    print(f"Starting {num_servers} RTSP servers on ports {base_port}-{base_port + num_servers - 1}")
    server = MultiStreamRTSPServer(video_dir, base_port, num_servers)
    print("RTSP Servers running. Press Ctrl+C to stop.")
    loop = GLib.MainLoop()
    loop.run()
