#
# test_video_streamer.py: unit tests for video streaming functionality
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements unit tests to test media server and video streaming functionality
#

import degirum_tools
import numpy as np
import time
from degirum_tools import streams


def test_video_streamer():
    """
    Test for VideoStreamerGizmo gizmo and MediaServer class
    """

    w = h = 100
    fps = 30.0
    localhost = "127.0.0.1"

    class TestSourceGizmo(streams.Gizmo):

        def __init__(self, nframes=-1):
            super().__init__([])
            self._nframes = nframes

        def run(self):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, 2] = 255  # Set red channel to max
            while not self._abort:
                if self._nframes == -1 or self._nframes > 0:
                    self.send_result(streams.StreamData(frame))
                time.sleep(1 / fps)
                if self._nframes > 0:
                    self._nframes -= 1

    class TestSinkGizmo(streams.Gizmo):

        def __init__(self, nframes):
            super().__init__([(0, False)])
            self._nframes = nframes

        def run(self):
            cnt = 0
            for data in self.get_input(0):
                if self._abort:
                    break
                assert data.data.shape == (h, w, 3)
                cnt += 1
                assert np.all(data.data[:, :, 0] == 0)
                assert np.all(data.data[:, :, 1] == 0)
                assert np.all(data.data[:, :, 2] > 250)
                if cnt >= self._nframes:  # Limit frames for the test
                    break
            if self.composition:
                self.composition.stop()
            assert cnt == self._nframes, f"Expected {self._nframes} frames, got {cnt}"

    def do_rtsp_test(in_frames, out_frames, do_detect_test):
        port = 8554
        url = f"rtsp://{localhost}:{port}/mystream"

        src = TestSourceGizmo(in_frames)
        dst = streams.VideoStreamerGizmo(url, fps=fps)
        rcv = streams.VideoSourceGizmo(url)
        chk = TestSinkGizmo(out_frames)

        # start streaming of red frames to RTSP server
        src_composition = streams.Composition(src >> dst)
        src_composition.start(wait=False)

        time.sleep(1)  # Allow some time for the stream to start

        # test detect_rtsp_cameras()
        if do_detect_test:
            detected = degirum_tools.detect_rtsp_cameras(f"{localhost}/32", port=port)
            assert (
                len(detected) == 1 and localhost in detected
            ), "RTSP camera detection failed"
            assert (
                detected[localhost].get("require_auth") is False
            ), "RTSP server should not require authentication"

        # receive and check frames from RTSP server
        rcv_composition = streams.Composition(rcv >> chk)
        rcv_composition.start()
        src_composition.stop()

    def do_rtmp_test(in_frames, out_frames):
        port = 1935
        url = f"rtmp://{localhost}:{port}/live/mystream"

        src = TestSourceGizmo(in_frames)
        dst = streams.VideoStreamerGizmo(url, fps=fps)

        # start streaming of red frames to RTMP server
        src_composition = streams.Composition(src >> dst)
        src_composition.start(wait=False)

        # Allow more time for RTMP stream to establish and be available
        time.sleep(3)

        # For RTMP, we'll just verify the streaming works by checking
        # that the composition runs without errors for a bit
        time.sleep(2)  # Let it stream for a few seconds

        src_composition.stop()
        print("RTMP streaming test completed successfully")

    with degirum_tools.MediaServer():
        # test infinite source + camera detection
        do_rtsp_test(-1, 10, True)
        # test ability to recover when source is dead
        do_rtsp_test(10, 40, False)

        # test RTMP streaming
        do_rtmp_test(15, 10)
        do_rtmp_test(20, 15)
