#
# test_video_streamer.py: unit tests for video streaming functionality
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements unit tests to test media server and video streaming functionality
#


def test_video_streamer():
    """
    Test for EventNotifier analyzer
    """

    import degirum_tools
    import numpy as np
    import time
    from degirum_tools import streams

    w = h = 100
    fps = 30.0

    class TestSourceGizmo(streams.Gizmo):
        def run(self):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, 2] = 255  # Set red channel to max
            while not self._abort:
                self.send_result(streams.StreamData(frame))
                time.sleep(1 / fps)

    class TestSinkGizmo(streams.Gizmo):

        def __init__(self):
            super().__init__([(0, False)])

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
                if cnt >= 10:  # Limit to 10 frames for the test
                    break
            if self.composition:
                self.composition.stop()

    url = "rtsp://localhost:8554/mystream"
    src = TestSourceGizmo()
    dst = streams.VideoStreamerGizmo(url, fps=fps)
    rcv = streams.VideoSourceGizmo(url)
    chk = TestSinkGizmo()

    with degirum_tools.MediaServer():

        # start streaming of red frames to RTSP server
        src_composition = streams.Composition(src >> dst)
        src_composition.start(wait=False)

        time.sleep(1)  # Allow some time for the stream to start

        # receive and check frames from RTSP server
        rcv_composition = streams.Composition(rcv >> chk)
        rcv_composition.start()
        src_composition.stop()
