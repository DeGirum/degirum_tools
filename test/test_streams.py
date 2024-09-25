#
# test_stream.py: unit tests for streams functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test composition and gizmos functionality
#

import time
import pytest
import degirum_tools
import degirum_tools.streams as streams
import cv2


class VideoSink(streams.Gizmo):
    """Video sink gizmo: counts frames and saves them"""

    def __init__(self):
        super().__init__([(0, False)])
        self.frames = []
        self.frames_cnt = 0

    def run(self):
        for data in self.get_input(0):
            self.frames.append(data)
            self.frames_cnt += 1


def test_streams_video_source(short_video):

    with degirum_tools.open_video_stream(short_video) as src:
        frame_width = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = src.get(cv2.CAP_PROP_FPS)
        frame_count = int(src.get(cv2.CAP_PROP_FRAME_COUNT))

    source = streams.VideoSourceGizmo(short_video)
    sink = VideoSink()
    sink.connect_to(source)
    streams.Composition(source, sink).start()

    assert streams.tag_video in source.get_tags()
    assert sink.frames_cnt == frame_count

    for frame in sink.frames:
        assert frame.data.shape == (frame_height, frame_width, 3)
        video_meta = frame.meta.find_last(streams.tag_video)
        assert video_meta is not None
        assert video_meta[source.key_frame_width] == frame_width
        assert video_meta[source.key_frame_height] == frame_height
        assert video_meta[source.key_fps] == fps
        assert video_meta[source.key_frame_count] == frame_count


def test_streams_video_display(short_video):

    source = streams.VideoSourceGizmo(short_video)
    display = streams.VideoDisplayGizmo()
    sink = VideoSink()
    display.connect_to(source)
    sink.connect_to(source)
    streams.Composition(source, display, sink).start()
    assert display._frames == sink.frames


def test_streams_video_saver(short_video, temp_dir):

    result_path = temp_dir / "test.mp4"
    source = streams.VideoSourceGizmo(short_video)
    saver = streams.VideoSaverGizmo(result_path)
    sink = VideoSink()
    saver.connect_to(source)
    sink.connect_to(source)
    streams.Composition(source, saver, sink).start()

    meta = sink.frames[0].meta.find_last(streams.tag_video)
    assert meta is not None
    result = cv2.VideoCapture(result_path)
    try:
        assert meta[source.key_frame_width] == int(result.get(cv2.CAP_PROP_FRAME_WIDTH))
        assert meta[source.key_frame_height] == int(
            result.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        assert meta[source.key_fps] == result.get(cv2.CAP_PROP_FPS)
        assert meta[source.key_frame_count] == int(result.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        result.release()


def test_streams_resizer(short_video):

    w = 320
    h = 240
    pad_method = "stretch"
    resize_method = cv2.INTER_NEAREST
    source = streams.VideoSourceGizmo(short_video)
    resizer = streams.ResizingGizmo(
        w, h, pad_method=pad_method, resize_method=resize_method
    )
    sink = VideoSink()
    sink.connect_to(resizer.connect_to(source))
    streams.Composition(source, resizer, sink).start()

    video_meta = sink.frames[0].meta.find_last(streams.tag_video)
    assert video_meta is not None
    assert video_meta[source.key_frame_count] == sink.frames_cnt

    for frame in sink.frames:
        resize_meta = frame.meta.find_last(streams.tag_resize)
        assert resize_meta is not None
        assert resize_meta[resizer.key_frame_width] == w
        assert resize_meta[resizer.key_frame_height] == h
        assert resize_meta[resizer.key_pad_method] == pad_method
        assert resize_meta[resizer.key_resize_method] == resize_method
        assert frame.data.shape == (h, w, 3)


def test_streams_error_handling():
    """Test for error handling in streams"""

    #
    # Exception in sink
    #

    class InfiniteSource(streams.Gizmo):
        def __init__(self):
            super().__init__()
            self.n = 0

        def run(self):
            while not self._abort:
                self.send_result(streams.StreamData(self.n, streams.StreamMeta(self.n)))
                self.n += 1

    class SinkWithException(streams.Gizmo):
        error_msg = "Test exception 1"

        def __init__(self):
            self.limit = 10
            super().__init__([(self.limit, False)])

        def run(self):
            inp = self.get_input(0)

            # wait until input queue is full
            while inp.qsize() < self.limit:
                time.sleep(0)

            for data in inp:
                if data.data == self.limit // 2:
                    # wait until input queue is full so source will block
                    while inp.qsize() < self.limit:
                        time.sleep(0)
                    raise ValueError(SinkWithException.error_msg)

    c = streams.Composition()
    src = c.add(InfiniteSource())
    dst = c.add(SinkWithException())
    src >> dst
    c.start(wait=False)

    with pytest.raises(Exception, match=SinkWithException.error_msg):
        c.wait()

    assert src.n > dst.limit  # type: ignore[attr-defined]

    #
    # Exception in source
    #
    class SourceWithException(streams.Gizmo):
        error_msg = "Test exception 2"

        def __init__(self):
            self.limit = 12
            super().__init__()

        def run(self):
            n = 0
            while not self._abort:
                self.send_result(streams.StreamData(n, streams.StreamMeta(n)))
                n += 1
                if n == self.limit:
                    # wait until output queue is empty so sink will block
                    while self._output_refs[0].qsize() > 0:
                        time.sleep(0)
                    raise ValueError(SourceWithException.error_msg)

    class InfiniteSink(streams.Gizmo):

        def __init__(self):
            super().__init__([(0, False)])
            self.n = 0

        def run(self):
            for data in self.get_input(0):
                self.n += 1
                if self._abort:
                    break
                time.sleep(0)

    c = streams.Composition()
    src = c.add(SourceWithException())
    dst = c.add(InfiniteSink())
    src >> dst
    c.start(wait=False)

    with pytest.raises(Exception, match=SourceWithException.error_msg):
        c.wait()

    assert dst.n == src.limit  # type: ignore[attr-defined]
