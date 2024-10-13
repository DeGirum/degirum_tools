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


class FaninGizmo(streams.Gizmo):
    """Gizmo to merge multiple inputs into one"""

    def __init__(self, n_inputs):
        super().__init__([(0, False)] * n_inputs)

    def run(self):
        while True:
            if self._abort:
                break

            for input in self.get_inputs():
                data = input.get()
                if data == input._poison:
                    self._abort = True
                    break
                self.send_result(data)


def test_streams_connection(short_video):
    """Test for streams connection"""

    # simple two-element connection
    source = streams.VideoSourceGizmo(short_video)
    sink = VideoSink()
    c = streams.Composition(source >> sink)

    assert source._connected_gizmos == {sink}
    assert sink._connected_gizmos == {source}
    assert set(c._gizmos) == {source, sink}

    # tree connection
    N = 3
    fanin = FaninGizmo(N)
    sources = [streams.VideoSourceGizmo(short_video) for _ in range(N)]
    sinks = [VideoSink() for _ in range(N)]
    c = streams.Composition(
        sources[0] >> fanin[0] >> sinks[0],
        sources[1] >> fanin[1] >> sinks[1],
        sources[2] >> fanin[2] >> sinks[2],
    )

    assert all(
        sources[i]._connected_gizmos == {fanin}
        and fanin in sources[i]._connected_gizmos
        for i in range(N)
    )
    assert all(
        sinks[i]._connected_gizmos == {fanin} and fanin in sinks[i]._connected_gizmos
        for i in range(N)
    )
    assert set(c._gizmos) == set(sources) | set(sinks) | {fanin}


def test_streams_video_source(short_video):
    """Test for VideoSourceGizmo"""

    with degirum_tools.open_video_stream(short_video) as src:
        frame_width = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = src.get(cv2.CAP_PROP_FPS)
        frame_count = int(src.get(cv2.CAP_PROP_FRAME_COUNT))

    source = streams.VideoSourceGizmo(short_video)
    sink = VideoSink()
    streams.Composition(source >> sink).start()

    assert streams.tag_video in source.get_tags()
    assert sink.frames_cnt == frame_count

    for i, frame in enumerate(sink.frames):
        assert frame.data.shape == (frame_height, frame_width, 3)
        video_meta = frame.meta.find_last(streams.tag_video)
        assert video_meta is not None
        assert video_meta[source.key_frame_width] == frame_width
        assert video_meta[source.key_frame_height] == frame_height
        assert video_meta[source.key_fps] == fps
        assert video_meta[source.key_frame_count] == frame_count
        assert video_meta[source.key_frame_id] == i


def test_streams_video_display(short_video):
    """Test for VideoDisplayGizmo"""

    source = streams.VideoSourceGizmo(short_video)
    display = streams.VideoDisplayGizmo()
    sink = VideoSink()
    streams.Composition(source >> display, source >> sink).start()

    assert all(
        (display._frames[i].data == sink.frames[i].data).all()
        for i in range(sink.frames_cnt)
    )


def test_streams_video_saver(short_video, temp_dir):
    """Test for VideoSaverGizmo"""

    result_path = temp_dir / "test.mp4"
    source = streams.VideoSourceGizmo(short_video)
    saver = streams.VideoSaverGizmo(result_path)
    sink = VideoSink()
    streams.Composition(source >> saver, source >> sink).start()

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
    """Test for ResizingGizmo"""

    w = 320
    h = 240
    pad_method = "stretch"
    resize_method = cv2.INTER_NEAREST
    source = streams.VideoSourceGizmo(short_video)
    resizer = streams.ResizingGizmo(
        w, h, pad_method=pad_method, resize_method=resize_method
    )
    sink = VideoSink()
    streams.Composition(source >> resizer >> sink).start()

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


def test_streams_simple_ai(short_video, classification_model):
    """Test for AiSimpleGizmo"""

    import degirum as dg

    model = classification_model

    source = streams.VideoSourceGizmo(short_video)
    ai = streams.AiSimpleGizmo(model)
    sink = VideoSink()

    streams.Composition(source >> ai >> sink).start()

    for frame in sink.frames:
        ai_meta = frame.meta.find_last(streams.tag_inference)
        assert ai_meta is not None
        assert isinstance(ai_meta, dg.postprocessor.InferenceResults)
        # expect meta in frame info
        assert isinstance(ai_meta.info, streams.StreamMeta)
        assert ai_meta.info.find_last(streams.tag_inference) == ai_meta


def test_streams_cropping_ai(short_video, detection_model):
    """Test for AiObjectDetectionCroppingGizmo"""

    import degirum as dg

    model = detection_model

    source = streams.VideoSourceGizmo(short_video)
    ai = streams.AiSimpleGizmo(model)
    cropper = streams.AiObjectDetectionCroppingGizmo(["Car"])
    sink = VideoSink()

    streams.Composition(source >> ai >> cropper >> sink).start()

    expected_crops = 0
    for frame in sink.frames:
        ai_meta = frame.meta.find_last(streams.tag_inference)
        assert ai_meta is not None
        assert isinstance(ai_meta, dg.postprocessor.DetectionResults)

        crop_meta = frame.meta.find_last(streams.tag_crop)
        assert crop_meta is not None
        assert cropper.key_original_result in crop_meta
        assert cropper.key_cropped_result in crop_meta
        assert cropper.key_cropped_index in crop_meta
        assert cropper.key_is_last_crop in crop_meta
        assert crop_meta[cropper.key_original_result] == ai_meta

        if expected_crops == 0:
            expected_crops = len(ai_meta.results)

        if expected_crops > 0:
            assert (
                crop_meta[cropper.key_is_last_crop]
                if expected_crops == 1
                else not crop_meta[cropper.key_is_last_crop]
            )
            i = crop_meta[cropper.key_cropped_index]
            r = crop_meta[cropper.key_cropped_result]

            bbox = r["bbox"]
            assert frame.data.shape == (
                int(bbox[3]) - int(bbox[1]),
                int(bbox[2]) - int(bbox[0]),
                3,
            )
            assert r == ai_meta.results[i]

            expected_crops -= 1
        else:
            assert crop_meta[cropper.key_is_last_crop]
            assert crop_meta[cropper.key_cropped_index] == -1
            assert crop_meta[cropper.key_cropped_result] is None


def test_streams_combining_ai(short_video, zoo_dir, detection_model_name):
    """Test for AiResultCombiningGizmo"""

    import degirum as dg

    N = 3
    zoo = dg.connect(dg.LOCAL, zoo_dir)
    models = [zoo.load_model(detection_model_name) for _ in range(N)]

    source = streams.VideoSourceGizmo(short_video)
    ai = [streams.AiSimpleGizmo(models[i]) for i in range(N)]

    combiner = streams.AiResultCombiningGizmo(N)
    sink = VideoSink()

    streams.Composition(
        (source >> ai[i] >> combiner[i] for i in range(N)), combiner >> sink
    ).start()

    for frame in sink.frames:
        ai_meta = frame.meta.find_last(streams.tag_inference)
        assert ai_meta is not None
        L = len(ai_meta.results)
        assert L % N == 0
        for i in range(1, N):
            assert (
                ai_meta.results[i * L // N : (i + 1) * L // N]
                == ai_meta.results[0 : L // N]
            )


def test_streams_preprocess_ai(short_video, classification_model):
    """Test for AiPreprocessGizmo"""

    import numpy as np

    model = classification_model
    model.save_model_image = True
    source = streams.VideoSourceGizmo(short_video)
    preprocessor = streams.AiPreprocessGizmo(model)
    sink = VideoSink()

    streams.Composition(source >> preprocessor >> sink).start()

    model_shape = tuple(model.input_shape[0][1:])

    for frame in sink.frames:
        assert isinstance(frame.data, bytes)
        pre_meta = frame.meta.find_last(streams.tag_preprocess)
        assert pre_meta is not None
        assert (
            preprocessor.key_image_input in pre_meta
            and preprocessor.key_converter in pre_meta
            and preprocessor.key_image_result in pre_meta
        )
        video_meta = frame.meta.find_last(streams.tag_video)
        assert video_meta is not None
        assert pre_meta[preprocessor.key_image_input].shape == (
            video_meta[source.key_frame_height],
            video_meta[source.key_frame_width],
            3,
        )

        image_result = pre_meta[preprocessor.key_image_result]
        assert image_result is not None and isinstance(image_result, np.ndarray)
        assert image_result.shape == model_shape


def test_streams_analyzer_ai(short_video, classification_model):
    """Test for AiAnalyzerGizmo"""

    model = classification_model

    class TestAnalyzer(degirum_tools.ResultAnalyzerBase):

        def __init__(self, attribute: str):
            self.attribute = attribute

        def analyze(self, result):
            setattr(result, self.attribute, 1)
            return result

    N = 3
    analyzers = [TestAnalyzer(f"attr{i}") for i in range(N)]

    source = streams.VideoSourceGizmo(short_video)
    ai = streams.AiSimpleGizmo(model)
    analyzer = streams.AiAnalyzerGizmo(analyzers)
    sink = VideoSink()

    streams.Composition(source >> ai >> analyzer >> sink).start()

    for frame in sink.frames:
        ai_meta = frame.meta.find_last(streams.tag_inference)
        assert ai_meta is not None
        for i in range(N):
            assert hasattr(ai_meta, f"attr{i}") and getattr(ai_meta, f"attr{i}") == 1


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

    src1 = InfiniteSource()
    dst1 = SinkWithException()

    with pytest.raises(Exception, match=SinkWithException.error_msg):
        streams.Composition(src1 >> dst1).start()

    assert src1.n > dst1.limit  # type: ignore[attr-defined]

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

    src2 = SourceWithException()
    dst2 = InfiniteSink()

    with pytest.raises(Exception, match=SourceWithException.error_msg):
        streams.Composition(src2 >> dst2).start()

    assert dst2.n == src2.limit  # type: ignore[attr-defined]
