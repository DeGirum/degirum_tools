#
# test_stream.py: unit tests for streams functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test composition and gizmos functionality
#

import time
import os
import pytest
import degirum_tools
import degirum_tools.streams as streams
import degirum as dg
import cv2
import numpy as np


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

    # double connection: should be the same as single connection
    source = streams.VideoSourceGizmo(short_video)
    sink = VideoSink()
    c = streams.Composition(source >> sink, source >> sink)
    assert len(source._output_refs) == 1 and source._output_refs[0] is sink.get_input(0)
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


def test_streams_iterator_source():
    """Test for IteratorSourceGizmo"""

    # Test with numpy arrays
    test_images = []
    expected_shapes = []

    # Create test images as numpy arrays
    img1 = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)  # BGR color
    img2 = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)  # Different size
    img3 = np.random.randint(0, 255, (50, 75), dtype=np.uint8)  # Grayscale

    test_images.extend([img1, img2, img3])
    expected_shapes.extend(
        [img1.shape, img2.shape, (img3.shape[0], img3.shape[1], 3)]
    )  # Grayscale converted to BGR

    def image_generator():
        for img in test_images:
            yield img

    # Test with generator
    source = streams.IteratorSourceGizmo(image_generator())
    sink = VideoSink()
    streams.Composition(source >> sink).start()

    # Verify tags
    assert streams.tag_video in source.get_tags()
    assert source.name in source.get_tags()

    # Verify correct number of frames
    assert sink.frames_cnt == len(test_images)

    # Verify each frame
    for i, (frame, expected_shape) in enumerate(zip(sink.frames, expected_shapes)):
        assert frame.data.shape == expected_shape

        # Check metadata
        video_meta = frame.meta.find_last(streams.tag_video)
        assert video_meta is not None
        assert video_meta[streams.VideoSourceGizmo.key_frame_width] == expected_shape[1]
        assert (
            video_meta[streams.VideoSourceGizmo.key_frame_height] == expected_shape[0]
        )
        assert (
            video_meta[streams.VideoSourceGizmo.key_fps] == 0.0
        )  # No FPS for generator
        assert (
            video_meta[streams.VideoSourceGizmo.key_frame_count] == -1
        )  # Unknown for generator
        assert video_meta[streams.VideoSourceGizmo.key_frame_id] == i
        assert streams.VideoSourceGizmo.key_timestamp in video_meta


def test_streams_iterator_source_file_paths():
    """Test IteratorSourceGizmo with file paths"""

    # Get some test image files
    import glob

    image_files = glob.glob("test/sample_dataset/chair/*.jpg")[:3]  # Use first 3 images

    if not image_files:
        pytest.skip("No test images found")

    def file_generator():
        for file_path in image_files:
            yield file_path

    source = streams.IteratorSourceGizmo(file_generator())
    sink = VideoSink()
    streams.Composition(source >> sink).start()

    assert sink.frames_cnt == len(image_files)

    # Verify each frame is properly loaded
    for i, frame in enumerate(sink.frames):
        assert isinstance(frame.data, np.ndarray)
        assert len(frame.data.shape) == 3  # Should be color image
        assert frame.data.shape[2] == 3  # Should have 3 channels (BGR)

        # Check metadata
        video_meta = frame.meta.find_last(streams.tag_video)
        assert video_meta is not None
        assert video_meta[streams.VideoSourceGizmo.key_frame_id] == i


def test_streams_iterator_source_pil_images():
    """Test IteratorSourceGizmo with PIL Images"""

    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not available")

    # Create test PIL images
    pil_images = []
    pil_images.append(Image.new("RGB", (100, 200), color="red"))
    pil_images.append(Image.new("RGB", (150, 100), color="green"))
    pil_images.append(Image.new("L", (80, 120), color=128))  # Grayscale

    def pil_generator():
        for img in pil_images:
            yield img

    source = streams.IteratorSourceGizmo(pil_generator())
    sink = VideoSink()
    streams.Composition(source >> sink).start()

    assert sink.frames_cnt == len(pil_images)

    expected_shapes = [
        (200, 100, 3),
        (100, 150, 3),
        (120, 80, 3),
    ]  # PIL uses (width, height), OpenCV uses (height, width)

    for i, (frame, expected_shape) in enumerate(zip(sink.frames, expected_shapes)):
        assert frame.data.shape == expected_shape
        assert isinstance(frame.data, np.ndarray)

        # Verify conversion from RGB to BGR happened correctly
        video_meta = frame.meta.find_last(streams.tag_video)
        assert video_meta is not None
        assert video_meta[streams.VideoSourceGizmo.key_frame_width] == expected_shape[1]
        assert (
            video_meta[streams.VideoSourceGizmo.key_frame_height] == expected_shape[0]
        )


def test_streams_iterator_source_mixed_formats():
    """Test IteratorSourceGizmo with mixed input formats"""

    # Create mixed format inputs
    test_inputs: list = []

    # Numpy array
    test_inputs.append(np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8))

    # Try to add PIL image if available
    try:
        from PIL import Image

        test_inputs.append(Image.new("RGB", (70, 80), color="blue"))
    except ImportError:
        pass

    # Try to add file path if images exist
    import glob

    image_files = glob.glob("test/sample_dataset/chair/*.jpg")
    if image_files:
        test_inputs.append(image_files[0])

    if len(test_inputs) < 2:
        pytest.skip("Not enough test inputs available")

    def mixed_generator():
        for inp in test_inputs:
            yield inp

    source = streams.IteratorSourceGizmo(mixed_generator())
    sink = VideoSink()
    streams.Composition(source >> sink).start()

    assert sink.frames_cnt == len(test_inputs)

    # Verify all frames are properly converted to numpy arrays
    for i, frame in enumerate(sink.frames):
        assert isinstance(frame.data, np.ndarray)
        assert len(frame.data.shape) == 3  # Should be 3D (H, W, C)
        assert frame.data.shape[2] == 3  # Should have 3 color channels

        video_meta = frame.meta.find_last(streams.tag_video)
        assert video_meta is not None
        assert video_meta[streams.VideoSourceGizmo.key_frame_id] == i


def test_streams_iterator_source_error_handling():
    """Test IteratorSourceGizmo error handling"""

    def invalid_generator():
        yield "non_existent_file.jpg"  # File that doesn't exist

    source = streams.IteratorSourceGizmo(invalid_generator())
    sink = VideoSink()

    # Should raise FileNotFoundError for non-existent file
    with pytest.raises(Exception):  # Could be wrapped in other exceptions
        streams.Composition(source >> sink).start()

    def unsupported_type_generator():
        yield 12345  # Unsupported type

    source2 = streams.IteratorSourceGizmo(unsupported_type_generator())
    sink2 = VideoSink()

    # Should raise RuntimeError for unsupported type
    with pytest.raises(Exception):  # Could be wrapped in other exceptions
        streams.Composition(source2 >> sink2).start()


def test_streams_iterator_source_empty():
    """Test IteratorSourceGizmo with empty iterator"""

    def empty_generator():
        return
        yield  # This will never execute

    source = streams.IteratorSourceGizmo(empty_generator())
    sink = VideoSink()
    streams.Composition(source >> sink).start()

    # Should handle empty generator gracefully
    assert sink.frames_cnt == 0


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
    resize_method = "nearest"
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


def test_streams_simple_ai(short_video, detection_model):
    """Test for AiSimpleGizmo"""

    bboxes = [[0, 0, 10, 20], [20, 30, 100, 200]]
    model = degirum_tools.RegionExtractionPseudoModel(bboxes, detection_model)

    source = streams.VideoSourceGizmo(short_video)
    ai = streams.AiSimpleGizmo(model)
    sink = VideoSink()

    streams.Composition(source >> ai >> sink).start()

    for frame in sink.frames:
        ai_meta = frame.meta.find_last(streams.tag_inference)
        assert ai_meta is not None
        assert isinstance(ai_meta, dg.postprocessor.InferenceResults)
        orig_meta = ai_meta.info
        assert isinstance(orig_meta, streams.StreamMeta)
        assert not (orig_meta is ai_meta)
        assert all(bbox == obj["bbox"] for bbox, obj in zip(bboxes, ai_meta.results))


def test_streams_simple_ai_construction(detection_model_name, zoo_dir):
    """Test for AiSimpleGizmo construction"""

    ai_base = streams.AiSimpleGizmo(
        detection_model_name,
        inference_host_address=dg.LOCAL,
        zoo_url=zoo_dir,
    )
    assert isinstance(ai_base.model, dg.model.Model)

    ai2 = streams.AiSimpleGizmo(
        detection_model_name,
        inference_host_address=dg.LOCAL,
        zoo_url=zoo_dir,
        measure_time=True,
        overlay_show_probabilities=True,
    )

    assert ai2.model.measure_time != ai_base.model.measure_time
    assert (
        ai2.model.overlay_show_probabilities != ai_base.model.overlay_show_probabilities
    )

    txt = f"""
    gizmos:
        source: VideoSourceGizmo
        ai:
            AiSimpleGizmo:
                model: {detection_model_name}
                inference_host_address: '@local'
                zoo_url: '{zoo_dir}'
                measure_time: true
                overlay_show_probabilities: true

    connections:
        - [source, ai]
    """

    c = streams.load_composition(txt)
    assert len(c._gizmos) == 2
    ai3 = c._gizmos[1]
    assert isinstance(ai3, streams.AiSimpleGizmo)
    assert ai3.model.measure_time
    assert ai3.model.overlay_show_probabilities


def test_streams_cropping_ai(short_video, detection_model):
    """Test for AiObjectDetectionCroppingGizmo"""

    model = detection_model

    source = streams.VideoSourceGizmo(short_video)
    ai = streams.AiSimpleGizmo(model)
    cropper = streams.AiObjectDetectionCroppingGizmo(["Car"], crop_extent=10.0)
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
            assert r != ai_meta.results[i]

            expected_crops -= 1
        else:
            assert crop_meta[cropper.key_is_last_crop]
            assert crop_meta[cropper.key_cropped_index] == -1
            assert crop_meta[cropper.key_cropped_result] is None

    # test for validate_bbox()

    class NoValidCropsGizmo(streams.AiObjectDetectionCroppingGizmo):
        def validate_bbox(
            self, result: dg.postprocessor.InferenceResults, idx: int
        ) -> bool:
            return False

    source = streams.VideoSourceGizmo(short_video)
    ai = streams.AiSimpleGizmo(model)
    non_valid_cropper = NoValidCropsGizmo(["Car"], send_original_on_no_objects=False)
    sink = VideoSink()
    streams.Composition(source >> ai >> non_valid_cropper >> sink).start()
    assert sink.frames_cnt == 0

    source = streams.VideoSourceGizmo(short_video)
    ai = streams.AiSimpleGizmo(model)
    non_valid_cropper = NoValidCropsGizmo(["Car"], send_original_on_no_objects=True)
    sink = VideoSink()
    streams.Composition(source >> ai >> non_valid_cropper >> sink).start()
    assert sink.frames_cnt == source.result_cnt

    class SomeValidCropsGizmo(streams.AiObjectDetectionCroppingGizmo):
        def validate_bbox(
            self, result: dg.postprocessor.InferenceResults, idx: int
        ) -> bool:
            return idx == 0

    source = streams.VideoSourceGizmo(short_video)
    ai = streams.AiSimpleGizmo(model)
    valid_cropper = SomeValidCropsGizmo(["Car"], send_original_on_no_objects=False)
    sink = VideoSink()
    streams.Composition(source >> ai >> valid_cropper >> sink).start()
    assert sink.frames_cnt > 0 and sink.frames_cnt < source.result_cnt
    for frame in sink.frames:
        crop_meta = frame.meta.find_last(streams.tag_crop)
        assert crop_meta is not None
        assert crop_meta[valid_cropper.key_cropped_index] == 0
        assert crop_meta[valid_cropper.key_is_last_crop]


def test_streams_combining_ai(short_video, zoo_dir, detection_model):
    """Test for AiResultCombiningGizmo"""

    N = 3
    bboxes = [[0, 0, 10 * i, 10 * i] for i in range(1, N + 1)]

    models = [
        degirum_tools.RegionExtractionPseudoModel([bboxes[i]], detection_model)
        for i in range(N)
    ]

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
        assert L == N
        for i, r in enumerate(ai_meta.results):
            assert r["bbox"] == bboxes[i]


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


def test_streams_analyzer_ai(short_video, detection_model):
    """Test for AiAnalyzerGizmo"""

    model = degirum_tools.RegionExtractionPseudoModel([[0, 0, 10, 10]], detection_model)

    class TestAnalyzer(degirum_tools.ResultAnalyzerBase):

        def __init__(self, level: int, *, event: str = "", notification: str = ""):
            self.level = level
            self.event = event
            self.notification = notification

        def analyze(self, result):
            setattr(result, f"attr{self.level}", 1)
            if self.event:
                result.events_detected = {self.event}
            if self.notification:
                result.notifications = {self.notification}
            return result

        def annotate(self, result, image):
            image[self.level, self.level] = (self.level, 0, 0)
            return image

    N = 3
    analyzers = [TestAnalyzer(i) for i in range(N)]

    source = streams.VideoSourceGizmo(short_video)
    ai = streams.AiSimpleGizmo(model)
    analyzer = streams.AiAnalyzerGizmo(analyzers)
    sink = VideoSink()

    streams.Composition(source >> ai >> analyzer >> sink).start()

    for frame in sink.frames:
        ai_meta = frame.meta.find_last(streams.tag_inference)
        assert ai_meta is not None
        img = ai_meta.image_overlay
        for i in range(N):
            assert hasattr(ai_meta, f"attr{i}") and getattr(ai_meta, f"attr{i}") == 1
            assert np.array_equal(img[i, i], [i, 0, 0])

    stream_size = sink.frames_cnt

    #
    # test filters
    #
    event_name = "event1"
    notification_name = "notification1"

    def check_filters(
        generate_event_name, generate_notification_name, expected_result_cnt
    ):
        analyzers = [
            TestAnalyzer(
                0, event=generate_event_name, notification=generate_notification_name
            )
        ]
        source = streams.VideoSourceGizmo(short_video)
        ai = streams.AiSimpleGizmo(model)
        analyzer = streams.AiAnalyzerGizmo(
            analyzers, filters={event_name, notification_name}
        )
        sink = VideoSink()
        streams.Composition(source >> ai >> analyzer >> sink).start()
        assert sink.frames_cnt == expected_result_cnt

    check_filters("", "", 0)
    check_filters(event_name, "", stream_size)
    check_filters("", notification_name, stream_size)


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


def test_streams_sink(short_video):
    """Test for SinkGizmo"""

    source = streams.VideoSourceGizmo(short_video)
    sink = streams.SinkGizmo()

    with streams.Composition(source >> sink):
        nresults = 0
        for r in sink():
            assert r.meta.find_last(streams.tag_video) is not None
            nresults += 1

        assert nresults == source.result_cnt


def test_streams_crop_combining(
    short_video, detection_model, classification_model, regression_model
):
    """Test for CropCombiningGizmo"""

    obj_label = "Car"
    source = streams.VideoSourceGizmo(short_video)
    detector = streams.AiSimpleGizmo(detection_model)
    crop = streams.AiObjectDetectionCroppingGizmo([obj_label])
    classifier1 = streams.AiSimpleGizmo(classification_model)
    classifier2 = streams.AiSimpleGizmo(regression_model)
    combiner = streams.CropCombiningGizmo(2)
    sink = VideoSink()

    streams.Composition(
        source >> detector >> crop,
        source >> combiner[0],
        crop >> classifier1 >> combiner[1],
        crop >> classifier2 >> combiner[2],
        combiner >> sink,
    ).start()

    nframes = source.result_cnt
    assert (
        detector.result_cnt == nframes
        and crop.result_cnt > nframes
        and classifier1.result_cnt > nframes
        and classifier2.result_cnt > nframes
        and combiner.result_cnt == nframes
        and sink.frames_cnt == nframes
    )

    for frame in sink.frames:
        ai_meta = frame.meta.find_last(streams.tag_inference)
        if ai_meta is not None:
            assert isinstance(ai_meta, dg.postprocessor.DetectionResults)
            for r in ai_meta.results:
                assert streams.CropCombiningGizmo.key_extra_results in r
                extra_results = r[streams.CropCombiningGizmo.key_extra_results]
                assert len(extra_results) == 2
                assert all(
                    isinstance(r, dg.postprocessor.ClassificationResults)
                    for r in extra_results
                )
                assert extra_results[0].results[0]["label"] == obj_label
                assert extra_results[1].results[0]["label"] == "Age"


def test_streams_load_composition(zoo_dir, detection_model_name):
    """Test for load_composition"""

    import tempfile, yaml

    #
    # check various ways to load composition
    #

    txt = """
    gizmos:
        source: VideoSourceGizmo
        resizer:
            ResizingGizmo:
                w: 320
                h: 240
                pad_method: "stretch"
        display:
            VideoDisplayGizmo:
                window_titles: ["Original", "Resized"]
                show_fps: True

    connections:
        - [source, [display, 0]]
        - [source, resizer, [display, 1]]
    """

    def check(desc):
        c = streams.load_composition(desc)
        assert c is not None
        assert len(c._gizmos) == 3
        gizmo_classes = [g.__class__ for g in c._gizmos]
        assert streams.VideoSourceGizmo is gizmo_classes[0]
        assert streams.ResizingGizmo is gizmo_classes[1]
        assert streams.VideoDisplayGizmo is gizmo_classes[2]

        assert len(c._gizmos[0]._output_refs) == 2
        assert len(c._gizmos[1]._output_refs) == 1
        assert len(c._gizmos[2]._output_refs) == 0
        assert c._gizmos[0]._output_refs[0] is c._gizmos[2].get_input(0)
        assert c._gizmos[0]._output_refs[1] is c._gizmos[1].get_input(0)
        assert c._gizmos[1]._output_refs[0] is c._gizmos[2].get_input(1)

    # check YAML text
    check(txt)

    # check Python dict
    check(yaml.safe_load(txt))

    fname = ""

    # check YAML file
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False
        ) as yaml_file:
            yaml_file.write(txt)
            fname = yaml_file.name
        check(fname)
    finally:
        os.unlink(fname)

    #
    # check for gizmos in context
    #
    class MyGizmo(streams.Gizmo):
        def __init__(self):
            super().__init__([(0, False)])

        def run(self):
            pass

    txt2 = """
    gizmos:
        source: VideoSourceGizmo
        mygizmo: MyGizmo

    connections:
        - [source, mygizmo]
    """

    c = streams.load_composition(txt2, local_context=locals())
    assert len(c._gizmos) == 2
    assert MyGizmo is c._gizmos[1].__class__

    #
    # check for custom YAML constructors
    #

    class MyGizmo2(streams.Gizmo):
        def __init__(
            self, crop_extent: streams.CropExtentOptions, cv2_interpolation: int, model
        ):
            super().__init__([(0, False)])
            self.crop_extent = crop_extent
            self.cv2_interpolation = cv2_interpolation
            self.model = model

        def run(self):
            pass

    txt2 = """
    vars:
        model:
            dg.load_model:
                model_name: $(detection_model_name)
                inference_host_address: $(dg.LOCAL)
                zoo_url: $(zoo_dir)
    gizmos:
        source: VideoSourceGizmo
        mygizmo:
            MyGizmo2:
                crop_extent: $(CropExtentOptions.ASPECT_RATIO_ADJUSTMENT_BY_AREA)
                cv2_interpolation: $(cv2.INTER_LINEAR)
                model: $(model)

    connections:
        - [source, mygizmo]
    """

    c = streams.load_composition(txt2, local_context=locals())
    assert len(c._gizmos) == 2
    g2 = c._gizmos[1]
    assert isinstance(g2, MyGizmo2)
    assert g2.crop_extent == streams.CropExtentOptions.ASPECT_RATIO_ADJUSTMENT_BY_AREA
    assert g2.cv2_interpolation == cv2.INTER_LINEAR
    assert (
        isinstance(g2.model, dg.model.Model)
        and g2.model._model_name == detection_model_name
    )

    #
    # test variables
    #

    class ParamGizmo(streams.Gizmo):
        def __init__(self, **kwargs):
            super().__init__([(0, False)])
            self.params = kwargs

        def run(self):
            pass

    def test_func(val):
        return val

    txt3 = """
    vars:
        tracker:
            ObjectTracker:
                class_list: ["Car"]
                anchor_point: $(AnchorPoint.CENTER)
        t2: $(tracker)
        i: 10
        f: 3.14
        b: true
        a:
            test_func:
                val: [1, 2]

    gizmos:
        source: VideoSourceGizmo
        sink:
            ParamGizmo:
                t: $(t2)
                i: $(i)
                f: $(f)
                b: $(b)
                a: $(a)

    connections:
        - [source, sink]
    """

    c = streams.load_composition(txt3, local_context=locals())
    assert len(c._gizmos) == 2
    g2 = c._gizmos[1]
    assert isinstance(g2, ParamGizmo)
    assert isinstance(g2.params["t"], degirum_tools.ObjectTracker)
    assert g2.params["t"]._anchor_point == degirum_tools.AnchorPoint.CENTER
    assert g2.params["i"] == 10
    assert g2.params["f"] == 3.14
    assert g2.params["b"] is True
    assert g2.params["a"] == [1, 2]

    #
    # check for various errors
    #

    with pytest.raises(RuntimeError, match="fail to evaluate expression"):
        streams.load_composition(
            """
            gizmos:
                source: $(undefined_variable)
            connections:
                - [source]
            """
        )

    with pytest.raises(ValueError, match="class is not defined"):
        streams.load_composition(
            """
            gizmos:
                source:
                    UndefinedClass: {}
            connections:
                - [source]
            """
        )

    class FailingGizmo(streams.Gizmo):
        pass

    with pytest.raises(RuntimeError, match="error creating instance of"):
        streams.load_composition(
            """
            gizmos:
                source:
                    FailingGizmo: {}
            connections:
                - [source]
            """,
            local_context=locals(),
        )

    def failing_func():
        raise ValueError("Test exception")

    with pytest.raises(RuntimeError, match="error creating instance from"):
        streams.load_composition(
            """
            gizmos:
                source: failing_func
            connections:
                - [source]
            """,
            local_context=locals(),
        )

    with pytest.raises(ValueError, match="does not define a Gizmo"):
        streams.load_composition(
            """
            gizmos:
                source: UndefinedClass
            connections:
                - [source]
            """
        )

    with pytest.raises(ValueError, match="must have at least two elements"):
        streams.load_composition(
            """
            gizmos:
                source: VideoSourceGizmo
            connections:
                - [source]
            """
        )

    with pytest.raises(ValueError, match="'unknown' gizmo is not defined"):
        streams.load_composition(
            """
            gizmos:
                source: VideoSourceGizmo
            connections:
                - [unknown, source]
            """
        )

    with pytest.raises(ValueError, match="'unknown' gizmo is not defined"):
        streams.load_composition(
            """
            gizmos:
                source: VideoSourceGizmo
            connections:
                - [source, unknown]
            """
        )
