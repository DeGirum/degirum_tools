#
# test_gst_support.py: unit tests for GstPipelineHandler and GstElementBase
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#

import struct
import threading
import pytest
from unittest.mock import patch
from typing import List
from pathlib import Path


def test_gst_pipeline_handler():
    """Test setup_gst_environment, GstPipelineHandler, and GstElementBase."""

    # ------------------------------------------------------------------
    # 1. Initialise GStreamer
    # ------------------------------------------------------------------
    from degirum_tools.gst import (
        GstElementBase,
        GstPipelineHandler,
        map_gst_buffer,
        setup_gst_environment,
    )

    try:
        setup_gst_environment()
    except ImportError:
        pytest.skip("gi not available")

    from gi.repository import Gst

    # ------------------------------------------------------------------
    # 2. Define custom elements
    # ------------------------------------------------------------------

    # Number of integer frames each CounterSource will emit.
    NUM_FRAMES = 1000

    # Caps used throughout: a single-channel 8-byte "application" buffer.
    _CAPS = "application/x-raw"

    class CounterSource(GstElementBase):
        """Source element: emits NUM_FRAMES buffers, each containing a single
        big-endian uint64 counter value (0, 1, …, NUM_FRAMES-1), then EOS."""

        @classmethod
        def get_metadata(cls) -> GstElementBase.ElementMetadata:
            return GstElementBase.ElementMetadata(
                longname="Counter Source",
                klass="Source",
                description="Generates sequential integer buffers",
                author="Test",
            )

        @classmethod
        def get_pads(cls) -> List[GstElementBase.PadInfo]:
            return [
                GstElementBase.PadInfo(
                    name="src",
                    direction=GstElementBase.PadDirection.SRC,
                    caps=_CAPS,
                )
            ]

        def _worker_func(self) -> None:
            src_pad = self.sources["src"]
            src_pad.start_stream()
            for i in range(NUM_FRAMES):
                ret = src_pad.push_bytes(struct.pack(">Q", i))
                if ret != Gst.FlowReturn.OK:
                    break
            src_pad.stop_stream()

    class SplitterElement(GstElementBase):
        """Filter element: one sink, two src pads.

        * ``src_double`` — the counter value multiplied by 2 (uint64 BE).
        * ``src_string`` — the counter value as an ASCII decimal string.
        """

        @classmethod
        def get_metadata(cls) -> GstElementBase.ElementMetadata:
            return GstElementBase.ElementMetadata(
                longname="Splitter Element",
                klass="Filter",
                description="Doubles the value on one pad, stringifies on another",
                author="Test",
            )

        @classmethod
        def get_pads(cls) -> List[GstElementBase.PadInfo]:
            return [
                GstElementBase.PadInfo(
                    name="sink",
                    direction=GstElementBase.PadDirection.SINK,
                    caps=_CAPS,
                    queue_drop=False,  # keep all points
                ),
                GstElementBase.PadInfo(
                    name="src_double",
                    direction=GstElementBase.PadDirection.SRC,
                    caps=_CAPS,
                ),
                GstElementBase.PadInfo(
                    name="src_string",
                    direction=GstElementBase.PadDirection.SRC,
                    caps=_CAPS,
                ),
            ]

        def _worker_func(self) -> None:

            sink = self.sinks["sink"]
            src_double = self.sources["src_double"]
            src_string = self.sources["src_string"]
            src_double.start_stream()
            src_string.start_stream()

            for buf in sink.queue:
                # Extract value
                with sink.map_buffer(buf) as data:
                    (value,) = struct.unpack(">Q", bytes(data))

                # Push doubled value
                src_double.push_bytes(struct.pack(">Q", value * 2))

                # Push string representation (may return NOT_LINKED
                # when src_string is not connected in the pipeline).
                src_string.push_bytes(str(value).encode())

            # Worker owns EOS for both src pads.
            src_double.stop_stream()
            src_string.stop_stream()

    # ------------------------------------------------------------------
    # 3. Register elements
    # ------------------------------------------------------------------

    COUNTER_SOURCE = "test_countersource"
    SPLITTER_ELEMENT = "test_splitterelement"
    SINK_DOUBLE = "sink_double"
    SINK_STRING = "sink_string"

    assert CounterSource.register(COUNTER_SOURCE)
    assert SplitterElement.register(SPLITTER_ELEMENT)

    # ------------------------------------------------------------------
    # Helper: run one pipeline and verify both appsink outputs
    # ------------------------------------------------------------------
    concurrent_errors: List[str] = []

    def run_pipeline(tag: str):
        pipe_str = (
            f"{COUNTER_SOURCE} ! {SPLITTER_ELEMENT} name=split "
            f"split.src_double ! queue ! appsink name={SINK_DOUBLE} "
            f"split.src_string ! queue ! appsink name={SINK_STRING}"
        )
        h = GstPipelineHandler(pipe_str, appsink_names=[SINK_DOUBLE, SINK_STRING])
        h.start()
        h.wait()

        errors: List[str] = []

        # Check doubled-value path
        received_doubles: List[int] = []
        for sample in h.appsinks[SINK_DOUBLE].queue:
            try:
                with map_gst_buffer(sample) as data:
                    (val,) = struct.unpack(">Q", bytes(data))
            except RuntimeError:
                errors.append(f"{tag}: double buffer map failed")
                break
            received_doubles.append(val)

        expected_doubles = [i * 2 for i in range(NUM_FRAMES)]
        if received_doubles != expected_doubles:
            errors.append(
                f"{tag}: doubles expected {expected_doubles}, got {received_doubles}"
            )

        # Check string path
        received_strings: List[str] = []
        for sample in h.appsinks[SINK_STRING].queue:
            try:
                with map_gst_buffer(sample) as data:
                    received_strings.append(bytes(data).decode())
            except RuntimeError:
                errors.append(f"{tag}: string buffer map failed")
                break

        expected_strings = [str(i) for i in range(NUM_FRAMES)]
        if received_strings != expected_strings:
            errors.append(
                f"{tag}: strings expected {expected_strings}, got {received_strings}"
            )

        if errors:
            concurrent_errors.extend(errors)
            assert not errors, f"Pipeline errors: {errors}"

    # ------------------------------------------------------------------
    # 4. Build and run a single pipeline, verify appsink output
    # ------------------------------------------------------------------
    run_pipeline("single")

    # ------------------------------------------------------------------
    # 5. Concurrent execution of two independent pipelines
    # ------------------------------------------------------------------
    threads = [
        threading.Thread(target=run_pipeline, args=(f"concurrent-{i}",))
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "Concurrent pipeline thread timed out"

    assert not concurrent_errors, f"Concurrent pipeline errors: {concurrent_errors}"


def test_gst_pipeline_builder(tmp_path):
    """Test build_gst_pipeline"""

    from degirum_tools.gst import build_gst_pipeline

    # --- Custom GStreamer pipeline strings are returned unchanged ---
    custom = "v4l2src ! videoconvert ! appsink name=sink"
    assert build_gst_pipeline(custom) == custom

    custom_complex = "videotestsrc ! video/x-raw,format=BGR ! appsink name=out"
    assert build_gst_pipeline(custom_complex) == custom_complex

    # --- RTSP URLs produce an rtspsrc-based pipeline ---
    url = "rtsp://192.168.1.10/stream"
    result = build_gst_pipeline(url)
    assert f'rtspsrc location="{url}"' in result
    assert "decodebin" in result
    assert "appsink" in result
    assert "media=video" in result  # audio-isolation cap filter
    assert "format=BGR" in result

    # RTSP detection is case-insensitive
    result = build_gst_pipeline("RTSP://CAM/stream")
    assert "rtspsrc" in result
    assert "format=BGR" in result

    # --- File paths produce a filesrc ! decodebin pipeline ---
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"")
    result = build_gst_pipeline(str(video_file))
    assert "filesrc" in result
    assert "decodebin" in result
    assert "appsink" in result
    assert "format=BGR" in result
    # GStreamer requires forward slashes in location even on Windows
    location = result.split('location="')[1].split('"')[0]
    assert "\\" not in location

    # Nested subdirectory path also uses forward slashes
    nested = tmp_path / "subdir" / "video.avi"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(b"")
    nested_result = build_gst_pipeline(str(nested))
    location = nested_result.split('location="')[1].split('"')[0]
    assert "\\" not in location
    assert "format=BGR" in nested_result

    # --- Unknown sources raise ValueError ---
    with pytest.raises(ValueError):
        build_gst_pipeline("/nonexistent/does_not_exist.mp4")
    with pytest.raises(ValueError):
        build_gst_pipeline("totally_unknown_source")

    _PB = "degirum_tools.gst.pipeline_builder"

    # --- Camera: Windows, mfvideosrc available ---
    with (
        patch(f"{_PB}._detect_platform", return_value="windows"),
        patch(f"{_PB}._detect_camera_type", return_value="usb"),
        patch(f"{_PB}._check_element_exists", return_value=True),
    ):
        result = build_gst_pipeline(0)
    assert (
        "mfvideosrc" in result
        and "device-index=0" in result
        and "appsink" in result
        and "format=BGR" in result
    )

    # --- Camera: Windows, fallback to ksvideosrc ---
    with (
        patch(f"{_PB}._detect_platform", return_value="windows"),
        patch(f"{_PB}._detect_camera_type", return_value="usb"),
        patch(f"{_PB}._check_element_exists", return_value=False),
    ):
        result = build_gst_pipeline(2)
    assert (
        "ksvideosrc" in result
        and "device-index=2" in result
        and "appsink" in result
        and "format=BGR" in result
    )

    # --- Camera: Linux generic USB ---
    with (
        patch(f"{_PB}._detect_platform", return_value="generic"),
        patch(f"{_PB}._detect_camera_type", return_value="usb"),
    ):
        result = build_gst_pipeline(1)
    assert (
        "v4l2src" in result
        and "device=/dev/video1" in result
        and "appsink" in result
        and "format=BGR" in result
    )

    # Digit string is treated as a device index
    with (
        patch(f"{_PB}._detect_platform", return_value="generic"),
        patch(f"{_PB}._detect_camera_type", return_value="usb"),
    ):
        result = build_gst_pipeline("3")
    assert (
        "v4l2src" in result
        and "device=/dev/video3" in result
        and "format=BGR" in result
    )

    # --- Camera: Raspberry Pi CSI, libcamerasrc available ---
    with (
        patch(f"{_PB}._detect_platform", return_value="raspberrypi"),
        patch(f"{_PB}._detect_camera_type", return_value="rpi_csi"),
        patch(f"{_PB}._check_element_exists", return_value=True),
    ):
        result = build_gst_pipeline(0)
    assert "libcamerasrc" in result and "appsink" in result and "format=BGR" in result

    # --- Camera: Raspberry Pi CSI, fallback to v4l2src ---
    with (
        patch(f"{_PB}._detect_platform", return_value="raspberrypi"),
        patch(f"{_PB}._detect_camera_type", return_value="rpi_csi"),
        patch(f"{_PB}._check_element_exists", return_value=False),
    ):
        result = build_gst_pipeline(0)
    assert (
        "v4l2src" in result
        and "device=/dev/video0" in result
        and "format=BGR" in result
    )


def test_gst_open_video_stream():
    """Test open_video_stream() with use_gstreamer=True on a video file."""

    from degirum_tools.tools.video_support import (
        open_video_stream,
        video_source,
        VideoCaptureGst,
    )

    # Skip if GStreamer is not available
    try:
        from degirum_tools.gst import setup_gst_environment

        setup_gst_environment()
    except ImportError:
        pytest.skip("gi module not available")

    VIDEO_FILE = str(Path(__file__).parent / "images" / "Traffic2_short.mp4")

    # Verify open_video_stream does not raise and returns a VideoCaptureGst object
    with open_video_stream(VIDEO_FILE, use_gstreamer=True) as stream:
        assert isinstance(
            stream, VideoCaptureGst
        ), f"Expected VideoCaptureGst, got {type(stream)}"

        # Read all frames via GStreamer
        gst_frames = sum(1 for _ in video_source(stream))

    # Read all frames via OpenCV baseline
    with open_video_stream(VIDEO_FILE, use_gstreamer=False) as stream_cv:
        cv_frames = sum(1 for _ in video_source(stream_cv))

    assert gst_frames > 0, "GStreamer read zero frames"
    assert (
        gst_frames == cv_frames
    ), f"Frame count mismatch: GStreamer={gst_frames}, OpenCV={cv_frames}"


def test_gst_element_properties():
    """Test GstElementBase GObject property support.

    Covers:
    - All scalar types (bool, int, float, str) and Python-object type.
    - Default values after element creation.
    - Round-trip via set_property / get_property.
    - Round-trip via gi props attribute shorthand (el.props.xxx).
    - Direct access to the internal _props dict.
    - Values set from a pipeline string.
    """

    from degirum_tools.gst import (
        GstElementBase,
        GstPipelineHandler,
        setup_gst_environment,
    )

    try:
        setup_gst_environment()
    except ImportError:
        pytest.skip("gi not available")

    from gi.repository import Gst

    _CAPS = "application/x-raw"

    class PropTestElement(GstElementBase):
        """Sink-only element exposing all property types for testing."""

        @classmethod
        def get_metadata(cls) -> GstElementBase.ElementMetadata:
            return GstElementBase.ElementMetadata(
                longname="Prop Test Element",
                klass="Sink",
                description="Element for testing GObject property support",
                author="Test",
            )

        @classmethod
        def get_pads(cls) -> List[GstElementBase.PadInfo]:
            return [
                GstElementBase.PadInfo(
                    name="sink",
                    direction=GstElementBase.PadDirection.SINK,
                    caps=_CAPS,
                )
            ]

        @classmethod
        def get_properties(cls) -> List[GstElementBase.PropInfo]:
            return [
                GstElementBase.PropInfo("prop-bool", False, "Boolean property"),
                GstElementBase.PropInfo(
                    "prop-int", 0, "Integer property", min=-1000, max=1000
                ),
                GstElementBase.PropInfo("prop-float", 0.0, "Float property"),
                GstElementBase.PropInfo("prop-str", "", "String property"),
                GstElementBase.PropInfo("prop-obj", None, "Python-object property"),
            ]

        def _worker_func(self) -> None:
            for _ in self.sinks["sink"].queue:
                pass

    PROPS_ELEMENT = "test_proptestelem"
    assert PropTestElement.register(PROPS_ELEMENT)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def assert_props(el, expected: dict):
        """Assert all expected values via get_property, gi shorthand, and _props."""
        for hyphen_name, value in expected.items():
            underscore_name = hyphen_name.replace("-", "_")

            actual_get = el.get_property(hyphen_name)
            actual_gi = getattr(el.props, underscore_name)
            actual_dict = el._props[hyphen_name]

            for actual, label in [
                (actual_get, "get_property"),
                (actual_gi, "props shorthand"),
                (actual_dict, "_props dict"),
            ]:
                if isinstance(value, float):
                    assert (
                        abs(actual - value) < 1e-9
                    ), f"{hyphen_name} via {label}: expected {value}, got {actual}"
                else:
                    assert (
                        actual == value
                    ), f"{hyphen_name} via {label}: expected {value!r}, got {actual!r}"

    def set_props_api(el, values: dict):
        """Set properties via set_property."""
        for name, value in values.items():
            el.set_property(name, value)

    def set_props_gi(el, values: dict):
        """Set properties via gi props shorthand."""
        for hyphen_name, value in values.items():
            setattr(el.props, hyphen_name.replace("-", "_"), value)

    # ------------------------------------------------------------------
    # 1. Default values after element creation
    # ------------------------------------------------------------------
    el = Gst.ElementFactory.make(PROPS_ELEMENT, "el_defaults")
    assert el is not None

    assert_props(
        el,
        {
            "prop-bool": False,
            "prop-int": 0,
            "prop-float": 0.0,
            "prop-str": "",
            "prop-obj": None,
        },
    )

    # ------------------------------------------------------------------
    # 2. Round-trip via set_property / get_property
    # ------------------------------------------------------------------
    set_props_api(
        el,
        {
            "prop-bool": True,
            "prop-int": 42,
            "prop-float": 3.14,
            "prop-str": "hello",
            "prop-obj": [1, 2, 3],
        },
    )
    assert_props(
        el,
        {
            "prop-bool": True,
            "prop-int": 42,
            "prop-float": 3.14,
            "prop-str": "hello",
            "prop-obj": [1, 2, 3],
        },
    )

    # ------------------------------------------------------------------
    # 3. Round-trip via gi props shorthand
    # ------------------------------------------------------------------
    set_props_gi(
        el,
        {
            "prop-bool": False,
            "prop-int": -7,
            "prop-float": 2.718,
            "prop-str": "world",
            "prop-obj": {"key": "value"},
        },
    )
    assert_props(
        el,
        {
            "prop-bool": False,
            "prop-int": -7,
            "prop-float": 2.718,
            "prop-str": "world",
            "prop-obj": {"key": "value"},
        },
    )

    # ------------------------------------------------------------------
    # 4. Values set from a pipeline string (scalar types only)
    # ------------------------------------------------------------------
    # Wrap in a GstPipelineHandler so element() can retrieve by name.
    pipe_str = (
        f"{PROPS_ELEMENT} name=pe "
        f"prop-bool=true prop-int=99 prop-float=1.5 prop-str=pipeline"
    )
    handler = GstPipelineHandler(pipe_str)
    pe = handler.element("pe")

    assert_props(
        pe,
        {
            "prop-bool": True,
            "prop-int": 99,
            "prop-float": 1.5,
            "prop-str": "pipeline",
            # Object property is not settable from a pipeline string; default must be kept.
            "prop-obj": None,
        },
    )

    pe.props.prop_bool = False
    pe.props.prop_int = -123
    pe.props.prop_float = 0.001
    pe.props.prop_str = "modified"
    assert_props(
        pe,
        {
            "prop-bool": False,
            "prop-int": -123,
            "prop-float": 0.001,
            "prop-str": "modified",
            "prop-obj": None,
        },
    )


def test_gst_worker_exception():
    """Test that an exception raised inside _worker_func is propagated by handler.wait()."""

    from degirum_tools.gst import (
        GstElementBase,
        GstPipelineHandler,
        setup_gst_environment,
    )

    try:
        setup_gst_environment()
    except ImportError:
        pytest.skip("gi not available")

    class BrokenSink(GstElementBase):
        """Sink element whose worker always raises a RuntimeError."""

        @classmethod
        def get_metadata(cls) -> GstElementBase.ElementMetadata:
            return GstElementBase.ElementMetadata(
                longname="Broken Sink",
                klass="Sink",
                description="Always raises in _worker_func",
                author="Test",
            )

        @classmethod
        def get_pads(cls) -> List[GstElementBase.PadInfo]:
            return [
                GstElementBase.PadInfo(
                    name="sink",
                    direction=GstElementBase.PadDirection.SINK,
                    caps="video/x-raw",
                )
            ]

        def _worker_func(self) -> None:
            raise RuntimeError("intentional worker failure")

    BROKEN_SINK = "test_brokensink"
    assert BrokenSink.register(BROKEN_SINK)

    pipe_str = f"videotestsrc ! {BROKEN_SINK}"
    handler = GstPipelineHandler(pipe_str)
    handler.start()

    with pytest.raises(RuntimeError, match="intentional worker failure"):
        handler.wait()


def test_gst_ai_element():
    """Test GstAiElement in four operating modes using Traffic2_short.mp4."""

    import json as _json
    import numpy as np

    from degirum_tools.gst import (
        GstAiElement,
        GstPipelineHandler,
        map_gst_buffer,
        setup_gst_environment,
    )

    try:
        setup_gst_environment()
    except ImportError:
        pytest.skip("gi not available")

    AI_ELEMENT = "test_aielement2"
    assert GstAiElement.register(AI_ELEMENT)

    video_path = str(Path(__file__).parent / "images" / "Traffic2_short.mp4").replace(
        "\\", "/"
    )

    _MODEL_NAME = "yolov8n_relu6_coco--640x640_quant_n2x_orca1_1"
    _ZOO_URL = "degirum/degirum"
    _HOST = "@cloud"

    _AI_PROPS = (
        f' model_name="{_MODEL_NAME}"'
        f' zoo_url="{_ZOO_URL}"'
        f' inference_host_address="{_HOST}"'
    )

    def _video_source() -> str:
        """Return the decode + colorconvert portion of the pipeline."""
        return (
            f'filesrc location="{video_path}" ! qtdemux ! h264parse ! avdec_h264 ! '
            f"videoconvert ! video/x-raw,format=RGB"
        )

    def _collect_frames(appsink_name: str, handler: GstPipelineHandler):
        """Return list of numpy frames from the named appsink."""
        frames = []
        for sample in handler.appsinks[appsink_name].queue:
            caps = sample.get_caps()
            s = caps.get_structure(0)
            w = s.get_value("width")
            h = s.get_value("height")
            with map_gst_buffer(sample) as data:
                frames.append(
                    np.frombuffer(bytes(data), dtype=np.uint8).reshape((h, w, 3)).copy()
                )
        return frames

    def _assert_frames_identical(frames_a, frames_b, tag: str) -> None:
        """Assert two frame sequences have the same length and identical pixel data."""
        assert len(frames_a) > 0, f"{tag}: no frames in first sequence"
        assert len(frames_b) > 0, f"{tag}: no frames in second sequence"
        assert len(frames_a) == len(
            frames_b
        ), f"{tag}: frame count mismatch {len(frames_a)} vs {len(frames_b)}"
        for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
            assert fa.shape == fb.shape, f"{tag}: frame {i} shape mismatch"
            assert np.array_equal(fa, fb), f"{tag}: frame {i} pixel data differs"

    def _assert_frames_annotated(
        frames_a, frames_b, tag: str, max_diff_fraction: float = 0.05
    ) -> None:
        """Assert that frames differ (AI overlay applied) but not excessively.

        The mean absolute per-pixel difference, normalized to [0, 1], must be
        greater than zero and no more than *max_diff_fraction*.
        """
        assert len(frames_a) > 0, f"{tag}: no frames in first sequence"
        assert len(frames_b) > 0, f"{tag}: no frames in second sequence"
        total_diff = 0.0
        total_pixels = 0
        for fa, fb in zip(frames_a, frames_b):
            assert fa.shape == fb.shape, f"{tag}: frame shape mismatch"
            total_diff += float(np.abs(fa.astype(np.int32) - fb.astype(np.int32)).sum())
            total_pixels += fa.size
        mean_diff_fraction = total_diff / (total_pixels * 255.0)
        assert (
            mean_diff_fraction > 0.0
        ), f"{tag}: no annotated frames — all output frames are identical to input"
        assert mean_diff_fraction <= max_diff_fraction, (
            f"{tag}: mean pixel difference {mean_diff_fraction:.3%} exceeds "
            f"threshold {max_diff_fraction:.0%}"
        )

    def _assert_json_has_detections(
        json_sink_name: str, handler: GstPipelineHandler, tag: str
    ) -> None:
        """Assert that at least one JSON sample has non-empty _inference_results
        and every result dict in that sample contains a 'bbox' key."""
        samples = list(handler.appsinks[json_sink_name].queue)
        assert len(samples) > 0, f"{tag}: no JSON samples received"
        for sample in samples:
            with map_gst_buffer(sample) as data:
                obj = _json.loads(bytes(data).decode())
            results = obj.get("_inference_results", [])
            if not results:
                continue
            assert all(
                "bbox" in r and "label" in r and "score" in r for r in results
            ), f"{tag}: some results in '_inference_results' are missing required keys"
            return
        assert False, f"{tag}: no non-empty '_inference_results' found in JSON output"

    # ------------------------------------------------------------------
    # Mode 1: Single input, single output, no JSON, no AI annotations.
    # Validate that output frames are identical to input frames.
    # ------------------------------------------------------------------
    pipe1 = (
        f"{_video_source()} ! tee name=t "
        f"t. ! appsink name=sink_in async=false "
        f"t. ! {AI_ELEMENT} name=ai1 ai_overlay=false {_AI_PROPS} "
        f"! appsink name=sink_out async=false"
    )
    h1 = GstPipelineHandler(pipe1, appsink_names=["sink_in", "sink_out"])
    h1.start()
    h1.wait()

    _assert_frames_identical(
        _collect_frames("sink_in", h1),
        _collect_frames("sink_out", h1),
        "Mode 1",
    )
    print("\nTest 1 done")

    # ------------------------------------------------------------------
    # Mode 2: Single input, single output, no JSON, with AI annotations.
    # Validate that output frames differ from input (annotations present).
    # ------------------------------------------------------------------
    pipe2 = (
        f"{_video_source()} ! tee name=t "
        f"t. ! queue max-size-buffers=0 ! appsink name=sink_in2 "
        f"t. ! queue ! {AI_ELEMENT} name=ai2 ai_overlay=true {_AI_PROPS} ! "
        f"appsink name=sink_out2"
    )
    h2 = GstPipelineHandler(pipe2, appsink_names=["sink_in2", "sink_out2"])
    h2.start()
    h2.wait()

    _assert_frames_annotated(
        _collect_frames("sink_in2", h2),
        _collect_frames("sink_out2", h2),
        "Mode 2",
    )
    print("Test 2 done")

    # ------------------------------------------------------------------
    # Mode 3: Single input, two outputs (video + JSON), no AI annotations.
    # Validate JSON contains detected objects.
    # ------------------------------------------------------------------
    pipe3 = (
        f"{_video_source()} ! "
        f"{AI_ELEMENT} name=ai3 ai_overlay=false {_AI_PROPS} "
        f"ai3.src ! appsink name=sink_video3 "
        f"ai3.src_json ! appsink name=sink_json3"
    )
    h3 = GstPipelineHandler(pipe3, appsink_names=["sink_video3", "sink_json3"])
    h3.start()
    h3.wait()

    _assert_json_has_detections("sink_json3", h3, "Mode 3")
    print("Test 3 done")

    # ------------------------------------------------------------------
    # Mode 4: Two inputs (main full-size + resized model input), two outputs
    # (video + JSON), no AI annotations.
    # Validate JSON contains detected objects.
    # Validate that output video frames are identical to full-size input frames.
    # ------------------------------------------------------------------
    pipe4 = (
        f"{_video_source()} ! tee name=t4 "
        # Full-size branch → sink_full pad AND input capture
        f"t4. ! queue ! tee name=t4_full "
        f"t4_full. ! queue ! ai4.sink_full "
        f"t4_full. ! queue max-size-buffers=0 ! appsink name=sink_full_in "
        # Resized branch → model input sink pad
        f"t4. ! queue ! videoscale ! video/x-raw,format=RGB,width=320,height=180 ! "
        f"ai4.sink "
        f"{AI_ELEMENT} name=ai4 ai_overlay=false {_AI_PROPS} "
        f"ai4.src ! appsink name=sink_video4 "
        f"ai4.src_json ! appsink name=sink_json4"
    )
    h4 = GstPipelineHandler(
        pipe4,
        appsink_names=["sink_full_in", "sink_video4", "sink_json4"],
    )
    h4.start()
    h4.wait()

    _assert_frames_identical(
        _collect_frames("sink_full_in", h4),
        _collect_frames("sink_video4", h4),
        "Mode 4",
    )
    _assert_json_has_detections("sink_json4", h4, "Mode 4")
    print("Test 4 done")
