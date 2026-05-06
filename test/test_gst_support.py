#
# test_gst_support.py: unit tests for GstPipelineHandler and GstElementBase
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#

import struct
import threading
from typing import List

import pytest


def test_gst_support():
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

    class CounterSource(GstElementBase, Gst.Element):
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
            self.wait_for_playing()
            src_pad = self.sources["src"]
            src_pad.start_stream()
            for i in range(NUM_FRAMES):
                ret = src_pad.push_bytes(struct.pack(">Q", i))
                if ret != Gst.FlowReturn.OK:
                    break
            src_pad.stop_stream()

    class SplitterElement(GstElementBase, Gst.Element):
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
            self.wait_for_playing()

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
