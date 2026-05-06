#
# pipeline_handler.py: GstPipelineHandler for managing GStreamer pipeline lifecycle
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# Implements GstPipelineHandler for managing GStreamer pipeline lifecycle
#

"""
Pipeline Handler
================

This module provides ``GstPipelineHandler``, a class that manages the full lifecycle of a
single GStreamer pipeline — from creation and playback through to orderly teardown.

Key Features

- Accepts any valid ``gst-launch``-style pipeline description string.
- Optional throughput probing via a named element's ``src`` pad, accessible through ``check()``.
- Named ``appsink`` elements are automatically wrapped in ``AppSink`` helper objects that
  expose a ``streams.Stream`` queue for frame-by-frame consumption from Python.
- A single ``GLib.MainLoop`` is shared across all ``GstPipelineHandler`` instances and
  started automatically on first use.

Typical Usage

1. Construct a ``GstPipelineHandler`` with a pipeline string.
2. Optionally name ``appsink`` elements to receive decoded frames.
3. Call ``start()`` to transition the pipeline to ``PLAYING``.
4. Iterate over ``handler.appsinks["name"].queue`` to consume ``Gst.Sample`` objects.
5. Call ``wait()`` to block until EOS, or ``stop()`` to tear down early.

Example::

    from degirum_tools.gst import setup_gst_environment, GstPipelineHandler

    setup_gst_environment()

    handler = GstPipelineHandler(
        'filesrc location="video.mp4" ! decodebin ! videoconvert'
        ' ! video/x-raw,format=BGR ! appsink name=sink',
        appsink_names=["sink"],
        appsink_queue_maxsize=4,
        appsink_queue_drop=True,
    )
    handler.start()

    for sample in handler.appsinks["sink"].queue:
        # sample is a Gst.Sample; use map_gst_buffer() to access raw bytes
        ...

    handler.wait()

Public API

- ``GstPipelineHandler`` — pipeline lifecycle manager.
- ``GstPipelineHandler.AppSink`` — appsink wrapper with a frame queue.
- ``GstPipelineHandler.start()`` — set the pipeline to ``PLAYING``.
- ``GstPipelineHandler.stop()`` — transition the pipeline to ``NULL``.
- ``GstPipelineHandler.wait()`` — block until the pipeline reaches EOS or raises on error.
- ``GstPipelineHandler.check()`` — return ``(is_active, fps)`` throughput from the probed element.
"""

from __future__ import annotations

import threading
from typing import Dict, List

import concurrent.futures

from ..tools import Watchdog


class GstPipelineHandler:
    """Manages a single GStreamer pipeline lifecycle.

    Pass a fully-formed gst-launch pipeline string to the constructor;
    the pipeline starts immediately. Use `wait()` to block until it ends
    or `stop()` to tear it down early.

    A single GLib main loop shared across all instances is started
    automatically on first use.
    """

    _glib_loop = None
    _glib_loop_lock = threading.Lock()

    class AppSink:
        """Wraps a GStreamer appsink element with a sample queue.

        Installs a `new-sample` callback on the appsink so that every decoded
        frame is pulled and pushed into `queue` for downstream consumption.

        Args:
            name: Name of the `appsink` element in the pipeline.
            handler: Owning `GstPipelineHandler` instance whose pipeline
                contains the element.
            queue_maxsize: Maximum number of samples buffered in the queue.
                `0` means unlimited.
            queue_drop: When `True`, the oldest sample is silently discarded
                when the queue is full. When `False`, the producer blocks
                until space is available.

        Attributes:
            name: Element name passed at construction.
            element: The underlying ``appsink`` ``Gst.Element``.
            queue: `streams.Stream` that receives `Gst.Sample` objects.
                Yields samples as they arrive; a `None` sentinel is pushed
                when the pipeline reaches EOS.
        """

        def __init__(
            self,
            name: str,
            handler: "GstPipelineHandler",
            *,
            queue_maxsize: int,
            queue_drop: bool,
        ):
            from gi.repository import Gst

            self._Gst = Gst
            self.name = name

            self.element = handler._pipeline.get_by_name(name)
            if self.element is None:
                raise ValueError(f"appsink element '{name}' not found in pipeline")

            from .. import streams

            self.queue = streams.Stream(queue_maxsize, queue_drop)

            self.element.set_property("emit-signals", True)
            self.element.set_property("sync", False)
            self.element.connect("new-sample", self._on_new_sample)

        def _on_new_sample(self, appsink):
            sample = appsink.emit("pull-sample")
            if sample is not None:
                self.queue.put(sample)
            return self._Gst.FlowReturn.OK

        @property
        def sink_pad(self):
            """The static sink pad of the appsink element."""
            return self.element.get_static_pad("sink")

    @classmethod
    def _ensure_main_loop(cls):
        from gi.repository import GLib

        with cls._glib_loop_lock:
            if cls._glib_loop is None or not cls._glib_loop.is_running():
                cls._glib_loop = GLib.MainLoop()
                threading.Thread(
                    target=cls._glib_loop.run,
                    daemon=True,
                    name="glib-main-loop",
                ).start()

    def __init__(
        self,
        pipeline_str,
        probe_element_name="",
        appsink_names: List[str] = [],
        *,
        appsink_queue_maxsize: int = 0,
        appsink_queue_drop: bool = False,
    ):
        """Create and configure a GStreamer pipeline.

        The pipeline is parsed and configured but not started. Call `start()` to
        set it to PLAYING state.

        Args:
            pipeline_str: A gst-launch-style pipeline description string.
            probe_element_name: Name of an element whose `src` pad will be probed
                to measure throughput via `check()`. Leave empty to skip probing.
            appsink_names: Names of `appsink` elements to wrap with `AppSink`
                instances. The resulting objects are available in `self.appsinks`.
            appsink_queue_maxsize: Maximum number of samples buffered in each
                `AppSink` queue. `0` means unlimited. Defaults to `0`.
            appsink_queue_drop: When `True`, the oldest sample is silently
                discarded when the queue is full. When `False`, the producer
                blocks until space is available. Defaults to `False`.

        Raises:
            ValueError: If `probe_element_name` is set but the element or its
                `src` pad is not found, or if any name in `appsink_names` does
                not match an element in the pipeline.
        """
        from gi.repository import Gst

        self._Gst = Gst
        self._pipeline = self._Gst.parse_launch(pipeline_str)
        self._future: concurrent.futures.Future = concurrent.futures.Future()
        self._watchdog = (
            Watchdog(time_limit=5.0, tps_threshold=0.0) if probe_element_name else None
        )

        if probe_element_name:
            element = self._pipeline.get_by_name(probe_element_name)
            if element is None:
                raise ValueError(
                    f"Element '{probe_element_name}' not found in pipeline"
                )
            pad = element.get_static_pad("src")
            if pad is None:
                raise ValueError(f"Element '{probe_element_name}' has no src pad")
            pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer)

        #: Dict of appsink name → AppSink
        self.appsinks: Dict[str, GstPipelineHandler.AppSink] = {
            name: GstPipelineHandler.AppSink(
                name,
                self,
                queue_maxsize=appsink_queue_maxsize,
                queue_drop=appsink_queue_drop,
            )
            for name in appsink_names
        }

        def on_bus_message(bus, message):
            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                self._pipeline.set_state(Gst.State.NULL)
                if not self._future.done():
                    self._future.set_exception(RuntimeError(f"{err.message} ({debug})"))
            elif message.type == Gst.MessageType.EOS:
                for sink in self.appsinks.values():
                    sink.queue.put(None)
                if not self._future.done():
                    self._future.set_result(None)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", on_bus_message)

    def start(self, wait_timeout_s: float = 0.0):
        """Start the pipeline. Returns self for chaining.

        Args:
            wait_timeout_s: If > 0, block until the pipeline reaches PLAYING state
                or the timeout elapses, then raise ``RuntimeError`` if the state was
                not reached. Defaults to ``0.0`` (no wait).

        Raises:
            RuntimeError: If ``wait_timeout_s`` > 0 and the pipeline did not reach
                PLAYING within the specified timeout.
        """
        self._ensure_main_loop()
        self._pipeline.set_state(self._Gst.State.PLAYING)
        if wait_timeout_s > 0:
            timeout_ns = int(wait_timeout_s * self._Gst.SECOND)
            _, state, _ = self._pipeline.get_state(timeout_ns)
            if state != self._Gst.State.PLAYING:
                raise RuntimeError(
                    f"Pipeline did not reach PLAYING state within {wait_timeout_s}s"
                )
        return self

    def _on_buffer(self, pad, info):
        assert self._watchdog is not None
        self._watchdog.tick()
        return self._Gst.PadProbeReturn.OK

    def check(self):
        """Return current throughput.

        Returns:
            Tuple (is_active, fps).

        Raises:
            RuntimeError: If `probe_element_name` was not set at construction.
        """
        if self._watchdog is None:
            raise RuntimeError("check() requires probe_element_name to be set")
        return self._watchdog.check()

    def wait(self):
        """Block until the pipeline reaches EOS.

        Raises:
            RuntimeError: On pipeline error.
        """
        self._future.result()

    @property
    def pipeline(self):
        """The underlying ``Gst.Pipeline`` object."""
        return self._pipeline

    def stop(self):
        """Gracefully stop the pipeline."""
        self._pipeline.set_state(self._Gst.State.NULL)
        self._pipeline.get_state(timeout=2 * self._Gst.SECOND)
        if not self._future.done():
            self._future.set_result(None)
