#
# GStreamer support subpackage
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#

"""
GStreamer Support Subpackage
============================

This subpackage provides a complete toolkit for building and running GStreamer-based
video pipelines from Python, including high-level pipeline management, a framework for
custom GStreamer elements, and automatic pipeline string construction for common video
sources.

Key Features

- One-call GStreamer environment initialisation with optional custom plugin discovery.
- ``GstPipelineHandler`` for launching, monitoring, and gracefully stopping pipelines.
- ``GstElementBase`` mixin for implementing custom GStreamer Python elements with minimal
  boilerplate: pad templates, worker threads, and state management are handled
  automatically.
- ``build_gst_pipeline`` to create a ready-to-use pipeline string from a camera index,
  RTSP URL, file path, or a verbatim pipeline string.

Typical Usage

1. Call ``setup_gst_environment()`` once at startup (before any ``gi`` import) to
   initialise GStreamer and, optionally, register custom plugin directories.
2. Build a pipeline string with ``build_gst_pipeline`` or compose one manually.
3. Wrap it in a ``GstPipelineHandler``, call ``start()``, and consume frames from
   ``appsinks`` or block on ``wait()``.
4. To write a custom element, subclass ``GstElementBase`` (and ``Gst.Element``),
   implement ``get_metadata()``, ``get_pads()``, and ``_worker_func()``, then call
   ``MyElement.register()`` before building a pipeline that uses it.

Example — playing a file through a pipeline::

    from degirum_tools.gst import setup_gst_environment, GstPipelineHandler, build_gst_pipeline

    setup_gst_environment()

    pipeline_str = build_gst_pipeline("/path/to/video.mp4")
    handler = GstPipelineHandler(pipeline_str, appsink_names=["sink"])
    handler.start()

    for sample in handler.appsinks["sink"].queue:
        # process Gst.Sample
        ...

    handler.wait()

Example — custom Python GStreamer element::

    from degirum_tools.gst import setup_gst_environment, GstElementBase

    setup_gst_environment("/path/to/plugin_dir")

    from gi.repository import Gst

    class MyFilter(GstElementBase, Gst.Element):
        @classmethod
        def get_metadata(cls):
            return GstElementBase.ElementMetadata(
                "My Filter", "Filter/Video", "A simple passthrough filter", "Author"
            )

        @classmethod
        def get_pads(cls):
            return [
                GstElementBase.PadInfo("sink", GstElementBase.PadDirection.SINK, "video/x-raw", forward_to="src"),
                GstElementBase.PadInfo("src",  GstElementBase.PadDirection.SRC,  "video/x-raw"),
            ]

        def _worker_func(self):
            for buf in self.sinks["sink"].queue:
                self.sources["src"].push(buf)

    MyFilter.register()

Public API

- ``setup_gst_environment(*plugin_dirs)`` — initialise GStreamer; returns the ``gi`` module.
- ``GstPipelineHandler`` — pipeline lifecycle manager with optional appsink queues and throughput probing.
- ``GstElementBase`` — mixin base class for custom GStreamer Python elements.
- ``map_gst_buffer(buf_or_sample, readonly)`` — context manager for safe ``Gst.Buffer`` memory mapping.
- ``build_gst_pipeline(source)`` — build a pipeline string for a camera, RTSP stream, or file.
"""

from __future__ import annotations

import os
import inspect

# flake8: noqa
from .element_base import *
from .pipeline_handler import *
from .pipeline_builder import *


def setup_gst_environment(*plugin_dirs):
    """Set GST_PLUGIN_PATH before gi is imported, then import and return gi.

    GStreamer scans the plugin registry when the library first loads, so
    GST_PLUGIN_PATH must be in place before any gi import.

    To use custom Python plugins, place your plugin .py files under a
    subdirectory named `python` inside any directory passed here, and pass
    that parent directory as one of the arguments.

    Args:
        *plugin_dirs: Directory paths to prepend to GST_PLUGIN_PATH.
            Relative paths are resolved relative to the calling file.

    Returns:
        The imported gi module.
    """

    if plugin_dirs:
        caller_dir = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))

        abs_dirs = [
            os.path.join(caller_dir, d) if not os.path.isabs(d) else d
            for d in plugin_dirs
        ]

        existing = os.environ.get("GST_PLUGIN_PATH", "")
        all_dirs = abs_dirs + ([existing] if existing else [])
        if all_dirs:
            os.environ["GST_PLUGIN_PATH"] = ":".join(all_dirs)

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")

    from gi.repository import Gst

    Gst.init(None)

    return gi
