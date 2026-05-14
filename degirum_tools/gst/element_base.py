#
# element_base.py: GstElementBase and map_gst_buffer for custom GStreamer Python elements
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# Implements the GstElementBase mixin and map_gst_buffer context manager
#

"""
GStreamer Element Base
======================

This module provides the ``GstElementBase`` mixin class and the ``map_gst_buffer`` context
manager for building custom GStreamer Python elements with minimal boilerplate.

Key Features

- Declare element metadata and pad templates via ``get_metadata()`` and ``get_pads()``;
  ``__gstmetadata__``, ``__gsttemplates__``, and ``__gproperties__`` are built automatically
  when ``register()`` is called.
- Sink pads are backed by ``streams.Stream`` queues with configurable depth and drop policy;
  iterate in a worker thread without additional synchronization code.
- Source pads expose helpers for pushing stream-start, segment, EOS events, and raw buffers.
- Worker thread is started on ``READY → PAUSED`` so that Python-object properties
  set between ``parse_launch()`` and ``start()`` are visible to it, and joined on
  ``PAUSED → READY``.
- ``register()`` calls ``GObject.type_register`` and ``Gst.Element.register`` in one step.
- ``map_gst_buffer`` safely maps a ``Gst.Buffer`` (or extracts and maps one from a
  ``Gst.Sample``) using a context manager, guaranteeing ``unmap`` even on error.

``GstElementBase`` must be listed **before** ``Gst.Element`` in the MRO so that
cooperative ``super()`` calls and gst-python's GObject metaclass machinery work correctly::

    class MyElement(GstElementBase, Gst.Element): ...

Subclass Contract

Every concrete subclass must implement:

- ``get_metadata()`` — return an ``ElementMetadata`` instance.
- ``get_pads()`` — return a list of ``PadInfo`` instances.
- ``_worker_func()`` — runs in a dedicated thread; returns when all sink queues are exhausted.

Example::

    from degirum_tools.gst import setup_gst_environment, GstElementBase

    setup_gst_environment()

    from gi.repository import Gst

    class Passthrough(GstElementBase, Gst.Element):
        @classmethod
        def get_metadata(cls):
            return GstElementBase.ElementMetadata(
                "Passthrough", "Filter/Video", "Copies buffers unchanged", "Author"
            )

        @classmethod
        def get_pads(cls):
            return [
                GstElementBase.PadInfo(
                    "sink", GstElementBase.PadDirection.SINK,
                    "video/x-raw", forward_to="src"
                ),
                GstElementBase.PadInfo(
                    "src", GstElementBase.PadDirection.SRC, "video/x-raw"
                ),
            ]

        def _worker_func(self):
            for buf in self.sinks["sink"].queue:
                self.sources["src"].push(buf)

    Passthrough.register("my_passthrough") # use this name in gst pipeline string

Public API

- ``map_gst_buffer(buf_or_sample, readonly)`` — context manager yielding a ``memoryview``.
- ``GstElementBase`` — mixin base for custom GStreamer Python elements.
- ``GstElementBase.SinkPad`` — sink pad wrapper with buffer queue and CAPS/EOS event handling.
- ``GstElementBase.SrcPad`` — source pad wrapper with stream-start helpers and push methods.
- ``GstElementBase.ElementMetadata`` — dataclass describing element identity for GStreamer registration.
- ``GstElementBase.PadInfo`` — dataclass describing a pad template.
- ``GstElementBase.PadDirection`` — enum selecting ``SINK`` or ``SRC``.
- ``GstElementBase.PropInfo`` — dataclass describing a GObject property.
- ``GstElementBase.get_properties()`` — return a list of ``PropInfo`` instances (optional override).
- ``GstElementBase.register(name)`` — register the element class with GStreamer.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

if TYPE_CHECKING:
    from gi.repository import Gst


@contextmanager
def map_gst_buffer(buf_or_sample: "Gst.Buffer | Gst.Sample", readonly: bool = True):
    """Map a `Gst.Buffer` (or extract one from a `Gst.Sample`), yield the raw data, then unmap.

    Args:
        buf_or_sample: A `Gst.Buffer` to map directly, or a `Gst.Sample` whose
            buffer will be extracted and mapped.
        readonly: When ``True`` (default), map with READ-only access.
            When ``False``, map with READ+WRITE access.

    Yields:
        A ``memoryview`` of the mapped buffer data.

    Raises:
        RuntimeError: If the buffer cannot be mapped.
    """
    from gi.repository import Gst

    buf = (
        buf_or_sample.get_buffer()
        if isinstance(buf_or_sample, Gst.Sample)
        else buf_or_sample
    )
    flags = Gst.MapFlags.READ if readonly else Gst.MapFlags.READ | Gst.MapFlags.WRITE
    ok, mapinfo = buf.map(flags)
    if not ok:
        raise RuntimeError("Failed to map GStreamer buffer")
    try:
        yield mapinfo.data
    finally:
        buf.unmap(mapinfo)


# ---------------------------------------------------------------------------
# Base element class for custom GStreamer Python elements
# ---------------------------------------------------------------------------
class GstElementBase:
    """Mixin base for custom GStreamer Python elements.

    Because `gi` must not be imported before `setup_gst_environment` patches `GST_PLUGIN_PATH`,
    this class intentionally does **not** inherit from `Gst.Element`. Concrete plugin classes
    must use multiple inheritance so that `Gst.Element` appears in the MRO:

        class MyElement(GstElementBase, Gst.Element): ...

    **Inheritance order matters:** `GstElementBase` must be listed *before* `Gst.Element` so that
    cooperative `super()` calls resolve correctly and gst-python's GObject metaclass machinery works as expected.

    Subclasses must implement:

    * `get_metadata()` — return an `ElementMetadata` instance.
    * `get_pads()`     — return a list of `PadInfo` instances.
    * `_worker_func()` — worker thread body; runs until all sink queues are exhausted.

    Subclasses may optionally override:

    * `get_properties()` — return a list of `PropInfo` instances to expose GObject properties.

    `register()` builds `__gstmetadata__`, `__gsttemplates__`, and `__gproperties__` from those declarations
    and registers the element with GStreamer.

    `__init__` creates all pads (SRC first so SINK pads can resolve `forward_to` by name), populates `self.sources`
    and `self.sinks`, then creates (but does not start) the worker thread.

    `do_change_state` starts the worker thread on ``READY → PAUSED``, and closes all sink queues
    and joins the worker thread on ``PAUSED → READY``.
    """

    # ------------------------------------------------------------------
    # Nested types used in the public API
    # ------------------------------------------------------------------

    if TYPE_CHECKING:
        # Stubs for attributes/methods provided by Gst.Element at runtime via
        # multiple inheritance.  Declaring them here avoids mypy attr-defined
        # errors in GstElementBase subclasses without requiring gi type stubs.
        from typing import Any as _Any

        # Populated by register() / GObject metaclass at class-creation time.
        __gstmetadata__: tuple
        __gsttemplates__: tuple

        props: _Any  # GObject property proxy

        def get_bus(self) -> _Any: ...  # noqa: E704
        def get_name(self) -> str: ...  # noqa: E704
        def add_pad(self, pad: _Any) -> bool: ...  # noqa: E704
        def get_pad_template(self, name: str) -> _Any: ...  # noqa: E704

    class SinkPad:
        """Wraps a GStreamer sink pad with its buffer queue and frame metadata.

        Args:
            element: Parent `Gst.Element` that owns this pad.
            template_name: Name of the pad template (also used as the pad name).
            forward_to: Source pad to forward non-CAPS/non-EOS events to.
            forward_caps: Whether to forward CAPS events to `forward_to`.
            forward_eos: Whether to forward EOS events to `forward_to`.

        Attributes:
            pad: The underlying `Gst.Pad` added to the parent element.
            width: Frame width in pixels, populated on the first CAPS event.
            height: Frame height in pixels, populated on the first CAPS event.
            format: Pixel format string (e.g. `"NV12"`), populated on the first CAPS event.
            stride: Per-plane byte strides read from `GstVideoMeta` on the first
                buffer. `None` until the first buffer arrives, or if the buffer
                carries no `GstVideoMeta`.
            queue: Buffer queue fed by the chain function. Iterate over it in the
                worker thread; yields `Gst.Buffer` items and terminates on EOS.
        """

        def __init__(
            self,
            element: Gst.Element,
            template_name: str,
            forward_to: Optional[Gst.Pad] = None,
            *,
            forward_caps: bool = True,
            forward_eos: bool = True,
            queue_maxsize: int = 10,
            queue_drop: bool = True,
        ):
            from gi.repository import Gst

            self._Gst = Gst
            self.pad = Gst.Pad.new_from_template(
                element.get_pad_template(template_name), template_name
            )
            self.pad.set_chain_function_full(self._chain)
            self.pad.set_event_function_full(self._event)
            element.add_pad(self.pad)

            self._forward_to = forward_to
            self._forward_caps = forward_caps
            self._forward_eos = forward_eos

            self.width: Optional[int] = None
            self.height: Optional[int] = None
            self.format: Optional[str] = None
            self.stride: Optional[List[int]] = None

            # Holds Gst.Buffer items; None is the stop sentinel.
            from .. import streams

            self.queue = streams.Stream(queue_maxsize, queue_drop)

        def is_initialized(self) -> bool:
            return self.width is not None and self.height is not None

        def is_linked(self) -> bool:
            """Return ``True`` if this pad is currently linked to a peer pad."""
            return self.pad.is_linked()

        @contextmanager
        def map_buffer(
            self,
            buf: "Gst.Buffer",
            readonly: bool = True,
        ) -> "Generator[memoryview, None, None]":
            """Context manager that maps *buf*, yields the raw data, then unmaps.

            Args:
                buf: The `Gst.Buffer` to map.
                readonly: When ``True`` (default), map with READ-only access.
                    When ``False``, map with READ+WRITE access.

            Yields:
                A ``memoryview`` of the mapped buffer data.

            Raises:
                RuntimeError: If the buffer cannot be mapped.
            """
            with map_gst_buffer(buf, readonly) as data:
                yield data

        def _chain(self, pad: Gst.Pad, parent, buf: Gst.Buffer) -> Gst.FlowReturn:
            if self.stride is None:
                from gi.repository import GstVideo

                meta = GstVideo.buffer_get_video_meta(buf)
                if meta is not None:
                    self.stride = list(meta.stride)
            self.queue.put(buf)
            return self._Gst.FlowReturn.OK

        def _event(self, pad: Gst.Pad, parent, event: Gst.Event) -> bool:
            if event.type == self._Gst.EventType.CAPS:
                caps = event.parse_caps()
                s = caps.get_structure(0)
                _, self.width = s.get_int("width")
                _, self.height = s.get_int("height")
                self.format = s.get_string("format")
                if self._forward_caps and self._forward_to:
                    return self._forward_to.push_event(event)

            elif event.type == self._Gst.EventType.EOS:
                self.queue.close()  # sentinel
                if self._forward_eos and self._forward_to:
                    return self._forward_to.push_event(event)

            elif self._forward_to:
                return self._forward_to.push_event(event)

            return True

    class SrcPad:
        """Wraps a GStreamer source pad with stream-start helpers.

        Args:
            element: Parent `Gst.Element` that owns this pad.
            template_name: Name of the pad template (also used as the pad name).

        Attributes:
            pad: The underlying `Gst.Pad` added to the parent element.
        """

        def __init__(self, element: Gst.Element, template_name: str):
            from gi.repository import Gst

            self._Gst = Gst
            self.pad = Gst.Pad.new_from_template(
                element.get_pad_template(template_name), template_name
            )
            element.add_pad(self.pad)
            self._stream_id = f"{element.get_name()}-{self.pad.get_name()}"

        def push_event(self, event: Gst.Event) -> bool:
            """Push an event downstream."""
            return self.pad.push_event(event)

        def start_stream(self, caps: Optional[str] = None) -> bool:
            """Push ``stream-start``, optional ``caps``, and ``segment`` events on this pad.

            Call once from the worker thread before pushing any buffers.

            Args:
                caps: Optional caps string (e.g. ``"application/json"``).  When provided,
                    a ``caps`` event is pushed between ``stream-start`` and ``segment``,
                    which is required for source pads that have no upstream sink pad to
                    forward caps automatically.

            Returns:
                ``True`` if all events were pushed successfully, ``False`` otherwise.
            """
            if not self.push_event(self._Gst.Event.new_stream_start(self._stream_id)):
                return False
            if caps is not None:
                if not self.push_event(
                    self._Gst.Event.new_caps(self._Gst.Caps.from_string(caps))
                ):
                    return False
            seg = self._Gst.Segment.new()
            seg.init(self._Gst.Format.TIME)
            return self.push_event(self._Gst.Event.new_segment(seg))

        def stop_stream(self) -> bool:
            """Push an EOS event on this pad to signal end of stream."""
            return self.push_event(self._Gst.Event.new_eos())

        def push_bytes(self, data: bytes) -> Gst.FlowReturn:
            """Wrap *data* in a new `Gst.Buffer` and push it downstream."""
            return self.pad.push(self._Gst.Buffer.new_wrapped(data))

        def push(self, buf: Gst.Buffer) -> Gst.FlowReturn:
            """Push a buffer downstream."""
            return self.pad.push(buf)

    class PadDirection(Enum):
        """Direction of a pad template declared in `get_pads`."""

        SINK = 1
        SRC = 2

    @dataclass
    class ElementMetadata:
        """GStreamer element metadata returned by `get_metadata`.

        Attributes:
            longname: Human-readable element name.
            klass: Element classification (e.g. "Filter/Video").
            description: Short description of what the element does.
            author: Author / organization string.
        """

        longname: str
        klass: str
        description: str
        author: str

    @dataclass
    class PadInfo:
        """Descriptor for a single pad template returned by `get_pads`.

        Attributes:
            name: Pad name (used as both template name and pad name).
            direction: `PadDirection.SINK` or `PadDirection.SRC`.
            caps: Gst pad capabilities string (e.g. `"video/x-raw,format=NV12"`).
            forward_to: Name of the SRC pad to forward events to. `None` disables forwarding entirely.
            forward_caps: Whether to forward CAPS events to `forward_to`.
            forward_eos: Whether to forward EOS events to `forward_to`.
            queue_maxsize: Maximum number of buffers held in the sink queue.
                `0` means unlimited. Defaults to `3`.
            queue_drop: When `True`, the oldest buffer is silently discarded
                when the queue is full. When `False`, the chain function blocks.
                Defaults to `True`.
            optional: When ``True``, the pad template uses ``PadPresence.SOMETIMES``
                so the pipeline does not require the pad to be linked. Defaults to ``False``.
        """

        name: str
        direction: GstElementBase.PadDirection
        caps: str
        # SINK-pad forwarding controls
        forward_to: Optional[str] = None
        forward_caps: bool = True
        forward_eos: bool = True
        # Queue controls
        queue_maxsize: int = 3
        queue_drop: bool = True
        # Pad presence
        optional: bool = False

    @dataclass
    class PropInfo:
        """Descriptor for a single GObject property returned by `get_properties`.

        The GObject type is inferred from `default`:

        * `bool`  → ``G_TYPE_BOOLEAN``
        * `int`   → ``G_TYPE_INT64``
        * `float` → ``G_TYPE_DOUBLE``
        * `str`   → ``G_TYPE_STRING``
        * anything else (including `None`) → ``G_TYPE_PYOBJECT``

        Scalar properties (bool/int/float/str) are settable from a pipeline
        string.  Python-object properties are only settable from Python code.

        Attributes:
            name: GObject property name (use underscores, e.g. ``"model_path"``).
            default: Default value.  Its type determines the GObject type.
            desc: Human-readable description shown by ``gst-inspect``.
            min: Lower bound for numeric properties.  ``None`` uses the type minimum.
            max: Upper bound for numeric properties.  ``None`` uses the type maximum.
        """

        name: str
        default: Any
        desc: str = ""
        min: Any = None
        max: Any = None

        def _resolved_type(self) -> type:
            """Return the Python type used for GObject type mapping."""
            t = type(self.default)
            return t if t in (bool, int, float, str) else object

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()

        from gi.repository import Gst

        self._Gst = Gst
        #: Dict of SRC pad name → GstElementBase.SrcPad
        self.sources: Dict[str, GstElementBase.SrcPad] = {}
        #: Dict of SINK pad name → GstElementBase.SinkPad
        self.sinks: Dict[str, GstElementBase.SinkPad] = {}

        pad_infos = self.get_pads()

        # Create SRC pads first so SINK pads can resolve forward_to by name.
        for info in pad_infos:
            if info.direction == GstElementBase.PadDirection.SRC:
                src = GstElementBase.SrcPad(self, info.name)
                self.sources[info.name] = src

        # Create SINK pads.
        for info in pad_infos:
            if info.direction == GstElementBase.PadDirection.SINK:
                forward_pad = (
                    self.sources[info.forward_to].pad
                    if info.forward_to is not None
                    else None
                )
                sink = GstElementBase.SinkPad(
                    self,
                    info.name,
                    forward_to=forward_pad,
                    forward_caps=info.forward_caps,
                    forward_eos=info.forward_eos,
                    queue_maxsize=info.queue_maxsize,
                    queue_drop=info.queue_drop,
                )
                self.sinks[info.name] = sink

        # Seed _props from PropInfo defaults so reads before any set_property work.
        self._props: Dict[str, Any] = {p.name: p.default for p in self.get_properties()}

        # Worker thread is created but not started until READY_TO_PAUSED so that
        # Python-object properties set after parse_launch() are visible to it.
        self._worker = threading.Thread(
            target=self._run_worker,
            name=f"{type(self).__name__}-worker",
            daemon=True,
        )

    def _run_worker(self) -> None:
        """Internal wrapper that runs ``_worker_func`` and handles exceptions."""
        try:
            self._worker_func()
        except Exception as exc:
            import traceback
            from gi.repository import GLib

            # Drain / close all sink queues so nothing else blocks.
            for sink in self.sinks.values():
                sink.queue.close(force=True)
            # Post a fatal error on the bus so the pipeline surfaces the exception.
            debug_info = traceback.format_exc()
            gerror = GLib.Error.new_literal(
                self._Gst.CoreError.quark(), str(exc), self._Gst.CoreError.FAILED
            )
            msg = self._Gst.Message.new_error(self, gerror, debug_info)
            bus = self.get_bus()
            if bus is not None:
                bus.post(msg)

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def get_metadata(cls) -> GstElementBase.ElementMetadata:
        """Return element metadata.

        Returns:
            An `ElementMetadata` instance.

        Raises:
            NotImplementedError: Must be overridden in every concrete subclass.
        """
        raise NotImplementedError(f"{cls.__name__} must implement get_metadata()")

    @classmethod
    def get_pads(cls) -> List[GstElementBase.PadInfo]:
        """Return the list of pad descriptors.

        Returns:
            A list of `PadInfo` instances.

        Raises:
            NotImplementedError: Must be overridden in every concrete subclass.
        """
        raise NotImplementedError(f"{cls.__name__} must implement get_pads()")

    @classmethod
    def get_properties(cls) -> List[GstElementBase.PropInfo]:
        """Return the list of GObject property descriptors.

        Override in subclasses to declare GObject properties settable from
        pipeline strings or Python code.  The base implementation returns an
        empty list (no properties).

        Returns:
            A list of `PropInfo` instances, or an empty list.
        """
        return []

    def _worker_func(self) -> None:
        """Worker thread body.

        Iterates over `self.sinks` queues, performs processing, and pushes
        results to the appropriate pad in `self.sources`. Returns when all
        sink queues have yielded their `None` sentinel.

        Raises:
            NotImplementedError: Must be overridden in every concrete subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _worker_func()"
        )

    # ------------------------------------------------------------------
    # GObject virtual method overrides
    # ------------------------------------------------------------------

    def do_get_property(self, prop):
        """Default GObject property getter backed by ``_props``."""
        return self._props.get(prop.name)

    def do_set_property(self, prop, value):
        """Default GObject property setter backed by ``_props``."""
        self._props[prop.name] = value

    def do_change_state(self, transition: Gst.StateChange) -> Gst.StateChangeReturn:
        """Start/stop worker resources on explicit GStreamer state transitions."""
        ret = self._Gst.Element.do_change_state(self, transition)
        if ret == self._Gst.StateChangeReturn.FAILURE:
            return ret

        if transition == self._Gst.StateChange.READY_TO_PAUSED:
            # Start worker here, not in __init__, so properties set between
            # parse_launch() and start() (e.g. a pre-built model object) are
            # already visible to the worker thread.
            if not self._worker.is_alive():
                self._worker.start()
            if not self.sinks:
                # Live sources (no sink pads) must report NO_PREROLL so the
                # pipeline does not wait for preroll data before going to PLAYING.
                ret = self._Gst.StateChangeReturn.NO_PREROLL

        elif transition == self._Gst.StateChange.PAUSED_TO_READY:
            for sink in self.sinks.values():
                sink.queue.close(force=True)

            if self._worker.is_alive():
                self._worker.join(timeout=2.0)

        return ret

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    @classmethod
    def _build_gst_metadata(cls) -> tuple:
        """Return a ``__gstmetadata__`` tuple from ``get_metadata()``."""
        meta = cls.get_metadata()
        return (meta.longname, meta.klass, meta.description, meta.author)

    @classmethod
    def _build_gst_templates(cls, Gst) -> tuple:
        """Return a ``__gsttemplates__`` tuple from ``get_pads()``."""
        templates = []
        for pad in cls.get_pads():
            gst_direction = (
                Gst.PadDirection.SINK
                if pad.direction == GstElementBase.PadDirection.SINK
                else Gst.PadDirection.SRC
            )
            presence = (
                Gst.PadPresence.SOMETIMES if pad.optional else Gst.PadPresence.ALWAYS
            )
            templates.append(
                Gst.PadTemplate.new(
                    pad.name,
                    gst_direction,
                    presence,
                    Gst.Caps.from_string(pad.caps),
                )
            )
        return tuple(templates)

    @classmethod
    def _build_gst_properties(cls, GObject) -> Dict[str, Any]:
        """Return a ``__gproperties__`` dict from ``get_properties()``."""
        import sys

        props = cls.get_properties()
        if not props:
            return {}

        flags = GObject.ParamFlags.READWRITE
        gprops: Dict[str, Any] = {}
        for p in props:
            t = p._resolved_type()
            if t is object:
                gprops[p.name] = (object, p.name, p.desc, flags)
            elif t is str:
                gprops[p.name] = (str, p.name, p.desc, p.default or "", flags)
            elif t is bool:
                gprops[p.name] = (bool, p.name, p.desc, bool(p.default), flags)
            elif t is float:
                lo = p.min if p.min is not None else -1e308
                hi = p.max if p.max is not None else 1e308
                gprops[p.name] = (
                    float,
                    p.name,
                    p.desc,
                    lo,
                    hi,
                    float(p.default),
                    flags,
                )
            elif t is int:
                lo = p.min if p.min is not None else -sys.maxsize
                hi = p.max if p.max is not None else sys.maxsize
                gprops[p.name] = (int, p.name, p.desc, lo, hi, int(p.default), flags)
        return gprops

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: Optional[str] = None,
    ) -> bool:
        """Register this element class with GStreamer.

        Builds ``__gstmetadata__``, ``__gsttemplates__``, and ``__gproperties__``
        from ``get_metadata()``, ``get_pads()``, and ``get_properties()``, then
        calls ``GObject.type_register`` and ``Gst.Element.register``.

        If ``Gst.Element`` is not already in the MRO (i.e. the class was defined
        as a plain ``GstElementBase`` subclass without explicitly inheriting
        ``Gst.Element``), a concrete subclass that adds ``Gst.Element`` is
        created automatically.

        Args:
            name: Factory name used to instantiate the element (e.g. ``"myelement"``).
                Defaults to the class name converted to lower-case.

        Returns:
            ``True`` on success, ``False`` otherwise.
        """
        from gi.repository import GObject, Gst

        if name is None:
            name = cls.__name__.lower()

        # Collect all GStreamer class attributes up-front so they are visible
        # to the GObject metaclass during ``type()`` (auto-subclass path).
        class_dict: Dict[str, Any] = {
            "__gstmetadata__": cls._build_gst_metadata(),
            "__gsttemplates__": cls._build_gst_templates(Gst),
            "__gproperties__": cls._build_gst_properties(GObject),
            "do_get_property": cls.do_get_property,
            "do_set_property": cls.do_set_property,
            "do_change_state": cls.do_change_state,
        }

        if Gst.Element not in cls.__mro__:
            # Auto-create concrete subclass with Gst.Element in MRO.
            # All attributes are passed in the dict so the GObject metaclass
            # picks them up during class creation and auto-registers the type.
            cls = type(cls.__name__, (cls, Gst.Element), class_dict)
        else:
            # Gst.Element already in MRO — inject attributes directly and
            # register the type manually.
            for key, value in class_dict.items():
                setattr(cls, key, value)
            GObject.type_register(cls)
        return Gst.Element.register(None, name, Gst.Rank.NONE, cls)
