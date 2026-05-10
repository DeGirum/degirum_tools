#
# ai_element.py: GstAiElement — GStreamer filter element with AI inference support
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# Implements a GStreamer filter element that accepts BGR/RGB raw video on its sink pad,
# runs AI inference via a DeGirum model, and pushes results on its source pad.
#

"""
GstAiElement
============

A GStreamer filter element that consumes raw BGR or RGB video frames, runs AI inference
via a DeGirum model, and forwards the annotated frames downstream.

The element exposes four GObject properties:

- **model** (Python object): A pre-constructed :class:`ModelLike` instance.
  When set, ``model_name``, ``zoo_url``, and ``inference_host_address`` are ignored.
- **model-name** (string): Model name to load from the zoo when ``model`` is not set.
- **zoo-url** (string): Zoo URL used together with ``model-name`` and
  ``inference-host-address`` to connect to the model zoo.
- **inference-host-address** (string): Inference host address (e.g. ``"@cloud"`` or an
  AI-server hostname).
- **ai-overlay** (bool): When ``True`` (default), the model's ``image_overlay`` is pushed
  downstream; when ``False``, the original unmodified frame is forwarded.
- **model-properties** (Python object): A ``dict`` of model attribute name -> value pairs applied
  to the model after it is loaded. Only settable from Python code.
- **model-properties-json** (string): A JSON object string of model attribute name -> value pairs
  applied after model load (e.g. ``model_properties_json='{"threshold":0.5}'``). Merged with
  ``model_properties`` when both are set; ``model_properties`` takes precedence on key conflicts.

The element has an optional second sink pad **sink_full** (BGR/RGB/NV12). When connected:

- ``ai_overlay=False``: ``sink_full`` frames are forwarded directly to the output; the model
  still runs on ``sink`` but its rendered output is discarded.
- ``ai_overlay=True``: not yet implemented.

The element has an optional source pad **src_json**. When connected, each frame's inference
result is serialized to a UTF-8 JSON string and pushed as an ``application/json`` buffer.
This pad operates independently of ``sink_full`` and ``ai_overlay``.
"""

import numpy as np
import degirum as dg
import json
from typing import List
from .element_base import GstElementBase

# ---------------------------------------------------------------------------
# GstAiElement
# ---------------------------------------------------------------------------


class GstAiElement(GstElementBase):
    """GStreamer filter element with AI inference support.

    Accepts raw BGR or RGB video on a single sink pad and forwards frames
    (optionally annotated with inference results) on a single source pad of the
    same type.

    This class is intended to be used with multiple inheritance alongside
    ``Gst.Element``::

        class MyElement(GstAiElement, Gst.Element): ...

    Properties
    ----------
    model : ModelLike or None
        A pre-constructed model object.  Takes precedence over the string
        properties below when set.
    model_name : str
        Model name to load from the zoo (used when *model* is ``None``).
    zoo_url : str
        Zoo URL for loading the model (used when *model* is ``None``).
    inference_host_address : str
        Inference host address, e.g. ``"@cloud"`` or an AI-server hostname
        (used when *model* is ``None``).
    ai_overlay : bool
        When ``True`` (default), push the model's annotated ``image_overlay`` downstream.
        When ``False``, forward the original unmodified frame.
    model_properties : dict or None
        Optional ``dict`` of model attribute name -> value pairs applied after the model is
        loaded. Only settable from Python code.
    model_properties_json : str
        Optional JSON object string of model attribute name -> value pairs applied after the
        model is loaded. Settable from a pipeline string. Merged with ``model_properties``
        when both are set; ``model_properties`` values take precedence on key conflicts.

    Notes
    -----
    An optional ``sink_full`` pad (BGR/RGB/NV12) can be linked to provide a full-size frame
    stream. When ``ai_overlay=False`` its buffers are forwarded directly to the output pad.
    When ``ai_overlay=True`` this mode is not yet implemented.

    An optional ``src_json`` source pad outputs each frame's inference result serialized as
    a UTF-8 JSON buffer (``application/json`` caps). It operates independently of
    ``sink_full`` and ``ai_overlay``.
    """

    # ------------------------------------------------------------------
    # Caps shared by both pads
    # ------------------------------------------------------------------

    _RAW_BGR_RGB_CAPS: str = "video/x-raw,format=(string){BGR,RGB}"
    _FULL_INPUT_CAPS: str = "video/x-raw,format=(string){BGR,RGB,NV12}"

    # ------------------------------------------------------------------
    # GstElementBase interface
    # ------------------------------------------------------------------

    @classmethod
    def get_metadata(cls) -> GstElementBase.ElementMetadata:
        """Return element metadata for GStreamer registration."""
        return cls.ElementMetadata(
            longname="AI Inference Filter",
            klass="Filter/Video",
            description="Runs AI inference on BGR/RGB video frames",
            author="DeGirum Corporation",
        )

    @classmethod
    def get_pads(cls) -> List[GstElementBase.PadInfo]:
        """Return sink, optional sink_full, and src pads."""
        return [
            cls.PadInfo(
                name="sink",
                direction=cls.PadDirection.SINK,
                caps=cls._RAW_BGR_RGB_CAPS,
                forward_to="src",
                forward_caps=True,
                forward_eos=False,
                queue_drop=False,
            ),
            cls.PadInfo(
                name="sink_full",
                direction=cls.PadDirection.SINK,
                caps=cls._FULL_INPUT_CAPS,
                forward_to=None,
                forward_eos=False,
                queue_maxsize=0,  # unbounded queue to avoid blocking on backpressure; assumes sink_full is consumed in lockstep
                optional=True,  # this pad is optional and may be unlinked or left unconnected
            ),
            cls.PadInfo(
                name="src",
                direction=cls.PadDirection.SRC,
                caps=cls._RAW_BGR_RGB_CAPS,
            ),
            cls.PadInfo(
                name="src_json",
                direction=cls.PadDirection.SRC,
                caps="application/json",
                optional=True,
            ),
        ]

    @classmethod
    def get_properties(cls) -> List[GstElementBase.PropInfo]:
        """Return GObject property descriptors for model configuration."""
        return [
            cls.PropInfo(
                name="model",
                default=None,
                desc="Pre-constructed ModelLike object (takes precedence over model-name/zoo-url)",
            ),
            cls.PropInfo(
                name="model_name",
                default="",
                desc="Model name to load from the zoo",
            ),
            cls.PropInfo(
                name="zoo_url",
                default="",
                desc="Zoo URL used to connect to the model zoo",
            ),
            cls.PropInfo(
                name="inference_host_address",
                default="",
                desc="Inference host address (e.g. '@cloud' or an AI-server hostname)",
            ),
            cls.PropInfo(
                name="ai_overlay",
                default=True,
                desc="Push model's annotated image_overlay downstream; False forwards the original frame",
            ),
            cls.PropInfo(
                name="model_properties",
                default=None,
                desc="Dict of model attribute name->value pairs applied after model load (Python only)",
            ),
            cls.PropInfo(
                name="model_properties_json",
                default="",
                desc="JSON object string of model attribute name->value pairs applied after model load",
            ),
        ]

    # ------------------------------------------------------------------
    # GStreamer state machine override
    # ------------------------------------------------------------------

    def do_change_state(self, transition):
        """Reconfigure pad event-forwarding when sink_full is linked."""
        if transition == self._Gst.StateChange.READY_TO_PAUSED:
            sink_full = self.sinks["sink_full"]
            if sink_full.is_linked():
                # sink_full is connected: route stream events (caps, segment, EOS)
                # from sink_full to src; suppress forwarding from the model-input
                # sink whose resolution/format differs from the full-size output.
                sink = self.sinks["sink"]
                sink._forward_to = None
                sink._forward_caps = False
                sink_full._forward_to = self.sources["src"].pad
                sink_full._forward_caps = True
        return super().do_change_state(transition)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _SERIALIZABLE_TYPES = (bool, int, float, str, list, dict, tuple, type(None))

    @staticmethod
    def _serialize_result(result) -> bytes:
        """Serialize an inference result to a UTF-8 JSON byte string.

        Includes all public attributes and ``_inference_results``, filtered to
        JSON-serializable built-in types.

        Args:
            result: A single inference result returned by ``predict_batch``.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        serializable = {
            k: v
            for k, v in result.__dict__.items()
            if (not k.startswith("_") or k == "_inference_results")
            and isinstance(v, GstAiElement._SERIALIZABLE_TYPES)
        }
        return json.dumps(serializable).encode()

    def _load_model(self):
        """Resolve, load, and configure the AI model from element properties.

        Uses the pre-built ``model`` property when available; otherwise constructs
        one from ``model_name`` / ``zoo_url`` / ``inference_host_address``.  Then
        applies ``model_properties_json`` and ``model_properties`` on top.

        Returns:
            A configured model instance ready for inference.
        """
        from ..tools.model_registry import ModelSpec
        from ..compound_models import ModelLike

        model = self.props.model
        if model is not None and (
            not isinstance(model, ModelLike) or not isinstance(model, dg.model.Model)
        ):
            raise TypeError(
                f"'model' property must be a Model-like instance, got {type(model).__name__}"
            )

        if model is None:
            spec = ModelSpec(
                model_name=self.props.model_name,
                zoo_url=self.props.zoo_url,
                inference_host_address=self.props.inference_host_address,
            )
            model = spec.load_model()

        # Apply model_properties_json (strip outer quotes added by GStreamer parser).
        json_str = (self.props.model_properties_json or "").strip()
        if (
            len(json_str) >= 2
            and json_str[0] == json_str[-1]
            and json_str[0] in ("'", '"')
        ):
            json_str = json_str[1:-1]
        model_properties = json.loads(json_str) if json_str else {}

        # model_properties (Python-only dict) takes precedence on key conflicts.
        if self.props.model_properties:
            model_properties.update(self.props.model_properties)
        for attr, value in model_properties.items():
            setattr(model, attr, value)

        return model

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_func(self) -> None:
        """Read BGR/RGB buffers from the sink pad, run AI inference, push results to the src pad."""
        model = self._load_model()

        # pads
        sink = self.sinks["sink"]
        sink_full = self.sinks["sink_full"]
        src = self.sources["src"]
        src_json = self.sources["src_json"]

        has_full_input = sink_full.is_linked()
        has_json_output = src_json.pad.is_linked()

        ai_overlay: bool = self.props.ai_overlay

        h, w = sink.height, sink.width
        assert h is not None and w is not None, "sink CAPS not yet negotiated"

        assert sink.format in ("BGR", "RGB"), f"Unsupported format {sink.format}"
        model.input_numpy_colorspace = sink.format

        if has_full_input and ai_overlay:
            raise NotImplementedError(
                "ai_overlay=True with sink_full pad is not yet implemented"
            )

        if has_json_output:
            src_json.start_stream("application/json")

        def frame_source():
            for buf in sink.queue:
                with sink.map_buffer(buf) as data:
                    yield np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()

        for result in model.predict_batch(frame_source()):
            # Video output: forward full-size frame (sink_full mode) or model output.
            if has_full_input:
                src.push(sink_full.queue.get())
            else:
                img = result.image_overlay if ai_overlay else result.image
                src.push_bytes(img.tobytes())

            # JSON output: serialize and push inference result if src_json is linked.
            if has_json_output:
                src_json.push_bytes(self._serialize_result(result))

        src.stop_stream()
        if has_json_output:
            src_json.stop_stream()
