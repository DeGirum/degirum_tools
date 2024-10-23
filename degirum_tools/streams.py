#
# streams.py: streaming toolkit for PySDK samples
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Please refer to `dgstreams_demo.ipynb` PySDKExamples notebook for examples of toolkit usage.
#

import threading
import queue
import copy
import time
import cv2
import numpy
import yaml
import degirum as dg

from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict, Union, Iterator
from contextlib import ExitStack
from .environment import get_test_mode
from .video_support import open_video_stream, open_video_writer
from .ui_support import Display
from .image_tools import crop_image, resize_image, image_size, to_opencv
from .result_analyzer_base import image_overlay_substitute
from .crop_extent import CropExtentOptions, extend_bbox

#
# predefined meta tags
#
tag_video = "video"  # tag for video source data
tag_resize = "resize"  # tag for resizer result
tag_inference = "inference"  # tag for inference result
tag_preprocess = "preprocess"  # tag for preprocessor result
tag_crop = "crop"  # tag for cropping result
tag_analyzer = "analyzer"  # tag for analyzer result


class StreamMeta:
    """Stream metainfo class

    Keeps a list of metainfo objects, which are produced by Gizmos in the pipeline.
    A gizmo appends its own metainfo to the end of the list tagging it with a set of tags.
    Tags are used to find metainfo objects by a specific tag combination.

    CAUTION: never modify received metainfo object, always do clone() before modifying it
    to avoid side effects in upstream gizmos.

    """

    def __init__(self, meta: Optional[Any] = None, tags: Union[str, List[str]] = []):
        """Constructor.

        - meta: initial metainfo object (optional)
        - tags: tag or list of tags associated with this metainfo object
        """

        self._meta_list: list = []
        self._tags: Dict[str, List[int]] = {}
        if meta is not None:
            self.append(meta, tags)

    def clone(self):
        """Shallow clone metainfo object: clones all internal structures, but only references to
        metainfo objects, not the objects themselves.
        """
        ret = StreamMeta()
        ret._meta_list = copy.copy(self._meta_list)
        ret._tags = copy.deepcopy(self._tags)
        return ret

    def append(self, meta: Any, tags: Union[str, List[str]] = []):
        """Append a metainfo object to the list and tag it with a set of tags.

        - meta: metainfo object
        - tags: tag or list of tags associated with this metainfo object
        """
        idx = len(self._meta_list)
        self._meta_list.append(meta)
        for tag in tags if isinstance(tags, list) else [tags]:
            if tag in self._tags:
                self._tags[tag].append(idx)
            else:
                self._tags[tag] = [idx]

    def find(self, tag: str) -> List[Any]:
        """Find metainfo objects by a set of tags.

        - tags: tag or list of tags to search for

        Returns a list of metainfo objects matching this tag(s).
        """

        tag_indexes = self._tags.get(tag)
        return [self._meta_list[idx] for idx in tag_indexes] if tag_indexes else []

    def find_last(self, tag: str):
        """Find metainfo objects by a set of tags.

        - tags: tag or list of tags to search for

        Returns last metainfo object matching this tag(s) or None.
        """

        tag_indexes = self._tags.get(tag)
        return self._meta_list[tag_indexes[-1]] if tag_indexes else None

    def get(self, idx: int) -> Any:
        """Get metainfo object by index"""
        return self._meta_list[idx]

    def remove_last(self, tag: str):
        """Remove last metainfo object by tag

        - tag: tag to search for
        """

        tag_indexes = self._tags.get(tag)
        if tag_indexes:
            del tag_indexes[-1]
            if not tag_indexes:
                del self._tags[tag]


class StreamData:
    """Single data element of the streaming pipelines"""

    def __init__(self, data: Any, meta: StreamMeta = StreamMeta()):
        """Constructor.

        - data: data payload
        - meta: metainfo"""
        self.data = data
        self.meta = meta

    def append_meta(self, meta: Any, tags: List[str] = []):
        """Append metainfo to the existing metainfo"""
        self.meta.append(meta, tags)


class Stream(queue.Queue):
    """Queue-based iterable class with optional item drop"""

    # minimum queue size to avoid deadlocks:
    # one for stray result, one for poison pill in request_stop(),
    # and one for poison pill gizmo_run()
    min_queue_size = 3

    def __init__(self, maxsize=0, allow_drop: bool = False):
        """Constructor

        - maxsize: maximum stream depth; 0 for unlimited depth
        - allow_drop: allow dropping elements on put() when stream is full
        """

        if maxsize < self.min_queue_size and maxsize != 0:
            raise Exception(
                f"Incorrect stream depth: {maxsize}. Should be 0 (unlimited) or at least {self.min_queue_size}"
            )

        super().__init__(maxsize)
        self.allow_drop = allow_drop
        self.dropped_cnt = 0  # number of dropped items

    _poison = None

    def put(
        self, item: Any, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        """Put an item into the stream

        - item: item to put
        If there is no space left, and allow_drop flag is set, then oldest item will
        be popped to free space
        """
        if self.allow_drop:
            while True:
                try:
                    super().put(item, False)
                    break
                except queue.Full:
                    self.dropped_cnt += 1
                    try:
                        self.get_nowait()
                    finally:
                        pass
        else:
            super().put(item, block, timeout)

    def __iter__(self):
        """Iterator method"""
        return iter(self.get, self._poison)

    def close(self):
        """Close stream: put poison pill"""
        self.put(self._poison)


def clone_result(result):
    """Create a clone of PySDK result object. Clone inherits image references, but duplicates inference results."""
    clone = copy.copy(result)
    clone._inference_results = copy.deepcopy(result._inference_results)
    return clone


class Gizmo(ABC):
    """Base class for all gizmos: streaming pipeline processing blocks.
    Each gizmo owns zero of more input streams, which are used to deliver
    the data to that gizmo for processing. For data-generating gizmos
    there is no input stream.

    A gizmo can be connected to other gizmo to receive a data from that
    gizmo into one of its own input streams. Multiple gizmos can be connected to
    a single gizmo, so one gizmo can broadcast data to multiple destinations.

    A data element is a tuple containing raw data object as a first element, and meta info
    object as a second element.

    Abstract run() method should be implemented in derived classes to run gizmo
    processing loop. It is not called directly, but is launched by Composition class
    in a separate thread.

    run() implementation should periodically check for _abort flag (set by abort())
    and run until there will be no more data in its input(s).

    """

    def __init__(self, input_stream_sizes: List[tuple] = []):
        """Constructor

        - input_stream_size: a list of tuples containing constructor parameters of input streams;
            pass empty list to have no inputs; zero means unlimited depth
        """

        self._inputs: List[Stream] = []
        for s in input_stream_sizes:
            self._inputs.append(Stream(*s))

        self._output_refs: List[Stream] = []
        self._connected_gizmos: set = set()
        self._abort = False
        self.composition: Optional[Composition] = None
        self.error: Optional[Exception] = None
        self.name = self.__class__.__name__
        self.result_cnt = 0  # gizmo result counter
        self.start_time_s = time.time()  # gizmo start time
        self.elapsed_s = 0
        self.fps = 0  # achieved FPS rate

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name]

    def get_input(self, inp: int) -> Stream:
        """Get inp-th input stream"""
        if inp >= len(self._inputs):
            raise Exception(f"Input {inp} is not assigned")
        return self._inputs[inp]

    def get_inputs(self) -> List[Stream]:
        """Get list of input streams"""
        return self._inputs

    def connect_to(self, other_gizmo, inp: Union[int, Stream] = 0):
        """Connect given input to other gizmo.

        - other_gizmo: gizmo to connect to
        - inp: input stream or input index to use for connection

        Returns self
        """
        other_gizmo._output_refs.append(
            self.get_input(inp) if isinstance(inp, int) else inp
        )
        self._connected_gizmos.add(other_gizmo)
        other_gizmo._connected_gizmos.add(self)
        return self

    def get_connected(self) -> set:
        """Get a set of all gizmos recursively connected to this gizmo"""

        def _get_connected(gizmo, visited):
            visited.add(gizmo)
            for g in gizmo._connected_gizmos:
                if g not in visited:
                    _get_connected(g, visited)
            return visited

        ret: set = set()
        return _get_connected(self, ret)

    def __getitem__(self, index):
        """Overloaded operator [], which returns tuple of self and input stream which corresponds to provided index.
        Used to connect gizmos by `>>` operator: `g1 >> g2[1]`

        - index: input index

        Returns tuple of self and input stream object
        """
        return (self, self.get_input(index))

    def __rshift__(self, other_gizmo: Union[Any, tuple]):
        """Operator antonym for connect_to(): connects other_gizmo to self

        - other_gizmo: either gizmo object or a tuple of a gizmo and it's input stream to connect to;
            if tuple is provided, input stream is taken from the second element of tuple, otherwise input 0 is assumed.

        When combined with `>>` operator, it allows to connect gizmos like this: `g1 >> g2[1]`

        Returns gizmo object, passed as an argument.
        """

        g, inp = other_gizmo if isinstance(other_gizmo, tuple) else (other_gizmo, 0)
        g.connect_to(self, inp)
        return g

    def send_result(self, data: Optional[StreamData]):
        """Send result to all connected outputs.

        - data: a tuple containing raw data object as a first element, and meta info object as a second element.
        When None is passed, all outputs will be closed.
        """
        if data != Stream._poison:
            self.result_cnt += 1
        for out in self._output_refs:
            if data == Stream._poison or data is None:
                out.close()
            else:
                out.put(data)

    @abstractmethod
    def run(self):
        """Run gizmo: get data from input(s), if any, process it,
        and send results to outputs (if any).

        The properly-behaving implementation should check for `self._abort` flag
        and exit `run()` when `self._abort` is set.

        Also, in case when retrieving data from input streams by `get()` or `get_nowait()`
        (as opposed to using input stream as iterator), the received value(s) should be
        compared with `Stream._poison`, and if poison pill is detected, need to exit `run()`
        as well.

        Typical single-input loop should look like this:

        ```
        for data in self.get_input(0):
            if self._abort:
                break
            result = self.process(data)
            self.send_result(result)
        ```

        No need to send poison pill to outputs: `Composition` class will do it automatically.
        """

    def abort(self, abort: bool = True):
        """Set abort flag"""
        self._abort = abort


class Composition:
    """Class, which holds and animates multiple connected gizmos.
    First you add all necessary gizmos to your composition using add() or __call()__ method.
    Then you connect all added gizmos in proper order using connect_to() method or `>>` operator.
    Then you start your composition by calling start() method.
    You may stop your composition by calling stop() method."""

    def __init__(self, *gizmos: Union[Gizmo, Iterator[Gizmo]]):
        """Constructor.

        - gizmos: optional list of gizmos or tuples of gizmos to add to composition
        """

        self._threads: List[threading.Thread] = []

        # collect all connected gizmos
        all_gizmos: set = set()
        for g in gizmos:
            if isinstance(g, Iterator):
                for gi in g:
                    all_gizmos |= gi.get_connected()
            elif isinstance(g, Gizmo):
                all_gizmos |= g.get_connected()
            else:
                raise Exception(f"Invalid argument type {type(g)}")

        self._gizmos: List[Gizmo] = list(all_gizmos)
        for g in self._gizmos:
            g.composition = self

    def add(self, gizmo: Gizmo) -> Gizmo:
        """Add a gizmo to composition

        - gizmo: gizmo to add

        Returns same gizmo
        """
        gizmo.composition = self
        self._gizmos.append(gizmo)
        return gizmo

    def __call__(self, gizmo: Gizmo) -> Gizmo:
        """Operator synonym for add()"""
        return self.add(gizmo)

    def start(self, *, wait: bool = True, detect_bottlenecks: bool = False):
        """Start gizmo animation: launch run() method of every registered gizmo in a separate thread.

        Args:
            wait: True to wait until all gizmos finished.
            detect_bottlenecks: True to switch all streams into dropping mode to detect bottlenecks.
            Use get_bottlenecks() method to return list of gizmos-bottlenecks
        """

        if len(self._threads) > 0:
            raise Exception("Composition already started")

        def gizmo_run(gizmo):
            try:
                gizmo.result_cnt = 0
                if detect_bottlenecks:
                    for i in gizmo.get_inputs():
                        i.allow_drop = True
                gizmo.start_time_s = time.time()
                gizmo.run()
                gizmo.elapsed_s = time.time() - gizmo.start_time_s
                gizmo.fps = (
                    gizmo.result_cnt / gizmo.elapsed_s if gizmo.elapsed_s > 0 else 0
                )
                gizmo.send_result(Stream._poison)
            except Exception as e:
                gizmo.error = e
                gizmo.composition.request_stop()

        for gizmo in self._gizmos:
            gizmo.abort(False)
            t = threading.Thread(target=gizmo_run, args=(gizmo,))
            t.name = t.name + "-" + type(gizmo).__name__
            self._threads.append(t)

        for t in self._threads:
            t.start()

        if wait or get_test_mode():
            self.wait()

    def get_bottlenecks(self) -> List[dict]:
        """Return a list of gizmos, which experienced bottlenecks during last run.
        Each list element is a dictionary. Key is gizmo name, value is # of dropped frames.
        Composition should be started with detect_bottlenecks=True to use this feature.
        """
        ret = []
        for gizmo in self._gizmos:
            for i in gizmo.get_inputs():
                if i.dropped_cnt > 0:
                    ret.append({gizmo.name: i.dropped_cnt})
                    break
        return ret

    def get_current_queue_sizes(self) -> List[dict]:
        """Return a list of gizmo input queue size at point of call.
        Can be used to analyze deadlocks
        Each list element is a dictionary. Key is gizmo name, value is a list of current queue sizes.
        """
        ret = []
        for gizmo in self._gizmos:
            qsizes = [gizmo.result_cnt]
            for i in gizmo.get_inputs():
                qsizes.append(i.qsize())
            ret.append({gizmo.name: qsizes})
        return ret

    def request_stop(self):
        """Signal abort to all registered gizmos"""

        # first signal abort to all gizmos
        for gizmo in self._gizmos:
            gizmo.abort()

        # then empty all streams to speedup completion
        for gizmo in self._gizmos:
            for i in gizmo._inputs:
                while not i.empty():
                    try:
                        i.get_nowait()
                    except queue.Empty:
                        break

        # finally, send poison pills from all gizmos to unblock all gets()
        for gizmo in self._gizmos:
            gizmo.send_result(Stream._poison)

    def wait(self):
        """Wait until all threads stopped"""

        if len(self._threads) == 0:
            raise Exception("Composition not started")

        # wait for completion of all threads
        for t in self._threads:
            if t.name != threading.current_thread().name:
                t.join()

        self._threads = []

        # error handling
        errors = ""
        for gizmo in self._gizmos:
            if gizmo.error is not None:
                errors += f"Error detected during execution of {gizmo.name}:\n  {type(gizmo.error)}: {str(gizmo.error)}\n\n"
        if errors:
            raise Exception(errors)

    def stop(self):
        """Signal abort to all registered gizmos and wait until all threads stopped"""

        self.request_stop()
        self.wait()

    def __enter__(self):
        """Context manager enter handler: start composition but do not wait for completion"""
        self.start(wait=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit handler: wait for composition completion"""
        self.wait()


# schema YAML
Key_Gizmos = "gizmos"
Key_ClassName = "class"
Key_ConstructorParams = "params"
Key_Connections = "connections"

composition_definition_schema_text = f"""
type: object
additionalProperties: false
properties:
    {Key_Gizmos}:
        type: object
        description: The collection of gizmos, keyed by gizmo instance name
        additionalProperties: false
        patternProperties:
            "^[a-zA-Z_][a-zA-Z0-9_]*$":
                type: object
                additionalProperties: false
                properties:
                    {Key_ClassName}:
                        type: string
                        description: The class name of the gizmo
                    {Key_ConstructorParams}:
                        type: object
                        description: The constructor parameters of the gizmo
                        additionalProperties: true
    {Key_Connections}:
        type: array
        description: The list of connections between gizmos
        items:
            type: array
            description: The connection between gizmos
            items:
                oneOf:
                    - type: string
                    - type: array
                      description: Gizmo with input index
                      prefixItems:
                        - type: string
                        - type: number
                      items: false
"""
composition_definition_schema = yaml.safe_load(composition_definition_schema_text)


def load_composition(
    description: Union[str, dict], context: Optional[dict] = None
) -> Composition:
    """Load composition from provided description of gizmos and connections.
    The description can be either JSON file, YAML file, YAML string, or Python dictionary
    conforming to JSON schema defined in `composition_definition_schema`.

    - description: text description of the composition in YAML format, or a file name with .json, .yaml, or .yml extension
      containing such text description, or Python dictionary with the same structure
    - context: optional context to look for gizmo classes (like globals())

    Returns: composition object
    """

    import json, jsonschema

    # custom YAML constructors

    def constructor_CropExtentOptions(loader, node):
        enum_name = loader.construct_scalar(node)
        value = CropExtentOptions.__members__.get(enum_name)
        if value is None:
            raise ValueError(f"Unknown CropExtentOptions value: {enum_name}")
        return CropExtentOptions(value)

    def constructor_cv2_constants(loader, node):
        const_name = loader.construct_scalar(node)
        value = cv2.__dict__.get(const_name)
        if value is None:
            raise ValueError(f"Unknown OpenCV value: {const_name}")
        return value

    yaml.add_constructor(
        "!CropExtentOptions", constructor_CropExtentOptions, yaml.SafeLoader
    )
    yaml.add_constructor("!OpenCV", constructor_cv2_constants, yaml.SafeLoader)

    description_dict: dict = {}
    if isinstance(description, str):
        if description.endswith(".json"):
            description_dict = json.load(open(description))
        elif description.endswith((".yaml", ".yml")):
            description_dict = yaml.safe_load(open(description))
        else:
            description_dict = yaml.safe_load(description)

    elif isinstance(description, dict):
        description_dict = description
    else:
        raise ValueError("load_composition: unsupported description type")

    jsonschema.validate(instance=description_dict, schema=composition_definition_schema)

    composition = Composition()

    # create all gizmos
    gizmos = {}
    for name, desc in description_dict[Key_Gizmos].items():
        gizmo_class_name = desc[Key_ClassName]

        gizmo_class = globals().get(gizmo_class_name)
        if gizmo_class is None:
            if context is not None:
                gizmo_class = context.get(gizmo_class_name)

        if gizmo_class is None:
            raise ValueError(
                f"load_composition: gizmo class {gizmo_class_name} not defined"
            )

        try:
            gizmo = gizmo_class(**desc.get(Key_ConstructorParams, {}))
        except Exception as e:
            raise ValueError(
                f"load_composition: error creating instance of {gizmo_class_name}"
            ) from e

        composition.add(gizmo)
        gizmos[name] = gizmo

    # create pipelines
    for p in description_dict[Key_Connections]:
        if len(p) < 2:
            raise ValueError(
                f"load_composition: pipeline {p} must have at least two elements"
            )
        if not isinstance(p[0], str):
            raise ValueError(
                f"load_composition: pipeline first element {p[0]} must be a gizmo name"
            )
        g0 = gizmos.get(p[0])
        if g0 is None:
            raise ValueError(f"load_composition: gizmo {p[0]} is not defined")

        for el in p[1:]:
            if isinstance(el, str):
                gizmo_name = el
                input_index = 0
            else:
                gizmo_name = el[0]
                input_index = el[1]

            g1 = gizmos.get(gizmo_name)
            if g1 is None:
                raise ValueError(f"load_composition: gizmo {gizmo_name} is not defined")

            g0 = g0 >> g1[input_index]

    return composition


class VideoSourceGizmo(Gizmo):
    """OpenCV-based video source gizmo"""

    # meta keys
    key_frame_width = "frame_width"  # frame width
    key_frame_height = "frame_height"  # frame height
    key_fps = "fps"  # stream frame rate
    key_frame_count = "frame_count"  # total stream frame count
    key_frame_id = "frame_id"  # frame index

    def __init__(self, video_source=None, *, stop_composition_on_end: bool = False):
        """Constructor.

        - video_source: cv2.VideoCapture-compatible video source designator
        - stop_composition_on_end: stop composition when video source is over
        """
        super().__init__()
        self._video_source = video_source
        self._stop_composition_on_end = stop_composition_on_end and not get_test_mode()

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_video]

    def run(self):
        """Run gizmo"""
        with open_video_stream(self._video_source) as src:

            meta = {
                self.key_frame_width: int(src.get(cv2.CAP_PROP_FRAME_WIDTH)),
                self.key_frame_height: int(src.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                self.key_fps: src.get(cv2.CAP_PROP_FPS),
                self.key_frame_count: int(src.get(cv2.CAP_PROP_FRAME_COUNT)),
            }

            while not self._abort:
                ret, data = src.read()
                if not ret:
                    break
                else:
                    meta2 = copy.copy(meta)
                    meta2[self.key_frame_id] = self.result_cnt
                    self.send_result(
                        StreamData(data, StreamMeta(meta2, self.get_tags()))
                    )

            if self._stop_composition_on_end:
                # stop composition if video source is over;
                # needed to stop other branches of the pipeline, like video sources, which are not over yet
                if self.composition is not None and not self._abort:
                    self.composition.stop()


class VideoDisplayGizmo(Gizmo):
    """OpenCV-based video display gizmo"""

    def __init__(
        self,
        window_titles: Union[str, List[str]] = "Display",
        *,
        show_ai_overlay: bool = False,
        show_fps: bool = False,
        stream_depth: int = 10,
        allow_drop: bool = False,
        multiplex: bool = False,
    ):
        """Constructor.

        - window_titles: window title string or array of title strings for multiple displays
        - show_fps: True to show FPS
        - show_ai_overlay: True to show AI inference overlay image when possible
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        - multiplex: then True, single input data is multiplexed to multiple displays;
          when False, each input is displayed on individual display
        """

        if isinstance(window_titles, str):
            window_titles = [
                window_titles,
            ]

        inp_cnt = 1 if multiplex else len(window_titles)
        super().__init__([(stream_depth, allow_drop)] * inp_cnt)
        self._window_titles = window_titles
        self._show_fps = show_fps
        self._show_ai_overlay = show_ai_overlay
        if multiplex and allow_drop:
            raise Exception("Frame multiplexing with frame dropping is not supported")
        self._multiplex = multiplex

    def run(self):
        """Run gizmo"""

        with ExitStack() as stack:
            ndisplays = len(self._window_titles)
            ninputs = len(self.get_inputs())

            test_mode = get_test_mode()
            displays = [
                stack.enter_context(Display(w, self._show_fps))
                for w in self._window_titles
            ]
            first_run = [True] * ndisplays

            di = -1  # di is display index
            try:
                while True:
                    if self._abort:
                        break

                    for ii, input in enumerate(self.get_inputs()):  # ii is input index
                        try:
                            if ninputs > 1 and not test_mode:
                                # non-multiplexing multi-input case (do not use it in test mode to avoid race conditions)
                                data = input.get_nowait()
                            else:
                                # single input or multiplexing case
                                data = input.get()
                                self.result_cnt += 1
                        except queue.Empty:
                            continue

                        # select display to show this frame
                        di = (di + 1) % ndisplays if self._multiplex else ii

                        if data == Stream._poison:
                            self._abort = True
                            break

                        img = data.data
                        if self._show_ai_overlay:
                            inference_meta = data.meta.find_last(tag_inference)
                            if inference_meta:
                                # show AI inference overlay if possible
                                img = inference_meta.image_overlay

                        displays[di].show(img)

                        if first_run[di] and not displays[di]._no_gui:
                            cv2.setWindowProperty(
                                self._window_titles[di], cv2.WND_PROP_TOPMOST, 1
                            )
                            first_run[di] = False

            except KeyboardInterrupt:
                if self.composition is not None:
                    self.composition.stop()


class VideoSaverGizmo(Gizmo):
    """OpenCV-based gizmo to save video to a file"""

    def __init__(
        self,
        filename: str = "",
        *,
        show_ai_overlay=False,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - filename: output video file name
        - show_ai_overlay: True to show AI inference overlay image when possible
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])
        self._filename = filename
        self._show_ai_overlay = show_ai_overlay

    def run(self):
        """Run gizmo"""

        h = w = 0

        def get_img(data: StreamData) -> numpy.ndarray:
            frame = data.data
            if self._show_ai_overlay:
                inference_meta = data.meta.find_last(tag_inference)
                if inference_meta:
                    frame = inference_meta.image_overlay

            w1, h1 = image_size(frame)
            if w != 0 and h != 0 and (w1 != w or h1 != h):
                frame = resize_image(frame, w, h)
            return to_opencv(frame)

        img = get_img(self.get_input(0).get())
        w, h = image_size(img)
        with open_video_writer(self._filename, w, h) as writer:
            self.result_cnt += 1
            writer.write(img)
            for data in self.get_input(0):
                if self._abort:
                    break
                writer.write(get_img(data))


class ResizingGizmo(Gizmo):
    """OpenCV-based image resizing/padding gizmo"""

    # meta keys
    key_frame_width = "frame_width"  # frame width
    key_frame_height = "frame_height"  # frame height
    key_pad_method = "pad_method"  # padding method
    key_resize_method = "resize_method"  # resampling method

    def __init__(
        self,
        w: int,
        h: int,
        pad_method: str = "letterbox",
        resize_method: str = "bilinear",
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - w, h: resulting image width/height
        - pad_method: padding method - one of "stretch", "letterbox", "crop-first", "crop-last"
        - resize_method: resampling method - one of "nearest", "bilinear", "area", "bicubic", "lanczos"
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])
        self._h = h
        self._w = w
        self._pad_method = pad_method
        self._resize_method = resize_method

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_resize]

    def _resize(self, image):
        is_opencv = isinstance(image, numpy.ndarray)
        mparams = dg.aiclient.ModelParams()
        mparams.InputRawDataType = ["DG_UINT8"]
        mparams.InputImgFmt = ["RAW"]
        mparams.InputW = [self._w]
        mparams.InputH = [self._h]
        mparams.InputColorSpace = ["BGR" if is_opencv else "RGB"]

        pp = dg._preprocessor.create_image_preprocessor(
            model_params=mparams,
            resize_method=self._resize_method,
            pad_method=self._pad_method,
            image_backend="opencv" if is_opencv else "pil",
        )
        pp.generate_image_result = True

        return pp.forward(image)["image_result"]

    def run(self):
        """Run gizmo"""

        my_meta = {
            self.key_frame_width: self._w,
            self.key_frame_height: self._h,
            self.key_pad_method: self._pad_method,
            self.key_resize_method: self._resize_method,
        }

        for data in self.get_input(0):
            if self._abort:
                break
            resized = self._resize(data.data)
            new_meta = data.meta.clone()
            new_meta.append(my_meta, self.get_tags())
            self.send_result(StreamData(resized, new_meta))


class AiGizmoBase(Gizmo):
    """Base class for AI inference gizmos"""

    def __init__(
        self,
        model,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
        inp_cnt: int = 1,
    ):
        """Constructor.

        - model: PySDK model object
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)] * inp_cnt)

        self.model = model

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_inference]

    def run(self):
        """Run gizmo"""

        def source():
            has_data = True
            while has_data:
                # get data from all inputs
                for inp in self.get_inputs():
                    d = inp.get()
                    if d == Stream._poison:
                        has_data = False
                        break
                    yield (d.data, d.meta)

                if self._abort:
                    break

        for result in self.model.predict_batch(source()):
            meta = result.info
            if isinstance(result._input_image, bytes):
                # most likely, we have preprocessing gizmo in the pipeline
                preprocess_meta = meta.find_last(tag_preprocess)
                if preprocess_meta:
                    # indeed, we have preprocessing gizmo in the pipeline

                    # patch raw bytes image in result when possible to provide better result visualization
                    result._input_image = preprocess_meta[
                        AiPreprocessGizmo.key_image_input
                    ]

                    # recalculate bbox coordinates to original image
                    converter = preprocess_meta.get(AiPreprocessGizmo.key_converter)
                    if converter is not None:
                        for res in result._inference_results:
                            if "bbox" in res:
                                box = res["bbox"]
                                res["bbox"] = [
                                    *converter(*box[:2]),
                                    *converter(*box[2:]),
                                ]

            self.on_result(result)
            if self._abort:
                break

    @abstractmethod
    def on_result(self, result):
        """Result handler to be overloaded in derived classes.

        - result: inference result; result.info contains reference to data frame used for inference
        """


class AiSimpleGizmo(AiGizmoBase):
    """AI inference gizmo with no result processing: it passes through input frames
    attaching inference results as meta info"""

    def on_result(self, result):
        """Result handler to be overloaded in derived classes.

        - result: inference result; result.info contains reference to data frame used for inference
        """
        new_meta = result.info.clone()
        new_meta.append(result, self.get_tags())
        self.send_result(StreamData(result.image, new_meta))


class AiObjectDetectionCroppingGizmo(Gizmo):
    """A gizmo, which receives object detection results, then for each detected object
    it crops input image and sends cropped result.

    Output image is the crop of original image.
    Output meta-info is a dictionary with the following keys:

    - "original_result": reference to original AI object detection result
    - "cropped_result": reference to sub-result for particular crop
    - "cropped_index": the number of that sub-result
    - "is_last_crop": the flag indicating that this is the last crop in the frame
    The last two key are present only if at least one object is detected in a frame.

    The validate_bbox() method can be overloaded in derived classes to filter out
    undesirable objects.
    """

    # meta keys
    key_original_result = "original_result"  # original AI object detection result
    key_cropped_result = "cropped_result"  # sub-result for particular crop
    key_cropped_index = "cropped_index"  # the number of that sub-result
    key_is_last_crop = "is_last_crop"  # 'last crop in the frame' flag

    def __init__(
        self,
        labels: List[str],
        *,
        send_original_on_no_objects: bool = True,
        crop_extent: float = 0.0,
        crop_extent_option: CropExtentOptions = CropExtentOptions.ASPECT_RATIO_NO_ADJUSTMENT,
        crop_aspect_ratio: float = 1.0,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - labels: list of class labels to process
        - send_original_on_no_objects: send original frame when no objects detected
        - crop_extent: extent of cropping in percent of bbox size
        - crop_extent_option: method of applying extending crop to the input image
        - crop_aspect_ratio: desired aspect ratio of the cropped image (W/H)
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """

        # we use unlimited frame queue #0 to not block any source gizmo
        super().__init__([(stream_depth, allow_drop)])

        self._labels = labels
        self._send_original_on_no_objects = send_original_on_no_objects
        self._crop_extent = crop_extent
        self._crop_extent_option = crop_extent_option
        self._crop_aspect_ratio = crop_aspect_ratio

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_crop]

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            img = data.data
            result = data.meta.find_last(tag_inference)
            if result is None:
                raise Exception(
                    f"{self.__class__.__name__}: inference meta not found: you need to have object detection gizmo in upstream"
                )

            # prepare all crops
            crops = []
            crop_metas = []
            for i, r in enumerate(result.results):

                bbox = r.get("bbox")
                if not bbox:
                    continue

                # discard objects which do not pass validation
                if not self.validate_bbox(result, i):
                    continue

                # apply crop extent
                ext_bbox = extend_bbox(
                    bbox,
                    self._crop_extent_option,
                    self._crop_extent,
                    self._crop_aspect_ratio,
                    image_size(result.image),
                )

                crops.append(crop_image(result.image, ext_bbox))

                cropped_obj = copy.deepcopy(r)
                cropped_obj["bbox"] = ext_bbox

                crop_meta = {
                    self.key_original_result: result,
                    self.key_cropped_result: cropped_obj,
                    self.key_cropped_index: i,
                    self.key_is_last_crop: False,  # will adjust later
                }
                new_meta = data.meta.clone()
                new_meta.append(crop_meta, self.get_tags())
                crop_metas.append(new_meta)

            if crop_metas:
                # adjust last crop flag
                crop_metas[-1].find_last(tag_crop)[self.key_is_last_crop] = True
                # and send all croped results
                for crop, meta in zip(crops, crop_metas):
                    self.send_result(StreamData(crop, meta))

            elif self._send_original_on_no_objects:
                # no objects detected: send original result
                crop_meta = {
                    self.key_original_result: result,
                    self.key_cropped_result: None,
                    self.key_cropped_index: -1,
                    self.key_is_last_crop: True,
                }
                new_meta = data.meta.clone()
                new_meta.append(crop_meta, self.get_tags())
                self.send_result(StreamData(img, new_meta))

    def validate_bbox(
        self, result: dg.postprocessor.InferenceResults, idx: int
    ) -> bool:
        """Validate detected object. To be overloaded in derived classes.

        - result: inference result
        - idx: index of object within `result.results[]` list to validate

        Returns True if object is accepted to be used for crop, False otherwise
        """
        return True


class CropCombiningGizmo(Gizmo):
    """Gizmo to combine results coming after AiObjectDetectionCroppingGizmo cropping gizmo(s).
    It has N+1 inputs: one for a stream of original frames, and N other inputs for streams of cropped
    and AI-processed results. It combines results from all crop streams
    and attaches them to the original frames.
    """

    # key for extra results list in inference detection results
    key_extra_results = "extra_results"

    def __init__(
        self,
        crop_inputs_num: int = 1,
        *,
        stream_depth: int = 10,
    ):
        """Constructor.

        - crop_inputs_num: number of crop inputs
        - stream_depth: input stream depth for crop inputs
        """

        # input 0 is for original frames, it should be unlimited depth
        # we never drop frames in combiner to avoid mis-synchronization between original and crop streams
        input_def = [(0, False)] + [(stream_depth, False)] * crop_inputs_num
        super().__init__(input_def)

    def run(self):
        """Run gizmo"""

        frame_input = self.get_input(0)
        crops: list[StreamData] = []
        combined_meta: Optional[StreamMeta] = None

        for full_frame in frame_input:
            if self._abort:
                break

            # get index of full frame
            video_meta = full_frame.meta.find_last(tag_video)
            if video_meta is None:
                raise Exception(
                    f"{self.__class__.__name__}: video meta not found: you need to have {VideoSourceGizmo.__class__.__name__} in upstream of input 0"
                )
            frame_id = video_meta[VideoSourceGizmo.key_frame_id]

            # collect all crops for this frame
            while True:
                if not crops:
                    # read all crop inputs
                    crops = []
                    for inp in self.get_inputs()[1:]:
                        crop = inp.get()
                        if crop == Stream._poison:
                            self._abort = True
                            break
                        crops.append(crop)

                if self._abort:
                    break

                # get index of cropped frame
                video_metas = [crop.meta.find_last(tag_video) for crop in crops]
                if None in video_metas:
                    raise Exception(
                        f"{self.__class__.__name__}: video meta not found: you need to have {VideoSourceGizmo.__class__.__name__} in upstream of all crop inputs"
                    )
                crop_frame_ids = [
                    meta[VideoSourceGizmo.key_frame_id] for meta in video_metas if meta
                ]

                if len(set(crop_frame_ids)) != 1:
                    raise Exception(
                        f"{self.__class__.__name__}: crop frame IDs are not synchronized. Make sure all crop inputs have the same {AiObjectDetectionCroppingGizmo.__class__.__name__} in upstream"
                    )
                crop_frame_id = crop_frame_ids[0]

                if crop_frame_id > frame_id:
                    # crop is for the next frame: send full frame
                    self.send_result(full_frame)
                    break

                if crop_frame_id < frame_id:
                    # crop is for the previous frame: skip it
                    crops = []
                    continue

                # at this point we do have both crop and after-crop inference metas for the current frame
                # even if there are no real crops (cropped_index==-1), because we just checked that
                # all crop_frame_ids are equal to frame_id, so crop inference was not skipped

                crop_metas = [crop.meta.find_last(tag_crop) for crop in crops]
                if None in crop_metas:
                    raise Exception(
                        f"{self.__class__.__name__}: crop meta(s) not found: you need to have {AiObjectDetectionCroppingGizmo.__class__.__name__} in upstream of all crop inputs"
                    )

                crop_meta = crop_metas[0]
                assert crop_meta

                if not all(crop_meta == cm for cm in crop_metas[1:]):
                    raise Exception(
                        f"{self.__class__.__name__}: crop metas are not synchronized. Make sure all crop inputs have the same {AiObjectDetectionCroppingGizmo.__class__.__name__} in upstream"
                    )

                orig_result = crop_meta[
                    AiObjectDetectionCroppingGizmo.key_original_result
                ]
                bbox_idx = crop_meta[AiObjectDetectionCroppingGizmo.key_cropped_index]

                # create combined meta if not done yet
                if combined_meta is None:
                    combined_meta = crops[0].meta.clone()
                    # remove all crop-related metas
                    combined_meta.remove_last(tag_crop)
                    combined_meta.remove_last(tag_inference)
                    # append original object detection result clone
                    combined_meta.append(self._clone_result(orig_result), tag_inference)

                # append all crop inference results to the combined meta
                if bbox_idx >= 0:

                    inference_metas = [
                        crop.meta.find_last(tag_inference) for crop in crops
                    ]
                    if any(im is orig_result for im in inference_metas):
                        raise Exception(
                            f"{self.__class__.__name__}: after-crop inference meta(s) not found: you need to have some inference-type gizmo in upstream of all crop inputs"
                        )

                    result = combined_meta.find_last(tag_inference)
                    assert result is not None
                    result._inference_results[bbox_idx][self.key_extra_results] = (
                        self._adjust_results(result, bbox_idx, inference_metas)
                    )

                crops = []  # mark crops as processed

                if crop_meta[AiObjectDetectionCroppingGizmo.key_is_last_crop]:
                    # last crop in the frame: send full frame
                    self.send_result(StreamData(full_frame.data, combined_meta))
                    combined_meta = None  # mark combined_meta as processed
                    break

    def _adjust_results(
        self, orig_result, bbox_idx: int, cropped_results: list
    ) -> list:
        """Adjust inference results for the crop: recalculates bbox coordinates to original image,
        attach original image to the result, and return the list of adjusted results"""

        bbox = orig_result._inference_results[bbox_idx].get("bbox")
        assert bbox
        tl = bbox[:2]

        ret = []
        for crop_res in cropped_results:
            cr = clone_result(crop_res)
            # attach original image to the result
            cr._input_image = orig_result._input_image
            # adjust all found coordinates to original image
            for r in cr._inference_results:
                if "bbox" in r:
                    r["bbox"][:] = [a + b for a, b in zip(r["bbox"], tl + tl)]
                if "landmarks" in r:
                    for lm_list in r["landmarks"]:
                        lm = lm_list["landmark"]
                        lm[:2] = [lm[0] + tl[0], lm[1] + tl[1]]
            ret.append(cr)

        return ret

    def _clone_result(self, result):
        """Clone inference result object with deepcopy of `_inference_results` list"""

        def _overlay_extra_results(result):
            """Produce image overlay with drawing all extra results"""

            orig_image = result._input_image
            overlay_image = result._orig_image_overlay_extra_results

            for res in result._inference_results:
                if self.key_extra_results in res:
                    bbox = res.get("bbox")
                    if bbox:
                        for extra_res in res[self.key_extra_results]:
                            extra_res._input_image = overlay_image
                            overlay_image = extra_res.image_overlay
                            extra_res._input_image = orig_image
            return overlay_image

        clone = clone_result(result)

        # redefine `image_overlay` property to `_overlay_extra_results` function so
        # that it will be called instead of the original one to annotate the image with extra results;
        # preserve original `image_overlay` property as `_orig_image_overlay_extra_results` property;
        clone.__class__ = type(
            clone.__class__.__name__ + "_overlay_extra_results",
            (clone.__class__,),
            {
                "image_overlay": property(_overlay_extra_results),
                "_orig_image_overlay_extra_results": clone.__class__.image_overlay,
            },
        )

        return clone


class AiResultCombiningGizmo(Gizmo):
    """Gizmo to combine results from multiple AI gizmos with similar-typed results"""

    def __init__(
        self,
        inp_cnt: int,
        *,
        stream_depth: int = 10,
    ):
        """Constructor.

        - inp_cnt: number of inputs to combine
        - stream_depth: input stream depth
        """
        self._inp_cnt = inp_cnt
        super().__init__([(stream_depth, False)] * inp_cnt)

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_inference]

    def run(self):
        """Run gizmo"""

        while True:
            # get data from all inputs
            all_data = []
            for inp in self.get_inputs():
                d = inp.get()
                if d == Stream._poison:
                    self._abort = True
                    break
                all_data.append(d)

            if self._abort:
                break

            base_data: Optional[StreamData] = None
            base_result: Optional[dg.postprocessor.InferenceResults] = None
            for d in all_data:
                inference_meta = d.meta.find_last(tag_inference)
                if inference_meta is None:
                    continue  # no inference results

                if base_result is None:
                    base_data = d
                    base_result = clone_result(inference_meta)
                else:
                    base_result._inference_results += copy.deepcopy(
                        inference_meta._inference_results
                    )

            assert base_data is not None
            new_meta = base_data.meta.clone()
            new_meta.append(base_result, self.get_tags())
            self.send_result(StreamData(base_data.data, new_meta))


class AiPreprocessGizmo(Gizmo):
    """Preprocessing gizmo. It applies AI model preprocessor to the input images.

    Output data is the result of preprocessing: raw bytes array to be sent to AI model.
    Output meta-info is a dictionary with the following keys
        "image_input": reference to the input image
        "converter": coordinate conversion lambda
        "image_result": pre-processed image (optional, only when model.save_model_image is set)
    """

    key_image_input = "image_input"  # reference to the input image
    key_converter = "converter"  # coordinate conversion lambda
    key_image_result = "image_result"  # pre-processed image

    def __init__(
        self,
        model,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.
        - model: PySDK model object
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        super().__init__([(stream_depth, allow_drop)])
        self._preprocessor = model._preprocessor

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_preprocess]

    def run(self):
        """Run gizmo"""
        for data in self.get_input(0):
            if self._abort:
                break

            res = self._preprocessor.forward(data.data)
            new_meta = data.meta.clone()
            new_meta.append(res[1], self.get_tags())
            self.send_result(StreamData(res[0], new_meta))


class AiAnalyzerGizmo(Gizmo):
    """Gizmo to apply a chain of analyzers to the inference result"""

    def __init__(
        self,
        analyzers: list,
        *,
        filters: Optional[set] = None,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - analyzers: list of analyzer objects
        - filters: a set of event names or notifications. When filters are specified, only results, which have
            at least one of the specified events and/or notifications, are passed through, other results are discarded.
            Event names are specified in the configuration of `EventDetector` analyzers, and notification names are
            specified in the configuration of `EventNotifier` analyzers. Filtering feature is useful when you need to
            suppress unwanted results to reduce the load on the downstream gizmos.
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """

        self._analyzers = analyzers
        self._filters = filters
        super().__init__([(stream_depth, allow_drop)])

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_inference, tag_analyzer]

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            filter_ok = True
            new_meta = data.meta.clone()
            inference_meta = new_meta.find_last(tag_inference)
            if inference_meta is not None and isinstance(
                inference_meta, dg.postprocessor.InferenceResults
            ):
                inference_clone = clone_result(inference_meta)
                for analyzer in self._analyzers:
                    analyzer.analyze(inference_clone)
                image_overlay_substitute(inference_clone, self._analyzers)
                new_meta.append(inference_clone, self.get_tags())

                if self._filters:
                    if not (
                        hasattr(inference_clone, "notifications")
                        and (self._filters & inference_clone.notifications)
                    ) and not (
                        hasattr(inference_clone, "events_detected")
                        and (self._filters & inference_clone.events_detected)
                    ):
                        filter_ok = False

            if filter_ok:
                self.send_result(StreamData(data.data, new_meta))


class SinkGizmo(Gizmo):
    """Gizmo to receive results and accumulate them in the queue. This queue can be used as a
    generator source for further processing in the main thread."""

    def __init__(
        self,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """

        super().__init__([(stream_depth, allow_drop)])

    def run(self):
        """Run gizmo"""
        # return immediately to not to consume thread resources
        return

    def __call__(self):
        """Return input queue. To be used in for-loops to get results"""
        return self._inputs[0]
