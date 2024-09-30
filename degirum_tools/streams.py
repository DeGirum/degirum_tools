#
# streams.py: streaming toolkit for PySDK samples
#
# Copyright DeGirum Corporation 2023
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

from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict, Union
from contextlib import ExitStack
from .environment import get_test_mode
from .video_support import open_video_stream, open_video_writer
from .ui_support import Display

#
# predefined meta tags
#
tag_video = "video"  # tag for video source data
tag_resize = "resize"  # tag for resizer result
tag_inference = "inference"  # tag for inference result
tag_preprocess = "preprocess"  # tag for preprocessor result
tag_crop = "crop"  # tag for cropping result


class StreamMeta:
    """Stream metainfo class

    Keeps a list of metainfo objects, which are produced by Gizmos in the pipeline.
    A gizmo appends its own metainfo to the end of the list tagging it with a set of tags.
    Tags are used to find metainfo objects by a specific tag combination.

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

    def __init__(self, maxsize=0, allow_drop: bool = False):
        """Constructor

        - maxsize: maximum stream depth; 0 for unlimited depth
        - allow_drop: allow dropping elements on put() when stream is full
        """
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
        self.result_cnt += 1
        for out in self._output_refs:
            if data is None:
                out.close()
            else:
                # clone meta to avoid cross-modifications by downstream gizmos in tree-like pipelines
                out.put(StreamData(data.data, data.meta.clone()))

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

    def __init__(self, *gizmos):
        """Constructor.

        - gizmos: optional list of gizmos to add to composition
        """

        self._threads: List[threading.Thread] = []

        # collect all connected gizmos
        all_gizmos: set = set()
        for g in gizmos:
            all_gizmos |= g.get_connected()

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


class VideoSourceGizmo(Gizmo):
    """OpenCV-based video source gizmo"""

    # meta keys
    key_frame_width = "frame_width"  # frame width
    key_frame_height = "frame_height"  # frame height
    key_fps = "fps"  # stream frame rate
    key_frame_count = "frame_count"  # total stream frame count

    def __init__(self, video_source=None):
        """Constructor.

        - video_source: cv2.VideoCapture-compatible video source designator
        """
        super().__init__()
        self._video_source = video_source

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
                    self._abort = True
                else:
                    self.send_result(
                        StreamData(data, StreamMeta(meta, self.get_tags()))
                    )


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
        self._multiplex = multiplex
        self._frames: list = []  # saved frames for tests

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

            di = 0  # di is display index
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
                        if test_mode:
                            self._frames.append(data)

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

        def get_img(data):
            if self._show_ai_overlay:
                inference_meta = data.meta.find_last(tag_inference)
                if inference_meta:
                    return inference_meta.image_overlay
            return data.data

        img = get_img(self.get_input(0).get())
        with open_video_writer(self._filename, img.shape[1], img.shape[0]) as writer:
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
        resize_method: int = cv2.INTER_LINEAR,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - w, h: resulting image width/height
        - pad_method: padding method - one of 'stretch', 'letterbox'
        - resize_method: resampling method - one of cv2.INTER_xxx constants
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
        dx = dy = 0  # offset of left top corner of original image in resized image

        image_ret = image
        if image.shape[1] != self._w or image.shape[0] != self._h:
            if self._pad_method == "stretch":
                image_ret = cv2.resize(
                    image, (self._w, self._h), interpolation=self._resize_method
                )
            elif self._pad_method == "letterbox":
                iw = image.shape[1]
                ih = image.shape[0]
                scale = min(self._w / iw, self._h / ih)
                nw = int(iw * scale)
                nh = int(ih * scale)

                # resize preserving aspect ratio
                scaled_image = cv2.resize(
                    image, (nw, nh), interpolation=self._resize_method
                )

                # create new canvas image and paste into it
                image_ret = numpy.zeros((self._h, self._w, 3), image.dtype)

                dx = (self._w - nw) // 2
                dy = (self._h - nh) // 2
                image_ret[dy : dy + nh, dx : dx + nw, :] = scaled_image

        return image_ret

    def run(self):
        """Run gizmo"""

        meta = {
            self.key_frame_width: self._w,
            self.key_frame_height: self._h,
            self.key_pad_method: self._pad_method,
            self.key_resize_method: self._resize_method,
        }

        for data in self.get_input(0):
            if self._abort:
                break
            resized = self._resize(data.data)
            data.meta.append(meta, self.get_tags())
            self.send_result(StreamData(resized, data.meta))


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
            while True:
                # get data from all inputs
                for inp in self.get_inputs():
                    d = inp.get()
                    if d == Stream._poison:
                        self._abort = True
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
            # finish processing all frames for tests
            if self._abort and not get_test_mode():
                break

    @abstractmethod
    def on_result(self, result):
        """Result handler to be overloaded in derived classes.

        - result: inference result; result.info contains reference to data frame used for inference
        """


class AiSimpleGizmo(AiGizmoBase):
    """AI inference gizmo with no result processing: it passes through input frames
    attaching inference results as meta info"""

    def get_tags(self) -> List[str]:
        """Get list of tags assigned to this gizmo"""
        return [self.name, tag_inference]

    def on_result(self, result):
        """Result handler to be overloaded in derived classes.

        - result: inference result; result.info contains reference to data frame used for inference
        """
        meta = result.info
        meta.append(result, self.get_tags())
        self.send_result(StreamData(result.image, meta))


class AiObjectDetectionCroppingGizmo(Gizmo):
    """A gizmo, which receives object detection results, then for each detected object
    it crops input image and sends cropped result.

    Output image is the crop of original image.
    Output meta-info is a dictionary with the following keys:

    - "original_result": reference to original AI object detection result (contained only in the first crop)
    - "cropped_result": reference to sub-result for particular crop
    - "cropped_index": the number of that sub-result
    The last two key are present only if at least one object is detected in a frame.
    """

    # meta keys
    key_original_result = "original_result"  # original AI object detection result
    key_cropped_result = "cropped_result"  # sub-result for particular crop
    key_cropped_index = "cropped_index"  # the number of that sub-result

    def __init__(
        self,
        labels: List[str],
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - labels: list of class labels to process
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """

        # we use unlimited frame queue #0 to not block any source gizmo
        super().__init__([(0, allow_drop), (stream_depth, allow_drop)])

        self._labels = labels

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
                self.send_result(data)
                continue

            is_first = True
            for i, r in enumerate(result.results):
                bbox = r.get("bbox")
                label = r.get("label")
                if not bbox or not label or label not in self._labels:
                    continue
                crop = Display.crop(result.image, bbox)

                crop_meta = {}
                if is_first:
                    # send whole result first
                    crop_meta[self.key_original_result] = result

                crop_meta[self.key_cropped_result] = r
                crop_meta[self.key_cropped_index] = i
                is_first = False

                new_meta = data.meta.clone()
                new_meta.append(crop_meta, self.get_tags())
                self.send_result(StreamData(crop, new_meta))

            if is_first:  # no objects detected: send just original result
                data.meta.append({self.key_original_result: result}, self.get_tags())
                self.send_result(StreamData(img, data.meta))


class AiResultCombiningGizmo(Gizmo):
    """Gizmo to combine results from multiple AI gizmos with similar-typed results"""

    def __init__(
        self,
        inp_cnt: int,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - inp_cnt: number of inputs to combine
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """
        self._inp_cnt = inp_cnt
        super().__init__([(stream_depth, allow_drop)] * inp_cnt)

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
            base_result: Optional[list] = None
            for d in all_data:
                inference_meta = d.meta.find_last(tag_inference)
                if inference_meta is None:
                    continue  # no inference results
                sub_result = inference_meta._inference_results

                if base_data is None:
                    base_data = d
                    base_result = sub_result
                else:
                    base_result += sub_result

            assert base_data is not None
            self.send_result(base_data)


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
            data.meta.append(res[1], self.get_tags())
            self.send_result(StreamData(res[0], data.meta))


class AiAnalyzerGizmo(Gizmo):
    """Gizmo to apply a chain of analyzers to the inference result"""

    def __init__(
        self,
        analyzers: list,
        *,
        stream_depth: int = 10,
        allow_drop: bool = False,
    ):
        """Constructor.

        - analyzers: list of analyzer objects
        - stream_depth: input stream depth
        - allow_drop: allow dropping frames from input stream on overflow
        """

        self._analyzers = analyzers
        super().__init__([(stream_depth, allow_drop)])

    def run(self):
        """Run gizmo"""

        for data in self.get_input(0):
            if self._abort:
                break

            inference_meta = data.meta.find_last(tag_inference)
            if inference_meta is not None:
                for analyzer in self._analyzers:
                    analyzer.analyze(inference_meta)
            self.send_result(data)
