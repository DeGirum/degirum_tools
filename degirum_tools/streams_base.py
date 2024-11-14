#
# streams_base.py: streaming toolkit: base classes
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements base classes for streaming toolkit:
#  - StreamMeta class is used to pass metainfo objects between gizmos.
#  - StreamData class is used to pass data and metainfo objects between gizmos.
#  - Stream class is a queue-based iterable class with optional item drop.
#  - Gizmo class is a base class for all gizmos: streaming pipeline processing blocks.
#  - Composition class is a class, which holds and animates multiple connected gizmos.
#

import threading, queue, copy, time
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict, Union, Iterator
from .environment import get_test_mode


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
        inp_stream = self.get_input(inp) if isinstance(inp, int) else inp
        if inp_stream not in other_gizmo._output_refs:
            other_gizmo._output_refs.append(inp_stream)
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

    def _do_start(self):
        """Start gizmo animation (internal method):
        launch run() method of every registered gizmo in a separate thread."""

        if len(self._threads) > 0:
            raise Exception("Composition already started")

        def gizmo_run(gizmo):
            try:
                gizmo.result_cnt = 0
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

    def start(self, *, wait: bool = True, detect_bottlenecks: bool = False):
        """Start gizmo animation: launch run() method of every registered gizmo in a separate thread.

        Args:
            wait: True to wait until all gizmos finished.
            detect_bottlenecks: True to switch all streams into dropping mode to detect bottlenecks.
            Use get_bottlenecks() method to return list of gizmos-bottlenecks
        """

        if detect_bottlenecks:
            for gizmo in self._gizmos:
                for i in gizmo.get_inputs():
                    i.allow_drop = True

        self._do_start()

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
        self._do_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit handler: wait for composition completion"""
        self.wait()
