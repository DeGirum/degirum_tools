#
# streams.base.py: streaming toolkit: base classes
#
# Copyright DeGirum Corporation 2025
# All rights reserved
#
# Implements base classes for streaming toolkit:
#  - StreamMeta class is used to pass metainfo objects between gizmos.
#  - StreamData class is used to pass data and metainfo objects between gizmos.
#  - Stream class is a queue-based iterable class with optional item drop.
#  - Gizmo class is a base class for all gizmos: streaming pipeline processing blocks.
#  - Composition class is a class that holds and animates multiple connected gizmos.
#

import threading, queue, copy, time
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict, Union, Iterator, Tuple
from ..environment import get_test_mode
from degirum.exceptions import DegirumException


class StreamMeta:
    """Stream metainfo class (metadata container).

    **Overview**

    - A [StreamMeta](streams_base.md#streammeta) instance is a container that holds a chronologically ordered list
      of metainfo objects (called "meta infos") produced by gizmos in a streaming pipeline.
    - Each time a gizmo adds new metadata (e.g., inference results, resizing information),
      it is *appended* to the tail of this list.
    - The gizmo may associate the appended metadata with one or more *tags*, so that
      downstream gizmos or the user can retrieve specific metadata objects by those tags.

    **Appending and Tagging**

    - To store new metadata, a gizmo calls `self.meta.append(meta_obj, tags)`,
      where `meta_obj` is the metadata to attach, and `tags` is a string or list of strings
      labeling that metadata (e.g., "tag_inference", "tag_resize").
    - Internally, [StreamMeta](streams_base.md#streammeta) keeps track of a list of appended objects and a mapping
      of tags to the indices in that list.

    **Retrieving Metadata**

    - You can retrieve all metadata objects tagged with a certain tag via `find(tag)`,
      which returns a list of all matching objects in the order they were appended.
    - You can retrieve only the most recently appended object with `find_last(tag)`.
    - For example, an inference gizmo might attach an inference result with the tag
      `"tag_inference"`, so a downstream gizmo can do:
      `inference_result = stream_data.meta.find_last("tag_inference")`.
    - If no metadata matches the requested tag, these methods return `[]` or `None`.

    **Modifications and Cloning**

    - **Important**: Never modify a received [StreamMeta](streams_base.md#streammeta) or its stored objects in-place,
      because it may create side effects for upstream components.
      Call `clone()` if you need to make changes.
      `clone()` creates a shallow copy of the metainfo list and a copy of the tag-index map.
    - If you want to remove the most recent entry associated with a certain tag,
      call `remove_last(tag)` (occasionally useful in advanced pipeline scenarios).

    **Typical Usage**

    A typical processing pipeline might look like:
        1. A video source gizmo creates a new [StreamMeta](streams_base.md#streammeta), appends frame info under tag `"Video"`.
        2. A resizing gizmo appends new dimension info under tag `"Resize"`.
        3. An AI inference gizmo appends the inference result under tag `"Inference"`.
        4. A display gizmo reads the final metadata to overlay bounding boxes, etc.

    This incremental metadata accumulation is extremely flexible and allows each gizmo
    to contribute to a unified record of the data's journey.

    **Example**:

    ```python
    # In a gizmo, produce meta and append:
    data.meta.append({"new_width": 640, "new_height": 480}, "Resize")

    # In a downstream gizmo:
    resize_info = data.meta.find_last("Resize")
    if resize_info:
        w, h = resize_info["new_width"], resize_info["new_height"]
    ```

    **CAUTION**:
    Never modify the existing metadata objects in place. If you need to
    adapt previously stored metadata for your own use, first copy the
    data structure or call `clone()` on the [StreamMeta](streams_base.md#streammeta).
    """

    def __init__(self, meta: Optional[Any] = None, tags: Union[str, List[str]] = []):
        """Constructor.

        Args:
            meta (Any, optional): Initial metainfo object. Defaults to None.
            tags (Union[str, List[str]]): Tag or list of tags to associate with the initial metainfo object.
        """
        self._meta_list: list = []
        self._tags: Dict[str, List[int]] = {}
        if meta is not None:
            self.append(meta, tags)

    def clone(self):
        """Shallow clone this StreamMeta.

        This creates a copy of the internal list and tags dictionary, but does not deep-copy the metainfo objects.

        Returns:
            StreamMeta (streams_base.StreamMeta): A cloned StreamMeta instance.
        """
        ret = StreamMeta()
        ret._meta_list = copy.copy(self._meta_list)
        ret._tags = copy.deepcopy(self._tags)
        return ret

    def append(self, meta: Any, tags: Union[str, List[str]] = []):
        """Append a metainfo object to this StreamMeta.

        Args:
            meta (Any): The metainfo object to append.
            tags (Union[str, List[str]]): Tag or list of tags to associate with the metainfo object.
        """
        idx = len(self._meta_list)
        self._meta_list.append(meta)
        for tag in tags if isinstance(tags, list) else [tags]:
            if tag in self._tags:
                self._tags[tag].append(idx)
            else:
                self._tags[tag] = [idx]

    def find(self, tag: str) -> List[Any]:
        """Find metainfo objects by tag.

        Args:
            tag (str): The tag to search for.

        Returns:
            List[Any]: A list of metainfo objects that have the given tag (empty if none).
        """
        tag_indexes = self._tags.get(tag)
        return [self._meta_list[idx] for idx in tag_indexes] if tag_indexes else []

    def find_last(self, tag: str):
        """Find the last metainfo object with a given tag.

        Args:
            tag (str): The tag to search for.

        Returns:
            Any (optional): The last metainfo object associated with the tag, or None if not found.
        """
        tag_indexes = self._tags.get(tag)
        return self._meta_list[tag_indexes[-1]] if tag_indexes else None

    def get(self, idx: int) -> Any:
        """Get a metainfo object by index.

        Args:
            idx (int): The index of the metainfo object in the list.

        Returns:
            Any: The metainfo object at the specified index.
        """
        return self._meta_list[idx]

    def remove_last(self, tag: str):
        """Remove the last metainfo object associated with a tag.

        Args:
            tag (str): The tag whose last associated metainfo object should be removed.
        """
        tag_indexes = self._tags.get(tag)
        if tag_indexes:
            del tag_indexes[-1]
            if not tag_indexes:
                del self._tags[tag]


class StreamData:
    """Single data element of the streaming pipeline."""

    def __init__(self, data: Any, meta: StreamMeta = StreamMeta()):
        """Constructor.

        Args:
            data (Any): The data payload.
            meta (StreamMeta, optional): The metainfo associated with the data. Defaults to a new empty StreamMeta.
        """
        self.data = data
        self.meta = meta

    def append_meta(self, meta: Any, tags: List[str] = []):
        """Append an additional metainfo object to this [StreamData](streams_base.md#streamdata)'s metadata.

        Args:
            meta (Any): The metainfo object to append.
            tags (List[str], optional): Tags to associate with this metainfo object. Defaults to [].
        """
        self.meta.append(meta, tags)


class Stream(queue.Queue):
    """Queue-based iterable stream with optional item drop."""

    # Minimum queue size to avoid deadlocks:
    # one for stray result, one for poison pill in request_stop(),
    # and one for poison pill in gizmo_run().
    min_queue_size = 3

    def __init__(self, maxsize: int = 0, allow_drop: bool = False):
        """Constructor.

        Args:
            maxsize (int): Maximum stream depth (queue size); use 0 for unlimited depth. Defaults to 0.
            allow_drop (bool): If True, allow dropping the oldest item when the stream is full on put(). Defaults to False.

        Raises:
            Exception: If maxsize is non-zero and less than `min_queue_size`.
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
        """Put an item into the stream, with optional dropping.

        If the stream is full and `allow_drop` is True, the oldest item will be removed to make room.

        Args:
            item (Any): The item to put.
            block (bool): Whether to block if the stream is full (ignored if dropping is enabled). Defaults to True.
            timeout (float, optional): Timeout in seconds for the blocking put. Defaults to None (no timeout).
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
        """Return an iterator over the stream's items."""
        return iter(self.get, self._poison)

    def close(self):
        """Close the stream by inserting a poison pill."""
        self.put(self._poison)


class Watchdog:
    """Monitors activity rate and timing using tick events and a filtered TPS estimate.

    Tracks the frequency of `tick()` calls and the time since the last one. The `check()` method
    evaluates whether the activity is recent enough and meets a minimum TPS (ticks per second) threshold,
    using a single-pole low-pass filter to smooth TPS estimation.
    """

    def __init__(self, time_limit: float, tps_threshold: float, smoothing: float = 0.9):
        """Initializes the Watchdog.

        Args:
            time_limit (float): Maximum allowed time (in seconds) since the last tick.
            tps_threshold (float): Minimum required filtered ticks per second.
            smoothing (float): Smoothing factor for the low-pass filter (0 < smoothing < 1).
        """

        self._time_limit = time_limit
        self._tps_threshold = tps_threshold
        self._smoothing = smoothing
        self._last_tick: Optional[float] = None
        self._average_tick = -1.0
        self._lock = threading.Lock()

    def tick(self):
        """Records the current timestamp and updates the filtered TPS estimate.

        Should be called regularly to track system activity. Uses the time between ticks to calculate
        instantaneous TPS and applies a low-pass filter to smooth the estimate.
        """

        with self._lock:
            now = time.time()
            if self._last_tick is not None:
                dt = now - self._last_tick
                self._average_tick = (
                    dt
                    if self._average_tick < 0
                    else (
                        self._smoothing * self._average_tick
                        + (1 - self._smoothing) * dt
                    )
                )
            self._last_tick = now

    def check(self) -> Tuple[bool, float]:
        """Checks whether the watchdog is within the allowed timing and TPS threshold.

        Returns:
            Tuple[bool, float]: A tuple containing:
                - bool: True if the watchdog is active (recent enough and meets TPS threshold), False otherwise.
                - float: The current TPS value.

        """
        with self._lock:
            if self._last_tick is None:
                return True, 0  # No ticks yet, consider it active
            age = time.time() - self._last_tick
            tps = 1 / self._average_tick if self._average_tick > 0 else 0
            return age <= self._time_limit and tps >= self._tps_threshold, tps


class Gizmo(ABC):
    """Base class for all gizmos (streaming pipeline processing blocks).

    Each gizmo owns zero or more input streams that deliver data for processing (data-generating gizmos have
    no input stream).

    A gizmo can be connected to other gizmos to receive data from them. One gizmo can broadcast data to
    multiple others (a single gizmo output feeding multiple destinations).

    A data element moving through the pipeline is a tuple `(data, meta)` where:
        - `data` is the raw data (e.g., an image, a frame, or any object),
        - `meta` is a [StreamMeta](streams_base.md#streammeta) object containing accumulated metadata.

    Subclasses must implement the abstract `run()` method to define a gizmo processing loop. The `run()`
    method is launched in a separate thread by the [Composition](streams_base.md#composition) and should run until no more data is available
    or until an abort signal is set.

    The `run()` implementation should:
        - Periodically check the `_abort` flag (set via `abort()`) to see if it should terminate.
        - Handle poison pills (`Stream._poison`) if they appear in the input streams, which signal "no more data."

    Below is a minimal example similar to `ResizingGizmo`. This gizmo simply reads items from its single input,
    processes them, and sends results downstream until either `_abort` is set or the input stream is exhausted:

    ```python
        def run(self):
            # For a single-input gizmo, iterate over all data in input #0
            for item in self.get_input(0):
                # If we were asked to abort, break out immediately
                if self._abort:
                    break

                # item is a StreamData object: item.data is the image/frame, item.meta is the metadata
                input_image = item.data

                # 1) Do the resizing (your logic can use OpenCV, PIL, etc.)
                resized_image = do_resize(input_image, width=640, height=480)
                # 'do_resize' is just a placeholder; you'd implement your own resizing function.

                # 2) Update the metadata
                #    - Clone the existing metadata first.
                #    - In Python all objects are passed by reference, so if you do not clone but try A >> B and A >> C, C will receive the meta object modified by B.
                out_meta = item.meta.clone()
                out_meta.append(
                    {
                        "frame_width": 640,
                        "frame_height": 480,
                        "method": "your_resize_method"
                    },
                    tags=self.get_tags()
                )

                # 3) Send the processed item downstream
                self.send_result(StreamData(resized_image, out_meta))
    ```

    Notes:
        - If your gizmo has multiple inputs, you can call `self.get_input(i)` for each input or iterate over
            `self.get_inputs()` if you need to merge or synchronize multiple streams.
        - Always check `_abort` periodically inside your main loop if your gizmo could run for a long time or block on I/O.
        - When done, you do not need to manually send poison pills; the [Composition](streams_base.md#composition) handles closing any downstream streams once each gizmo `run()` completes.
        - If, instead of `self.get_input(0)`, you use `self.get_input(0).get()` or `.get_nowait()`, you must check if you receive a poison pill.
            - In simple loops, `self.get_input(0)` will terminate the loop.
            - In multi-input gizmos where simple nested for-loops aren't usable, get_nowait() is typically used to read input streams.
            - This way the gizmo code may query all inputs on a non-blocking manner and properly terminate loops.
    """

    def __init__(self, input_stream_sizes: List[tuple] = []):
        """Constructor.

        Args:
            input_stream_sizes (List[tuple]): List of (maxsize, allow_drop) tuples for each input stream.
                Use an empty list for no inputs. Each tuple defines the input stream's depth (0 means unlimited) and whether dropping is allowed.
        """
        self._inputs: List[Stream] = []
        for s in input_stream_sizes:
            self._inputs.append(Stream(*s))
        self._output_refs: List[Stream] = []
        self._connected_gizmos: set = set()
        self._abort = False

        # public attributes
        self.composition: Optional[Composition] = None
        self.error: Optional[DegirumException] = None
        self.name = self.__class__.__name__
        self.result_cnt = 0  # gizmo result counter
        self.start_time_s = time.time()  # gizmo start time
        self.elapsed_s = 0
        self.fps = 0  # achieved FPS rate
        self.watchdog: Optional[Watchdog] = None  # optional watchdog

    def get_tags(self) -> List[str]:
        """Get the list of meta tags for this gizmo.

        Returns:
            List[str]: Tags associated with this gizmo (by default, just its class name).
        """
        return [self.name]

    def require_tags(self, inp: int) -> List[str]:
        """Get the list of meta tags this gizmo requires in upstream meta for a specific input.

        Returns:
            List[str]: Tags required by this gizmo in upstream meta for the specified input.
        """
        return []

    def get_input(self, inp: int) -> Stream:
        """Get a specific input stream by index.

        Args:
            inp (int): Index of the input stream to retrieve.

        Returns:
            Stream: The input stream at the given index.

        Raises:
            Exception: If the requested input index does not exist.
        """
        if inp >= len(self._inputs):
            raise Exception(f"Input {inp} is not assigned")
        return self._inputs[inp]

    def get_inputs(self) -> List[Stream]:
        """Get all input streams of this gizmo.

        Returns:
            List[Stream]: List of input stream objects.
        """
        return self._inputs

    def connect_to(self, other_gizmo, inp: Union[int, Stream] = 0) -> "Gizmo":
        """Connect an input stream of this gizmo to another gizmo's output.

        Args:
            other_gizmo (Gizmo): The source gizmo to connect from.
            inp (int or Stream): The input index of this gizmo (or an input Stream) to use for the connection. Defaults to 0.

        Returns:
            Gizmo: This gizmo (to allow chaining).
        """
        inp_stream = self.get_input(inp) if isinstance(inp, int) else inp
        if inp_stream not in other_gizmo._output_refs:
            other_gizmo._output_refs.append(inp_stream)
        self._connected_gizmos.add(other_gizmo)
        other_gizmo._connected_gizmos.add(self)
        return self

    def get_connected(self) -> set:
        """Recursively gather all gizmos connected to this gizmo.

        Returns:
            set: A set of Gizmo objects that are connected (directly or indirectly) to this gizmo.
        """

        def _get_connected(gizmo, visited):
            visited.add(gizmo)
            for g in gizmo._connected_gizmos:
                if g not in visited:
                    _get_connected(g, visited)
            return visited

        return _get_connected(self, set())

    def __getitem__(self, index):
        """Enable `gizmo[index]` syntax for specifying connections.

        Returns a tuple `(self, input_stream)` which can be used on the right side of the `>>` operator
        for connecting gizmos (e.g., `source_gizmo >> target_gizmo[index]`).

        Args:
            index (int): The input stream index on this gizmo.

        Returns:
            (Gizmo, Stream):
                tuple: A tuple of (this gizmo, the Stream at the given input index).
        """
        return (self, self.get_input(index))

    def __rshift__(self, other_gizmo: Union[Any, tuple, None]) -> "Gizmo":
        """Connect another gizmo to this gizmo using the `>>` operator.

        This implements the right-shift operator, allowing syntax like `source >> target` or `source >> target[input_index]`.

        Args:
            other_gizmo (Gizmo or tuple): Either a Gizmo to connect (assumes input 0), or a tuple `(gizmo, inp)` where `inp` is the input index or Stream of that gizmo.

        Returns:
            Gizmo: The source gizmo (other_gizmo), enabling chaining of connections.
        """
        if other_gizmo is not None:
            g, inp = other_gizmo if isinstance(other_gizmo, tuple) else (other_gizmo, 0)
            g.connect_to(self, inp)
            return g
        else:
            # to allow chaining with Nones, skipping them
            return self

    def send_result(self, data: Optional[StreamData]):
        """Send a result to all connected output streams.

        Args:
            data (StreamData or None): The data result to send. If None (or a poison pill) is provided, all connected outputs will be closed.
        """
        if data != Stream._poison:
            self.result_cnt += 1
            if self.watchdog is not None:
                self.watchdog.tick()

        for out in self._output_refs:
            if data == Stream._poison or data is None:
                out.close()
            else:
                out.put(data)

    @abstractmethod
    def run(self):
        """Run the gizmo's processing loop.

        This method should retrieve data from input streams (if any), process it, and send results to outputs. Subclasses implement this method to define the gizmo's behavior.

        Important guidelines for implementation:
            - Check `self._abort` periodically and exit the loop if it becomes True.
            - If reading from an input stream via `get()` or `get_nowait()`, check for the poison pill (`Stream._poison`). If encountered, exit the loop.
            - For example, a typical single-input loop could be:
        ```
        for data in self.get_input(0):
            if self._abort:
                break
            result = self.process(data)
            self.send_result(result)
        ```

        There is no need to send a poison pill to outputs; the [Composition](streams_base.md#composition) will handle closing output streams.
        """
        pass

    def abort(self, abort: bool = True):
        """Set or clear the abort flag to stop the run loop.

        Args:
            abort (bool): True to request aborting the run loop, False to clear the abort signal.
        """
        self._abort = abort


class Composition:
    """Orchestrates and runs a set of connected gizmos.

    Usage:
        1. Add all gizmos to the composition using `add()` or by calling the composition instance.
        2. Connect the gizmos together using `connect_to()` or the `>>` operator.
        3. Start the execution by calling `start()`.
        4. To stop the execution, call `stop()` (or use the composition as a context manager).
    """

    def __init__(self, *gizmos: Union[Gizmo, Iterator[Gizmo]]):
        """Initialize the composition with optional initial gizmos.

        Args:
            *gizmos (Gizmo or Iterator[Gizmo]): Optional gizmos (or iterables of gizmos) to add initially.
                If a Gizmo is provided, all gizmos connected to it (including itself) are added.
                If an iterator of gizmos is provided, all those gizmos (and their connected gizmos) are added.
        """
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []

        def check_requirements(g, connected):
            required_tags = {
                tag for inp in range(len(g.get_inputs())) for tag in g.require_tags(inp)
            }
            upstream_tags = {tag for c in connected if c != g for tag in c.get_tags()}
            if not required_tags.issubset(upstream_tags):
                raise Exception(
                    f"Gizmo {g.name} has unmet tag requirements in the provided gizmo set: "
                    f"required tags: {required_tags}, tags found in upstream gizmos: {upstream_tags}"
                )

        # collect all connected gizmos
        all_gizmos: set = set()
        for g in gizmos:
            if isinstance(g, Iterator):
                for gi in g:
                    connected = gi.get_connected()
                    check_requirements(gi, connected)
                    all_gizmos |= connected
            elif isinstance(g, Gizmo):
                connected = g.get_connected()
                check_requirements(g, connected)
                all_gizmos |= connected
            else:
                raise Exception(f"Invalid argument type {type(g)}")

        self._gizmos: List[Gizmo] = list(all_gizmos)
        for g in self._gizmos:
            g.composition = self

    def add(self, gizmo: Gizmo) -> Gizmo:
        """Add a gizmo to this composition.

        Args:
            gizmo (Gizmo): The gizmo to add.

        Returns:
            Gizmo: The same gizmo, for convenience.
        """
        gizmo.composition = self
        self._gizmos.append(gizmo)
        return gizmo

    def __call__(self, gizmo: Gizmo) -> Gizmo:
        """Add a gizmo to this composition (callable syntax).

        Equivalent to calling `add(gizmo)`.

        Args:
            gizmo (Gizmo): The gizmo to add.

        Returns:
            Gizmo: The same gizmo.
        """
        return self.add(gizmo)

    def _do_start(self):
        """Internal helper to start all gizmo threads.

        Launches the `run()` method of every added gizmo in a separate thread.

        Raises:
            Exception: If the composition is already started.
        """
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
                gizmo.error = DegirumException(str(e))
                gizmo.composition.request_stop()

        for gizmo in self._gizmos:
            gizmo.abort(False)
            t = threading.Thread(target=gizmo_run, args=(gizmo,))
            t.name = t.name + "-" + type(gizmo).__name__
            self._threads.append(t)

        for t in self._threads:
            t.start()

    def start(self, *, wait: bool = True, detect_bottlenecks: bool = False):
        """Start the execution of all gizmos (each in its own thread): launch run() method of every registered gizmo.

        Args:
            wait (bool): If True, wait until all gizmos have finished. Defaults to True.
            detect_bottlenecks (bool): If True, enable frame dropping on all streams to detect bottlenecks (see `get_bottlenecks()`). Defaults to False.
        """
        if detect_bottlenecks:
            for gizmo in self._gizmos:
                for i in gizmo.get_inputs():
                    i.allow_drop = True

        self._do_start()
        if wait or get_test_mode():
            self.wait()

    def get_bottlenecks(self) -> List[dict]:
        """Get gizmos that experienced input queue bottlenecks in the last run.

        For this to be meaningful, the composition must have been started with `detect_bottlenecks=True`.

        Returns:
            A list of dictionaries where each key is a gizmo name and the value is the number of frames dropped for that gizmo.
        """
        ret = []
        for gizmo in self._gizmos:
            for i in gizmo.get_inputs():
                if i.dropped_cnt > 0:
                    ret.append({gizmo.name: i.dropped_cnt})
                    break
        return ret

    def get_current_queue_sizes(self) -> List[dict]:
        """Get current sizes of each gizmo's input queues.
        Can be used to analyze deadlocks.

        Returns:
            A list of dictionaries where each key is a gizmo name and the value is a list containing the gizmo's result count followed by the size of each of its input queues.
        """
        ret = []
        for gizmo in self._gizmos:
            qsizes = [gizmo.result_cnt]
            for i in gizmo.get_inputs():
                qsizes.append(i.qsize())
            ret.append({gizmo.name: qsizes})
        return ret

    def request_stop(self):
        """Signal all gizmos in this composition to stop (abort).

        This sets each gizmo's abort flag, clears all remaining items from their input queues, and sends poison pills to unblock any waiting gets. This method does not wait for threads to finish; call `wait()` to join threads.
        """

        with self._lock:
            # signal abort to all gizmos
            for gizmo in self._gizmos:
                gizmo.abort()

            # empty all streams to speed up completion
            for gizmo in self._gizmos:
                for i in gizmo._inputs:
                    while not i.empty():
                        try:
                            i.get_nowait()
                        except queue.Empty:
                            break

            # send poison pills from all gizmos to unblock gets()
            for gizmo in self._gizmos:
                gizmo.send_result(Stream._poison)

    def wait(self):
        """Wait for all gizmo threads in the composition to finish.

        Raises:
            Exception: If the composition has not been started, or if any gizmo raised an error during execution (the exception message will contain details).
        """
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
                errors += f"Error detected during execution of {gizmo.name}:\n  {type(gizmo.error)}: {str(gizmo.error)}\n\n{gizmo.error.traceback}\n\n"
        if errors:
            raise Exception(errors)

    def stop(self):
        """Stop the composition by aborting all gizmos and waiting for all threads to finish."""
        self.request_stop()
        self.wait()

    def __enter__(self) -> "Composition":
        """Start the composition when entering a context (without waiting).

        Returns:
            Composition: The composition itself (so that context manager usage is possible).
        """
        self._do_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """On exiting a context, wait for all gizmos to finish (and raise any errors).

        Automatically calls `wait()` to ensure all threads have completed.
        """
        if exc_type is not None:
            self.request_stop()
        self.wait()
