#
# test_stream.py: unit tests for streams functionality
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#
# Implements unit tests to test composition and gizmos functionality
#

import time
import pytest
import degirum_tools.streams as streams


def test_streams_error_handling():
    """Test for error handling in streams"""

    #
    # Exception in sink
    #

    class InfiniteSource(streams.Gizmo):
        def __init__(self):
            super().__init__()
            self.n = 0

        def run(self):
            while not self._abort:
                self.send_result(streams.StreamData(self.n, self.n))
                self.n += 1

    class SinkWithException(streams.Gizmo):
        error_msg = "Test exception 1"

        def __init__(self):
            self.limit = 10
            super().__init__([(self.limit, False)])

        def run(self):
            inp = self.get_input(0)

            # wait until input queue is full
            while inp.qsize() < self.limit:
                time.sleep(0)

            for data in inp:
                if data.data == self.limit // 2:
                    # wait until input queue is full so source will block
                    while inp.qsize() < self.limit:
                        time.sleep(0)
                    raise ValueError(SinkWithException.error_msg)

    c = streams.Composition()
    src = c.add(InfiniteSource())
    dst = c.add(SinkWithException())
    src >> dst
    c.start(wait=False)

    with pytest.raises(Exception, match=SinkWithException.error_msg):
        c.wait()

    assert src.n > dst.limit  # type: ignore[attr-defined]

    #
    # Exception in source
    #
    class SourceWithException(streams.Gizmo):
        error_msg = "Test exception 2"

        def __init__(self):
            self.limit = 12
            super().__init__()

        def run(self):
            n = 0
            while not self._abort:
                self.send_result(streams.StreamData(n, n))
                n += 1
                if n == self.limit:
                    # wait until output queue is empty so sink will block
                    while self._output_refs[0].qsize() > 0:
                        time.sleep(0)
                    raise ValueError(SourceWithException.error_msg)

    class InfiniteSink(streams.Gizmo):

        def __init__(self):
            super().__init__([(0, False)])
            self.n = 0

        def run(self):
            for data in self.get_input(0):
                self.n += 1
                if self._abort:
                    break
                time.sleep(0)

    c = streams.Composition()
    src = c.add(SourceWithException())
    dst = c.add(InfiniteSink())
    src >> dst
    c.start(wait=False)

    with pytest.raises(Exception, match=SourceWithException.error_msg):
        c.wait()

    assert dst.n == src.limit  # type: ignore[attr-defined]