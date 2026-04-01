#
# ipc_test_workers.py: IPCBase worker classes imported by test_ipc.py
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# These classes must live in their own importable module (not __main__) so that
# the child process spawned by IPCBase._start_server can re-import them.
#

import numpy as np
from degirum_tools import IPCBase, ipc


class ArrayWorker(IPCBase):
    """Worker that operates on numpy arrays."""

    @ipc
    def scale(self, arr, factor):
        """Return arr * factor, preserving dtype."""
        return arr * factor

    @ipc
    def dot(self, a, b):
        """Return dot product of two 1-D arrays."""
        return np.dot(a, b)

    @ipc
    def identity(self, arr):
        """Return the array unchanged (round-trip check)."""
        return arr

    @ipc
    def sink(self, data):
        """Just accept any data"""
        return None


class MultiplyWorker(IPCBase):
    """Worker that multiplies values by a configured factor."""

    def __init__(self, factor: int):
        self.factor = factor

    @ipc
    def multiply(self, x):
        return x * self.factor

    @ipc
    def add(self, a, b=0):
        return a + b

    @ipc
    def fail(self):
        raise ValueError("intentional error")


class ExitingWorker(IPCBase):
    """Worker whose @ipc method calls os._exit(), simulating a crash mid-call."""

    @ipc
    def exit_now(self):
        import os

        os._exit(1)


class BrokenWorker(IPCBase):
    """Worker whose constructor always raises, so the server exits before advertising its endpoint."""

    def __init__(self):
        raise RuntimeError("constructor failure")

    @ipc
    def noop(self):
        pass


class NestingWorker(IPCBase):
    """Worker that itself creates a sub-worker (MultiplyWorker) in a separate subprocess."""

    def __init__(self, factor: int):
        self.factor = factor

    @ipc
    def compute(self, x: int) -> int:
        """Multiply x by factor using a nested MultiplyWorker IPC call."""
        sub = MultiplyWorker(factor=self.factor)
        try:
            return sub.multiply(x)
        finally:
            sub._ipc_shutdown()

    @ipc
    def add_via_sub(self, a: int, b: int) -> int:
        """Add two numbers using a nested MultiplyWorker (factor=1) IPC call."""
        sub = MultiplyWorker(factor=1)
        try:
            return sub.add(a, b)
        finally:
            sub._ipc_shutdown()
