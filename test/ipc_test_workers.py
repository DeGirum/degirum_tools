#
# ipc_test_workers.py: worker classes imported by test_ipc.py
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# These classes must live in their own importable module (not __main__) so that
# the child process spawned by ipc() can re-import them.
#

import numpy as np


class ArrayWorker:
    """Worker that operates on numpy arrays."""

    def scale(self, arr, factor):
        """Return arr * factor, preserving dtype."""
        return arr * factor

    def dot(self, a, b):
        """Return dot product of two 1-D arrays."""
        return np.dot(a, b)

    def identity(self, arr):
        """Return the array unchanged (round-trip check)."""
        return arr

    def sink(self, data):
        """Just accept any data"""
        return None


class MultiplyWorker:
    """Worker that multiplies values by a configured factor."""

    def __init__(self, factor: int):
        self.factor = factor

    def multiply(self, x):
        return x * self.factor

    def add(self, a, b=0):
        return a + b

    def fail(self):
        raise ValueError("intentional error")


class ExitingWorker:
    """Worker whose method calls os._exit(), simulating a crash mid-call."""

    def exit_now(self):
        import os

        os._exit(1)


class BrokenWorker:
    """Worker whose constructor always raises, so the server exits before advertising its endpoint."""

    def __init__(self):
        raise RuntimeError("constructor failure")

    def noop(self):
        pass


class NestingWorker:
    """Worker that itself creates a sub-worker (MultiplyWorker) in a separate subprocess."""

    def __init__(self, factor: int):
        self.factor = factor

    def compute(self, x: int) -> int:
        from degirum_tools import ipc

        """Multiply x by factor using a nested MultiplyWorker IPC call."""
        sub = ipc(MultiplyWorker, factor=self.factor)
        try:
            return sub.multiply(x)
        finally:
            del sub  # ensure sub-worker is cleaned up immediately

    def add_via_sub(self, a: int, b: int) -> int:
        """Add two numbers using a nested MultiplyWorker (factor=1) IPC call."""

        from degirum_tools import ipc

        sub = ipc(MultiplyWorker, factor=1)
        try:
            return sub.add(a, b)
        finally:
            del sub  # ensure sub-worker is cleaned up immediately
