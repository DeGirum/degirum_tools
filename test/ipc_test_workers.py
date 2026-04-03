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
        """Multiply x by factor using a nested MultiplyWorker IPC call."""

        from degirum_tools.ipc import spawn

        sub = spawn(MultiplyWorker, factor=self.factor)
        try:
            return sub.multiply(x)
        finally:
            del sub  # ensure sub-worker is cleaned up immediately

    def add_via_sub(self, a: int, b: int) -> int:
        """Add two numbers using a nested MultiplyWorker (factor=1) IPC call."""

        from degirum_tools.ipc import spawn

        sub = spawn(MultiplyWorker, factor=1)
        try:
            return sub.add(a, b)
        finally:
            del sub  # ensure sub-worker is cleaned up immediately


class InOutWorker:
    """Worker used to test Out/InOut writeback arguments."""

    def fill_list(self, data: list, value: int) -> None:
        """Replace every element of *data* with *value* in-place."""
        for i in range(len(data)):
            data[i] = value

    def append_list(self, data: list, value: int) -> None:
        """Append *value* to *data* (Out: data arrives empty from client)."""
        data.append(value)

    def fill_dict(self, data: dict, key: str, value: int) -> None:
        """Set *data[key] = value* in-place."""
        data[key] = value

    def fill_bytes(self, data: bytearray, value: int) -> None:
        """Overwrite every byte of *data* with *value* in-place."""
        for i in range(len(data)):
            data[i] = value

    def scale_array(self, arr, factor: float) -> None:
        """Multiply every element of *arr* by *factor* in-place."""
        arr *= factor

    def fill_array(self, arr, value: float) -> None:
        """Overwrite every element of *arr* with *value* in-place."""
        arr[:] = value

    def swap_lists(self, a: list, b: list) -> None:
        """Copy contents: a gets b's elements, b gets a's elements."""
        tmp = list(a)
        a[:] = b
        b[:] = tmp


class NestedArrayWorker:
    """Worker that passes numpy arrays nested inside lists, dicts, and tuples."""

    def list_result(self, arr):
        """Return a list containing arr and arr*2."""
        return [arr, arr * 2]

    def dict_result(self, arr):
        """Return a dict mapping 'a'->arr and 'b'->arr*2."""
        return {"a": arr, "b": arr * 2}

    def tuple_result(self, arr):
        """Return a tuple (arr, arr*2)."""
        return (arr, arr * 2)

    def deep_result(self, arr):
        """Return a deeply nested structure: {"outer": [arr, {"inner": arr*2}]}."""
        return {"outer": [arr, {"inner": arr * 2}]}

    def accept_list(self, arrays):
        """Accept a list of arrays and return their element-wise sum."""
        result = arrays[0].copy()
        for a in arrays[1:]:
            result = result + a
        return result

    def accept_dict(self, arrays):
        """Accept a dict of arrays and return the element-wise sum of all values."""
        values = list(arrays.values())
        result = values[0].copy()
        for a in values[1:]:
            result = result + a
        return result

    def accept_tuple(self, t):
        """Accept a tuple/list of two arrays and return their element-wise sum."""
        return t[0] + t[1]
