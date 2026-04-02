#
# IPC subpackage: inter-process communication via ZeroMQ RPC
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# Implements spawn() function to transparently execute class instances in
# isolated child processes, communicating via MsgPack-encoded RPC over
# ZeroMQ REQ/REP sockets.
#

"""
IPC Subpackage Overview
=======================

This subpackage provides transparent inter-process communication (IPC) by executing
class instances in isolated child processes and communicating with them via
MsgPack-encoded RPC over ZeroMQ REQ/REP sockets.

All public instance methods of any class are automatically available as RPC endpoints.

Usage::

    from degirum_tools import ipc

    class MyWorker:
        def __init__(self, param):
            self.param = param

        def compute(self, x):
            return x * self.param

    worker = ipc.spawn(MyWorker, param=3)  # spawns child process transparently
    result = worker.compute(10)             # executes in child process, returns 30
    del worker                              # shuts down child process cleanly

Limitations
-----------

- Only works for classes defined in importable modules (not ``__main__``).
- Only public instance methods are exposed (not classmethods, staticmethods,
  or methods starting with ``_``).
- Method arguments and return values must be serializable by MsgPack
  (with msgpack_numpy patch).
- Method arguments are passed by value by default: modifications on the
  server side are not visible to the client.  Wrap a mutable argument in
  :class:`InOut` (read-write) or :class:`Out` (write-only) to request writeback;
  the original object is patched in-place after the call returns.
  Supported types: ``list``, ``dict``, ``bytearray``, ``numpy.ndarray``.
"""

# flake8: noqa

from .client import (
    IPCRemoteError,
    Out,
    InOut,
    spawn,
    # for unit tests:
    _pack,
    _unpack,
    _pack_multipart,
    _unpack_multipart,
    _get_public_methods,
    _KEY_ARGS,
    _KEY_RESULT,
    _InOutSupport,
)

__all__ = ["IPCRemoteError", "Out", "InOut", "spawn"]
