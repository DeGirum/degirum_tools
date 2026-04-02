#
# server.py: server-side entry point for ipc() child processes
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# This module is loaded directly by file path in each child process spawned by
# ipc(), using importlib.util, to avoid importing the full degirum_tools package.
# It must remain self-contained: import only what the server loop itself needs.
#

import sys
import os
import traceback
import psutil
import numpy as np
import zmq
import msgpack
import msgpack_numpy
from typing import Any, FrozenSet

# Patch msgpack to transparently encode/decode numpy arrays.
msgpack_numpy.patch()

# ---------------------------------------------------------------------------
# RPC protocol constants
# ---------------------------------------------------------------------------

# Special method name for graceful shutdown.
_SHUTDOWN_METHOD = "__shutdown__"

# Polling interval (ms) for the server's recv loop; allows the orphan guard
# to detect a dead client even when no requests arrive.
_RPC_RECV_TIMEOUT_MS = 1_000

# Keys used in MsgPack request/response dicts.
_KEY_METHOD = "method"
_KEY_ARGS = "args"
_KEY_KWARGS = "kwargs"
_KEY_RESULT = "result"
_KEY_ERROR = "error"
_KEY_TYPE = "type"
_KEY_MESSAGE = "message"
_KEY_TRACEBACK = "traceback"
_KEY_INOUT_ARGS = "inout_args"
_KEY_INOUT_KWARGS = "inout_kwargs"
_KEY_OUT_ARGS = "out_args"

# Sentinel key used in the msgpack envelope to mark positions of numpy arrays
# that are transmitted as separate ZMQ multipart frames.
_NP_FRAME_SENTINEL = "__np__"

# ---------------------------------------------------------------------------
# Serialization helpers  (re-exported so ipc.py can import them)
# ---------------------------------------------------------------------------


def _pack(obj: Any) -> bytes:
    """Pack a Python object to bytes using MsgPack."""
    return msgpack.packb(obj, use_bin_type=True)


def _unpack(data: bytes) -> Any:
    """Unpack bytes to a Python object using MsgPack."""
    return msgpack.unpackb(data, raw=False)


def _pack_multipart(envelope: dict) -> list:
    """Pack *envelope* as a multipart ZMQ message, extracting numpy arrays.

    Numpy arrays found under ``_KEY_ARGS``, ``_KEY_KWARGS``, ``_KEY_RESULT``,
    and ``_KEY_OUT_ARGS`` are pulled out and appended as raw buffer frames so
    ZMQ can transmit them without any Python-layer copy. Extraction is
    recursive within tuples, lists, and dicts reachable from those top-level
    envelope fields. Their positions in the envelope are replaced by a
    sentinel dict carrying shape and dtype.

    Returns:
        A list ``[envelope_bytes, arr0, arr1, ...]`` suitable for
        ``socket.send_multipart(..., copy=False)``.
    """
    arrays: list = []

    def _extract(val: Any) -> Any:
        if isinstance(val, np.ndarray):
            arr = val if val.flags["C_CONTIGUOUS"] else np.ascontiguousarray(val)
            idx = len(arrays)
            arrays.append(arr)
            return {_NP_FRAME_SENTINEL: idx, "d": arr.dtype.str, "s": list(arr.shape)}
        elif isinstance(val, tuple):
            return tuple(_extract(v) for v in val)
        elif isinstance(val, list):
            return [_extract(v) for v in val]
        elif isinstance(val, dict):
            return {k: _extract(v) for k, v in val.items()}
        return val

    new_env = dict(envelope)
    for key in (_KEY_ARGS, _KEY_KWARGS, _KEY_RESULT, _KEY_OUT_ARGS):
        if key not in new_env:
            continue
        new_env[key] = _extract(new_env[key])

    return [_pack(new_env)] + arrays


def _unpack_multipart(frames: list) -> dict:
    """Unpack a multipart ZMQ message produced by ``_pack_multipart``.

    ``frames[0]`` is the msgpack envelope; ``frames[1:]`` are raw numpy array
    buffers.  Sentinel dicts in the envelope are replaced with zero-copy
    ``numpy.frombuffer`` views backed by the ZMQ frame buffers.

    Args:
        frames: List of ``zmq.Frame`` objects (or bytes) returned by
            ``socket.recv_multipart(copy=False)``.

    Returns:
        The decoded envelope dict with numpy arrays restored.
    """
    envelope = _unpack(bytes(frames[0]))
    if len(frames) == 1:
        return envelope

    array_frames = frames[1:]

    def _restore(val: Any) -> Any:
        if isinstance(val, dict) and _NP_FRAME_SENTINEL in val:
            idx = val[_NP_FRAME_SENTINEL]
            dtype = np.dtype(val["d"])
            shape = tuple(val["s"])
            return np.frombuffer(memoryview(array_frames[idx]), dtype=dtype).reshape(
                shape
            )
        elif isinstance(val, tuple):
            return tuple(_restore(v) for v in val)
        elif isinstance(val, list):
            return [_restore(v) for v in val]
        elif isinstance(val, dict):
            return {k: _restore(v) for k, v in val.items()}
        return val

    for key in (_KEY_ARGS, _KEY_KWARGS, _KEY_RESULT, _KEY_OUT_ARGS):
        if key not in envelope:
            continue
        envelope[key] = _restore(envelope[key])

    return envelope


# ---------------------------------------------------------------------------
# Server-side entry point (called inside the child process via -c)
# ---------------------------------------------------------------------------


def _run_server_loop(server_class: Any, methods: FrozenSet[str]) -> None:
    """Instantiate *server_class* from stdin args, then dispatch incoming RPC requests.

    Reads MsgPack-encoded constructor arguments from stdin, instantiates
    *server_class*, binds a ZMQ REP socket to an OS-assigned port, writes the
    actual endpoint to stdout so the client can connect without a race condition,
    then enters the dispatch loop.  Exits cleanly when a shutdown command is
    received or when the parent process disappears (orphan guard).  Called only
    in the child process.

    Args:
        server_class: The user class to instantiate in the child process.
        methods: Frozenset of method names that are valid RPC targets.
    """
    ctor_data = _unpack(sys.stdin.buffer.read())
    server_instance = server_class(*ctor_data[_KEY_ARGS], **ctor_data[_KEY_KWARGS])

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, _RPC_RECV_TIMEOUT_MS)
    sock.bind("tcp://127.0.0.1:0")
    endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)

    # Advertise the endpoint to the parent process by writing it to stdout and flushing immediately
    sys.stdout.write(endpoint + "\n")
    sys.stdout.flush()

    # IPC dispatch loop
    parent_pid = os.getppid()
    while True:
        # Orphan guard: exit if the parent process is gone.
        if not psutil.pid_exists(parent_pid):
            break

        try:
            frames = sock.recv_multipart(copy=False)
        except zmq.Again:
            continue  # timeout: re-check orphan guard
        except zmq.ZMQError:
            break

        # parse the request
        request = _unpack_multipart(frames)
        method = request.get(_KEY_METHOD, "")
        req_args = request.get(_KEY_ARGS, [])
        req_kwargs = request.get(_KEY_KWARGS, {})
        inout_arg_indices = request.get(_KEY_INOUT_ARGS, [])
        inout_kwarg_keys = request.get(_KEY_INOUT_KWARGS, [])

        # msgpack decodes bytearray as bytes (immutable).  Since bytes cannot
        # be wrapped in InOut/Out on the client (check_type rejects them), any
        # bytes value at an inout position must have been a bytearray originally
        # — convert it back so the server method can mutate it in-place.
        # msgpack_numpy unpacks arrays as read-only memory views; copy them so
        # that in-place mutations (arr *= factor, etc.) work on the server.
        for i in inout_arg_indices:
            if isinstance(req_args[i], bytes):
                req_args[i] = bytearray(req_args[i])
            elif (
                isinstance(req_args[i], np.ndarray) and not req_args[i].flags.writeable
            ):
                req_args[i] = req_args[i].copy()
        for k in inout_kwarg_keys:
            if isinstance(req_kwargs[k], bytes):
                req_kwargs[k] = bytearray(req_kwargs[k])
            elif (
                isinstance(req_kwargs[k], np.ndarray)
                and not req_kwargs[k].flags.writeable
            ):
                req_kwargs[k] = req_kwargs[k].copy()

        if method == _SHUTDOWN_METHOD:
            sock.send_multipart([_pack({_KEY_RESULT: "bye"})], copy=False)
            break

        if method not in methods:
            sock.send_multipart(
                [
                    _pack(
                        {
                            _KEY_ERROR: True,
                            _KEY_TYPE: "AttributeError",
                            _KEY_MESSAGE: f"Method '{method}' is not an IPC method",
                            _KEY_TRACEBACK: "",
                        }
                    )
                ],
                copy=False,
            )
            continue

        try:
            result = getattr(server_instance, method)(*req_args, **req_kwargs)
            out_args = {}
            for i in inout_arg_indices:
                out_args[str(i)] = req_args[i]
            for k in inout_kwarg_keys:
                out_args[k] = req_kwargs[k]
            sock.send_multipart(
                _pack_multipart({_KEY_RESULT: result, _KEY_OUT_ARGS: out_args}),
                copy=False,
            )
        except Exception as exc:
            sock.send_multipart(
                [
                    _pack(
                        {
                            _KEY_ERROR: True,
                            _KEY_TYPE: type(exc).__name__,
                            _KEY_MESSAGE: str(exc),
                            _KEY_TRACEBACK: traceback.format_exc(),
                        }
                    )
                ],
                copy=False,
            )

    sock.close(linger=0)
    ctx.term()
