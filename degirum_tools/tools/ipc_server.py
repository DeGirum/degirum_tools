#
# ipc_server.py: server-side entry point for ipc() child processes
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

# ---------------------------------------------------------------------------
# Serialization helpers  (re-exported so ipc.py can import them)
# ---------------------------------------------------------------------------


def _pack(obj: Any) -> bytes:
    """Pack a Python object to bytes using MsgPack."""
    return msgpack.packb(obj, use_bin_type=True)


def _unpack(data: bytes) -> Any:
    """Unpack bytes to a Python object using MsgPack."""
    return msgpack.unpackb(data, raw=False)


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
        if os.getppid() != parent_pid:
            break

        try:
            raw = sock.recv()
        except zmq.Again:
            continue  # timeout: re-check orphan guard
        except zmq.ZMQError:
            break

        # parse the request
        request = _unpack(raw)
        method = request.get(_KEY_METHOD, "")
        req_args = request.get(_KEY_ARGS, [])
        req_kwargs = request.get(_KEY_KWARGS, {})

        if method == _SHUTDOWN_METHOD:
            sock.send(_pack({_KEY_RESULT: "bye"}))
            break

        if method not in methods:
            sock.send(
                _pack(
                    {
                        _KEY_ERROR: True,
                        _KEY_TYPE: "AttributeError",
                        _KEY_MESSAGE: f"Method '{method}' is not an IPC method",
                        _KEY_TRACEBACK: "",
                    }
                )
            )
            continue

        try:
            result = getattr(server_instance, method)(*req_args, **req_kwargs)
            sock.send(_pack({_KEY_RESULT: result}))
        except Exception as exc:
            sock.send(
                _pack(
                    {
                        _KEY_ERROR: True,
                        _KEY_TYPE: type(exc).__name__,
                        _KEY_MESSAGE: str(exc),
                        _KEY_TRACEBACK: traceback.format_exc(),
                    }
                )
            )

    sock.close(linger=0)
    ctx.term()
