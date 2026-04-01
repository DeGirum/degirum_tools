#
# ipc_client.py: inter-process communication support via ZeroMQ RPC
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# Implements ipc() function to transparently execute class instances in
# isolated child processes, communicating via MsgPack-encoded RPC over
# ZeroMQ REQ/REP sockets.  All public instance methods of any class are
# automatically available as RPC endpoints.
#
# Usage:
#   class MyWorker:
#       def __init__(self, param):
#           self.param = param
#
#       def compute(self, x):
#           return x * self.param
#
#   worker = ipc(MyWorker, param=3)  # spawns child process transparently
#   result = worker.compute(10)      # executes in child process, returns 30
#   del worker                       # shuts down child process cleanly
#
# Limitations:
#   - Only works for classes defined in importable modules (not __main__).
#   - Only public instance methods are exposed (not classmethods, staticmethods,
#     or methods starting with '_').
#   - Method arguments and return values must be serializable by MsgPack
#     (with msgpack_numpy patch).
#   - Method arguments are passed by value: modifications on the server side
#     are not visible to the client.
#

import subprocess
import sys
import os
import queue
import threading
import weakref
import functools
import zmq
from typing import Any, Optional, Set, Type, TypeVar, cast

from .ipc_server import _pack, _unpack

__all__ = ["IPCRemoteError", "ipc"]


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class IPCRemoteError(Exception):
    """Raised on the client when the server-side method raises an exception.

    Attributes:
        remote_type: The name of the exception class raised on the server.
        remote_traceback: The formatted traceback string from the server.
    """

    def __init__(self, remote_type: str, message: str, remote_traceback: str):
        """Args:
        remote_type: Name of the exception class raised on the server.
        message: Exception message from the server.
        remote_traceback: Formatted traceback string from the server.
        """
        self.remote_type = remote_type
        self.remote_traceback = remote_traceback
        super().__init__(
            f"Remote {remote_type}: {message}\n"
            f"--- Remote traceback ---\n{remote_traceback}"
        )


# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------

# Type variable for the return type of ipc(), representing the user class.
T = TypeVar("T")

# Path to ipc_server.py, loaded by file path in child processes to avoid
# triggering the full degirum_tools package __init__.
_IPC_SERVER_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ipc_server.py"
)

# Special method name for graceful shutdown.
_SHUTDOWN_METHOD = "__shutdown__"

# Maximum time (seconds) to wait for the server process to advertise its endpoint.
_SERVER_STARTUP_TIMEOUT_S = 30

# Timeout (seconds) when waiting for the child process to exit after kill.
_SERVER_TERMINATE_TIMEOUT_S = 5

# Send timeout for individual RPC calls (ms).  A timed-out REQ socket cannot be
# reused, so any timeout is treated as a fatal server failure.
_RPC_SEND_TIMEOUT_MS = 30_000

# Polling interval (ms) used in ipc() to check whether the server
# process is still alive while waiting for a response.
_RPC_POLL_INTERVAL_MS = 500

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
# Private helpers
# ---------------------------------------------------------------------------


def _get_public_methods(cls) -> Set[str]:
    """Return the set of public instance method names in cls's MRO (excluding object).

    Args:
        cls: The class to inspect.

    Returns:
        Set of method name strings that are public, non-static, non-class methods.
    """
    return {
        name
        for klass in cls.__mro__
        if klass is not object
        for name, val in vars(klass).items()
        if callable(val)
        and not name.startswith("_")
        and not isinstance(val, (classmethod, staticmethod))
    }


class _IPCConnection:
    """Manages the ZMQ socket, child process, and RPC dispatch for one IPC proxy.

    Attributes:
        _zmq_ctx: The ZeroMQ context.
        _socket: The ZeroMQ REQ socket.
        _poller: The ZeroMQ poller registered to the socket.
        _lock: Mutex serialising concurrent RPC calls.
        _process: The child subprocess.Popen instance.
    """

    def __init__(self, process: subprocess.Popen) -> None:
        """
        Construct an _IPCConnection for the given child process.
        Args:
            process: Child process to associate with this connection.
        """

        self._process: Optional[subprocess.Popen] = process
        self._lock: threading.Lock = threading.Lock()
        self._zmq_ctx: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._poller: Optional[zmq.Poller] = None

    def connect(self, endpoint: str) -> None:
        """Create and connect a ZMQ REQ socket to *endpoint*.

        Args:
            endpoint: ZeroMQ TCP endpoint string, e.g. ``tcp://127.0.0.1:5555``.
        """
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDTIMEO, _RPC_SEND_TIMEOUT_MS)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoint)
        self._zmq_ctx = ctx
        self._socket = sock
        self._poller = zmq.Poller()
        self._poller.register(sock, zmq.POLLIN)

    def call(self, method: str, args: tuple, kwargs: dict) -> Any:
        """Serialize and dispatch a single RPC call.

        Sends the request then polls for the response in a loop, waking every
        ``_RPC_POLL_INTERVAL_MS`` to check whether the server process is still
        alive.  This allows arbitrarily long-running methods while still
        detecting a crashed server promptly.

        Args:
            method: Name of the remote method to invoke.
            args: Positional arguments to forward.
            kwargs: Keyword arguments to forward.

        Returns:
            The return value of the remote method call.

        Raises:
            IPCRemoteError: If the server-side method raised an exception.
            RuntimeError: If a transport error or server crash is detected.
        """
        assert self._socket is not None
        assert self._poller is not None

        payload = _pack(
            {
                _KEY_METHOD: method,
                _KEY_ARGS: list(args),
                _KEY_KWARGS: kwargs,
            }
        )
        with self._lock:
            try:
                self._socket.send(payload)
            except zmq.ZMQError as exc:
                self.cleanup()
                raise RuntimeError(
                    f"IPC call '{method}' failed due to transport error: {exc}"
                ) from exc

            while True:
                ready = dict(self._poller.poll(timeout=_RPC_POLL_INTERVAL_MS))
                if self._socket in ready:
                    break
                # Server process died without responding.
                if self._process is not None and self._process.poll() is not None:
                    self.cleanup()
                    raise RuntimeError(
                        f"IPC call '{method}' failed: server process exited unexpectedly"
                    )

            try:
                raw = self._socket.recv()
            except zmq.ZMQError as exc:
                self.cleanup()
                raise RuntimeError(
                    f"IPC call '{method}' failed due to transport error: {exc}"
                ) from exc

        response = _unpack(raw)

        if response.get(_KEY_ERROR):
            raise IPCRemoteError(
                response.get(_KEY_TYPE, "RemoteError"),
                response.get(_KEY_MESSAGE, ""),
                response.get(_KEY_TRACEBACK, ""),
            )
        return response[_KEY_RESULT]

    def shutdown(self) -> None:
        """Send shutdown RPC to the server and release all resources.

        Attempts a graceful shutdown by sending the shutdown command and
        waiting for acknowledgement before tearing down ZMQ resources and
        reaping the child process.
        """
        graceful = False
        with self._lock:
            try:
                if self._socket is not None:
                    self._socket.send(
                        _pack(
                            {
                                _KEY_METHOD: _SHUTDOWN_METHOD,
                                _KEY_ARGS: [],
                                _KEY_KWARGS: {},
                            }
                        )
                    )
                    self._socket.recv()  # wait for ack
                    graceful = True
            except Exception:
                pass
        self.cleanup(graceful_shutdown=graceful)

    def cleanup(self, *, graceful_shutdown: bool = False) -> None:
        """Release ZMQ resources and reap the child process.

        Args:
            graceful_shutdown: When ``True`` the server has already acknowledged
                the shutdown command, so the process is given a chance to exit on
                its own before escalating to ``kill``.  When ``False`` the process
                is killed immediately.
        """
        try:
            if self._poller is not None and self._socket is not None:
                self._poller.unregister(self._socket)
        except Exception:
            pass
        self._poller = None

        try:
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None
        except Exception:
            graceful_shutdown = False

        try:
            if self._zmq_ctx is not None:
                self._zmq_ctx.term()
                self._zmq_ctx = None
        except Exception:
            graceful_shutdown = False

        proc = self._process
        self._process = None
        if proc is not None:
            if graceful_shutdown:
                try:
                    proc.wait(timeout=_SERVER_TERMINATE_TIMEOUT_S)
                    return
                except Exception:
                    pass
            try:
                proc.kill()
                proc.wait(timeout=_SERVER_TERMINATE_TIMEOUT_S)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# IPC proxy helpers — injected into the dynamic proxy subclass
# ---------------------------------------------------------------------------


def _proxy_ipc_shutdown(self: Any) -> None:
    """Gracefully shut down the server process and release all resources."""
    conn = getattr(self, "_ipc_conn", None)
    if conn is not None:
        conn.shutdown()


def _proxy_ipc_process(self: Any) -> Optional[subprocess.Popen]:
    """The child ``subprocess.Popen`` instance.  Exposed for testing and monitoring."""
    return self._ipc_conn._process  # type: ignore[attr-defined]


def _proxy_ipc_call(self: Any, method: str, args: tuple, kwargs: dict) -> Any:
    """Dispatch an RPC call directly, bypassing the per-method wrapper.

    Useful in tests to call methods by name without going through the
    installed closure attribute.

    Args:
        method: Name of the remote method to invoke.
        args: Positional arguments to forward.
        kwargs: Keyword arguments to forward.

    Returns:
        The return value of the remote method call.
    """
    return self._ipc_conn.call(method, args, kwargs)  # type: ignore[attr-defined]


def _install_rpc_wrapper(instance: Any, method_name: str, original_method: Any) -> None:
    """Install an instance-level RPC wrapper for *method_name* on *instance*.

    The wrapper is set as an instance attribute, shadowing the class-level
    method.  Annotations, docstring, and signature of *original_method* are
    copied via ``functools.wraps`` so that IDEs and ``inspect.signature()``
    reflect the original call signature.

    Args:
        instance: The proxy instance to install the wrapper on.
        method_name: Name of the method to wrap.
        original_method: The original unbound method from the user class,
            used as the ``functools.wraps`` source.
    """
    weak_instance = weakref.ref(instance)

    @functools.wraps(original_method)
    def _rpc_wrapper(*args, **kwargs):
        p = weak_instance()
        if p is None:
            raise RuntimeError(
                f"IPC call '{method_name}' failed: worker has been deleted"
            )
        return p._ipc_conn.call(method_name, args, kwargs)

    object.__setattr__(instance, method_name, _rpc_wrapper)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ipc(cls: Type[T], *args: Any, **kwargs: Any) -> T:
    """Spawn *cls* in an isolated child process and return a proxy stub.

    The stub exposes the same public methods as *cls*; each call is
    transparently forwarded to the child process via ZeroMQ RPC.
    Constructor arguments *args* / *kwargs* are passed to the child's
    ``__init__`` via stdin (MsgPack-encoded) to avoid shell-escaping issues.
    The child process is shut down when the proxy stub is garbage-collected.

    Args:
        cls: The class to instantiate in the child process.
        *args: Positional arguments forwarded to ``cls.__init__``.
        **kwargs: Keyword arguments forwarded to ``cls.__init__``.

    Returns:
        Proxy stub with the same public interface as *cls*.

    Raises:
        RuntimeError: If the child process fails to start or times out.
    """

    module_name = cls.__module__
    class_name = cls.__name__

    if module_name == "__main__":
        raise RuntimeError(
            "Classes defined in a __main__ script cannot be used with ipc(). "
            "Move the class to an importable module."
        )

    methods = _get_public_methods(cls)

    # -----------------------------------------------------------------
    # Build the -c command for the child process.
    # ipc_server.py is loaded by file path (via importlib.util) to avoid
    # triggering the full degirum_tools package __init__.
    # -----------------------------------------------------------------
    cmd_parts = [
        "import importlib.util;",
        f"_spec = importlib.util.spec_from_file_location('_ipc_server', {repr(_IPC_SERVER_FILE)});",
        "_ipc_server = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ipc_server);",
        f"import {module_name} as _user_module;",
        f"_ipc_server._run_server_loop(getattr(_user_module, {repr(class_name)}), frozenset({repr(list(methods))}))",
    ]
    cmd = "".join(cmd_parts)

    # Spawn the child process with the command
    proc = subprocess.Popen(
        [sys.executable, "-c", cmd],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Send constructor args via stdin; close to signal EOF.  The server
    # reads stdin until EOF, then writes the ZMQ endpoint to stdout —
    # so reading one line here is race-condition-free.
    assert proc.stdin is not None
    ctor_payload = _pack({_KEY_ARGS: list(args), _KEY_KWARGS: kwargs})
    proc.stdin.write(ctor_payload)
    proc.stdin.close()

    # Get response from the child process, which should be the advertised ZMQ endpoint.
    # Read the endpoint line with a bounded timeout so a hung child
    # (e.g. blocked import or constructor) does not block forever.
    _endpoint_q: queue.Queue[bytes] = queue.Queue()

    def _reader():
        assert proc.stdout is not None
        _endpoint_q.put(proc.stdout.readline())

    threading.Thread(target=_reader, daemon=True).start()
    startup_timed_out = False

    try:
        endpoint_line = _endpoint_q.get(timeout=_SERVER_STARTUP_TIMEOUT_S)
    except queue.Empty:
        endpoint_line = b""
        startup_timed_out = True

    if not endpoint_line:
        if startup_timed_out:
            try:
                # Kill before reading stderr: if the child is still alive,
                # proc.stderr.read() would block until it exits and closes the pipe.
                proc.kill()
                proc.wait(timeout=_SERVER_TERMINATE_TIMEOUT_S)
            except Exception:
                pass
        try:
            assert proc.stderr is not None
            stderr_output = proc.stderr.read().decode(errors="replace")
        except Exception:
            stderr_output = ""
        reason = (
            f"timed out after {_SERVER_STARTUP_TIMEOUT_S}s"
            if startup_timed_out
            else "exited before advertising endpoint"
        )
        raise RuntimeError(
            f"IPC server for {class_name} {reason}.\n"
            f"Server stderr:\n{stderr_output}"
        )

    endpoint = endpoint_line.decode().strip()

    # Connect ZMQ REQ socket.  The server writes the endpoint only after
    # bind() returns, so it is already ready to accept connections.
    conn = _IPCConnection(proc)
    conn.connect(endpoint)

    # Build a dynamic subclass of cls that adds IPC plumbing (__del__,
    # _ipc_process, _ipc_call) without running __init__.
    # Inheriting from cls means the return type is T, so linters see the full
    # original method signatures on the returned object.
    _proxy_cls = type(
        f"_IPC_{class_name}",
        (cls,),
        {
            "__del__": _proxy_ipc_shutdown,
            "_ipc_process": property(_proxy_ipc_process),
            "_ipc_call": _proxy_ipc_call,
        },
    )
    instance = cast(T, object.__new__(_proxy_cls))
    object.__setattr__(instance, "_ipc_conn", conn)

    # Install RPC wrappers for each public method
    for method_name in methods:
        _install_rpc_wrapper(instance, method_name, getattr(cls, method_name))

    return instance
