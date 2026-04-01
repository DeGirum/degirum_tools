#
# ipc.py: inter-process communication support via ZeroMQ RPC
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#
# Implements IPCBase class and @ipc decorator to transparently execute
# subclass instances in isolated child processes, communicating via
# MsgPack-encoded RPC over ZeroMQ REQ/REP sockets.
#
# Usage:
#   class MyWorker(IPCBase):
#       def __init__(self, param):
#           self.param = param
#
#       @ipc
#       def compute(self, x):
#           return x * self.param
#
#   worker = MyWorker(param=3)   # spawns child process transparently
#   result = worker.compute(10)  # executes in child process, returns 30
#   del worker                   # shuts down child process cleanly
#

import subprocess
import sys
import os
import enum
import queue
import threading
import weakref
import traceback
from typing import Any, Optional, Set, Type

import zmq
import msgpack
import msgpack_numpy

# Patch msgpack to transparently encode/decode numpy arrays
msgpack_numpy.patch()


# ---------------------------------------------------------------------------
# @ipc decorator
# ---------------------------------------------------------------------------


def ipc(fn: Any) -> Any:
    """Decorator that marks a method as remotely callable via IPC.

    On the server side the original function is called directly.
    On the client side (proxy) the call is forwarded to the server process
    via MsgPack-encoded RPC over a ZeroMQ REQ socket.

    Note: use this decorator only on regular instance methods of IPCBase subclasses.
    """
    _FORBIDDEN = {"__init__", "__new__", "__del__"}
    name = getattr(fn, "__name__", None)
    if name in _FORBIDDEN:
        raise TypeError(
            f"@ipc cannot be applied to '{name}': lifecycle methods are not remotely callable"
        )
    fn._is_ipc = True
    return fn


# ---------------------------------------------------------------------------
# Support classes
# ---------------------------------------------------------------------------


class IPCMode(enum.Enum):
    """Internal enum to distinguish client vs server execution mode."""

    CLIENT = "CLIENT"
    SERVER = "SERVER"


class IPCRemoteError(Exception):
    """Raised on the client when the server-side method raises an exception."""

    def __init__(self, remote_type: str, message: str, remote_traceback: str):
        self.remote_type = remote_type
        self.remote_traceback = remote_traceback
        super().__init__(
            f"Remote {remote_type}: {message}\n"
            f"--- Remote traceback ---\n{remote_traceback}"
        )


# ---------------------------------------------------------------------------
# IPCBase
# ---------------------------------------------------------------------------


class IPCBase:
    """Base class for objects that execute in an isolated child process.

    Subclasses mark methods with @ipc to expose them as RPC endpoints.
    Client-side construction automatically spawns the child process and
    returns a lightweight proxy; subclass __init__ runs only in the server.

    The server process is requested to shut down when the proxy object is deleted (__del__ is called).
    """

    # ---------------------------------------------------------------------------
    # Constants
    # ---------------------------------------------------------------------------

    # Special method name for graceful shutdown.
    _SHUTDOWN_METHOD = "__shutdown__"

    # Maximum time (seconds) to wait for the server process to advertise its endpoint.
    _SERVER_STARTUP_TIMEOUT_S = 30

    # Timeout for individual RPC calls (ms). A timed-out REQ socket cannot be
    # reused, so any timeout is treated as a fatal server failure.
    _RPC_TIMEOUT_MS = 30_000

    # Polling interval (ms) used in _ipc_call to check whether the server
    # process is still alive while waiting for a response.  Shorter values
    # make crash detection faster at the cost of more syscalls; 500 ms is a
    # good balance.
    _RPC_POLL_INTERVAL_MS = 500

    # Timeout (seconds) when waiting for the child process to exit after kill.
    _TERMINATE_TIMEOUT_S = 5

    # Polling interval (ms) for the server's recv loop; allows the orphan guard
    # to detect a dead client even if no requests arrive.
    _SERVER_RECV_TIMEOUT_MS = 1_000

    # Keys used in MsgPack request/response dicts.
    _KEY_METHOD = "method"
    _KEY_ARGS = "args"
    _KEY_KWARGS = "kwargs"
    _KEY_RESULT = "result"
    _KEY_ERROR = "error"
    _KEY_TYPE = "type"
    _KEY_MESSAGE = "message"
    _KEY_TRACEBACK = "traceback"

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pack(obj: Any) -> bytes:
        """Pack a Python object to bytes using MsgPack."""
        return msgpack.packb(obj, use_bin_type=True)

    @staticmethod
    def _unpack(data: bytes) -> Any:
        """Unpack bytes to a Python object using MsgPack."""
        return msgpack.unpackb(data, raw=False)

    # ------------------------------------------------------------------ #
    # Class-level state
    # ------------------------------------------------------------------ #

    # Set to IPCMode.SERVER in the child process before subclass instantiation.
    _mode: IPCMode = IPCMode.CLIENT

    # Populated per subclass by __init_subclass__.
    _ipc_methods: Set[str]

    # Instance attributes (client-mode instances only).
    _ipc_is_server: bool
    _ipc_zmq_ctx: Optional[zmq.Context[zmq.Socket[bytes]]]
    _ipc_socket: Optional[zmq.Socket[bytes]]
    _ipc_poller: Optional[zmq.Poller]
    _ipc_lock: threading.Lock
    _ipc_process: Optional[subprocess.Popen[bytes]]

    # ------------------------------------------------------------------ #
    # Subclass registration
    # ------------------------------------------------------------------ #

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Collect every name tagged @ipc anywhere in the MRO (below IPCBase),
        # then keep only those whose *effective* (most-derived) definition still
        # carries _is_ipc=True.  This means a subclass can un-export an inherited
        # @ipc method simply by overriding it without the decorator.
        candidates = {
            name
            for klass in cls.__mro__
            if klass is not IPCBase and klass is not object
            for name, val in vars(klass).items()
            if callable(val) and getattr(val, "_is_ipc", False)
        }
        cls._ipc_methods = {
            name
            for name in candidates
            if getattr(getattr(cls, name, None), "_is_ipc", False)
        }
        # Wrap the subclass __init__ so it is skipped on client-side proxies.
        # Wrapping is done here (class definition time) because Python's object
        # construction protocol calls type(instance).__init__, not any instance
        # attribute — so patching the instance in __new__ has no effect.
        # The _is_ipc_guarded sentinel prevents double-wrapping if
        # __init_subclass__ is somehow called more than once for the same class.
        if "__init__" in cls.__dict__:
            _original_init = cls.__dict__["__init__"]
            if not getattr(_original_init, "_is_ipc_guarded", False):

                def _guarded_init(self, *args, _orig=_original_init, **kwargs):
                    if not self._ipc_is_server:
                        return
                    _orig(self, *args, **kwargs)

                _guarded_init._is_ipc_guarded = True  # type: ignore[attr-defined]
                setattr(cls, "__init__", _guarded_init)

    # ------------------------------------------------------------------ #
    # Construction routing
    # ------------------------------------------------------------------ #

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        if cls._mode == IPCMode.SERVER:
            # Server path: __init__ runs normally after __new__ returns.
            instance._ipc_is_server = True
        else:
            # Client path: spawn server, install RPC wrappers, skip __init__.
            instance._ipc_is_server = False
            instance._ipc_zmq_ctx = None
            instance._ipc_socket = None
            instance._ipc_poller = None
            instance._ipc_process = None
            instance._ipc_lock = threading.Lock()
            instance._start_server(cls, args, kwargs)
        return instance

    # ------------------------------------------------------------------ #
    # Client side
    # ------------------------------------------------------------------ #

    def _start_server(self, cls: Type["IPCBase"], args: tuple, kwargs: dict) -> None:
        """Spawn the server child process via `python -c`, read the advertised
        endpoint from its stdout, and install per-instance RPC wrappers.
        Constructor args are passed to the child via stdin as a MsgPack blob
        to avoid shell-escaping issues.
        """
        module_name = cls.__module__
        class_name = cls.__name__

        if module_name == "__main__":
            raise RuntimeError(
                "IPCBase subclasses defined in a __main__ script are not supported. "
                "Move the class to an importable module."
            )

        # -----------------------------------------------------------------
        # Build the -c command.
        # The command:
        #   1. Imports IPCBase and switches to SERVER mode.
        #   2. Imports the module containing the derived class.
        #   3. Looks up the class by name in the imported module.
        #   4. Reads constructor args from stdin (until EOF).
        #   5. Instantiates the class (runs subclass __init__ normally).
        #   6. Calls _run_server() which binds to an OS-assigned port,
        #      writes the actual endpoint to stdout, then dispatches.
        # -----------------------------------------------------------------
        cmd_parts = [
            "import sys;",
            f"from {IPCBase.__module__} import IPCBase, IPCMode;",
            f"import {module_name} as user_module;",
            f"server_class = getattr(user_module, {repr(class_name)});",
            "server_class._mode = IPCMode.SERVER;",
            "ctor_data = IPCBase._unpack(sys.stdin.buffer.read());",
            "server = server_class(*ctor_data[IPCBase._KEY_ARGS], **ctor_data[IPCBase._KEY_KWARGS]);",
            "del server_class._mode;",
            "server._run_server()",
        ]
        cmd = "".join(cmd_parts)

        ctor_payload = self._pack(
            {self._KEY_ARGS: list(args), self._KEY_KWARGS: kwargs}
        )

        proc = subprocess.Popen(
            [sys.executable, "-c", cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Send constructor args via stdin; close to signal EOF.  The server
        # reads stdin until EOF, then writes the actual ZMQ endpoint to
        # stdout — so reading one line here is race-condition-free.
        assert proc.stdin is not None
        proc.stdin.write(ctor_payload)
        proc.stdin.close()
        self._ipc_process = proc

        # Read the endpoint line with a bounded timeout so a hung child
        # (e.g. blocked import or constructor) does not block forever.
        _endpoint_q: queue.Queue[bytes] = queue.Queue()

        def _reader():
            assert proc.stdout is not None
            _endpoint_q.put(proc.stdout.readline())

        threading.Thread(target=_reader, daemon=True).start()
        _startup_timed_out = False
        try:
            endpoint_line = _endpoint_q.get(timeout=self._SERVER_STARTUP_TIMEOUT_S)
        except queue.Empty:
            endpoint_line = b""
            _startup_timed_out = True
        if not endpoint_line:
            # Use the queue-timeout flag rather than proc.poll() to decide:
            # the process may close stdout and put b"" on the queue before
            # its exit code is visible to poll(), causing a false "timed out".
            timed_out = _startup_timed_out
            # Kill before reading stderr: if the child is still alive,
            # proc.stderr.read() would block until it exits and closes the pipe.
            if timed_out:
                try:
                    proc.kill()
                    proc.wait(timeout=self._TERMINATE_TIMEOUT_S)
                except Exception:
                    pass
            try:
                assert proc.stderr is not None
                stderr_output = proc.stderr.read().decode(errors="replace")
            except Exception:
                stderr_output = ""
            self._ipc_cleanup()
            reason = (
                f"timed out after {self._SERVER_STARTUP_TIMEOUT_S}s"
                if timed_out
                else "exited before advertising endpoint"
            )
            raise RuntimeError(
                f"IPC server for {class_name} {reason}.\n"
                f"Server stderr:\n{stderr_output}"
            )
        endpoint = endpoint_line.decode().strip()

        # Connect ZMQ REQ socket.  The server writes the endpoint only after
        # bind() returns, so it is already ready to accept connections.
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDTIMEO, self._RPC_TIMEOUT_MS)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoint)
        self._ipc_zmq_ctx = ctx
        self._ipc_socket = sock
        self._ipc_poller = zmq.Poller()
        self._ipc_poller.register(sock, zmq.POLLIN)

        for method_name in cls._ipc_methods:
            self._install_rpc_wrapper(method_name)

    def _install_rpc_wrapper(self, method_name: str) -> None:
        """Replace *method_name* on this instance with a closure that
        forwards calls to the server via _ipc_call.
        """
        weak_self = weakref.ref(self)

        def _rpc_wrapper(*args, **kwargs):
            instance = weak_self()
            if instance is None:
                raise RuntimeError(
                    f"IPC call '{method_name}' failed: worker has been deleted"
                )
            return instance._ipc_call(method_name, args, kwargs)

        _rpc_wrapper.__name__ = method_name
        object.__setattr__(self, method_name, _rpc_wrapper)

    def _ipc_call(self, method: str, args: tuple, kwargs: dict) -> Any:
        """Serialize and dispatch a single RPC call; return the result.

        Sends the request then polls for the response in a loop, waking every
        _RPC_POLL_INTERVAL_MS to check whether the server process is still
        alive.  This allows arbitrarily long-running methods while still
        detecting a crashed server promptly.
        """

        assert self._ipc_socket is not None
        assert self._ipc_poller is not None

        payload = self._pack(
            {
                self._KEY_METHOD: method,
                self._KEY_ARGS: list(args),
                self._KEY_KWARGS: kwargs,
            }
        )
        with self._ipc_lock:
            try:
                self._ipc_socket.send(payload)
            except zmq.ZMQError as exc:
                self._ipc_cleanup()
                raise RuntimeError(
                    f"IPC call '{method}' failed due to transport error: {exc}"
                ) from exc

            while True:
                ready = dict(self._ipc_poller.poll(timeout=self._RPC_POLL_INTERVAL_MS))
                if self._ipc_socket in ready:
                    break
                # Server process died without responding.
                if (
                    self._ipc_process is not None
                    and self._ipc_process.poll() is not None
                ):
                    self._ipc_cleanup()
                    raise RuntimeError(
                        f"IPC call '{method}' failed: server process exited unexpectedly"
                    )

            try:
                raw = self._ipc_socket.recv()
            except zmq.ZMQError as exc:
                self._ipc_cleanup()
                raise RuntimeError(
                    f"IPC call '{method}' failed due to transport error: {exc}"
                ) from exc

        response = self._unpack(raw)

        if response.get(self._KEY_ERROR):
            raise IPCRemoteError(
                response.get(self._KEY_TYPE, "RemoteError"),
                response.get(self._KEY_MESSAGE, ""),
                response.get(self._KEY_TRACEBACK, ""),
            )
        return response[self._KEY_RESULT]

    def __del__(self):
        """Deleter. On client instances, send shutdown RPC to the server and release resources."""

        if getattr(self, "_ipc_is_server", True):
            return
        self._ipc_shutdown()

    def _ipc_shutdown(self) -> None:
        """Send shutdown RPC to the server and release all resources."""

        graceful = False
        with self._ipc_lock:
            try:
                if self._ipc_socket is not None:
                    self._ipc_socket.send(
                        self._pack(
                            {
                                self._KEY_METHOD: self._SHUTDOWN_METHOD,
                                self._KEY_ARGS: [],
                                self._KEY_KWARGS: {},
                            }
                        )
                    )
                    self._ipc_socket.recv()  # wait for ack
                    graceful = True
            except Exception:
                pass
        self._ipc_cleanup(graceful_shutdown=graceful)

    def _ipc_cleanup(self, *, graceful_shutdown: bool = False) -> None:
        """Release ZMQ resources and reap the child process.

        If *graceful_shutdown* is True the server already acknowledged the
        shutdown command, so we give the process a chance to exit on its own
        before escalating to kill.  Otherwise we kill immediately.
        """

        try:
            if self._ipc_poller is not None and self._ipc_socket is not None:
                self._ipc_poller.unregister(self._ipc_socket)
        except Exception:
            pass
        self._ipc_poller = None

        try:
            if self._ipc_socket is not None:
                self._ipc_socket.close(linger=0)
                self._ipc_socket = None
        except Exception:
            graceful_shutdown = False

        try:
            if self._ipc_zmq_ctx is not None:
                self._ipc_zmq_ctx.term()
                self._ipc_zmq_ctx = None
        except Exception:
            graceful_shutdown = False

        proc = self._ipc_process
        self._ipc_process = None
        if proc is not None:
            if graceful_shutdown:
                try:
                    proc.wait(timeout=self._TERMINATE_TIMEOUT_S)
                    return
                except Exception:
                    pass
            try:
                proc.kill()
                proc.wait(timeout=self._TERMINATE_TIMEOUT_S)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Server side
    # ------------------------------------------------------------------ #

    def _run_server(self) -> None:
        """Bind a ZMQ REP socket and dispatch incoming RPC requests.

        Binds to an OS-assigned port, writes the actual endpoint to stdout
        so the client can connect without a race condition, then enters the
        dispatch loop.  Exits cleanly when a shutdown command is received or
        when the parent process disappears (orphan guard).  Called only in
        the child process, after subclass __init__ has already run.
        """

        if not getattr(self, "_ipc_is_server", False):
            raise RuntimeError(
                "_run_server() should only be called on server-mode instances"
            )

        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, self._SERVER_RECV_TIMEOUT_MS)
        sock.bind("tcp://127.0.0.1:0")
        endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
        sys.stdout.write(endpoint + "\n")
        sys.stdout.flush()

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

            request = self._unpack(raw)
            method = request.get(self._KEY_METHOD, "")
            req_args = request.get(self._KEY_ARGS, [])
            req_kwargs = request.get(self._KEY_KWARGS, {})

            if method == self._SHUTDOWN_METHOD:
                sock.send(self._pack({self._KEY_RESULT: "bye"}))
                break

            if method not in self._ipc_methods:
                sock.send(
                    self._pack(
                        {
                            self._KEY_ERROR: True,
                            self._KEY_TYPE: "AttributeError",
                            self._KEY_MESSAGE: f"Method '{method}' is not an @ipc method",
                            self._KEY_TRACEBACK: "",
                        }
                    )
                )
                continue

            try:
                result = getattr(self, method)(*req_args, **req_kwargs)
                sock.send(self._pack({self._KEY_RESULT: result}))
            except Exception as exc:
                sock.send(
                    self._pack(
                        {
                            self._KEY_ERROR: True,
                            self._KEY_TYPE: type(exc).__name__,
                            self._KEY_MESSAGE: str(exc),
                            self._KEY_TRACEBACK: traceback.format_exc(),
                        }
                    )
                )

        sock.close(linger=0)
        ctx.term()
