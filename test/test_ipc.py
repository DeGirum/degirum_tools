#
# test_ipc.py: unit tests for ipc.py inter-process communication module
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#

import os
import pytest
import psutil
import numpy as np

from degirum_tools import IPCBase, IPCRemoteError, ipc

_test_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def _pythonpath():
    """Prepend the test directory to PYTHONPATH for spawned child processes."""
    existing = os.environ.get("PYTHONPATH", "")
    new_path = (_test_dir + os.pathsep + existing) if existing else _test_dir
    os.environ["PYTHONPATH"] = new_path
    yield
    if existing:
        os.environ["PYTHONPATH"] = existing
    else:
        del os.environ["PYTHONPATH"]


# ===========================================================================
# Unit tests — no subprocess involved
# ===========================================================================


def test_ipc_unit():
    """Unit tests for ipc.py that do not involve subprocesses."""

    # --- @ipc decorator sets _is_ipc flag and returns the same object ---
    def my_method(self):
        pass

    decorated = ipc(my_method)
    assert decorated._is_ipc is True
    assert decorated is my_method

    # --- @ipc raises TypeError for lifecycle methods ---
    for forbidden_name in ("__init__", "__new__", "__del__"):

        def fn(self):
            pass

        fn.__name__ = forbidden_name
        with pytest.raises(TypeError, match=forbidden_name):
            ipc(fn)

    # --- IPCRemoteError stores attributes and formats its message ---
    err = IPCRemoteError(
        "ValueError", "bad value", "Traceback (most recent call last):\n  ..."
    )
    assert err.remote_type == "ValueError"
    assert err.remote_traceback == "Traceback (most recent call last):\n  ..."
    msg = str(err)
    assert "Remote ValueError" in msg
    assert "bad value" in msg
    assert "Traceback" in msg

    # --- _pack/_unpack round-trips basic Python objects ---
    for obj in [
        None,
        42,
        3.14,
        "hello",
        [1, 2, 3],
        {"key": "value", "n": 99},
        True,
        False,
    ]:
        assert IPCBase._unpack(IPCBase._pack(obj)) == obj

    # --- _pack/_unpack round-trips a numpy array ---
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = IPCBase._unpack(IPCBase._pack(arr))
    assert np.array_equal(result, arr)
    assert result.dtype == arr.dtype

    # --- _ipc_methods contains all and only @ipc-decorated methods ---
    class Worker(IPCBase):
        @ipc
        def exported(self):
            pass

        def not_exported(self):
            pass

    assert "exported" in Worker._ipc_methods
    assert "not_exported" not in Worker._ipc_methods

    # --- @ipc methods from a base class are inherited by subclasses ---
    class Base(IPCBase):
        @ipc
        def base_method(self):
            pass

    class Child(Base):
        @ipc
        def child_method(self):
            pass

    assert "base_method" in Child._ipc_methods
    assert "child_method" in Child._ipc_methods
    assert "child_method" not in Base._ipc_methods

    # --- overriding an @ipc method without the decorator un-exports it ---
    class Base2(IPCBase):
        @ipc
        def shared(self):
            pass

        @ipc
        def only_base(self):
            pass

    class Child2(Base2):
        def shared(self):  # override without @ipc -> un-exports
            pass

    assert "only_base" in Child2._ipc_methods
    assert "shared" not in Child2._ipc_methods
    assert "shared" in Base2._ipc_methods  # base class set is unaffected

    # --- subclass with __module__ == '__main__' raises RuntimeError ---
    class FakeMainWorker(IPCBase):
        @ipc
        def noop(self):
            pass

    FakeMainWorker.__module__ = "__main__"
    with pytest.raises(RuntimeError, match="__main__"):
        FakeMainWorker()


# ===========================================================================
# Integration tests — spawn real child processes
# ===========================================================================


def test_ipc_integration(_pythonpath):
    """Integration tests for ipc.py that spawn real child processes."""
    from ipc_test_workers import (
        MultiplyWorker,
        BrokenWorker,
        ExitingWorker,
        ArrayWorker,
        NestingWorker,
    )

    # --- basic method call: multiply(x) returns x * factor in child process ---
    w = MultiplyWorker(factor=3)
    try:
        # --- @ipc method on client instance is a proxy, not the original function ---
        assert "multiply" in w.__dict__
        assert w.__dict__["multiply"] is not MultiplyWorker.__dict__["multiply"]
        assert w.__dict__["multiply"].__name__ == "multiply"

        # --- method calls return correct results from the server process ---
        assert w.multiply(10) == 30
        assert w.multiply(0) == 0
        assert w.multiply(-4) == -12

        # --- keyword arguments are passed through RPC correctly ---
        assert w.add(5) == 5
        assert w.add(5, b=3) == 8
        assert w.add(a=1, b=2) == 3

        # --- multiple sequential calls all return correct results ---
        assert [w.multiply(i) for i in range(5)] == [0, 3, 6, 9, 12]

        # --- server-side exception is re-raised as IPCRemoteError ---
        with pytest.raises(IPCRemoteError) as exc_info:
            w.fail()
        err = exc_info.value
        assert err.remote_type == "ValueError"
        assert "intentional error" in str(err)
        assert err.remote_traceback  # non-empty traceback string

        # --- worker remains usable after a remote exception ---
        assert w.multiply(4) == 12

        # --- non-@ipc method raises IPCRemoteError(AttributeError) ---
        with pytest.raises(IPCRemoteError) as exc_info:
            w._ipc_call("nonexistent_method", (), {})
        assert exc_info.value.remote_type == "AttributeError"

        # --- child process is running while the worker is alive ---
        assert w._ipc_process is not None
        assert w._ipc_process.poll() is None
        server_pid = w._ipc_process.pid

    finally:
        # exc_info/err hold exception __traceback__ chains that reference
        # frames where 'self' (= w) is a local variable.  Clear them first
        # so that del w drops the refcount to zero and __del__ fires.
        err = None  # type: ignore[assignment]
        exc_info = None  # type: ignore[assignment]
        del w

    # --- after del, server process is no longer running ---
    assert not psutil.pid_exists(server_pid)

    # --- server process that exits mid-call raises RuntimeError ---
    w3 = ExitingWorker()
    with pytest.raises(RuntimeError, match="server process exited unexpectedly"):
        w3.exit_now()

    # --- server that crashes during __init__ raises RuntimeError with stderr ---
    with pytest.raises(
        RuntimeError, match="exited before advertising endpoint"
    ) as exc_info2:
        BrokenWorker()
    assert "constructor failure" in str(exc_info2.value)

    # --- numpy arrays are passed to remote methods and returned correctly ---
    wa = ArrayWorker()
    try:
        arr_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr_i16 = np.array([10, 20, 30], dtype=np.int16)
        arr_2d = np.arange(6, dtype=np.float64).reshape(2, 3)

        # scale: result dtype and values preserved
        result = wa.scale(arr_f32, 3.0)
        assert np.array_equal(result, arr_f32 * 3.0)
        assert result.dtype == arr_f32.dtype

        # integer dtype round-trip
        result = wa.identity(arr_i16)
        assert np.array_equal(result, arr_i16)
        assert result.dtype == arr_i16.dtype

        # 2-D array round-trip: shape and values preserved
        result = wa.identity(arr_2d)
        assert result.shape == arr_2d.shape
        assert np.array_equal(result, arr_2d)

        # dot product: scalar result
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        assert np.isclose(wa.dot(a, b), np.dot(a, b))
    finally:
        wa._ipc_shutdown()

    # --- server-side method can spawn and call a sub-worker in a third process ---
    wn = NestingWorker(factor=5)
    try:
        assert wn.compute(7) == 35
        assert wn.compute(0) == 0
        assert wn.add_via_sub(10, 3) == 13
    finally:
        wn._ipc_shutdown()


def test_ipc_array_throughput(_pythonpath):
    """Measure round-trip throughput for numpy arrays of different sizes."""
    import time
    from ipc_test_workers import ArrayWorker

    # (payload, iterations): None measures baseline RPC overhead with no data
    cases = [(None, 10000), (1_000, 10000), (1_000_000, 1000), (10_000_000, 100)]
    cases = [(10_000_000, 100)]

    w = ArrayWorker()
    try:
        print()
        for payload, iterations in cases:
            arr = None if payload is None else np.ones(payload, dtype=np.uint8)
            # warm-up
            w.sink(arr)

            t0 = time.perf_counter()
            for _ in range(iterations):
                w.sink(arr)
            elapsed = time.perf_counter() - t0

            qps = iterations / elapsed
            nbytes = 0 if arr is None else arr.nbytes
            mb_per_s = (nbytes * iterations) / elapsed / 1e6
            size_str = f"{'None':>10}" if payload is None else f"{payload:>10,}"
            print(
                f"  size={size_str}  iters={iterations:>5}  "
                f"QPS={qps:>8.1f}  throughput={mb_per_s:.1f} MB/s"
            )
    finally:
        w._ipc_shutdown()


# ===========================================================================
# Serialization micro-benchmark — no subprocess
# ===========================================================================


def test_ipc_pack_unpack_performance():
    """Measure _pack / _unpack throughput for numpy arrays of varying sizes."""
    import time

    # (n_elements, dtype, iterations)
    cases = [
        (1_000, np.uint8, 10_000),
        (1_000_000, np.uint8, 1_000),
        (10_000_000, np.uint8, 100),
        (1_000_000, np.float32, 1_000),
        (1_000_000, np.float64, 1_000),
    ]

    print()
    for n, dtype, iterations in cases:
        arr = np.arange(n, dtype=dtype)
        obj = {IPCBase._KEY_RESULT: arr}

        # warm-up
        IPCBase._unpack(IPCBase._pack(obj))

        # pack
        t0 = time.perf_counter()
        for _ in range(iterations):
            packed = IPCBase._pack(obj)
        pack_elapsed = time.perf_counter() - t0

        # unpack
        t0 = time.perf_counter()
        for _ in range(iterations):
            result = IPCBase._unpack(packed)
        unpack_elapsed = time.perf_counter() - t0

        # verify round-trip correctness
        np.testing.assert_array_equal(result[IPCBase._KEY_RESULT], arr)

        nbytes = arr.nbytes
        pack_mb_s = (nbytes * iterations) / pack_elapsed / 1e6
        unpack_mb_s = (nbytes * iterations) / unpack_elapsed / 1e6
        print(
            f"  dtype={np.dtype(dtype)}  len={n:>10,}  ({nbytes / 1e6:>6.1f} MB)"
            f"  pack={pack_mb_s:>7.0f} MB/s"
            f"  unpack={unpack_mb_s:>7.0f} MB/s"
        )


# ===========================================================================
# Thread-safety test
# ===========================================================================


def test_ipc_multithreaded(_pythonpath):
    """Multiple threads calling @ipc methods concurrently must not corrupt results."""
    import threading
    from ipc_test_workers import MultiplyWorker

    N_THREADS = 16
    CALLS_PER_THREAD = 50

    w = MultiplyWorker(factor=3)
    errors = []
    barrier = threading.Barrier(N_THREADS)

    def worker_thread(thread_id, _w=w):
        barrier.wait()  # all threads start their first call simultaneously
        for i in range(CALLS_PER_THREAD):
            a = thread_id * 1000 + i
            expected = a + a  # add(a, a) == 2*a, independent of factor
            try:
                result = _w.add(a, a)
                if result != expected:
                    errors.append(
                        f"thread {thread_id} call {i}: got {result}, want {expected}"
                    )
            except Exception as exc:
                errors.append(f"thread {thread_id} call {i}: {exc}")

    threads = [
        threading.Thread(target=worker_thread, args=(t,)) for t in range(N_THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    del w
    assert not errors, "\n".join(errors)
