#
# test_ipc.py: unit tests for ipc.py inter-process communication module
#
# Copyright DeGirum Corporation 2026
# All rights reserved
#

import os
from typing import Optional
import pytest
import psutil
import numpy as np

from degirum_tools import ipc

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

    # --- IPCRemoteError stores attributes and formats its message ---
    err = ipc.IPCRemoteError(
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
        assert ipc._unpack(ipc._pack(obj)) == obj

    # --- _pack/_unpack round-trips a numpy array ---
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = ipc._unpack(ipc._pack(arr))
    assert np.array_equal(result, arr)
    assert result.dtype == arr.dtype

    # --- _pack_multipart/_unpack_multipart: list of arrays in result ---
    arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    arr2 = np.array([4, 5, 6], dtype=np.int16)
    arr_2d = np.arange(6, dtype=np.float64).reshape(2, 3)

    def _roundtrip(envelope):
        return ipc._unpack_multipart(ipc._pack_multipart(envelope))

    out = _roundtrip({ipc._KEY_RESULT: [arr1, arr2]})
    assert np.array_equal(out[ipc._KEY_RESULT][0], arr1)
    assert out[ipc._KEY_RESULT][0].dtype == arr1.dtype
    assert np.array_equal(out[ipc._KEY_RESULT][1], arr2)
    assert out[ipc._KEY_RESULT][1].dtype == arr2.dtype

    # --- _pack_multipart/_unpack_multipart: dict of arrays in result ---
    out = _roundtrip({ipc._KEY_RESULT: {"a": arr1, "b": arr2}})
    assert np.array_equal(out[ipc._KEY_RESULT]["a"], arr1)
    assert np.array_equal(out[ipc._KEY_RESULT]["b"], arr2)

    # --- _pack_multipart/_unpack_multipart: tuple of arrays (decoded as list) ---
    out = _roundtrip({ipc._KEY_RESULT: (arr1, arr2)})
    assert np.array_equal(out[ipc._KEY_RESULT][0], arr1)
    assert np.array_equal(out[ipc._KEY_RESULT][1], arr2)

    # --- _pack_multipart/_unpack_multipart: plain scalar tuple (no arrays, decoded as list) ---
    out = _roundtrip({ipc._KEY_RESULT: (1, "hello", True)})
    assert out[ipc._KEY_RESULT] == [1, "hello", True]

    # --- _pack_multipart/_unpack_multipart: tuple as a direct arg value ---
    out = _roundtrip({ipc._KEY_ARGS: [(arr1, 99)]})
    assert np.array_equal(out[ipc._KEY_ARGS][0][0], arr1)
    assert out[ipc._KEY_ARGS][0][1] == 99

    # --- _pack_multipart/_unpack_multipart: 2-D array inside a list ---
    out = _roundtrip({ipc._KEY_RESULT: [arr_2d, arr1]})
    assert out[ipc._KEY_RESULT][0].shape == arr_2d.shape
    assert np.array_equal(out[ipc._KEY_RESULT][0], arr_2d)

    # --- _pack_multipart/_unpack_multipart: deeply nested dict -> list -> dict ---
    out = _roundtrip({ipc._KEY_RESULT: {"outer": [arr1, {"inner": arr2}]}})
    assert np.array_equal(out[ipc._KEY_RESULT]["outer"][0], arr1)
    assert np.array_equal(out[ipc._KEY_RESULT]["outer"][1]["inner"], arr2)

    # --- _pack_multipart/_unpack_multipart: list of arrays in args ---
    out = _roundtrip({ipc._KEY_ARGS: [arr1, arr2]})
    assert np.array_equal(out[ipc._KEY_ARGS][0], arr1)
    assert np.array_equal(out[ipc._KEY_ARGS][1], arr2)

    # --- _pack_multipart/_unpack_multipart: tuple nested inside list in args ---
    out = _roundtrip({ipc._KEY_ARGS: [(arr1, arr2), 42]})
    assert np.array_equal(out[ipc._KEY_ARGS][0][0], arr1)
    assert np.array_equal(out[ipc._KEY_ARGS][0][1], arr2)
    assert out[ipc._KEY_ARGS][1] == 42

    # --- _pack_multipart/_unpack_multipart: no arrays passes through unchanged ---
    out = _roundtrip({ipc._KEY_RESULT: {"x": [1, 2, 3], "y": "hello"}})
    assert out[ipc._KEY_RESULT] == {"x": [1, 2, 3], "y": "hello"}

    # --- _get_public_methods returns all public instance methods ---
    class Worker:
        def public_method(self):
            pass

        def _private_method(self):
            pass

    assert "public_method" in ipc._get_public_methods(Worker)
    assert "_private_method" not in ipc._get_public_methods(Worker)

    # --- public methods from a base class are included for subclasses ---
    class Base:
        def base_method(self):
            pass

    class Child(Base):
        def child_method(self):
            pass

    assert "base_method" in ipc._get_public_methods(Child)
    assert "child_method" in ipc._get_public_methods(Child)
    assert "child_method" not in ipc._get_public_methods(Base)

    # --- overriding a public method keeps it in the method set ---
    class Base2:
        def shared(self):
            pass

        def only_base(self):
            pass

    class Child2(Base2):
        def shared(self):  # override: still a public method
            pass

    assert "only_base" in ipc._get_public_methods(Child2)
    assert "shared" in ipc._get_public_methods(Child2)
    assert "shared" in ipc._get_public_methods(Base2)

    # --- class with __module__ == '__main__' raises RuntimeError ---
    class FakeMainWorker:
        def noop(self):
            pass

    FakeMainWorker.__module__ = "__main__"
    with pytest.raises(RuntimeError, match="__main__"):
        ipc.spawn(FakeMainWorker)


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
        NestedArrayWorker,
    )

    # --- basic method call: multiply(x) returns x * factor in child process ---
    w = ipc.spawn(MultiplyWorker, factor=3)
    try:
        # --- IPC method on client instance is a proxy, not the original function ---
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
        with pytest.raises(ipc.IPCRemoteError) as exc_info:
            w.fail()
        err = exc_info.value
        assert err.remote_type == "ValueError"
        assert "intentional error" in str(err)
        assert err.remote_traceback  # non-empty traceback string

        # --- worker remains usable after a remote exception ---
        assert w.multiply(4) == 12

        # --- non-existent method raises IPCRemoteError(AttributeError) ---
        with pytest.raises(ipc.IPCRemoteError) as exc_info:
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

    # --- context manager: __exit__ shuts down the server deterministically ---
    with ipc.spawn(MultiplyWorker, factor=2) as wcm:
        assert wcm.multiply(5) == 10
        cm_pid = wcm._ipc_process.pid
    assert not psutil.pid_exists(cm_pid)

    # --- __del__ is a no-op after __exit__ (double-shutdown safe) ---
    wcm2 = ipc.spawn(MultiplyWorker, factor=1)
    cm_pid2 = wcm2._ipc_process.pid
    wcm2.__exit__(None, None, None)
    assert not psutil.pid_exists(cm_pid2)
    del wcm2  # must not raise

    # --- server process that exits mid-call raises RuntimeError ---
    w3 = ipc.spawn(ExitingWorker)
    with pytest.raises(RuntimeError, match="server process exited unexpectedly"):
        w3.exit_now()

    # --- server that crashes during __init__ raises RuntimeError with stderr ---
    with pytest.raises(
        RuntimeError, match="exited before advertising endpoint"
    ) as exc_info2:
        ipc.spawn(BrokenWorker)
    assert "constructor failure" in str(exc_info2.value)

    # --- numpy arrays are passed to remote methods and returned correctly ---
    wa = ipc.spawn(ArrayWorker)
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
        del wa

    # --- nested arrays: list/dict/tuple/deep structures survive IPC round-trip ---
    wna = ipc.spawn(NestedArrayWorker)
    try:
        arr_n = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr_n2 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        arr_2d_n = np.arange(6, dtype=np.float64).reshape(2, 3)

        # list result: [arr, arr*2]
        res = wna.list_result(arr_n)
        assert np.array_equal(res[0], arr_n) and res[0].dtype == arr_n.dtype
        assert np.array_equal(res[1], arr_n * 2)

        # 2-D array in a list result
        res = wna.list_result(arr_2d_n)
        assert res[0].shape == arr_2d_n.shape and np.array_equal(res[0], arr_2d_n)
        assert np.array_equal(res[1], arr_2d_n * 2)

        # dict result: {"a": arr, "b": arr*2}
        res = wna.dict_result(arr_n)
        assert np.array_equal(res["a"], arr_n) and np.array_equal(res["b"], arr_n * 2)

        # tuple result: comes back as list (msgpack encodes tuples as arrays)
        res = wna.tuple_result(arr_n)
        assert np.array_equal(res[0], arr_n) and np.array_equal(res[1], arr_n * 2)

        # tuple as argument: (arr, arr2) passed as a tuple, server sums elements
        res = wna.accept_tuple((arr_n, arr_n2))
        assert np.array_equal(res, arr_n + arr_n2)

        # deeply nested result: {"outer": [arr, {"inner": arr*2}]}
        res = wna.deep_result(arr_n)
        assert np.array_equal(res["outer"][0], arr_n)
        assert np.array_equal(res["outer"][1]["inner"], arr_n * 2)

        # list of arrays as argument
        res = wna.accept_list([arr_n, arr_n2])
        assert np.array_equal(res, arr_n + arr_n2)

        # dict of arrays as argument
        res = wna.accept_dict({"x": arr_n, "y": arr_n2})
        assert np.array_equal(res, arr_n + arr_n2)
    finally:
        del wna

    # --- server-side method can spawn and call a sub-worker in a third process ---
    wn = ipc.spawn(NestingWorker, factor=5)
    try:
        assert wn.compute(7) == 35
        assert wn.compute(0) == 0
        assert wn.add_via_sub(10, 3) == 13
    finally:
        del wn


# ===========================================================================
# Tests for Out/InOut argument support in ipc.py
# ===========================================================================


def test_ipc_inout(_pythonpath):
    """Integration tests for Out/InOut writeback argument support."""
    from ipc_test_workers import InOutWorker

    w = ipc.spawn(InOutWorker)
    try:
        # --- InOut list: server mutates in-place, original is patched ---
        ll = [1, 2, 3]
        w.fill_list(ipc.InOut(ll), 99)
        assert ll == [99, 99, 99]

        # --- InOut list: identity after fill (no mutation) ---
        ll2 = [0, 0]
        w.fill_list(ipc.InOut(ll2), 7)
        assert ll2 == [7, 7]

        # --- Out list: empty sentinel sent; server appends, original patched ---
        out_list = [10, 20, 30]  # existing contents NOT sent
        w.append_list(ipc.Out(out_list), 42)
        assert out_list == [42]  # only what server appended

        # --- InOut dict: server sets key, original is patched ---
        d = {"a": 0}
        w.fill_dict(ipc.InOut(d), "a", 55)
        assert d == {"a": 55}

        # --- Out dict: empty sentinel sent; server sets key, original patched ---
        d2 = {"stale": 999}
        w.fill_dict(ipc.Out(d2), "x", 7)
        assert d2 == {"x": 7}

        # --- InOut bytearray: server overwrites bytes, original is patched ---
        buf = bytearray([0, 0, 0, 0])
        w.fill_bytes(ipc.InOut(buf), 0xFF)
        assert buf == bytearray([0xFF, 0xFF, 0xFF, 0xFF])

        # --- InOut numpy array: server scales in-place, original is patched ---
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original_ptr = arr.ctypes.data
        w.scale_array(ipc.InOut(arr), 2.0)
        assert np.allclose(arr, [2.0, 4.0, 6.0])
        assert arr.ctypes.data == original_ptr  # same buffer, patched via copyto

        # --- InOut as keyword argument ---
        kw = ["a", "b", "c"]
        w.fill_list(data=ipc.InOut(kw), value=0)
        assert kw == [0, 0, 0]

        # --- two InOut args simultaneously: swap ---
        x = [1, 2]
        y = [3, 4]
        w.swap_lists(ipc.InOut(x), ipc.InOut(y))
        assert x == [3, 4]
        assert y == [1, 2]

        # --- non-wrapped args are unaffected by the writeback path ---
        plain = [5, 6, 7]
        w.fill_list(plain, 0)  # no InOut wrapper: plain list, no writeback
        assert plain == [5, 6, 7]  # unchanged on the client side

    finally:
        del w

    # --- _InOutSupport.check_type raises TypeError for unsupported types ---
    for bad in (42, "hello", (1, 2), {1, 2}, object()):
        with pytest.raises(TypeError, match="Out/InOut does not support"):
            ipc._InOutSupport.check_type(bad)

    # --- Out/InOut constructors reject unsupported types immediately ---
    with pytest.raises(TypeError):
        ipc.Out(42)
    with pytest.raises(TypeError):
        ipc.InOut("string")

    # --- _InOutSupport.placeholder returns correct empty sentinels ---
    assert ipc._InOutSupport.placeholder([1, 2, 3]) == []
    assert ipc._InOutSupport.placeholder({"a": 1}) == {}
    assert ipc._InOutSupport.placeholder(bytearray(b"abc")) == bytearray()
    arr_ph = ipc._InOutSupport.placeholder(np.zeros((3, 2), dtype=np.float32))
    assert arr_ph.shape == (3, 2)
    assert arr_ph.dtype == np.float32

    # --- _InOutSupport.patch for numpy raises ValueError on shape mismatch ---
    a = np.zeros((3,), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        ipc._InOutSupport.patch(a, np.zeros((4,), dtype=np.float32))


# ===========================================================================
# Thread-safety test
# ===========================================================================


def test_ipc_multithreaded(_pythonpath):
    """Multiple threads calling IPC methods concurrently must not corrupt results."""
    import threading
    from ipc_test_workers import MultiplyWorker

    N_THREADS = 16
    CALLS_PER_THREAD = 50

    w = ipc.spawn(MultiplyWorker, factor=3)
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


# ===========================================================================
# Benchmarks
# ===========================================================================


def test_ipc_array_performance(_pythonpath):
    # (payload, iterations): None measures baseline RPC overhead with no data
    cases = [(None, 10000), (10_000_000, 100)]

    """Measure round-trip throughput for numpy arrays of different sizes."""
    import time
    from ipc_test_workers import ArrayWorker

    w = ipc.spawn(ArrayWorker)
    try:
        print()
        for payload, iterations in cases:
            data: Optional[np.ndarray] = (
                None if payload is None else np.ones(payload, dtype=np.uint8)
            )
            # warm-up
            w.sink(data)

            t0 = time.perf_counter()
            for _ in range(iterations):
                w.sink(data)
            elapsed = time.perf_counter() - t0

            qps = iterations / elapsed
            nbytes = 0 if data is None else data.nbytes
            mb_per_s = (nbytes * iterations) / elapsed / 1e6
            size_str = f"{'None':>10}" if payload is None else f"{payload:>10,}"
            print(
                f"  size={size_str}  iters={iterations:>5}  "
                f"QPS={qps:>8.1f}  throughput={mb_per_s:.1f} MB/s"
            )
    finally:
        del w


def test_ipc_pack_unpack_performance():
    """Measure _pack / _unpack throughput for numpy arrays of varying sizes."""
    import time

    # (n_elements, dtype, iterations)
    cases = [
        (1_000, np.uint8, 10_000),
        (1_000_000, np.uint8, 1_000),
        (10_000_000, np.uint8, 100),
        (1_000_000, np.float32, 250),
        (1_000_000, np.float64, 125),
    ]

    print()
    for n, dtype, iterations in cases:
        arr = np.arange(n, dtype=dtype)
        obj = {ipc._KEY_RESULT: arr}

        # warm-up
        ipc._unpack(ipc._pack(obj))

        # pack
        t0 = time.perf_counter()
        for _ in range(iterations):
            packed = ipc._pack(obj)
        pack_elapsed = time.perf_counter() - t0

        # unpack
        t0 = time.perf_counter()
        for _ in range(iterations):
            result = ipc._unpack(packed)
        unpack_elapsed = time.perf_counter() - t0

        # verify round-trip correctness
        np.testing.assert_array_equal(result[ipc._KEY_RESULT], arr)

        nbytes = arr.nbytes
        pack_mb_s = (nbytes * iterations) / pack_elapsed / 1e6
        unpack_mb_s = (nbytes * iterations) / unpack_elapsed / 1e6
        print(
            f"  dtype={str(np.dtype(dtype)):>8}  len={n:>11,}  ({nbytes / 1e6:>6.1f} MB)"
            f"  pack={pack_mb_s:>7.0f} MB/s"
            f"  unpack={unpack_mb_s:>7.0f} MB/s"
        )
