#!/usr/bin/env python3
"""
Comprehensive comparison: Sequential vs Free-threading vs Multiprocessing (pickled vs shared memory)

What this measures (with the *same* 32-bit integer buffer):
  1) Sequential baseline                              -> single core
  2) Threads (shared memory)                          -> requires free-threaded CPython to scale across cores
  3) Processes with pickled chunks                    -> copies data there & back (serialization overhead)
  4) Processes with shared memory (no copies)         -> (a) stdlib SharedMemory (portable)
                                                        (b) Linux memfd + mmap + pass_fds (zero-copy FD-backed)

Why the design:
  - Use a typed, contiguous buffer (32-bit ints) to avoid Python list object overhead
  - Keep CPU work purely in Python (no NumPy/C loops) so the parallelism effect is visible
  - Avoid misleading "copy-by-accident"; be explicit when we serialize vs when we share

Requires:
  - Python 3.10+ (for os.memfd_create you need 3.8+ on Linux; memfd mode is Linux-only)
  - For free-threaded scaling, Python 3.14 (officially supported) or 3.13 free-threaded build,
    and run with:  PYTHON_GIL=0  or  python -X gil=0 compare_parallel_modes.py
"""

from __future__ import annotations
import argparse
import gc
import math
import mmap
import os
import sys
import sysconfig
import time
from array import array
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from statistics import median

# Portable shared memory
try:
    from multiprocessing import shared_memory, get_context
    HAVE_SHM = True
except Exception:
    shared_memory = None
    HAVE_SHM = False


# ----------------------------
# Utilities
# ----------------------------

def is_free_thread_build() -> bool:
    """Does this interpreter SUPPORT free-threading?"""
    try:
        return sysconfig.get_config_var("Py_GIL_DISABLED") == 1
    except Exception:
        return False

def gil_enabled_runtime() -> bool | None:
    """Is the GIL currently enabled? (None if not supported on this Python)."""
    fn = getattr(sys, "_is_gil_enabled", None)
    if fn is None:
        return None
    try:
        return bool(fn())
    except Exception:
        return None

def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.1f} {u}"
        x /= 1024.0

def checksum(mv_uint32) -> int:
    """Cheap correctness check; sum of the first 1M values (or fewer)."""
    sample = min(1_000_000, len(mv_uint32))
    s = 0
    for i in range(sample):
        s = (s + mv_uint32[i]) & 0xFFFFFFFF
    return s

def chunk_bounds(n: int, parts: int, i: int) -> tuple[int,int]:
    """Return [start,end) for chunk i of parts over n elements."""
    base = n // parts
    extra = n % parts
    start = i * base + min(i, extra)
    end = start + base + (1 if i < extra else 0)
    return start, end

def perf_ms(f, repeats=3) -> tuple[float, list[float]]:
    """Run f() 'repeats' times; return (median_seconds, all_seconds)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        f()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0)/1e9)
    return median(times), times


# ----------------------------
# CPU-bound kernel (pure Python)
# ----------------------------

def cpu_bound_span_update(mv_uint32, start: int, end: int, iters: int) -> None:
    """
    Work per element: ~O(iters). Operates in place on a memoryview of 'I' (uint32).
    Uses arithmetic that keeps values in 32-bit space to avoid Python big-int growth.
    """
    # Constants are chosen to give a decent amount of ALU work without branches.
    for i in range(start, end):
        v = mv_uint32[i]
        acc = v ^ 0x9E3779B9  # golden ratio constant
        for j in range(iters):
            # Mix with a few operations; all masked to 32-bit
            acc = (acc + ((j + 1) * 2654435761)) & 0xFFFFFFFF
            acc ^= ((acc << 13) | (acc >> 19)) & 0xFFFFFFFF
            acc = (acc * 2246822519) & 0xFFFFFFFF
        mv_uint32[i] = acc


# ----------------------------
# Modes
# ----------------------------

def mk_uint32_buffer(n: int) -> tuple[array, memoryview]:
    """
    Allocate typed contiguous storage: array('I', n) and view it as memoryview('I').
    Initialize with 0..n-1 (vectorized-ish via frombytes would still need bytes; this is fine).
    """
    arr = array('I', range(n))
    mv = memoryview(arr)  # format='I'
    return arr, mv

# 1) Sequential baseline
def run_sequential(n: int, workers: int, iters: int, repeats: int) -> tuple[float, int]:
    def one():
        _arr, mv = mk_uint32_buffer(n)
        for i in range(workers):
            s, e = chunk_bounds(n, workers, i)
            cpu_bound_span_update(mv, s, e, iters)
        # keep checksum work outside timing in the harness
    med, _ = perf_ms(one, repeats)
    # Validate on a final run
    arr, mv = mk_uint32_buffer(n)
    for i in range(workers):
        s, e = chunk_bounds(n, workers, i)
        cpu_bound_span_update(mv, s, e, iters)
    return med, checksum(mv)

# 2) Threads: shared memory; real scaling only with -X gil=0 / PYTHON_GIL=0
def run_threads(n: int, workers: int, iters: int, repeats: int) -> tuple[float, int]:
    def one():
        _arr, mv = mk_uint32_buffer(n)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for i in range(workers):
                s, e = chunk_bounds(n, workers, i)
                futs.append(ex.submit(cpu_bound_span_update, mv, s, e, iters))
            for f in futs:
                f.result()
    med, _ = perf_ms(one, repeats)
    # Validate on a final run
    arr, mv = mk_uint32_buffer(n)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i in range(workers):
            s, e = chunk_bounds(n, workers, i)
            futs.append(ex.submit(cpu_bound_span_update, mv, s, e, iters))
        for f in futs:
            f.result()
    return med, checksum(mv)

# 3) Processes (pickled): copy chunks to workers, copy results back
def worker_pickled(chunk: list[int], start: int, iters: int) -> list[int]:
    # reconstruct typed view locally (list->loop), then return list (forces serialization)
    # To keep it pure-Python, operate directly on the chunk list:
    # use the same kernel but on Python ints
    acc_chunk = chunk[:]  # avoid modifying caller memory unexpectedly
    mv = acc_chunk  # alias
    # same math
    for idx in range(len(mv)):
        v = mv[idx]
        acc = v ^ 0x9E3779B9
        for j in range(iters):
            acc = (acc + ((j + 1) * 2654435761)) & 0xFFFFFFFF
            acc ^= ((acc << 13) | (acc >> 19)) & 0xFFFFFFFF
            acc = (acc * 2246822519) & 0xFFFFFFFF
        mv[idx] = acc
    return acc_chunk

def run_mp_pickled(n: int, workers: int, iters: int, repeats: int) -> tuple[float, int]:
    ctx = get_context("spawn")  # robust across platforms
    def one():
        # fresh buffer per run
        arr, mv = mk_uint32_buffer(n)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = []
            for i in range(workers):
                s, e = chunk_bounds(n, workers, i)
                # serialize only the slice we need
                chunk = [mv[j] for j in range(s, e)]
                futs.append(ex.submit(worker_pickled, chunk, s, iters))
            # stitch results back (extra serialization on result)
            off = 0
            for i, f in enumerate(futs):
                s, e = chunk_bounds(n, workers, i)
                out = f.result()
                for k, val in enumerate(out):
                    mv[s + k] = val
    med, _ = perf_ms(one, repeats)
    # Validate once
    arr, mv = mk_uint32_buffer(n)
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = []
        for i in range(workers):
            s, e = chunk_bounds(n, workers, i)
            chunk = [mv[j] for j in range(s, e)]
            futs.append(ex.submit(worker_pickled, chunk, s, iters))
        for i, f in enumerate(futs):
            s, e = chunk_bounds(n, workers, i)
            out = f.result()
            for k, val in enumerate(out):
                mv[s + k] = val
    return med, checksum(mv)

# 4a) Processes with stdlib SharedMemory (portable, no copies)
def worker_shm(name: str, n: int, start: int, end: int, iters: int):
    shm = shared_memory.SharedMemory(name=name)
    mv = memoryview(shm.buf).cast('I')
    try:
        cpu_bound_span_update(mv, start, end, iters)
    finally:
        mv.release()
        shm.close()

def run_mp_shm(n: int, workers: int, iters: int, repeats: int) -> tuple[float, int] | None:
    if not HAVE_SHM:
        return None
    ctx = get_context("spawn")
    def one():
        # create shared block and initialize quickly by blasting bytes from array
        arr, mv_local = mk_uint32_buffer(n)
        shm = shared_memory.SharedMemory(create=True, size=n * 4)
        try:
            buf_mv = memoryview(shm.buf)
            buf_mv[:] = mv_local.tobytes()  # fast init
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = []
                for i in range(workers):
                    s, e = chunk_bounds(n, workers, i)
                    futs.append(ex.submit(worker_shm, shm.name, n, s, e, iters))
                for f in futs:
                    f.result()
        finally:
            if buf_mv: buf_mv.release()
            shm.close()
            shm.unlink()
    med, _ = perf_ms(one, repeats)
    # Validate once
    arr, mv_local = mk_uint32_buffer(n)
    shm = shared_memory.SharedMemory(create=True, size=n * 4)
    try:
        buf_mv = memoryview(shm.buf)
        buf_mv[:] = mv_local.tobytes()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = []
            for i in range(workers):
                s, e = chunk_bounds(n, workers, i)
                futs.append(ex.submit(worker_shm, shm.name, n, s, e, iters))
            for f in futs:
                f.result()
        mv_check = memoryview(shm.buf).cast('I')
        ch = checksum(mv_check)
        mv_check.release()
    finally:
        if buf_mv: buf_mv.release()
        shm.close()
        shm.unlink()
    return med, ch

# 4b) Linux memfd + mmap + pass_fds (no copies, FD-backed). Implemented with subprocess for explicit FD passing.
def memfd_supported() -> bool:
    return hasattr(os, "memfd_create") and os.name == "posix" and sys.platform.startswith("linux")

def run_mp_memfd(n: int, workers: int, iters: int, repeats: int, script_path: str) -> tuple[float, int] | None:
    if not memfd_supported():
        return None

    def one():
        # allocate memfd-backed buffer and initialize
        fd = os.memfd_create("shared-bench", flags=0)  # not CLOEXEC; we'll pass explicitly
        size = n * 4
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        try:
            arr, mv_local = mk_uint32_buffer(n)
            mm[:] = mv_local.tobytes()
            # spawn N children with explicit pass_fds
            procs = []
            env_base = os.environ.copy()
            env_base["SHARED_FD"] = str(fd)
            env_base["SHARED_SIZE"] = str(size)
            env_base["N"] = str(n)
            env_base["ITERS"] = str(iters)
            for i in range(workers):
                s, e = chunk_bounds(n, workers, i)
                env = env_base.copy()
                env["START"] = str(s)
                env["END"] = str(e)
                procs.append(
                    # invoke this same script in "memfd-child" mode
                    # NOTE: pass_fds keeps 'fd' open in the child even if close_fds=True
                    # (POSIX-only)
                    __import__("subprocess").Popen(
                        [sys.executable, script_path, "--memfd-child"],
                        pass_fds=(fd,),
                        env=env,
                        stdout=None, stderr=None, close_fds=True
                    )
                )
            for p in procs:
                rc = p.wait()
                if rc != 0:
                    raise RuntimeError(f"memfd child exited with {rc}")
        finally:
            mm.close()
            os.close(fd)

    med, _ = perf_ms(one, repeats)

    # Validate once
    fd = os.memfd_create("shared-bench", flags=0)
    size = n * 4
    os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    try:
        arr, mv_local = mk_uint32_buffer(n)
        mm[:] = mv_local.tobytes()
        procs = []
        env_base = os.environ.copy()
        env_base["SHARED_FD"] = str(fd)
        env_base["SHARED_SIZE"] = str(size)
        env_base["N"] = str(n)
        env_base["ITERS"] = str(iters)
        for i in range(workers):
            s, e = chunk_bounds(n, workers, i)
            env = env_base.copy()
            env["START"] = str(s)
            env["END"] = str(e)
            procs.append(
                __import__("subprocess").Popen(
                    [sys.executable, script_path, "--memfd-child"],
                    pass_fds=(fd,),
                    env=env,
                    stdout=None, stderr=None, close_fds=True
                )
            )
        for p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"memfd child exited with {rc}")
        mv_check = memoryview(mm).cast('I')
        ch = checksum(mv_check)
        mv_check.release()
    finally:
        mm.close()
        os.close(fd)

    return med, ch


# ----------------------------
# Child entrypoint for memfd workers (launched via subprocess)
# ----------------------------

def memfd_child_entry():
    fd = int(os.environ["SHARED_FD"])
    size = int(os.environ["SHARED_SIZE"])
    n = int(os.environ["N"])
    iters = int(os.environ["ITERS"])
    start = int(os.environ["START"])
    end = int(os.environ["END"])
    mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    try:
        mv = memoryview(mm).cast('I')
        cpu_bound_span_update(mv, start, end, iters)
        mv.release()
    finally:
        mm.close()
        os.close(fd)


# ----------------------------
# Harness
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Compare concurrency/parallelism modes on shared data.")
    p.add_argument("--size", type=int, default=2_000_000, help="Number of uint32 elements (default: 2,000,000 ~= 7.6 MB)")
    p.add_argument("--iters", type=int, default=75, help="Inner iterations per element (default: 75)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)), help="Degree of parallelism (default: cpu_count())")
    p.add_argument("--repeats", type=int, default=3, help="Timing repeats; median is reported (default: 3)")
    p.add_argument("--skip-pickled", action="store_true", help="Skip the 'mp_pickled' test")
    p.add_argument("--skip-shm", action="store_true", help="Skip the 'mp_shared_memory' test")
    p.add_argument("--skip-memfd", action="store_true", help="Skip the Linux memfd test")
    p.add_argument("--memfd-child", action="store_true", help=argparse.SUPPRESS)

    args = p.parse_args()

    if args.memfd_child:
        memfd_child_entry()
        return

    # Banner
    print("=" * 88)
    print("Parallel modes on shared data: Sequential vs Threads (free-threaded) vs Processes")
    print("=" * 88)
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Free-thread build supported: {is_free_thread_build()}")
    ge = gil_enabled_runtime()
    if ge is None:
        print("Runtime GIL status: n/a on this Python (use 3.13+) for sys._is_gil_enabled()")
    else:
        print(f"Runtime GIL enabled   : {ge}  (Tip: run with 'PYTHON_GIL=0' or 'python -X gil=0')")

    total_bytes = args.size * 4
    print(f"\nDataset: {args.size:,} uint32 elements  (~{human_bytes(total_bytes)})")
    print(f"Workers: {args.workers}    Iters/elem: {args.iters}    Repeats: {args.repeats}")

    # GC off during timings to reduce noise
    gc_was_enabled = gc.isenabled()
    gc.disable()

    results = []

    # 1) Sequential
    print("\n[1/5] Sequential baseline ...")
    t_seq, ch_seq = run_sequential(args.size, args.workers, args.iters, args.repeats)
    print(f"  time = {t_seq:.3f}s    checksum = {ch_seq}")
    results.append(("Sequential", t_seq, 1.0, 100.0, ch_seq))

    # 2) Threads (shared memory)
    print("\n[2/5] Threads on shared memory ...")
    t_thr, ch_thr = run_threads(args.size, args.workers, args.iters, args.repeats)
    sp_thr = t_seq / t_thr if t_thr > 0 else float('inf')
    eff_thr = (sp_thr / max(1, args.workers)) * 100.0
    print(f"  time = {t_thr:.3f}s   speedup = {sp_thr:.2f}x   efficiency = {eff_thr:.1f}%   checksum = {ch_thr}")
    results.append(("Threads", t_thr, sp_thr, eff_thr, ch_thr))

    # 3) Processes (pickled chunks)
    if not args.skip_pickled:
        print("\n[3/5] Processes with pickled chunks (copies there & back) ...")
        t_mp_pick, ch_mp_pick = run_mp_pickled(args.size, args.workers, args.iters, args.repeats)
        sp_pick = t_seq / t_mp_pick if t_mp_pick > 0 else float('inf')
        eff_pick = (sp_pick / max(1, args.workers)) * 100.0
        print(f"  time = {t_mp_pick:.3f}s   speedup = {sp_pick:.2f}x   efficiency = {eff_pick:.1f}%   checksum = {ch_mp_pick}")
        results.append(("MP (pickled)", t_mp_pick, sp_pick, eff_pick, ch_mp_pick))

    # 4a) Processes with stdlib SharedMemory
    if not args.skip_shm and HAVE_SHM:
        print("\n[4/5] Processes with stdlib SharedMemory (no copies) ...")
        r = run_mp_shm(args.size, args.workers, args.iters, args.repeats)
        if r is None:
            print("  SharedMemory not available; skipping.")
        else:
            t_shm, ch_shm = r
            sp_shm = t_seq / t_shm if t_shm > 0 else float('inf')
            eff_shm = (sp_shm / max(1, args.workers)) * 100.0
            print(f"  time = {t_shm:.3f}s   speedup = {sp_shm:.2f}x   efficiency = {eff_shm:.1f}%   checksum = {ch_shm}")
            results.append(("MP (SharedMemory)", t_shm, sp_shm, eff_shm, ch_shm))
    elif not HAVE_SHM:
        print("\n[4/5] Processes with stdlib SharedMemory ... not available on this Python; skipping.")

    # 4b) Processes with Linux memfd + mmap + pass_fds
    if not args.skip_memfd:
        print("\n[5/5] Processes with Linux memfd + mmap + pass_fds (no copies) ...")
        r = run_mp_memfd(args.size, args.workers, args.iters, args.repeats, os.path.abspath(sys.argv[0]))
        if r is None:
            print("  memfd not supported on this platform; skipping.")
        else:
            t_mfd, ch_mfd = r
            sp_mfd = t_seq / t_mfd if t_mfd > 0 else float('inf')
            eff_mfd = (sp_mfd / max(1, args.workers)) * 100.0
            print(f"  time = {t_mfd:.3f}s   speedup = {sp_mfd:.2f}x   efficiency = {eff_mfd:.1f}%   checksum = {ch_mfd}")
            results.append(("MP (memfd + mmap)", t_mfd, sp_mfd, eff_mfd, ch_mfd))

    # Restore GC
    if gc_was_enabled:
        gc.enable()

    # Summary
    print("\n" + "=" * 88)
    print(f"{'Approach':<28} {'Time (s)':>10}   {'Speedup':>10}   {'Eff@N':>8}   {'Checksum':>10}")
    print("-" * 88)
    for name, t, sp, eff, ch in results:
        print(f"{name:<28} {t:>10.3f}   {sp:>10.2f}   {eff:>8.1f}%   {ch:>10}")
    print("=" * 88)

    # Quick guidance
    print("\nNotes:")
    print("  • For true multi-core threads, use a free-threaded build and run with -X gil=0 or PYTHON_GIL=0.")
    print("  • If you must use processes and pass big data, prefer SharedMemory or memfd over pickled chunks.")
    print("  • Checksums should match across modes for the same size/iters/workers (sanity check).")


if __name__ == "__main__":
    main()
