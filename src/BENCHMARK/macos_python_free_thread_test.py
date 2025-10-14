#!/usr/bin/env python3
"""
Parallel on macOS: Sequential vs Threads (free-threaded) vs Processes
- Processes (pickled): copies there & back
- Processes (SharedMemory): zero-copy, portable
- Processes (FD-backed mmap over unlinked tempfile): zero-copy, macOS-safe analogue to Linux memfd

Notes
-----
• os.memfd_create is Linux-only; on macOS use SharedMemory or an unlinked tempfile + mmap + pass_fds.
• Unlink semantics: removing the directory entry does NOT reclaim storage until the last open FD is closed.
• On macOS, ProcessPoolExecutor uses 'spawn' by default; we also request it explicitly.

Run examples
------------
python -X gil=0 compare_parallel_modes_macos.py
python -X gil=0 compare_parallel_modes_macos.py --size 10_000_000 --iters 75
"""

from __future__ import annotations
import argparse
import gc
import mmap
import os
import sys
import sysconfig
import tempfile
import time
from array import array
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import get_context
from statistics import median

try:
    from multiprocessing import shared_memory
    HAVE_SHM = True
except Exception:
    shared_memory = None
    HAVE_SHM = False


# ----------------------------
# Helpers
# ----------------------------

def is_free_thread_build() -> bool:
    try:
        return sysconfig.get_config_var("Py_GIL_DISABLED") == 1
    except Exception:
        return False

def gil_enabled_runtime():
    f = getattr(sys, "_is_gil_enabled", None)
    return None if f is None else bool(f())

def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.1f} {u}"
        x /= 1024.0

def chunk_bounds(n: int, parts: int, i: int) -> tuple[int,int]:
    base = n // parts
    extra = n % parts
    s = i*base + min(i, extra)
    e = s + base + (1 if i < extra else 0)
    return s, e

def perf_ms(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0)/1e9)
    return median(times), times

def checksum(mv_uint32) -> int:
    sample = min(1_000_000, len(mv_uint32))
    s = 0
    for i in range(sample):
        s = (s + mv_uint32[i]) & 0xFFFFFFFF
    return s

def mk_uint32_buffer(n: int):
    arr = array('I', range(n))
    return arr, memoryview(arr)  # format 'I'


# ----------------------------
# Kernel (pure Python arithmetic)
# ----------------------------

def cpu_bound_span_update(mv_uint32, start: int, end: int, iters: int) -> None:
    for i in range(start, end):
        v = mv_uint32[i]
        acc = v ^ 0x9E3779B9
        for j in range(iters):
            acc = (acc + ((j + 1) * 2654435761)) & 0xFFFFFFFF
            acc ^= ((acc << 13) | (acc >> 19)) & 0xFFFFFFFF
            acc = (acc * 2246822519) & 0xFFFFFFFF
        mv_uint32[i] = acc


# ----------------------------
# Modes
# ----------------------------

def run_sequential(n: int, workers: int, iters: int, repeats: int):
    def one():
        _arr, mv = mk_uint32_buffer(n)
        for i in range(workers):
            s, e = chunk_bounds(n, workers, i)
            cpu_bound_span_update(mv, s, e, iters)
    med, _ = perf_ms(one, repeats)

    arr, mv = mk_uint32_buffer(n)
    for i in range(workers):
        s, e = chunk_bounds(n, workers, i)
        cpu_bound_span_update(mv, s, e, iters)
    return med, checksum(mv)

def run_threads(n: int, workers: int, iters: int, repeats: int):
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

    arr, mv = mk_uint32_buffer(n)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i in range(workers):
            s, e = chunk_bounds(n, workers, i)
            futs.append(ex.submit(cpu_bound_span_update, mv, s, e, iters))
        for f in futs: f.result()
    return med, checksum(mv)

# --- Processes with pickled chunks (copies) ---
def _worker_pickled(chunk: list[int], iters: int) -> list[int]:
    out = chunk[:]
    for k in range(len(out)):
        v = out[k]
        acc = v ^ 0x9E3779B9
        for j in range(iters):
            acc = (acc + ((j + 1) * 2654435761)) & 0xFFFFFFFF
            acc ^= ((acc << 13) | (acc >> 19)) & 0xFFFFFFFF
            acc = (acc * 2246822519) & 0xFFFFFFFF
        out[k] = acc
    return out

def run_mp_pickled(n: int, workers: int, iters: int, repeats: int):
    ctx = get_context("spawn")
    def one():
        arr, mv = mk_uint32_buffer(n)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = []
            for i in range(workers):
                s, e = chunk_bounds(n, workers, i)
                futs.append(ex.submit(_worker_pickled, [mv[j] for j in range(s,e)], iters))
            for i, f in enumerate(futs):
                s, e = chunk_bounds(n, workers, i)
                out = f.result()
                for k, val in enumerate(out):
                    mv[s + k] = val
    med, _ = perf_ms(one, repeats)

    arr, mv = mk_uint32_buffer(n)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = []
        for i in range(workers):
            s, e = chunk_bounds(n, workers, i)
            futs.append(ex.submit(_worker_pickled, [mv[j] for j in range(s,e)], iters))
        for i, f in enumerate(futs):
            s, e = chunk_bounds(n, workers, i)
            out = f.result()
            for k, val in enumerate(out):
                mv[s + k] = val
    return med, checksum(mv)

# --- Processes with stdlib SharedMemory (portable, no copies) ---
def _worker_shm(name: str, start: int, end: int, iters: int):
    shm = shared_memory.SharedMemory(name=name)
    mv = memoryview(shm.buf).cast('I')
    try:
        cpu_bound_span_update(mv, start, end, iters)
    finally:
        mv.release()
        shm.close()

def run_mp_shm(n: int, workers: int, iters: int, repeats: int):
    if not HAVE_SHM:
        return None
    ctx = get_context("spawn")

    def one():
        arr, mv_local = mk_uint32_buffer(n)              # mv_local: format='I'
        shm = shared_memory.SharedMemory(create=True, size=n * 4)
        try:
            # Make the destination view the same structure as mv_local (uint32)
            count = len(mv_local)
            dest = shm.buf.cast('I')
            try:
                dest[:count] = mv_local                  # zero-copy view->view assignment (no tobytes())
            finally:
                dest.release()
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = []
                for i in range(workers):
                    s, e = chunk_bounds(n, workers, i)
                    futs.append(ex.submit(_worker_shm, shm.name, s, e, iters))
                for f in futs:
                    f.result()
        finally:
            shm.close()
            shm.unlink()

    med, _ = perf_ms(one, repeats)

    # Validate once and compute checksum
    arr, mv_local = mk_uint32_buffer(n)
    shm = shared_memory.SharedMemory(create=True, size=n * 4)
    try:
        count = len(mv_local)
        dest = shm.buf.cast('I')
        try:
            dest[:count] = mv_local
        finally:
            dest.release()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = []
            for i in range(workers):
                s, e = chunk_bounds(n, workers, i)
                futs.append(ex.submit(_worker_shm, shm.name, s, e, iters))
            for f in futs:
                f.result()
        mv_full = shm.buf.cast('I')
        try:
            mv_check = mv_full[:count]
            try:
                ch = checksum(mv_check)
            finally:
                mv_check.release()
        finally:
            mv_full.release()
    finally:
        shm.close()
        shm.unlink()
    return med, ch

# --- Processes with FD-backed mmap via unlinked tempfile (no copies, macOS-friendly) ---
def fd_child_entry():
    fd = int(os.environ["SHARED_FD"])
    size = int(os.environ["SHARED_SIZE"])
    start = int(os.environ["START"])
    end = int(os.environ["END"])
    iters = int(os.environ["ITERS"])

    mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    try:
        mv = memoryview(mm).cast('I')
        cpu_bound_span_update(mv, start, end, iters)
        mv.release()
    finally:
        mm.close()
        os.close(fd)

def run_mp_fd_unlinked(n: int, workers: int, iters: int, repeats: int, script_path: str):
    def one():
        # 1) Create a temp file, size it, and unlink the path (lifetime bound to open FDs)
        fd, path = tempfile.mkstemp(prefix="py-shared-")
        try:
            size = n * 4
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            try:
                arr, mv_local = mk_uint32_buffer(n)
                mm[:] = mv_local.tobytes()
                # remove directory entry; file persists until FDs are closed
                os.unlink(path)

                # 2) Spawn children and pass the FD explicitly
                procs = []
                base_env = os.environ.copy()
                base_env.update({"SHARED_FD": str(fd), "SHARED_SIZE": str(size), "ITERS": str(iters)})
                for i in range(workers):
                    s, e = chunk_bounds(n, workers, i)
                    env = base_env.copy(); env.update({"START": str(s), "END": str(e)})
                    procs.append(
                        __import__("subprocess").Popen(
                            [sys.executable, script_path, "--fd-child"],
                            pass_fds=(fd,), env=env, stdout=None, stderr=None, close_fds=True
                        )
                    )
                for p in procs:
                    rc = p.wait()
                    if rc != 0:
                        raise RuntimeError(f"fd-child exited {rc}")
            finally:
                mm.close()
        finally:
            os.close(fd)  # after last close, kernel reclaims storage

    med, _ = perf_ms(one, repeats)

    # Validate one run and compute checksum
    fd, path = tempfile.mkstemp(prefix="py-shared-")
    try:
        size = n * 4
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        try:
            arr, mv_local = mk_uint32_buffer(n)
            mm[:] = mv_local.tobytes()
            os.unlink(path)
            procs = []
            base_env = os.environ.copy()
            base_env.update({"SHARED_FD": str(fd), "SHARED_SIZE": str(size), "ITERS": str(iters)})
            for i in range(workers):
                s, e = chunk_bounds(n, workers, i)
                env = base_env.copy(); env.update({"START": str(s), "END": str(e)})
                procs.append(
                    __import__("subprocess").Popen(
                        [sys.executable, script_path, "--fd-child"],
                        pass_fds=(fd,), env=env, stdout=None, stderr=None, close_fds=True
                    )
                )
            for p in procs:
                if p.wait() != 0:
                    raise RuntimeError("fd-child failed")
            mv_chk = memoryview(mm).cast('I'); ch = checksum(mv_chk); mv_chk.release()
        finally:
            mm.close()
    finally:
        os.close(fd)
    return med, ch


# ----------------------------
# Harness
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="macOS-friendly parallel modes on shared data.")
    ap.add_argument("--size", type=int, default=2_000_000, help="number of uint32 elements (default 2,000,000)")
    ap.add_argument("--iters", type=int, default=75, help="inner iterations per element (default 75)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4), help="degree of parallelism (default cpu_count())")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats; median reported")
    ap.add_argument("--skip-pickled", action="store_true")
    ap.add_argument("--skip-shm", action="store_true")
    ap.add_argument("--skip-fd", action="store_true")
    ap.add_argument("--fd-child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.fd_child:
        fd_child_entry()
        return

    print("=" * 86)
    print("Parallel on macOS: Sequential vs Threads (free-threaded) vs Process strategies")
    print("=" * 86)
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Free-thread build supported: {is_free_thread_build()}")
    ge = gil_enabled_runtime()
    print(f"GIL enabled at runtime   : {ge} (tip: run with -X gil=0 or PYTHON_GIL=0)")
    print(f"Dataset: {args.size:,} uint32  (~{human_bytes(args.size*4)})  Workers: {args.workers}  Iters: {args.iters}")

    gc_was = gc.isenabled(); gc.disable()
    results = []

    t_seq, ch_seq = run_sequential(args.size, args.workers, args.iters, args.repeats)
    print(f"\n[1/5] Sequential          time={t_seq:.3f}s  checksum={ch_seq}")
    results.append(("Sequential", t_seq, 1.0, 100.0, ch_seq))

    t_thr, ch_thr = run_threads(args.size, args.workers, args.iters, args.repeats)
    sp = t_seq / t_thr if t_thr > 0 else float('inf'); eff = (sp / max(1,args.workers))*100
    print(f"[2/5] Threads (shared)    time={t_thr:.3f}s  speedup={sp:.2f}x  eff@N={eff:.1f}%  checksum={ch_thr}")
    results.append(("Threads", t_thr, sp, eff, ch_thr))

    if not args.skip_pickled:
        t_mp, ch_mp = run_mp_pickled(args.size, args.workers, args.iters, args.repeats)
        sp = t_seq / t_mp if t_mp > 0 else float('inf'); eff = (sp / max(1,args.workers))*100
        print(f"[3/5] MP (pickled)        time={t_mp:.3f}s  speedup={sp:.2f}x  eff@N={eff:.1f}%  checksum={ch_mp}")
        results.append(("MP (pickled)", t_mp, sp, eff, ch_mp))

    if not args.skip_shm and HAVE_SHM:
        t_shm, ch_shm = run_mp_shm(args.size, args.workers, args.iters, args.repeats)  # type: ignore
        sp = t_seq / t_shm if t_shm > 0 else float('inf'); eff = (sp / max(1,args.workers))*100
        print(f"[4/5] MP (SharedMemory)   time={t_shm:.3f}s  speedup={sp:.2f}x  eff@N={eff:.1f}%  checksum={ch_shm}")
        results.append(("MP (SharedMemory)", t_shm, sp, eff, ch_shm))
    elif not HAVE_SHM:
        print("[4/5] MP (SharedMemory)   not available on this Python")

    if not args.skip_fd:
        t_fd, ch_fd = run_mp_fd_unlinked(args.size, args.workers, args.iters, args.repeats, os.path.abspath(sys.argv[0]))
        sp = t_seq / t_fd if t_fd > 0 else float('inf'); eff = (sp / max(1,args.workers))*100
        print(f"[5/5] MP (FD + mmap)      time={t_fd:.3f}s  speedup={sp:.2f}x  eff@N={eff:.1f}%  checksum={ch_fd}")
        results.append(("MP (FD + mmap)", t_fd, sp, eff, ch_fd))

    if gc_was: gc.enable()

    print("\n" + "-"*86)
    print(f"{'Approach':<26} {'Time(s)':>10}   {'Speedup':>9}   {'Eff@N':>8}   {'Checksum':>10}")
    print("-"*86)
    for name, t, spd, eff, ch in results:
        print(f"{name:<26} {t:>10.3f}   {spd:>9.2f}   {eff:>8.1f}%   {ch:>10}")
    print("-"*86)
    print("Notes: Threads only scale on a free-threaded build (3.14+) when run with -X gil=0 / PYTHON_GIL=0.")
    print("       On macOS, prefer SharedMemory for portability; FD+mmap via unlinked tempfile mirrors memfd semantics.")
    print("="*86)

if __name__ == "__main__":
    main()
