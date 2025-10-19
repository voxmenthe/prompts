# Free-Threading (No-GIL) Benchmarks

Benchmarks comparing Python 3.14's free-threaded build (no GIL) against traditional GIL-based Python and multiprocessing.

## What is Free-Threading?

Python 3.14 introduces an optional build without the Global Interpreter Lock (GIL), enabling true parallel execution of Python threads on multi-core systems ([PEP 703](https://peps.python.org/pep-0703/), [PEP 779](https://peps.python.org/pep-0779/)).

**GIL-based Python:** Only one thread executes Python code at a time, even on multi-core systems.

**Free-threaded Python:** Multiple threads can execute Python code in parallel.

## Benchmark: `threading_vs_multiprocessing.py`

Compares three approaches for CPU-bound parallel computation with **shared data** (10M elements, ~267 MB):

1. **Sequential** - Single-threaded baseline
2. **Free-threading** - Multiple threads with no GIL, shared memory access
3. **Multiprocessing** - Multiple separate processes, data must be serialized/copied

**Key difference:** Threading accesses shared memory directly, while multiprocessing must pickle and copy data to each process.

### Results

#### Ubuntu Server 22.04 - Python 3.14t (16 cores, 10 workers)
```
Sequential:          40.295s  (baseline)
Free-threading:       8.007s  (5.03x speedup, 50.3% efficiency)
Multiprocessing:     10.647s  (3.78x speedup, 37.8% efficiency)
```

Free-threading is 33% faster. Shared memory access eliminates serialization overhead.

#### Ubuntu Server 22.04 - Python 3.14 with GIL (16 cores, 10 workers)
```
Sequential:              35.955s  (baseline)
Threading (with GIL):    38.301s  (0.94x - no speedup)
Multiprocessing:          7.499s  (4.79x speedup, 47.9% efficiency)
```

#### Ubuntu Server 22.04 - Python 3.13 with GIL (16 cores, 10 workers)
```
Sequential:              46.885s  (baseline)
Threading (with GIL):    48.598s  (0.96x - no speedup)
Multiprocessing:         10.625s  (4.41x speedup, 44.1% efficiency)
```

With the GIL (both Python 3.13 and 3.14), threading provides no speedup for CPU-bound tasks. The performance improvement requires both Python 3.14 AND the free-threaded build (3.14t).

### Key Findings

1. Free-threading is 33% faster than multiprocessing when data sharing is required
2. Multiprocessing must pickle/copy 267 MB per process, threading shares memory directly
3. Free-threading achieves 5x speedup vs 0.96x with GIL-based Python
4. 50% efficiency (5 of 10 cores utilized) is limited by memory bandwidth contention

## When to Use Each Approach

### Free-Threading

Use when:
- Shared memory access across workers is required
- Frequent, small tasks (low startup overhead)
- Low latency requirements (threads start faster than processes)
- Memory constrained (threads share memory)

Example use cases: game engines with shared world state, real-time simulations, in-memory databases/caches, ML model serving with shared parameters.

### Multiprocessing

Use when:
- Pure CPU-bound work with no shared state
- Complete isolation between workers is required
- Very long-running tasks (startup overhead doesn't matter)
- Using Python < 3.13 (no free-threading available)

Example use cases: embarrassingly parallel computations, independent data processing tasks, fault-tolerant systems.

## Setup

### Prerequisites

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.14 free-threaded build
uv python install 3.14t

# Verify installation
uv python list
```

### Running the Benchmark

```bash
# Navigate to the free-threading directory
cd free-threading

# Pin to Python 3.14 free-threaded build
uv python pin 3.14t

# Run the benchmark
uv run python threading_vs_multiprocessing.py
```

### Verify Free-Threading is Enabled

```bash
uv run python -c "import sys; print(f'Free-threaded: {not sys._is_gil_enabled()}')"
# Should print: Free-threaded: True
```

## Understanding the Results

**Speedup:** How much faster parallel execution is compared to sequential (5x = 1/5th the time).

**Efficiency:** How effectively the approach uses available CPU cores (50% efficiency on 10 cores = 5 cores fully utilized).

**Limitations:** Real-world parallel speedup is limited by memory bandwidth (all cores compete for RAM access), cache contention, synchronization overhead, and Python interpreter overhead.

## Comparing with GIL-based Python

Run with traditional Python to compare:

```bash
uv python install 3.13
uv python pin 3.13
uv run python threading_vs_multiprocessing.py
```

With the GIL, threading provides no speedup for CPU-bound tasks (0.96x - slower due to context switching overhead).

## Technical Details

- **Task:** CPU-intensive computation modifying shared data in-place (10M elements, ~267 MB)
- **Workers:** 10 parallel workers
- **Shared data:** Threading accesses by reference, multiprocessing must serialize/copy
- **Timing:** Uses `time.perf_counter()` for high-precision measurements
- **Fairness:** Identical computation executed by all approaches

## The GIL Was Hiding Your Bugs

Demo: `race_condition_demo.py`

Operations like `counter.value += 1` on shared state require explicit locks in multi-threaded code. This isn't atomic - it performs multiple bytecode operations (read, increment, write). When multiple threads execute this without locks, they can interleave these steps, causing race conditions where updates are lost.

The GIL made race conditions rare by serializing bytecode execution. Thread switches typically happened after enough bytecode instructions that short sequences like `+= 1` usually completed atomically. Code that lacked proper synchronization appeared to work.

Free-threading removes this accidental protection. Multiple threads execute simultaneously, and race conditions occur consistently.

```bash
# Run with Python 3.14t (free-threaded)
uv run --python 3.14t python race_condition_demo.py

# Run with Python 3.13 (GIL-based)
uv run --python 3.13 python race_condition_demo.py
```

### Results: 10 threads, 100k increments each

**Python 3.14t (Free-threaded):**
```
Expected: 1,000,000
Run 1: 195,277 (lost: 804,723 = 80.5%)
Run 2: 184,918 (lost: 815,082 = 81.5%)
Run 3: 183,916 (lost: 816,084 = 81.6%)
```

**Python 3.13 (GIL-based):**
```
Expected: 1,000,000
Run 1: 1,000,000 (lost: 0 = 0.0%)
Run 2: 1,000,000 (lost: 0 = 0.0%)
Run 3: 1,000,000 (lost: 0 = 0.0%)
```

Migrating to free-threaded Python requires auditing shared mutable state and adding explicit synchronization (locks, atomics, or thread-safe data structures).

## Learn More

- [PEP 703 - Making the Global Interpreter Lock Optional](https://peps.python.org/pep-0703/)
- [PEP 779 - Python 3.14 Free-threading Support](https://peps.python.org/pep-0779/)
- [Python 3.14 Documentation](https://docs.python.org/3.14/)
- [Free-threading Design Doc](https://docs.python.org/3.14/howto/free-threading-python.html)

## License

MIT - Free to use for testing and analysis.