# Avoiding Race Conditions on Python 3.14t Free-Threaded Builds

The `macos_python_free_thread_test.py` benchmark shows how much parallel throughput you gain once the Global Interpreter Lock (GIL) is disabled. That speed-up is only useful if threads stop trampling on shared state. This guide collects the high‑leverage practices we use in production to keep the blast radius small and make future changes safe on Python 3.14t.

---

## 1. Confirm You Are Actually Running Free Threading

Free-threaded Python remains an opt-in build as of October 12, 2025. Always inspect the runtime before you rely on true parallel execution:

```python
import sys
import sysconfig

print("build supports free threading:", sysconfig.get_config_var("Py_GIL_DISABLED") == 1)
print("GIL enabled at runtime:", sys._is_gil_enabled())
```

- Start your process with `PYTHON_GIL=0` or `python -X gil=0 …` to guarantee the GIL stays disabled.
- Watch for native extensions that silently re-enable the GIL when they do not advertise free-threaded support. If `sys._is_gil_enabled()` suddenly reports `True`, log it and fall back to a single-threaded code path until the dependency is patched.

---

## 2. Identify New Race Surfaces Created by Removing the GIL

Code that behaved “accidentally thread-safe” thanks to the GIL is now vulnerable. Classic examples include monetary transfers, dosage accumulators, and other read-modify-write sequences. Under 3.14t two threads can interleave these updates unless you add explicit coordination.

```python
# Unsafe once the GIL is gone:
def make_payment(buyer, seller, amount):
    buyer.account_balance -= amount
    seller.account_balance += amount
```

Instrument your system to list *every* mutable object that is shared across threads. Your target is either “immutable data crossing threads” or “a single choke point that owns mutation.”

---

## 3. Design Data-First, Not Lock-First

Before grabbing locks, ask whether the state needs to be shared at all. Prefer these transformations:

- **Immutable snapshots:** Broadcast frozen dataclasses, tuples, or `MappingProxyType` objects instead of live dictionaries.
- **Message passing:** Replace shared mutations with `queue.SimpleQueue` or `concurrent.futures.ThreadPoolExecutor.submit()` so each worker owns its portion of the state.
- **Sharded state:** If mutation is required, partition by key (e.g., user ID) and let each shard operate independently to minimize contention.

Remember that core containers (`dict`, `list`, `set`) now use internal locks in free-threaded CPython, but the documentation still treats their behaviour under concurrent mutation as an implementation detail. Treat them as unsafe without external coordination.

---

## 4. Synchronize With the Right Primitive

| Scenario | Prefer this primitive | Why it works on 3.14t |
| --- | --- | --- |
| Short critical section guarding plain Python objects | `threading.Lock()` | Small, uncontended lock keeps mutation atomic. |
| Re-entrant API layers (callbacks invoking callbacks) | `threading.RLock()` | Avoids self-deadlock when the same thread re-enters the protected region. |
| Coordinating writers and readers where one condition must hold | `threading.Condition()` with a pure predicate | Keeps waiters sleeping without spinning; isolate predicate editing to one function. |
| One-shot cross-thread signals (shutdown, refresh) | `threading.Event()` | Simple, idempotent flip with no shared payload. |
| Load-balancing CPU-bound work | `concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())` | Threads finally run in parallel, but keep tasks pure and side-effect free. |
| Producer/consumer pipelines | `queue.SimpleQueue()` or `queue.Queue()` | Implemented with internal locks that remain safe under free threading. |
| Coordinated startup/shutdown across N workers | `threading.Barrier(parties=N)` | Ensures all participants reach a checkpoint before continuing. |

Include a short docstring or comment near each primitive explaining what invariant it protects to make future refactors safer.

---

## 5. Use Atomic or Lock-Free Helpers When Contention Is High

When a metric counter or shared flag becomes hotly contested, switch to atomic constructs to avoid bouncing through the kernel on every increment:

```python
import atomics

request_count = atomics.atomic(width=8, atype=atomics.INT)

def record_request():
    request_count.inc()  # Lock-free increment
```

Third-party packages such as `atomics`, `atomicx`, or `pyatomix` publish wheels that target both standard and free-threaded interpreters. Keep them at the edges (metrics, feature flags) so the rest of your core stays lock-free and easy to test.

---

## 6. Respect Context Propagation Defaults

Free-threaded interpreters flip two runtime flags to make cross-thread coordination safer by default:

- `sys.flags.thread_inherit_context` defaults to `True`, so child threads start with a copy of the parent’s `contextvars.Context`. Override the `context` parameter when constructing a `threading.Thread` if you need isolation.
- `sys.flags.context_aware_warnings` defaults to `True`, which makes `warnings.catch_warnings()` use thread-local state. Ensure you leave this flag on (the default) so parallel workers do not stomp on shared warning filters.

---

## 7. Build Idempotent, Bounded Handlers

Treat every threaded callback like an idempotent event handler:

1. Validate inputs synchronously before hopping onto a worker.
2. Use pure functions to prepare outputs.
3. Commit side effects at the boundary, wrapped in a retry or compensation plan that can be rerun safely.

Pair each handler with a “rollback or resume” strategy so a crash in one thread does not corrupt shared structures.

---

## 8. Test Like You Expect Races

Thread bugs hide unless you stress them:

- Add a `pytest` stress test that runs the critical section thousands of times with `ThreadPoolExecutor` and random sleeps.
- Run the suite once with `PYTHON_GIL=1` (serial execution) and once with `PYTHON_GIL=0` to compare outcomes.
- Capture `threading.excepthook` output and send it to your structured logger so background thread crashes become visible.
- When a defect is hard to reproduce, lean on vendor tooling such as the `ft_utils` compatibility checks that accompany free-threaded utility suites.

Every bug fix should carry a regression test that fails without the fix and passes with it.

---

## 9. Deployment Checklist

- [ ] Verify `sysconfig.get_config_var("Py_GIL_DISABLED") == 1` and `sys._is_gil_enabled() is False` in startup diagnostics.
- [ ] Audit shared mutable state; convert to immutable snapshots, queues, or shard ownership.
- [ ] Wrap unavoidable mutation behind a small number of locks with clear invariants.
- [ ] Replace hot counters or flags with atomic helpers when lock contention shows up in profiling.
- [ ] Keep `thread_inherit_context` and `context_aware_warnings` enabled unless you have a proven reason to override them.
- [ ] Run targeted stress tests under both GIL-disabled and GIL-enabled modes before shipping.
- [ ] Document which modules re-enable the GIL so downstream teams know the blast radius.

Investing in these guardrails means free-threaded Python becomes a lever for throughput instead of a source of heisenbugs.
