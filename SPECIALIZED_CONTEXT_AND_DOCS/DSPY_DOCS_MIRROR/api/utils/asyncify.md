# dspy.asyncify

## dspy.asyncify

```python
def asyncify(program)
```

Wraps a DSPy program so that it can be called asynchronously. This is useful for running a
program in parallel with another task (e.g., another DSPy program).

This implementation propagates the current thread's configuration context to the worker thread.

Args:
    program: The DSPy program to be wrapped for asynchronous execution.

Returns:
    An async function: An async function that, when awaited, runs the program in a worker thread.
        The current thread's configuration context is inherited for each call.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/utils/asyncify.py` (lines 30–65)

