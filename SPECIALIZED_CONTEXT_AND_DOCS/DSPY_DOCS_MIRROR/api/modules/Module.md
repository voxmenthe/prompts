# dspy.Module

## dspy.Module

```python
class Module(callbacks=None)
```

### __call__

```python
def __call__(self, *args, **kwargs)
```

### acall

```python
async def acall(self, *args, **kwargs)
```

### batch

```python
def batch(self, examples, num_threads=None, max_errors=None, return_failed_examples=False, provide_traceback=None, disable_progress_bar=False)
```

Processes a list of dspy.Example instances in parallel using the Parallel module.

Args:
    examples: List of dspy.Example instances to process.
    num_threads: Number of threads to use for parallel processing.
    max_errors: Maximum number of errors allowed before stopping execution.
        If ``None``, inherits from ``dspy.settings.max_errors``.
    return_failed_examples: Whether to return failed examples and exceptions.
    provide_traceback: Whether to include traceback information in error logs.
    disable_progress_bar: Whether to display the progress bar.

Returns:
    List of results, and optionally failed examples and exceptions.


### get_lm

```python
def get_lm(self)
```

### inspect_history

```python
def inspect_history(self, n=1)
```

### map_named_predictors

```python
def map_named_predictors(self, func)
```

Applies a function to all named predictors.


### named_predictors

```python
def named_predictors(self)
```

### predictors

```python
def predictors(self)
```

### set_lm

```python
def set_lm(self, lm)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/primitives/module.py` (lines 40–190)

