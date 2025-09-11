# dspy.Evaluate

## dspy.Evaluate

```python
class Evaluate(*, devset, metric=None, num_threads=None, display_progress=False, display_table=False, max_errors=None, provide_traceback=None, failure_score=0.0, save_as_csv=None, save_as_json=None, **kwargs)
```

DSPy Evaluate class.

This class is used to evaluate the performance of a DSPy program. Users need to provide a evaluation dataset and
a metric function in order to use this class. This class supports parallel evaluation on the provided dataset.


### __call__

```python
def __call__(self, program, metric=None, devset=None, num_threads=None, display_progress=None, display_table=None, callback_metadata=None, save_as_csv=None, save_as_json=None)
```

Args:
    program (dspy.Module): The DSPy program to evaluate.
    metric (Callable): The metric function to use for evaluation. if not provided, use `self.metric`.
    devset (list[dspy.Example]): the evaluation dataset. if not provided, use `self.devset`.
    num_threads (Optional[int]): The number of threads to use for parallel evaluation. if not provided, use
        `self.num_threads`.
    display_progress (bool): Whether to display progress during evaluation. if not provided, use
        `self.display_progress`.
    display_table (Union[bool, int]): Whether to display the evaluation results in a table. if not provided, use
        `self.display_table`. If a number is passed, the evaluation results will be truncated to that number before displayed.
    callback_metadata (dict): Metadata to be used for evaluate callback handlers.

Returns:
    The evaluation results are returned as a dspy.EvaluationResult object containing the following attributes:

    - score: A float percentage score (e.g., 67.30) representing overall performance

    - results: a list of (example, prediction, score) tuples for each example in devset

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/evaluate/evaluate.py` (lines 64–298)

