# dspy.evaluate.answer_exact_match

## dspy.evaluate.answer_exact_match

```python
def answer_exact_match(example, pred, trace=None, frac=1.0)
```

Evaluate exact match or F1-thresholded match for an example/prediction pair.

If `example.answer` is a string, compare `pred.answer` against it. If it's a list,
compare against any of the references. When `frac >= 1.0` (default), use EM;
otherwise require that the maximum F1 across references is at least `frac`.

Args:
    example: `dspy.Example` object with field `answer` (str or list[str]).
    pred: `dspy.Prediction` object with field `answer` (str).
    trace: Unused; reserved for compatibility.
    frac (float, optional): Threshold in [0.0, 1.0]. `1.0` means EM.

Returns:
    bool: True if the match condition holds; otherwise False.

Example:
    ```python
    import dspy

    example = dspy.Example(answer=["Eiffel Tower", "Louvre"])
    pred = dspy.Prediction(answer="The Eiffel Tower")

    answer_exact_match(example, pred, frac=1.0)  # equivalent to EM, True
    answer_exact_match(example, pred, frac=0.5)  # True
    ```

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/evaluate/metrics.py` (lines 285–317)

