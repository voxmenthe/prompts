# dspy.evaluate.answer_passage_match

## dspy.evaluate.answer_passage_match

```python
def answer_passage_match(example, pred, trace=None)
```

Return True if any passage in `pred.context` contains the answer(s).

Strings are normalized (and passages also use DPR normalization internally).

Args:
    example: `dspy.Example` object with field `answer` (str or list[str]).
    pred: `dspy.Prediction` object with field `context` (list[str]) containing passages.
    trace: Unused; reserved for compatibility.

Returns:
    bool: True if any passage contains any reference answer; otherwise False.

Example:
    ```python
    import dspy

    example = dspy.Example(answer="Eiffel Tower")
    pred = dspy.Prediction(context=["The Eiffel Tower is in Paris.", "..."])

    answer_passage_match(example, pred)  # True
    ```

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/evaluate/metrics.py` (lines 320–348)

