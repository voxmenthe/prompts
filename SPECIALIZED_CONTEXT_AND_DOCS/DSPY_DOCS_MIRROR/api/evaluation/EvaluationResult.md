# dspy.evaluate.EvaluationResult


## dspy.evaluate.EvaluationResult

```python
class EvaluationResult(score, results)
```

A class that represents the result of an evaluation.
It is a subclass of `dspy.Prediction` that contains the following fields

- score: An float value (e.g., 67.30) representing the overall performance
- results: a list of (example, prediction, score) tuples for each example in devset

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/evaluate/evaluate.py` (lines 48–61)

