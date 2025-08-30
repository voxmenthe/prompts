# dspy.COPRO

## dspy.COPRO

```python
class COPRO(prompt_model=None, metric=None, breadth=10, depth=3, init_temperature=1.4, track_stats=False, **_kwargs)
```

### compile

```python
def compile(self, student, *, trainset, eval_kwargs)
```

optimizes `signature` of `student` program - note that it may be zero-shot or already pre-optimized (demos already chosen - `demos != []`)

parameters:
student: program to optimize and left modified.
trainset: iterable of `Example`s
eval_kwargs: optional, dict
   Additional keywords to go into `Evaluate` for the metric.

Returns optimized version of `student`.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/teleprompt/copro_optimizer.py` (lines 59–357)

