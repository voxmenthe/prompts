# dspy.Prediction

## dspy.Prediction

```python
class Prediction(*args, **kwargs)
```

A prediction object that contains the output of a DSPy module.

Prediction inherits from Example.

To allow feedback-augmented scores, Prediction supports comparison operations
(<, >, <=, >=) for Predictions with a `score` field. The comparison operations
compare the 'score' values as floats. For equality comparison, Predictions are equal
if their underlying data stores are equal (inherited from Example).

Arithmetic operations (+, /, etc.) are also supported for Predictions with a 'score'
field, operating on the score value.


### from_completions

```python
def from_completions(cls, list_or_dict, signature=None)
```

### get_lm_usage

```python
def get_lm_usage(self)
```

### set_lm_usage

```python
def set_lm_usage(self, value)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/primitives/prediction.py` (lines 4–116)

