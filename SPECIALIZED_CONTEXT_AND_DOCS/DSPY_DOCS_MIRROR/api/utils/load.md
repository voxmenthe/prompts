# dspy.load

## dspy.load

```python
def load(path, allow_pickle=False)
```

Load saved DSPy model.

This method is used to load a saved DSPy model with `save_program=True`, i.e., the model is saved with cloudpickle.

Args:
    path (str): Path to the saved model.
    allow_pickle (bool): Whether to allow loading the model with pickle. This is dangerous and should only be used if you are sure you trust the source of the model.

Returns:
    The loaded model, a `dspy.Module` instance.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/utils/saving.py` (lines 27–61)

