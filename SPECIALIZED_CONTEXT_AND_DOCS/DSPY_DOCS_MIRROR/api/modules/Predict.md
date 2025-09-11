# dspy.Predict

## dspy.Predict

```python
class Predict(signature, callbacks=None, **config)
```

Basic DSPy module that maps inputs to outputs using a language model.

Args:
    signature: The input/output signature describing the task.
    callbacks: Optional list of callbacks for instrumentation.
    **config: Default keyword arguments forwarded to the underlying
        language model. These values can be overridden for a single
        invocation by passing a ``config`` dictionary when calling the
        module. For example::

            predict = dspy.Predict("q -> a", rollout_id=1, temperature=1.0)
            predict(q="What is 1 + 52?", config={"rollout_id": 2, "temperature": 1.0})


### __call__

```python
def __call__(self, *args, **kwargs)
```

### acall

```python
async def acall(self, *args, **kwargs)
```

### aforward

```python
async def aforward(self, **kwargs)
```

### dump_state

```python
def dump_state(self, json_mode=True)
```

### forward

```python
def forward(self, **kwargs)
```

### get_config

```python
def get_config(self)
```

### load_state

```python
def load_state(self, state)
```

Load the saved state of a `Predict` object.

Args:
    state: The saved state of a `Predict` object.

Returns:
    Self to allow method chaining.


### reset

```python
def reset(self)
```

### update_config

```python
def update_config(self, **kwargs)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/predict/predict.py` (lines 19–216)

