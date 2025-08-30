# dspy.Predict

## dspy.Predict

```python
class Predict(signature, callbacks=None, **config)
```

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
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/predict/predict.py` (lines 19–202)

