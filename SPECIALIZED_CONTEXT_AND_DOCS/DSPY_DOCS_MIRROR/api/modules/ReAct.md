# dspy.ReAct

## dspy.ReAct

```python
class ReAct(signature, tools, max_iters=10)
```

### aforward

```python
async def aforward(self, **input_args)
```

### forward

```python
def forward(self, **input_args)
```

### truncate_trajectory

```python
def truncate_trajectory(self, trajectory)
```

Truncates the trajectory so that it fits in the context window.

Users can override this method to implement their own truncation logic.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/predict/react.py` (lines 17–184)

