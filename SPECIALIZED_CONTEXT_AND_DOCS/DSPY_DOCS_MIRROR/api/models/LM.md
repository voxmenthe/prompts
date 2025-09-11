# dspy.LM

## dspy.LM

```python
class LM(model, model_type='chat', temperature=0.0, max_tokens=4000, cache=True, callbacks=None, num_retries=3, provider=None, finetuning_model=None, launch_kwargs=None, train_kwargs=None, use_developer_role=False, **kwargs)
```

A language model supporting chat or text completion requests for use with DSPy modules.


### aforward

```python
async def aforward(self, prompt=None, messages=None, **kwargs)
```

### dump_state

```python
def dump_state(self)
```

### finetune

```python
def finetune(self, train_data, train_data_format, train_kwargs=None)
```

### forward

```python
def forward(self, prompt=None, messages=None, **kwargs)
```

### infer_provider

```python
def infer_provider(self)
```

### kill

```python
def kill(self, launch_kwargs=None)
```

### launch

```python
def launch(self, launch_kwargs=None)
```

### reinforce

```python
def reinforce(self, train_kwargs)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/clients/lm.py` (lines 25–293)

