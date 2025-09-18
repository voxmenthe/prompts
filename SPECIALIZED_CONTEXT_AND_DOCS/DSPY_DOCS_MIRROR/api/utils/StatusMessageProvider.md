# dspy.streaming.StatusMessageProvider

## dspy.streaming.StatusMessageProvider

```python
class StatusMessageProvider
```

Provides customizable status message streaming for DSPy programs.

This class serves as a base for creating custom status message providers. Users can subclass
and override its methods to define specific status messages for different stages of program execution,
each method must return a string.

Example:
```python
class MyStatusMessageProvider(StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return f"Calling LM with inputs {inputs}..."

    def module_end_status_message(self, outputs):
        return f"Module finished with output: {outputs}!"

program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())
```


### lm_end_status_message

```python
def lm_end_status_message(self, outputs)
```

Status message after a `dspy.LM` is called.


### lm_start_status_message

```python
def lm_start_status_message(self, instance, inputs)
```

Status message before a `dspy.LM` is called.


### module_end_status_message

```python
def module_end_status_message(self, outputs)
```

Status message after a `dspy.Module` or `dspy.Predict` is called.


### module_start_status_message

```python
def module_start_status_message(self, instance, inputs)
```

Status message before a `dspy.Module` or `dspy.Predict` is called.


### tool_end_status_message

```python
def tool_end_status_message(self, outputs)
```

Status message after a `dspy.Tool` is called.


### tool_start_status_message

```python
def tool_start_status_message(self, instance, inputs)
```

Status message before a `dspy.Tool` is called.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/streaming/messages.py` (lines 53–95)

