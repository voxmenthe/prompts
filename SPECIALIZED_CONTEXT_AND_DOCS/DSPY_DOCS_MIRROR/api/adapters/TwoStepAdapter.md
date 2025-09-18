# dspy.TwoStepAdapter

## dspy.TwoStepAdapter

```python
class TwoStepAdapter(extraction_model, **kwargs)
```

A two-stage adapter that:
    1. Uses a simpler, more natural prompt for the main LM
    2. Uses a smaller LM with chat adapter to extract structured data from the response of main LM
This adapter uses a common __call__ logic defined in base Adapter class.
This class is particularly useful when interacting with reasoning models as the main LM since reasoning models
are known to struggle with structured outputs.

Example:
```
import dspy
lm = dspy.LM(model="openai/o3-mini", max_tokens=16000, temperature = 1.0)
adapter = dspy.TwoStepAdapter(dspy.LM("openai/gpt-4o-mini"))
dspy.configure(lm=lm, adapter=adapter)
program = dspy.ChainOfThought("question->answer")
result = program("What is the capital of France?")
print(result)
```


### acall

```python
async def acall(self, lm, lm_kwargs, signature, demos, inputs)
```

### format

```python
def format(self, signature, demos, inputs)
```

Format a prompt for the first stage with the main LM.
This no specific structure is required for the main LM, we customize the format method
instead of format_field_description or format_field_structure.

Args:
    signature: The signature of the original task
    demos: A list of demo examples
    inputs: The current input

Returns:
    A list of messages to be passed to the main LM.


### format_assistant_message_content

```python
def format_assistant_message_content(self, signature, outputs, missing_field_message=None)
```

### format_task_description

```python
def format_task_description(self, signature)
```

Create a description of the task based on the signature


### format_user_message_content

```python
def format_user_message_content(self, signature, inputs, prefix='', suffix='')
```

### parse

```python
def parse(self, signature, completion)
```

Use a smaller LM (extraction_model) with chat adapter to extract structured data
from the raw completion text of the main LM.

Args:
    signature: The signature of the original task
    completion: The completion from the main LM

Returns:
    A dictionary containing the extracted structured data.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/adapters/two_step_adapter.py` (lines 21–229)

