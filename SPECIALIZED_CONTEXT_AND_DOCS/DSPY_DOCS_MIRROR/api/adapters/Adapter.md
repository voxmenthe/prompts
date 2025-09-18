# dspy.Adapter

## dspy.Adapter

```python
class Adapter(callbacks=None, use_native_function_calling=False)
```

### __call__

```python
def __call__(self, lm, lm_kwargs, signature, demos, inputs)
```

### acall

```python
async def acall(self, lm, lm_kwargs, signature, demos, inputs)
```

### format

```python
def format(self, signature, demos, inputs)
```

Format the input messages for the LM call.

This method converts the DSPy structured input along with few-shot examples and conversation history into
multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize
the formatting of the input messages.

In general we recommend the messages to have the following structure:
```
[
    {"role": "system", "content": system_message},
    # Begin few-shot examples
    {"role": "user", "content": few_shot_example_1_input},
    {"role": "assistant", "content": few_shot_example_1_output},
    {"role": "user", "content": few_shot_example_2_input},
    {"role": "assistant", "content": few_shot_example_2_output},
    ...
    # End few-shot examples
    # Begin conversation history
    {"role": "user", "content": conversation_history_1_input},
    {"role": "assistant", "content": conversation_history_1_output},
    {"role": "user", "content": conversation_history_2_input},
    {"role": "assistant", "content": conversation_history_2_output},
    ...
    # End conversation history
    {"role": "user", "content": current_input},
]

And system message should contain the field description, field structure, and task description.
```


Args:
    signature: The DSPy signature for which to format the input messages.
    demos: A list of few-shot examples.
    inputs: The input arguments to the DSPy module.

Returns:
    A list of multiturn messages as expected by the LM.


### format_assistant_message_content

```python
def format_assistant_message_content(self, signature, outputs, missing_field_message=None)
```

Format the assistant message content.

This method formats the assistant message content, which can be used in formatting few-shot examples,
conversation history.

Args:
    signature: The DSPy signature for which to format the assistant message content.
    outputs: The output fields to be formatted.
    missing_field_message: A message to be used when a field is missing.

Returns:
    A string that contains the assistant message content.


### format_conversation_history

```python
def format_conversation_history(self, signature, history_field_name, inputs)
```

Format the conversation history.

This method formats the conversation history and the current input as multiturn messages.

Args:
    signature: The DSPy signature for which to format the conversation history.
    history_field_name: The name of the history field in the signature.
    inputs: The input arguments to the DSPy module.

Returns:
    A list of multiturn messages.


### format_demos

```python
def format_demos(self, signature, demos)
```

Format the few-shot examples.

This method formats the few-shot examples as multiturn messages.

Args:
    signature: The DSPy signature for which to format the few-shot examples.
    demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
        the signature.

Returns:
    A list of multiturn messages.


### format_field_description

```python
def format_field_description(self, signature)
```

Format the field description for the system message.

This method formats the field description for the system message. It should return a string that contains
the field description for the input fields and the output fields.

Args:
    signature: The DSPy signature for which to format the field description.

Returns:
    A string that contains the field description for the input fields and the output fields.


### format_field_structure

```python
def format_field_structure(self, signature)
```

Format the field structure for the system message.

This method formats the field structure for the system message. It should return a string that dictates the
format the input fields should be provided to the LM, and the format the output fields will be in the response.
Refer to the ChatAdapter and JsonAdapter for an example.

Args:
    signature: The DSPy signature for which to format the field structure.


### format_task_description

```python
def format_task_description(self, signature)
```

Format the task description for the system message.

This method formats the task description for the system message. In most cases this is just a thin wrapper
over `signature.instructions`.

Args:
    signature: The DSPy signature of the DSpy module.

Returns:
    A string that describes the task.


### format_user_message_content

```python
def format_user_message_content(self, signature, inputs, prefix='', suffix='', main_request=False)
```

Format the user message content.

This method formats the user message content, which can be used in formatting few-shot examples, conversation
history, and the current input.

Args:
    signature: The DSPy signature for which to format the user message content.
    inputs: The input arguments to the DSPy module.
    prefix: A prefix to the user message content.
    suffix: A suffix to the user message content.

Returns:
    A string that contains the user message content.


### parse

```python
def parse(self, signature, completion)
```

Parse the LM output into a dictionary of the output fields.

This method parses the LM output into a dictionary of the output fields.

Args:
    signature: The DSPy signature for which to parse the LM output.
    completion: The LM output to be parsed.

Returns:
    A dictionary of the output fields.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/adapters/base.py` (lines 19–448)

