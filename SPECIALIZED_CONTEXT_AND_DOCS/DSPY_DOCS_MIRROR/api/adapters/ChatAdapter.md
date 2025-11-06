# dspy.ChatAdapter

## dspy.ChatAdapter

```python
class ChatAdapter(callbacks=None, use_native_function_calling=False, native_response_types=None, use_json_adapter_fallback=True)
```

### __call__

```python
def __call__(self, lm, lm_kwargs, signature, demos, inputs)
```

### acall

```python
async def acall(self, lm, lm_kwargs, signature, demos, inputs)
```

### format_assistant_message_content

```python
def format_assistant_message_content(self, signature, outputs, missing_field_message=None)
```

### format_field_description

```python
def format_field_description(self, signature)
```

### format_field_structure

```python
def format_field_structure(self, signature)
```

`ChatAdapter` requires input and output fields to be in their own sections, with section header using markers
`[[ ## field_name ## ]]`. An arbitrary field `completed` ([[ ## completed ## ]]) is added to the end of the
output fields section to indicate the end of the output fields.


### format_field_with_value

```python
def format_field_with_value(self, fields_with_values)
```

Formats the values of the specified fields according to the field's DSPy type (input or output),
annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
into a single string, which is is a multiline string if there are multiple fields.

Args:
    fields_with_values: A dictionary mapping information about a field to its corresponding
        value.

Returns:
    The joined formatted values of the fields, represented as a string


### format_finetune_data

```python
def format_finetune_data(self, signature, demos, inputs, outputs)
```

Format the call data into finetuning data according to the OpenAI API specifications.

For the chat adapter, this means formatting the data as a list of messages, where each message is a dictionary
with a "role" and "content" key. The role can be "system", "user", or "assistant". Then, the messages are
wrapped in a dictionary with a "messages" key.


### format_task_description

```python
def format_task_description(self, signature)
```

### format_user_message_content

```python
def format_user_message_content(self, signature, inputs, prefix='', suffix='', main_request=False)
```

### parse

```python
def parse(self, signature, completion)
```

### user_message_output_requirements

```python
def user_message_output_requirements(self, signature)
```

Returns a simplified format reminder for the language model.

In chat-based interactions, language models may lose track of the required output format
as the conversation context grows longer. This method generates a concise reminder of
the expected output structure that can be included in user messages.

Args:
    signature (Type[Signature]): The DSPy signature defining the expected input/output fields.

Returns:
    str: A simplified description of the required output format.

Note:
    This is a more lightweight version of `format_field_structure` specifically designed
    for inline reminders within chat messages.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/adapters/chat_adapter.py` (lines 28–270)

