# dspy.JSONAdapter

## dspy.JSONAdapter

```python
class JSONAdapter(callbacks=None, use_native_function_calling=True)
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

### format_field_structure

```python
def format_field_structure(self, signature)
```

### format_field_with_value

```python
def format_field_with_value(self, fields_with_values, role='user')
```

Formats the values of the specified fields according to the field's DSPy type (input or output),
annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
into a single string, which is a multiline string if there are multiple fields.

Args:
    fields_with_values: A dictionary mapping information about a field to its corresponding value.
Returns:
    The joined formatted values of the fields, represented as a string.


### format_finetune_data

```python
def format_finetune_data(self, signature, demos, inputs, outputs)
```

### parse

```python
def parse(self, signature, completion)
```

### user_message_output_requirements

```python
def user_message_output_requirements(self, signature)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/adapters/json_adapter.py` (lines 41–211)

