# dspy.ToolCalls

## dspy.ToolCalls

```python
class ToolCalls
```

### description

```python
def description(cls)
```

### format

```python
def format(self)
```

### from_dict_list

```python
def from_dict_list(cls, tool_calls_dicts)
```

Convert a list of dictionaries to a ToolCalls instance.

Args:
    dict_list: A list of dictionaries, where each dictionary should have 'name' and 'args' keys.

Returns:
    A ToolCalls instance.

Example:

    ```python
    tool_calls_dict = [
        {"name": "search", "args": {"query": "hello"}},
        {"name": "translate", "args": {"text": "world"}}
    ]
    tool_calls = ToolCalls.from_dict_list(tool_calls_dict)
    ```


### validate_input

```python
def validate_input(cls, data)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/adapters/types/tool.py` (lines 258–381)

