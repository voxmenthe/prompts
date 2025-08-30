# dspy.Signature

## dspy.Signature

```python
class Signature
```

### append

```python
def append(cls, name, field, type_=None)
```

### delete

```python
def delete(cls, name)
```

### dump_state

```python
def dump_state(cls)
```

### equals

```python
def equals(cls, other)
```

Compare the JSON schema of two Signature classes.


### insert

```python
def insert(cls, index, name, field, type_=None)
```

### load_state

```python
def load_state(cls, state)
```

### prepend

```python
def prepend(cls, name, field, type_=None)
```

### with_instructions

```python
def with_instructions(cls, instructions)
```

### with_updated_fields

```python
def with_updated_fields(cls, name, type_=None, **kwargs)
```

Create a new Signature class with the updated field information.

Returns a new Signature class with the field, name, updated
with fields[name].json_schema_extra[key] = value.

Args:
    name: The name of the field to update.
    type_: The new type of the field.
    kwargs: The new values for the field.

Returns:
    A new Signature class (not an instance) with the updated field information.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/signatures/signature.py` (lines 240–355)

