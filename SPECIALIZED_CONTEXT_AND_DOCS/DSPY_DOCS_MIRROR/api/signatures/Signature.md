# dspy.Signature

## dspy.Signature

```python
class Signature
```

### append

```python
def append(cls, name, field, type_=None)
```

Insert a field at the end of the `inputs` or `outputs` section.

Args:
    name (str): Field name to add.
    field: `InputField` or `OutputField` instance to insert.
    type_ (type | None): Optional explicit type annotation. If `type_` is `None`, the effective type is
        resolved by `insert`.

Returns:
    A new Signature class with the field appended.

Example:
    ```python
    import dspy

    class MySig(dspy.Signature):
        input_text: str = dspy.InputField(desc="Input sentence")
        output_text: str = dspy.OutputField(desc="Translated sentence")

    NewSig = MySig.append("confidence", dspy.OutputField(desc="Translation confidence"))
    print(list(NewSig.fields.keys()))
    ```


### delete

```python
def delete(cls, name)
```

Return a new Signature class without the given field.

If `name` is not present, the fields are unchanged (no error raised).

Args:
    name (str): Field name to remove.

Returns:
    A new Signature class with the field removed (or unchanged if the field was absent).

Example:
    ```python
    import dspy

    class MySig(dspy.Signature):
        input_text: str = dspy.InputField(desc="Input sentence")
        temp_field: str = dspy.InputField(desc="Temporary debug field")
        output_text: str = dspy.OutputField(desc="Translated sentence")

    NewSig = MySig.delete("temp_field")
    print(list(NewSig.fields.keys()))

    # No error is raised if the field is not present
    Unchanged = NewSig.delete("nonexistent")
    print(list(Unchanged.fields.keys()))
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

Insert a field at a specific position among inputs or outputs.

Negative indices are supported (e.g., `-1` appends). If `type_` is omitted, the field's
existing `annotation` is used; if that is missing, `str` is used.

Args:
    index (int): Insertion position within the chosen section; negatives append.
    name (str): Field name to add.
    field: InputField or OutputField instance to insert.
    type_ (type | None): Optional explicit type annotation.

Returns:
    A new Signature class with the field inserted.

Raises:
    ValueError: If `index` falls outside the valid range for the chosen section.

Example:
    ```python
    import dspy

    class MySig(dspy.Signature):
        input_text: str = dspy.InputField(desc="Input sentence")
        output_text: str = dspy.OutputField(desc="Translated sentence")

    NewSig = MySig.insert(0, "context", dspy.InputField(desc="Context for translation"))
    print(list(NewSig.fields.keys()))

    NewSig2 = NewSig.insert(-1, "confidence", dspy.OutputField(desc="Translation confidence"))
    print(list(NewSig2.fields.keys()))
    ```


### load_state

```python
def load_state(cls, state)
```

### prepend

```python
def prepend(cls, name, field, type_=None)
```

Insert a field at index 0 of the `inputs` or `outputs` section.

Args:
    name (str): Field name to add.
    field: `InputField` or `OutputField` instance to insert.
    type_ (type | None): Optional explicit type annotation. If `type_` is `None`, the effective type is
        resolved by `insert`.

Returns:
    A new `Signature` class with the field inserted first.

Example:
    ```python
    import dspy

    class MySig(dspy.Signature):
        input_text: str = dspy.InputField(desc="Input sentence")
        output_text: str = dspy.OutputField(desc="Translated sentence")

    NewSig = MySig.prepend("context", dspy.InputField(desc="Context for translation"))
    print(list(NewSig.fields.keys()))
    ```


### with_instructions

```python
def with_instructions(cls, instructions)
```

Return a new Signature class with identical fields and new instructions.

This method does not mutate `cls`. It constructs a fresh Signature
class using the current fields and the provided `instructions`.

Args:
    instructions (str): Instruction text to attach to the new signature.

Returns:
    A new Signature class whose fields match `cls.fields`
    and whose instructions equal `instructions`.

Example:
    ```python
    import dspy

    class MySig(dspy.Signature):
        input_text: str = dspy.InputField(desc="Input text")
        output_text: str = dspy.OutputField(desc="Output text")

    NewSig = MySig.with_instructions("Translate to French.")
    assert NewSig is not MySig
    assert NewSig.instructions == "Translate to French."
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

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/signatures/signature.py` (lines 261–506)

