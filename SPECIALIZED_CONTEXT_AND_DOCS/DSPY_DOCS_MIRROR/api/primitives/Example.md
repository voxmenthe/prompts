# dspy.Example

## dspy.Example

```python
class Example(base=None, **kwargs)
```

A flexible data container for DSPy examples and training data.

The `Example` class is the standard data format used in DSPy evaluation and optimization.

Key features:
    - Dictionary-like access patterns (item access, iteration, etc.)
    - Flexible initialization from dictionaries, other `Example` instances, or keyword arguments
    - Input/output field separation for training data
    - Serialization support for saving/loading `Example` instances
    - Immutable operations that return new `Example` instances

Examples:

    Basic usage with keyword arguments:

    ```python
    import dspy

    # Create an example with input and output fields
    example = dspy.Example(
        question="What is the capital of France?",
        answer="Paris",
    )
    print(example.question)  # "What is the capital of France?"
    print(example.answer)   # "Paris"
    ```

    Initialize from a dictionary:

    ```python
    data = {"question": "What is 2+2?", "answer": "4"}
    example = dspy.Example(data)
    print(example["question"])  # "What is 2+2?"
    ```

    Copy from another Example:

    ```python
    original = dspy.Example(question="Hello", answer="World")
    copy = dspy.Example(original)
    print(copy.question)  # "Hello"
    ```

    Working with input/output separation:

    ```python
    # Mark which fields are inputs for training
    example = dspy.Example(
        question="What is the weather?",
        answer="It's sunny",
    ).with_inputs("question")

    # Get only input fields
    inputs = example.inputs()
    print(inputs.question)  # "What is the weather?"

    # Get only output fields (labels)
    labels = example.labels()
    print(labels.answer)  # "It's sunny"
    ```

    Dictionary-like operations:

    ```python
    example = dspy.Example(name="Alice", age=30)

    # Check if key exists
    if "name" in example:
        print("Name field exists")

    # Get with default value
    city = example.get("city", "Unknown")
    print(city)  # "Unknown"
    ```


### copy

```python
def copy(self, **kwargs)
```

### get

```python
def get(self, key, default=None)
```

### inputs

```python
def inputs(self)
```

### items

```python
def items(self, include_dspy=False)
```

### keys

```python
def keys(self, include_dspy=False)
```

### labels

```python
def labels(self)
```

### toDict

```python
def toDict(self)
```

### values

```python
def values(self, include_dspy=False)
```

### with_inputs

```python
def with_inputs(self, *keys)
```

### without

```python
def without(self, *keys)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/primitives/example.py` (lines 4–213)

