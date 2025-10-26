# dspy.Code

## dspy.Code

```python
class Code
```

Code type in DSPy.

This type is useful for code generation and code analysis.

Example 1: dspy.Code as output type in code generation:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class CodeGeneration(dspy.Signature):
    '''Generate python code to answer the question.'''

    question: str = dspy.InputField(description="The question to answer")
    code: dspy.Code["java"] = dspy.OutputField(description="The code to execute")


predict = dspy.Predict(CodeGeneration)

result = predict(question="Given an array, find if any of the two numbers sum up to 10")
print(result.code)
```

Example 2: dspy.Code as input type in code analysis:

```python
import dspy
import inspect

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class CodeAnalysis(dspy.Signature):
    '''Analyze the time complexity of the function.'''

    code: dspy.Code["python"] = dspy.InputField(description="The function to analyze")
    result: str = dspy.OutputField(description="The time complexity of the function")


predict = dspy.Predict(CodeAnalysis)


def sleepsort(x):
    import time

    for i in x:
        time.sleep(i)
        print(i)

result = predict(code=inspect.getsource(sleepsort))
print(result.result)
```


### description

```python
def description(cls)
```

### format

```python
def format(self)
```

### serialize_model

```python
def serialize_model(self)
```

Override to bypass the <<CUSTOM-TYPE-START-IDENTIFIER>> and <<CUSTOM-TYPE-END-IDENTIFIER>> tags.


### validate_input

```python
def validate_input(cls, data)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/adapters/types/code.py` (lines 10–102)

