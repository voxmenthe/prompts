# dspy.ProgramOfThought

## dspy.ProgramOfThought

```python
class ProgramOfThought(signature, max_iters=3, interpreter=None)
```

A DSPy module that runs Python programs to solve a problem.
This module requires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/

Example:
```
import dspy

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
pot = dspy.ProgramOfThought("question -> answer")
pot(question="what is 1+1?")
```


### forward

```python
def forward(self, **kwargs)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/predict/program_of_thought.py` (lines 13–197)

