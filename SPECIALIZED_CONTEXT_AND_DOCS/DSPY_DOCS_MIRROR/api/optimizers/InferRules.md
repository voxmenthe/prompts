# dspy.InferRules

## dspy.InferRules

```python
class InferRules(num_candidates=10, num_rules=10, num_threads=None, teacher_settings=None, **kwargs)
```

### compile

```python
def compile(self, student, *, teacher=None, trainset, valset=None)
```

### evaluate_program

```python
def evaluate_program(self, program, dataset)
```

### format_examples

```python
def format_examples(self, demos, signature)
```

### get_predictor_demos

```python
def get_predictor_demos(self, trainset, predictor)
```

### induce_natural_language_rules

```python
def induce_natural_language_rules(self, predictor, trainset)
```

### update_program_instructions

```python
def update_program_instructions(self, predictor, natural_language_rules)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/teleprompt/infer_rules.py` (lines 13–124)

