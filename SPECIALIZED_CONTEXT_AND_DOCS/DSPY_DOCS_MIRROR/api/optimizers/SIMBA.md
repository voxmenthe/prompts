# dspy.SIMBA

## dspy.SIMBA

```python
class SIMBA(*, metric, bsize=32, num_candidates=6, max_steps=8, max_demos=4, demo_input_field_maxlen=100000, num_threads=None, temperature_for_sampling=0.2, temperature_for_candidates=0.2)
```

SIMBA (Stochastic Introspective Mini-Batch Ascent) optimizer for DSPy.

SIMBA is a DSPy optimizer that uses the LLM to analyze its own performance and 
generate improvement rules. It samples mini-batches, identifies challenging examples 
with high output variability, then either creates self-reflective rules or adds 
successful examples as demonstrations.

For more details, see: https://dspy.ai/api/optimizers/SIMBA/


### compile

```python
def compile(self, student, *, trainset, seed=0)
```

Compile and optimize the student module using SIMBA.

Args:
    student: The module to optimize
    trainset: Training examples for optimization
    seed: Random seed for reproducibility
    
Returns:
    The optimized module with candidate_programs and trial_logs attached

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/teleprompt/simba.py` (lines 14–366)


## Example Usage

```python
optimizer = dspy.SIMBA(metric=your_metric)
optimized_program = optimizer.compile(your_program, trainset=your_trainset)

# Save optimize program for future use
optimized_program.save(f"optimized.json")
```

## How `SIMBA` works
SIMBA (Stochastic Introspective Mini-Batch Ascent) is a DSPy optimizer that uses the LLM to analyze its own performance and generate improvement rules. It samples mini-batches, identifies challenging examples with high output variability, then either creates self-reflective rules or adds successful examples as demonstrations. See [this great blog post](https://blog.mariusvach.com/posts/dspy-simba) from [Marius](https://x.com/rasmus1610) for more details.