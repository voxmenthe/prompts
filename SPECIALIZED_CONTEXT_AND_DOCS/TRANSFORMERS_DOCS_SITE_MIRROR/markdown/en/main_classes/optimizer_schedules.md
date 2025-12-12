# Optimization

The `.optimization` module provides:

- an optimizer with weight decay fixed that can be used to fine-tuned models, and
- several schedules in the form of schedule objects that inherit from `_LRSchedule`:
- a gradient accumulation class to accumulate the gradients of multiple batches

## AdaFactor[[transformers.Adafactor]]

#### transformers.Adafactor[[transformers.Adafactor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L687)

AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

Paper: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* https://huggingface.co/papers/1804.04235 Note that
this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and
`warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
`relative_step=False`.

This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

Recommended T5 finetuning settings (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3):

- Training without LR warmup or clip_threshold is not recommended.

  - use scheduled LR warm-up to fixed LR
  - use clip_threshold=1.0 (https://huggingface.co/papers/1804.04235)
- Disable relative updates
- Use scale_parameter=False
- Additional optimizer operations like gradient clipping should not be used alongside Adafactor

Example:

```python
Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
```

Others reported the following combination to work well:

```python
Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
```

When using `lr=None` with [Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) you will most likely need to use `AdafactorSchedule`

scheduler as following:

```python
from transformers.optimization import Adafactor, AdafactorSchedule

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)
trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
```

Usage:

```python
# replace AdamW with Adafactor
optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)
```

steptransformers.Adafactor.stephttps://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L832[{"name": "closure", "val": " = None"}]- **closure** (callable, optional) -- A closure that reevaluates the model
  and returns the loss.0

Performs a single optimization step

**Parameters:**

params (`Iterable[nn.parameter.Parameter]`) : Iterable of parameters to optimize or dictionaries defining parameter groups.

lr (`float`, *optional*) : The external learning rate.

eps (`tuple[float, float]`, *optional*, defaults to `(1e-30, 0.001)`) : Regularization constants for square gradient and parameter scale respectively

clip_threshold (`float`, *optional*, defaults to 1.0) : Threshold of root mean square of final gradient update

decay_rate (`float`, *optional*, defaults to -0.8) : Coefficient used to compute running averages of square

beta1 (`float`, *optional*) : Coefficient used for computing running averages of gradient

weight_decay (`float`, *optional*, defaults to 0.0) : Weight decay (L2 penalty)

scale_parameter (`bool`, *optional*, defaults to `True`) : If True, learning rate is scaled by root mean square

relative_step (`bool`, *optional*, defaults to `True`) : If True, time-dependent learning rate is computed instead of external learning rate

warmup_init (`bool`, *optional*, defaults to `False`) : Time-dependent learning rate computation depends on whether warm-up initialization is being used

## Schedules

### SchedulerType[[transformers.SchedulerType]]

#### transformers.SchedulerType[[transformers.SchedulerType]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L391)

Scheduler names for the parameter `lr_scheduler_type` in [TrainingArguments](/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments).
By default, it uses "linear". Internally, this retrieves `get_linear_schedule_with_warmup` scheduler from [Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer).
Scheduler types:
- "linear" = [get_linear_schedule_with_warmup()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup)
- "cosine" = [get_cosine_schedule_with_warmup()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup)
- "cosine_with_restarts" = [get_cosine_with_hard_restarts_schedule_with_warmup()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_cosine_with_hard_restarts_schedule_with_warmup)
- "polynomial" = [get_polynomial_decay_schedule_with_warmup()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_polynomial_decay_schedule_with_warmup)
- "constant" =  [get_constant_schedule()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_constant_schedule)
- "constant_with_warmup" = [get_constant_schedule_with_warmup()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_constant_schedule_with_warmup)
- "inverse_sqrt" = [get_inverse_sqrt_schedule()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_inverse_sqrt_schedule)
- "reduce_lr_on_plateau" = [get_reduce_on_plateau_schedule()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_reduce_on_plateau_schedule)
- "cosine_with_min_lr" = [get_cosine_with_min_lr_schedule_with_warmup()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_cosine_with_min_lr_schedule_with_warmup)
- "cosine_warmup_with_min_lr" = [get_cosine_with_min_lr_schedule_with_warmup_lr_rate()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_cosine_with_min_lr_schedule_with_warmup_lr_rate)
- "warmup_stable_decay" = [get_wsd_schedule()](/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.get_wsd_schedule)

### get_scheduler[[transformers.get_scheduler]]

#### transformers.get_scheduler[[transformers.get_scheduler]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L593)

Unified API to get any scheduler from its name.

**Parameters:**

name (`str` or `SchedulerType`) : The name of the scheduler to use.

optimizer (`torch.optim.Optimizer`) : The optimizer that will be used during training.

num_warmup_steps (`int`, *optional*) : The number of warmup steps to do. This is not required by all schedulers (hence the argument being optional), the function will raise an error if it's unset and the scheduler type requires it.

num_training_steps (`int``, *optional*) : The number of training steps to do. This is not required by all schedulers (hence the argument being optional), the function will raise an error if it's unset and the scheduler type requires it.

scheduler_specific_kwargs (`dict`, *optional*) : Extra parameters for schedulers such as cosine with restarts. Mismatched scheduler types and scheduler parameters will cause the scheduler function to raise a TypeError.

### get_constant_schedule[[transformers.get_constant_schedule]]

#### transformers.get_constant_schedule[[transformers.get_constant_schedule]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L36)

Create a schedule with a constant learning rate, using the learning rate set in optimizer.

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

last_epoch (`int`, *optional*, defaults to -1) : The index of the last epoch when resuming training.

**Returns:**

`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

### get_constant_schedule_with_warmup[[transformers.get_constant_schedule_with_warmup]]

#### transformers.get_constant_schedule_with_warmup[[transformers.get_constant_schedule_with_warmup]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L77)

Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
increases linearly between 0 and the initial lr set in the optimizer.

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

num_warmup_steps (`int`) : The number of steps for the warmup phase.

last_epoch (`int`, *optional*, defaults to -1) : The index of the last epoch when resuming training.

**Returns:**

`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

### get_cosine_schedule_with_warmup[[transformers.get_cosine_schedule_with_warmup]]

#### transformers.get_cosine_schedule_with_warmup[[transformers.get_cosine_schedule_with_warmup]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L140)

Create a schedule with a learning rate that decreases following the values of the cosine function between the
initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
initial lr set in the optimizer.

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

num_warmup_steps (`int`) : The number of steps for the warmup phase.

num_training_steps (`int`) : The total number of training steps.

num_cycles (`float`, *optional*, defaults to 0.5) : The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0 following a half-cosine).

last_epoch (`int`, *optional*, defaults to -1) : The index of the last epoch when resuming training.

**Returns:**

`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

### get_cosine_with_hard_restarts_schedule_with_warmup[[transformers.get_cosine_with_hard_restarts_schedule_with_warmup]]

#### transformers.get_cosine_with_hard_restarts_schedule_with_warmup[[transformers.get_cosine_with_hard_restarts_schedule_with_warmup]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L185)

Create a schedule with a learning rate that decreases following the values of the cosine function between the
initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
linearly between 0 and the initial lr set in the optimizer.

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

num_warmup_steps (`int`) : The number of steps for the warmup phase.

num_training_steps (`int`) : The total number of training steps.

num_cycles (`int`, *optional*, defaults to 1) : The number of hard restarts to use.

last_epoch (`int`, *optional*, defaults to -1) : The index of the last epoch when resuming training.

**Returns:**

`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

### get_cosine_with_min_lr_schedule_with_warmup[[transformers.get_cosine_with_min_lr_schedule_with_warmup]]

#### transformers.get_cosine_with_min_lr_schedule_with_warmup[[transformers.get_cosine_with_min_lr_schedule_with_warmup]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L334)

Create a schedule with a learning rate that decreases following the values of the cosine function between the
initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
initial lr set in the optimizer.

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

num_warmup_steps (`int`) : The number of steps for the warmup phase.

num_training_steps (`int`) : The total number of training steps.

num_cycles (`float`, *optional*, defaults to 0.5) : The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0 following a half-cosine).

last_epoch (`int`, *optional*, defaults to -1) : The index of the last epoch when resuming training.

min_lr (`float`, *optional*) : The minimum learning rate to reach after the cosine schedule.

min_lr_rate (`float`, *optional*) : The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.

**Returns:**

`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

### get_cosine_with_min_lr_schedule_with_warmup_lr_rate[[transformers.get_cosine_with_min_lr_schedule_with_warmup_lr_rate]]

#### transformers.get_cosine_with_min_lr_schedule_with_warmup_lr_rate[[transformers.get_cosine_with_min_lr_schedule_with_warmup_lr_rate]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L411)

Create a schedule with a learning rate that decreases following the values of the cosine function between the
initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
initial lr set in the optimizer.

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

num_warmup_steps (`int`) : The number of steps for the warmup phase.

num_training_steps (`int`) : The total number of training steps.

num_cycles (`float`, *optional*, defaults to 0.5) : The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0 following a half-cosine).

last_epoch (`int`, *optional*, defaults to -1) : The index of the last epoch when resuming training.

min_lr (`float`, *optional*) : The minimum learning rate to reach after the cosine schedule.

min_lr_rate (`float`, *optional*) : The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.

warmup_lr_rate (`float`, *optional*) : The minimum learning rate as a ratio of the start learning rate. If not set, `warmup_lr_rate` will be treated as float(1/num_warmup_steps).

**Returns:**

`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

### get_linear_schedule_with_warmup[[transformers.get_linear_schedule_with_warmup]]

#### transformers.get_linear_schedule_with_warmup[[transformers.get_linear_schedule_with_warmup]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L104)

Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

num_warmup_steps (`int`) : The number of steps for the warmup phase.

num_training_steps (`int`) : The total number of training steps.

last_epoch (`int`, *optional*, defaults to -1) : The index of the last epoch when resuming training.

**Returns:**

`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

### get_polynomial_decay_schedule_with_warmup[[transformers.get_polynomial_decay_schedule_with_warmup]]

#### transformers.get_polynomial_decay_schedule_with_warmup[[transformers.get_polynomial_decay_schedule_with_warmup]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L239)

Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
initial lr set in the optimizer.

Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
implementation at
https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

num_warmup_steps (`int`) : The number of steps for the warmup phase.

num_training_steps (`int`) : The total number of training steps.

lr_end (`float`, *optional*, defaults to 1e-7) : The end LR.

power (`float`, *optional*, defaults to 1.0) : Power factor.

last_epoch (`int`, *optional*, defaults to -1) : The index of the last epoch when resuming training.

**Returns:**

`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

### get_inverse_sqrt_schedule[[transformers.get_inverse_sqrt_schedule]]

#### transformers.get_inverse_sqrt_schedule[[transformers.get_inverse_sqrt_schedule]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L293)

Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

num_warmup_steps (`int`) : The number of steps for the warmup phase.

timescale (`int`, *optional*, defaults to `num_warmup_steps`) : Time scale.

last_epoch (`int`, *optional*, defaults to -1) : The index of the last epoch when resuming training.

**Returns:**

`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

### get_reduce_on_plateau_schedule[[transformers.get_reduce_on_plateau_schedule]]

#### transformers.get_reduce_on_plateau_schedule[[transformers.get_reduce_on_plateau_schedule]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L53)

Create a schedule with a constant learning rate that decreases when a metric has stopped improving.

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

kwargs (`dict`, *optional*) : Extra parameters to be passed to the scheduler. See `torch.optim.lr_scheduler.ReduceLROnPlateau` for possible parameters.

**Returns:**

`torch.optim.lr_scheduler.ReduceLROnPlateau` with the appropriate schedule.

### get_wsd_schedule[[transformers.get_wsd_schedule]]

#### transformers.get_wsd_schedule[[transformers.get_wsd_schedule]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py#L505)

Create a schedule with a learning rate that has three stages:
1. warmup: increase from min_lr_ratio times the initial learning rate to the initial learning rate following a warmup_type.
2. stable: constant learning rate.
3. decay: decrease from the initial learning rate to min_lr_ratio times the initial learning rate following a decay_type.

**Parameters:**

optimizer (`~torch.optim.Optimizer`) : The optimizer for which to schedule the learning rate.

num_warmup_steps (`int`) : The number of steps for the warmup phase.

num_decay_steps (`int`) : The number of steps for the decay phase.

num_training_steps (`int`, *optional*) : The total number of training steps. This is the sum of the warmup, stable and decay steps. If `num_stable_steps` is not provided, the stable phase will be `num_training_steps - num_warmup_steps - num_decay_steps`.

num_stable_steps (`int`, *optional*) : The number of steps for the stable phase. Please ensure that `num_warmup_steps + num_stable_steps + num_decay_steps` equals `num_training_steps`, otherwise the other steps will default to the minimum learning rate.

warmup_type (`str`, *optional*, defaults to "linear") : The type of warmup to use. Can be 'linear', 'cosine' or '1-sqrt'.

decay_type (`str`, *optional*, defaults to "cosine") : The type of decay to use. Can be 'linear', 'cosine' or '1-sqrt'.

min_lr_ratio (`float`, *optional*, defaults to 0) : The minimum learning rate as a ratio of the initial learning rate.

num_cycles (`float`, *optional*, defaults to 0.5) : The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0 following a half-cosine).

last_epoch (`int`, *optional*, defaults to -1) : The index of the last epoch when resuming training.

**Returns:**

`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
