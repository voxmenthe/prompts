# Optimization

The `.optimization` module provides:

* an optimizer with weight decay fixed that can be used to fine-tuned models, and
* several schedules in the form of schedule objects that inherit from `_LRSchedule`:
* a gradient accumulation class to accumulate the gradients of multiple batches

## AdaFactor

### class transformers.Adafactor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L688)

( params lr = None eps = (1e-30, 0.001) clip\_threshold = 1.0 decay\_rate = -0.8 beta1 = None weight\_decay = 0.0 scale\_parameter = True relative\_step = True warmup\_init = False  )

Parameters

* **params** (`Iterable[nn.parameter.Parameter]`) —
  Iterable of parameters to optimize or dictionaries defining parameter groups.
* **lr** (`float`, *optional*) —
  The external learning rate.
* **eps** (`tuple[float, float]`, *optional*, defaults to `(1e-30, 0.001)`) —
  Regularization constants for square gradient and parameter scale respectively
* **clip\_threshold** (`float`, *optional*, defaults to 1.0) —
  Threshold of root mean square of final gradient update
* **decay\_rate** (`float`, *optional*, defaults to -0.8) —
  Coefficient used to compute running averages of square
* **beta1** (`float`, *optional*) —
  Coefficient used for computing running averages of gradient
* **weight\_decay** (`float`, *optional*, defaults to 0.0) —
  Weight decay (L2 penalty)
* **scale\_parameter** (`bool`, *optional*, defaults to `True`) —
  If True, learning rate is scaled by root mean square
* **relative\_step** (`bool`, *optional*, defaults to `True`) —
  If True, time-dependent learning rate is computed instead of external learning rate
* **warmup\_init** (`bool`, *optional*, defaults to `False`) —
  Time-dependent learning rate computation depends on whether warm-up initialization is being used

AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
<https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py>

Paper: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* <https://huggingface.co/papers/1804.04235> Note that
this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and
`warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
`relative_step=False`.

This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

Recommended T5 finetuning settings (<https://discuss.huggingface.co/t/t5-finetuning-tips/684/3>):

* Training without LR warmup or clip\_threshold is not recommended.

  + use scheduled LR warm-up to fixed LR
  + use clip\_threshold=1.0 (<https://huggingface.co/papers/1804.04235>)
* Disable relative updates
* Use scale\_parameter=False
* Additional optimizer operations like gradient clipping should not be used alongside Adafactor

Example:


```
Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
```

Others reported the following combination to work well:


```
Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
```

When using `lr=None` with [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) you will most likely need to use `AdafactorSchedule`

scheduler as following:


```
from transformers.optimization import Adafactor, AdafactorSchedule

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)
trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
```

Usage:


```
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

#### step

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L833)

( closure = None  )

Parameters

* **closure** (callable, optional) — A closure that reevaluates the model
  and returns the loss.

Performs a single optimization step

## Schedules

### Learning Rate Schedules

### class transformers.SchedulerType

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_utils.py#L419)

( value names = None module = None qualname = None type = None start = 1  )

Scheduler names for the parameter `lr_scheduler_type` in [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments).
By default, it uses “linear”. Internally, this retrieves `get_linear_schedule_with_warmup` scheduler from [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer).
Scheduler types:

* “linear” = get\_linear\_schedule\_with\_warmup
* “cosine” = get\_cosine\_schedule\_with\_warmup
* “cosine\_with\_restarts” = get\_cosine\_with\_hard\_restarts\_schedule\_with\_warmup
* “polynomial” = get\_polynomial\_decay\_schedule\_with\_warmup
* “constant” = get\_constant\_schedule
* “constant\_with\_warmup” = get\_constant\_schedule\_with\_warmup
* “inverse\_sqrt” = get\_inverse\_sqrt\_schedule
* “reduce\_lr\_on\_plateau” = get\_reduce\_on\_plateau\_schedule
* “cosine\_with\_min\_lr” = get\_cosine\_with\_min\_lr\_schedule\_with\_warmup
* “warmup\_stable\_decay” = get\_wsd\_schedule

#### transformers.get\_scheduler

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L594)

( name: typing.Union[str, transformers.trainer\_utils.SchedulerType] optimizer: Optimizer num\_warmup\_steps: typing.Optional[int] = None num\_training\_steps: typing.Optional[int] = None scheduler\_specific\_kwargs: typing.Optional[dict] = None  )

Parameters

* **name** (`str` or `SchedulerType`) —
  The name of the scheduler to use.
* **optimizer** (`torch.optim.Optimizer`) —
  The optimizer that will be used during training.
* **num\_warmup\_steps** (`int`, *optional*) —
  The number of warmup steps to do. This is not required by all schedulers (hence the argument being
  optional), the function will raise an error if it’s unset and the scheduler type requires it.
* **num\_training\_steps** (`int“, *optional*) —
  The number of training steps to do. This is not required by all schedulers (hence the argument being
  optional), the function will raise an error if it’s unset and the scheduler type requires it.
* **scheduler\_specific\_kwargs** (`dict`, *optional*) —
  Extra parameters for schedulers such as cosine with restarts. Mismatched scheduler types and scheduler
  parameters will cause the scheduler function to raise a TypeError.

Unified API to get any scheduler from its name.

#### transformers.get\_constant\_schedule

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L37)

( optimizer: Optimizer last\_epoch: int = -1  )

Parameters

* **optimizer** (`~torch.optim.Optimizer`) —
  The optimizer for which to schedule the learning rate.
* **last\_epoch** (`int`, *optional*, defaults to -1) —
  The index of the last epoch when resuming training.

Create a schedule with a constant learning rate, using the learning rate set in optimizer.

#### transformers.get\_constant\_schedule\_with\_warmup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L78)

( optimizer: Optimizer num\_warmup\_steps: int last\_epoch: int = -1  )

Parameters

* **optimizer** (`~torch.optim.Optimizer`) —
  The optimizer for which to schedule the learning rate.
* **num\_warmup\_steps** (`int`) —
  The number of steps for the warmup phase.
* **last\_epoch** (`int`, *optional*, defaults to -1) —
  The index of the last epoch when resuming training.

Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
increases linearly between 0 and the initial lr set in the optimizer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_constant_schedule.png)

#### transformers.get\_cosine\_schedule\_with\_warmup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L141)

( optimizer: Optimizer num\_warmup\_steps: int num\_training\_steps: int num\_cycles: float = 0.5 last\_epoch: int = -1  )

Parameters

* **optimizer** (`~torch.optim.Optimizer`) —
  The optimizer for which to schedule the learning rate.
* **num\_warmup\_steps** (`int`) —
  The number of steps for the warmup phase.
* **num\_training\_steps** (`int`) —
  The total number of training steps.
* **num\_cycles** (`float`, *optional*, defaults to 0.5) —
  The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
  following a half-cosine).
* **last\_epoch** (`int`, *optional*, defaults to -1) —
  The index of the last epoch when resuming training.

Create a schedule with a learning rate that decreases following the values of the cosine function between the
initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
initial lr set in the optimizer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_schedule.png)

#### transformers.get\_cosine\_with\_hard\_restarts\_schedule\_with\_warmup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L186)

( optimizer: Optimizer num\_warmup\_steps: int num\_training\_steps: int num\_cycles: int = 1 last\_epoch: int = -1  )

Parameters

* **optimizer** (`~torch.optim.Optimizer`) —
  The optimizer for which to schedule the learning rate.
* **num\_warmup\_steps** (`int`) —
  The number of steps for the warmup phase.
* **num\_training\_steps** (`int`) —
  The total number of training steps.
* **num\_cycles** (`int`, *optional*, defaults to 1) —
  The number of hard restarts to use.
* **last\_epoch** (`int`, *optional*, defaults to -1) —
  The index of the last epoch when resuming training.

Create a schedule with a learning rate that decreases following the values of the cosine function between the
initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
linearly between 0 and the initial lr set in the optimizer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_hard_restarts_schedule.png)

#### transformers.get\_linear\_schedule\_with\_warmup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L105)

( optimizer num\_warmup\_steps num\_training\_steps last\_epoch = -1  )

Parameters

* **optimizer** (`~torch.optim.Optimizer`) —
  The optimizer for which to schedule the learning rate.
* **num\_warmup\_steps** (`int`) —
  The number of steps for the warmup phase.
* **num\_training\_steps** (`int`) —
  The total number of training steps.
* **last\_epoch** (`int`, *optional*, defaults to -1) —
  The index of the last epoch when resuming training.

Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_linear_schedule.png)

#### transformers.get\_polynomial\_decay\_schedule\_with\_warmup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L240)

( optimizer num\_warmup\_steps num\_training\_steps lr\_end = 1e-07 power = 1.0 last\_epoch = -1  )

Parameters

* **optimizer** (`~torch.optim.Optimizer`) —
  The optimizer for which to schedule the learning rate.
* **num\_warmup\_steps** (`int`) —
  The number of steps for the warmup phase.
* **num\_training\_steps** (`int`) —
  The total number of training steps.
* **lr\_end** (`float`, *optional*, defaults to 1e-7) —
  The end LR.
* **power** (`float`, *optional*, defaults to 1.0) —
  Power factor.
* **last\_epoch** (`int`, *optional*, defaults to -1) —
  The index of the last epoch when resuming training.

Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
optimizer to end lr defined by *lr\_end*, after a warmup period during which it increases linearly from 0 to the
initial lr set in the optimizer.

Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
implementation at
<https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37>

#### transformers.get\_inverse\_sqrt\_schedule

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L294)

( optimizer: Optimizer num\_warmup\_steps: int timescale: typing.Optional[int] = None last\_epoch: int = -1  )

Parameters

* **optimizer** (`~torch.optim.Optimizer`) —
  The optimizer for which to schedule the learning rate.
* **num\_warmup\_steps** (`int`) —
  The number of steps for the warmup phase.
* **timescale** (`int`, *optional*, defaults to `num_warmup_steps`) —
  Time scale.
* **last\_epoch** (`int`, *optional*, defaults to -1) —
  The index of the last epoch when resuming training.

Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

#### transformers.get\_wsd\_schedule

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/optimization.py#L506)

( optimizer: Optimizer num\_warmup\_steps: int num\_decay\_steps: int num\_training\_steps: typing.Optional[int] = None num\_stable\_steps: typing.Optional[int] = None warmup\_type: str = 'linear' decay\_type: str = 'cosine' min\_lr\_ratio: float = 0 num\_cycles: float = 0.5 last\_epoch: int = -1  )

Parameters

* **optimizer** (`~torch.optim.Optimizer`) —
  The optimizer for which to schedule the learning rate.
* **num\_warmup\_steps** (`int`) —
  The number of steps for the warmup phase.
* **num\_decay\_steps** (`int`) —
  The number of steps for the decay phase.
* **num\_training\_steps** (`int`, *optional*) —
  The total number of training steps. This is the sum of the warmup, stable and decay steps. If `num_stable_steps` is not provided, the stable phase will be `num_training_steps - num_warmup_steps - num_decay_steps`.
* **num\_stable\_steps** (`int`, *optional*) —
  The number of steps for the stable phase. Please ensure that `num_warmup_steps + num_stable_steps + num_decay_steps` equals `num_training_steps`, otherwise the other steps will default to the minimum learning rate.
* **warmup\_type** (`str`, *optional*, defaults to “linear”) —
  The type of warmup to use. Can be ‘linear’, ‘cosine’ or ‘1-sqrt’.
* **decay\_type** (`str`, *optional*, defaults to “cosine”) —
  The type of decay to use. Can be ‘linear’, ‘cosine’ or ‘1-sqrt’.
* **min\_lr\_ratio** (`float`, *optional*, defaults to 0) —
  The minimum learning rate as a ratio of the initial learning rate.
* **num\_cycles** (`float`, *optional*, defaults to 0.5) —
  The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
  following a half-cosine).
* **last\_epoch** (`int`, *optional*, defaults to -1) —
  The index of the last epoch when resuming training.

Create a schedule with a learning rate that has three stages:

1. warmup: increase from min\_lr\_ratio times the initial learning rate to the initial learning rate following a warmup\_type.
2. stable: constant learning rate.
3. decay: decrease from the initial learning rate to min\_lr\_ratio times the initial learning rate following a decay\_type.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/optimizer_schedules.md)
