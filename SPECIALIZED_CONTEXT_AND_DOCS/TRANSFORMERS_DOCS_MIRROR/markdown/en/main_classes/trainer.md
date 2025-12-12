# Trainer

The [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) class provides an API for feature-complete training in PyTorch, and it supports distributed training on multiple GPUs/TPUs, mixed precision for [NVIDIA GPUs](https://nvidia.github.io/apex/), [AMD GPUs](https://rocm.docs.amd.com/en/latest/rocm.html), and [`torch.amp`](https://pytorch.org/docs/stable/amp.html) for PyTorch. [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) goes hand-in-hand with the [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) class, which offers a wide range of options to customize how a model is trained. Together, these two classes provide a complete training API.

[Seq2SeqTrainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Seq2SeqTrainer) and [Seq2SeqTrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments) inherit from the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) and [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) classes and they’re adapted for training models for sequence-to-sequence tasks such as summarization or translation.

The [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) class is optimized for 🤗 Transformers models and can have surprising behaviors
when used with other models. When using it with your own model, make sure:

* your model always return tuples or subclasses of [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput)
* your model can compute the loss if a `labels` argument is provided and that loss is returned as the first
  element of the tuple (if your model returns tuples)
* your model can accept multiple label arguments (use `label_names` in [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) to indicate their name to the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer)) but none of them should be named `"label"`

## Trainer

### class transformers.Trainer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L319)

( model: typing.Union[transformers.modeling\_utils.PreTrainedModel, torch.nn.modules.module.Module, NoneType] = None args: TrainingArguments = None data\_collator: typing.Optional[transformers.data.data\_collator.DataCollator] = None train\_dataset: typing.Union[torch.utils.data.dataset.Dataset, torch.utils.data.dataset.IterableDataset, ForwardRef('datasets.Dataset'), NoneType] = None eval\_dataset: typing.Union[torch.utils.data.dataset.Dataset, dict[str, torch.utils.data.dataset.Dataset], ForwardRef('datasets.Dataset'), NoneType] = None processing\_class: typing.Union[transformers.tokenization\_utils\_base.PreTrainedTokenizerBase, transformers.image\_processing\_utils.BaseImageProcessor, transformers.feature\_extraction\_utils.FeatureExtractionMixin, transformers.processing\_utils.ProcessorMixin, NoneType] = None model\_init: typing.Optional[typing.Callable[[], transformers.modeling\_utils.PreTrainedModel]] = None compute\_loss\_func: typing.Optional[typing.Callable] = None compute\_metrics: typing.Optional[typing.Callable[[transformers.trainer\_utils.EvalPrediction], dict]] = None callbacks: typing.Optional[list[transformers.trainer\_callback.TrainerCallback]] = None optimizers: tuple = (None, None) optimizer\_cls\_and\_kwargs: typing.Optional[tuple[type[torch.optim.optimizer.Optimizer], dict[str, typing.Any]]] = None preprocess\_logits\_for\_metrics: typing.Optional[typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None  )

Parameters

* **model** ([PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel) or `torch.nn.Module`, *optional*) —
  The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

  [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) is optimized to work with the [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel) provided by the library. You can still use
  your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
  models.
* **args** ([TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments), *optional*) —
  The arguments to tweak for training. Will default to a basic instance of [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) with the
  `output_dir` set to a directory named *tmp\_trainer* in the current directory if not provided.
* **data\_collator** (`DataCollator`, *optional*) —
  The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
  default to [default\_data\_collator()](/docs/transformers/v4.56.2/en/main_classes/data_collator#transformers.default_data_collator) if no `processing_class` is provided, an instance of
  [DataCollatorWithPadding](/docs/transformers/v4.56.2/en/main_classes/data_collator#transformers.DataCollatorWithPadding) otherwise if the processing\_class is a feature extractor or tokenizer.
* **train\_dataset** (Union[`torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `datasets.Dataset`], *optional*) —
  The dataset to use for training. If it is a [Dataset](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the
  `model.forward()` method are automatically removed.

  Note that if it’s a `torch.utils.data.IterableDataset` with some randomization and you are training in a
  distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
  `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
  manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
  sets the seed of the RNGs used.
* **eval\_dataset** (Union[`torch.utils.data.Dataset`, dict[str, `torch.utils.data.Dataset`, `datasets.Dataset`]), *optional*) —
  The dataset to use for evaluation. If it is a [Dataset](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the
  `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
  dataset prepending the dictionary key to the metric name.
* **processing\_class** (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*) —
  Processing class used to process the data. If provided, will be used to automatically process the inputs
  for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
  reuse the fine-tuned model.
  This supersedes the `tokenizer` argument, which is now deprecated.
* **model\_init** (`Callable[[], PreTrainedModel]`, *optional*) —
  A function that instantiates the model to be used. If provided, each call to [train()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.train) will start
  from a new instance of the model as given by this function.

  The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
  be able to choose different architectures according to hyper parameters (such as layer count, sizes of
  inner layers, dropout probabilities etc).
* **compute\_loss\_func** (`Callable`, *optional*) —
  A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
  batch (batch\_size \* gradient\_accumulation\_steps) and returns the loss. For example, see the default [loss function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618) used by [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer).
* **compute\_metrics** (`Callable[[EvalPrediction], Dict]`, *optional*) —
  The function that will be used to compute metrics at evaluation. Must take a [EvalPrediction](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.EvalPrediction) and return
  a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
  `True`, your compute\_metrics function must take a boolean `compute_result` argument. This will be triggered
  after the last eval batch to signal that the function needs to calculate and return the global summary
  statistics rather than accumulating the batch-level statistics
* **callbacks** (List of [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback), *optional*) —
  A list of callbacks to customize the training loop. Will add those to the list of default callbacks
  detailed in [here](callback).

  If you want to remove one of the default callbacks used, use the [Trainer.remove\_callback()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.remove_callback) method.
* **optimizers** (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`) —
  A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
  model and a scheduler given by [get\_linear\_schedule\_with\_warmup()](/docs/transformers/v4.56.2/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup) controlled by `args`.
* **optimizer\_cls\_and\_kwargs** (`tuple[Type[torch.optim.Optimizer], dict[str, Any]]`, *optional*) —
  A tuple containing the optimizer class and keyword arguments to use.
  Overrides `optim` and `optim_args` in `args`. Incompatible with the `optimizers` argument.

  Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before initializing the Trainer.
* **preprocess\_logits\_for\_metrics** (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*) —
  A function that preprocess the logits right before caching them at each evaluation step. Must take two
  tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
  by this function will be reflected in the predictions received by `compute_metrics`.

  Note that the labels (second parameter) will be `None` if the dataset does not have them.

Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

Important attributes:

* **model** — Always points to the core model. If using a transformers model, it will be a [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel)
  subclass.
* **model\_wrapped** — Always points to the most external model in case one or more other modules wrap the
  original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
  the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
  model hasn’t been wrapped, then `self.model_wrapped` is the same as `self.model`.
* **is\_model\_parallel** — Whether or not a model has been switched to a model parallel mode (different from
  data parallelism, this means some of the model layers are split on different GPUs).
* **place\_model\_on\_device** — Whether or not to automatically place the model on the device - it will be set
  to `False` if model parallel or deepspeed is used, or if the default
  `TrainingArguments.place_model_on_device` is overridden to return `False` .
* **is\_in\_train** — Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
  in `train`)

#### add\_callback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L874)

( callback  )

Parameters

* **callback** (`type` or [`~transformers.TrainerCallback]`) —
  A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) class or an instance of a [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback). In the
  first case, will instantiate a member of that class.

Add a callback to the current list of [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback).

#### autocast\_smart\_context\_manager

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L3955)

( cache\_enabled: typing.Optional[bool] = True  )

A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
arguments, depending on the situation.

#### compute\_loss

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4064)

( model: Module inputs: dict return\_outputs: bool = False num\_items\_in\_batch: typing.Optional[torch.Tensor] = None  )

Parameters

* **model** (`nn.Module`) —
  The model to compute the loss for.
* **inputs** (`dict[str, Union[torch.Tensor, Any]]`) —
  The input data for the model.
* **return\_outputs** (`bool`, *optional*, defaults to `False`) —
  Whether to return the model outputs along with the loss.
* **num\_items\_in\_batch** (Optional[torch.Tensor], *optional*) —
  The number of items in the batch. If num\_items\_in\_batch is not passed,

How the loss is computed by Trainer. By default, all models return the loss in the first element.

Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculating might be slightly inaccurate when performing gradient accumulation.

#### compute\_loss\_context\_manager

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L3943)

( )

A helper wrapper to group together context managers.

#### create\_model\_card

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4945)

( language: typing.Optional[str] = None license: typing.Optional[str] = None tags: typing.Union[str, list[str], NoneType] = None model\_name: typing.Optional[str] = None finetuned\_from: typing.Optional[str] = None tasks: typing.Union[str, list[str], NoneType] = None dataset\_tags: typing.Union[str, list[str], NoneType] = None dataset: typing.Union[str, list[str], NoneType] = None dataset\_args: typing.Union[str, list[str], NoneType] = None  )

Parameters

* **language** (`str`, *optional*) —
  The language of the model (if applicable)
* **license** (`str`, *optional*) —
  The license of the model. Will default to the license of the pretrained model used, if the original
  model given to the `Trainer` comes from a repo on the Hub.
* **tags** (`str` or `list[str]`, *optional*) —
  Some tags to be included in the metadata of the model card.
* **model\_name** (`str`, *optional*) —
  The name of the model.
* **finetuned\_from** (`str`, *optional*) —
  The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
  of the original model given to the `Trainer` (if it comes from the Hub).
* **tasks** (`str` or `list[str]`, *optional*) —
  One or several task identifiers, to be included in the metadata of the model card.
* **dataset\_tags** (`str` or `list[str]`, *optional*) —
  One or several dataset tags, to be included in the metadata of the model card.
* **dataset** (`str` or `list[str]`, *optional*) —
  One or several dataset identifiers, to be included in the metadata of the model card.
* **dataset\_args** (`str` or `list[str]`, *optional*) —
  One or several dataset arguments, to be included in the metadata of the model card.

Creates a draft of a model card using the information available to the `Trainer`.

#### create\_optimizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1281)

( )

Setup the optimizer.

We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
Trainer’s init through `optimizers`, or subclass and override this method in a subclass.

#### create\_optimizer\_and\_scheduler

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1253)

( num\_training\_steps: int  )

Setup the optimizer and the learning rate scheduler.

We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
Trainer’s init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
`create_scheduler`) in a subclass.

#### create\_scheduler

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1846)

( num\_training\_steps: int optimizer: Optimizer = None  )

Parameters

* **num\_training\_steps** (int) — The number of training steps to do.

Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
passed as an argument.

#### evaluate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4401)

( eval\_dataset: typing.Union[torch.utils.data.dataset.Dataset, dict[str, torch.utils.data.dataset.Dataset], NoneType] = None ignore\_keys: typing.Optional[list[str]] = None metric\_key\_prefix: str = 'eval'  )

Parameters

* **eval\_dataset** (Union[`Dataset`, dict[str, `Dataset`]), *optional*) —
  Pass a dataset if you wish to override `self.eval_dataset`. If it is a [Dataset](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset), columns
  not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
  evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
  `__len__` method.

  If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
  separate evaluations on each dataset. This can be useful to monitor how training affects other
  datasets or simply to get a more fine-grained evaluation.
  When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
  of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
  `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
  loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.
* **ignore\_keys** (`list[str]`, *optional*) —
  A list of keys in the output of your model (if it is a dictionary) that should be ignored when
  gathering predictions.
* **metric\_key\_prefix** (`str`, *optional*, defaults to `"eval"`) —
  An optional prefix to be used as the metrics key prefix. For example the metrics “bleu” will be named
  “eval\_bleu” if the prefix is “eval” (default)

Run evaluation and returns metrics.

The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
(pass it to the init `compute_metrics` argument).

You can also subclass and override this method to inject custom behavior.

#### evaluation\_loop

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4569)

( dataloader: DataLoader description: str prediction\_loss\_only: typing.Optional[bool] = None ignore\_keys: typing.Optional[list[str]] = None metric\_key\_prefix: str = 'eval'  )

Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

Works both with or without labels.

#### floating\_point\_ops

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4909)

( inputs: dict  ) → `int`

Parameters

* **inputs** (`dict[str, Union[torch.Tensor, Any]]`) —
  The inputs and targets of the model.

Returns

`int`

The number of floating-point operations.

For models that inherit from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel), uses that method to compute the number of floating point
operations for every backward + forward pass. If using another model, either implement such a method in the
model or subclass and override this method.

#### get\_batch\_samples

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L5570)

( epoch\_iterator: Iterator num\_batches: int device: device  )

Collects a specified number of batches from the epoch iterator and optionally counts the number of items in the batches to properly scale the loss.

#### get\_decay\_parameter\_names

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1269)

( model  )

Get all parameter names that weight decay will be applied to.

This function filters out parameters in two ways:

1. By layer type (instances of layers specified in ALL\_LAYERNORM\_LAYERS)
2. By parameter name patterns (containing ‘bias’, or variation of ‘norm’)

#### get\_eval\_dataloader

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1196)

( eval\_dataset: typing.Union[str, torch.utils.data.dataset.Dataset, NoneType] = None  )

Parameters

* **eval\_dataset** (`str` or `torch.utils.data.Dataset`, *optional*) —
  If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [Dataset](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the `model.forward()` method are automatically removed.

Returns the evaluation `~torch.utils.data.DataLoader`.

Subclass and override this method if you want to inject some custom behavior.

#### get\_learning\_rates

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1354)

( )

Returns the learning rate of each parameter from self.optimizer.

#### get\_num\_trainable\_parameters

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1348)

( )

Get the number of trainable parameters.

#### get\_optimizer\_cls\_and\_kwargs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1378)

( args: TrainingArguments model: typing.Optional[transformers.modeling\_utils.PreTrainedModel] = None  )

Parameters

* **args** (`transformers.training_args.TrainingArguments`) —
  The training arguments for the training session.

Returns the optimizer class and optimizer parameters based on the training arguments.

#### get\_optimizer\_group

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1362)

( param: typing.Union[str, torch.nn.parameter.Parameter, NoneType] = None  )

Parameters

* **param** (`str` or `torch.nn.parameter.Parameter`, *optional*) —
  The parameter for which optimizer group needs to be returned.

Returns optimizer group for a parameter if given, else returns all optimizer groups for params.

#### get\_test\_dataloader

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1235)

( test\_dataset: Dataset  )

Parameters

* **test\_dataset** (`torch.utils.data.Dataset`, *optional*) —
  The test dataset to use. If it is a [Dataset](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the
  `model.forward()` method are automatically removed. It must implement `__len__`.

Returns the test `~torch.utils.data.DataLoader`.

Subclass and override this method if you want to inject some custom behavior.

#### get\_total\_train\_batch\_size

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L2349)

( args  )

Calculates total batch size (micro\_batch *grad\_accum* dp\_world\_size).

Note: Only considers DP and TP (dp\_world\_size = world\_size // tp\_size).

#### get\_tp\_size

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L2335)

( )

Get the tensor parallel size from either the model or DeepSpeed config.

#### get\_train\_dataloader

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1131)

( )

Returns the training `~torch.utils.data.DataLoader`.

Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
training if necessary) otherwise.

Subclass and override this method if you want to inject some custom behavior.

#### hyperparameter\_search

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L3685)

( hp\_space: typing.Optional[typing.Callable[[ForwardRef('optuna.Trial')], dict[str, float]]] = None compute\_objective: typing.Optional[typing.Callable[[dict[str, float]], float]] = None n\_trials: int = 20 direction: typing.Union[str, list[str]] = 'minimize' backend: typing.Union[ForwardRef('str'), transformers.trainer\_utils.HPSearchBackend, NoneType] = None hp\_name: typing.Optional[typing.Callable[[ForwardRef('optuna.Trial')], str]] = None \*\*kwargs  ) → [`trainer_utils.BestRun` or `list[trainer_utils.BestRun]`]

Parameters

* **hp\_space** (`Callable[["optuna.Trial"], dict[str, float]]`, *optional*) —
  A function that defines the hyperparameter search space. Will default to
  `default_hp_space_optuna()` or `default_hp_space_ray()` or
  `default_hp_space_sigopt()` depending on your backend.
* **compute\_objective** (`Callable[[dict[str, float]], float]`, *optional*) —
  A function computing the objective to minimize or maximize from the metrics returned by the `evaluate`
  method. Will default to `default_compute_objective()`.
* **n\_trials** (`int`, *optional*, defaults to 100) —
  The number of trial runs to test.
* **direction** (`str` or `list[str]`, *optional*, defaults to `"minimize"`) —
  If it’s single objective optimization, direction is `str`, can be `"minimize"` or `"maximize"`, you
  should pick `"minimize"` when optimizing the validation loss, `"maximize"` when optimizing one or
  several metrics. If it’s multi objectives optimization, direction is `list[str]`, can be List of
  `"minimize"` and `"maximize"`, you should pick `"minimize"` when optimizing the validation loss,
  `"maximize"` when optimizing one or several metrics.
* **backend** (`str` or `~training_utils.HPSearchBackend`, *optional*) —
  The backend to use for hyperparameter search. Will default to optuna or Ray Tune or SigOpt, depending
  on which one is installed. If all are installed, will default to optuna.
* **hp\_name** (`Callable[["optuna.Trial"], str]]`, *optional*) —
  A function that defines the trial/run name. Will default to None.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional keyword arguments for each backend:
  + `optuna`: parameters from
    [optuna.study.create\_study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html)
    and also the parameters `timeout`, `n_jobs` and `gc_after_trial` from
    [optuna.study.Study.optimize](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize)
  + `ray`: parameters from [tune.run](https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run).
    If `resources_per_trial` is not set in the `kwargs`, it defaults to 1 CPU core and 1 GPU (if available).
    If `progress_reporter` is not set in the `kwargs`,
    [ray.tune.CLIReporter](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html) is used.
  + `sigopt`: the parameter `proxies` from
    [sigopt.Connection.set\_proxies](https://docs.sigopt.com/support/faq#how-do-i-use-sigopt-with-a-proxy).

Returns

[`trainer_utils.BestRun` or `list[trainer_utils.BestRun]`]

All the information about the best run or best
runs for multi-objective optimization. Experiment summary can be found in `run_summary` attribute for Ray
backend.

Launch an hyperparameter search using `optuna` or `Ray Tune` or `SigOpt`. The optimized quantity is determined
by `compute_objective`, which defaults to a function returning the evaluation loss when no metric is provided,
the sum of all metrics otherwise.

To use this method, you need to have provided a `model_init` when initializing your [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer): we need to
reinitialize the model at each new run. This is incompatible with the `optimizers` argument, so you need to
subclass [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) and override the method [create\_optimizer\_and\_scheduler()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.create_optimizer_and_scheduler) for custom
optimizer/scheduler.

#### init\_hf\_repo

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4927)

( token: typing.Optional[str] = None  )

Initializes a git repo in `self.args.hub_model_id`.

#### is\_local\_process\_zero

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4136)

( )

Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
machines) main process.

#### is\_world\_process\_zero

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4143)

( )

Whether or not this process is the global main process (when training in a distributed fashion on several
machines, this is only going to be `True` for one process).

#### log

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L3769)

( logs: dict start\_time: typing.Optional[float] = None  )

Parameters

* **logs** (`dict[str, float]`) —
  The values to log.
* **start\_time** (`Optional[float]`) —
  The start of training.

Log `logs` on the various objects watching training.

Subclass and override this method to inject custom behavior.

#### log\_metrics

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_pt_utils.py#L958)

( split metrics  )

Parameters

* **split** (`str`) —
  Mode/split name: one of `train`, `eval`, `test`
* **metrics** (`dict[str, float]`) —
  The metrics returned from train/evaluate/predictmetrics: metrics dict

Log metrics in a specially formatted way.

Under distributed environment this is done only for a process with rank 0.

Notes on memory reports:

In order to get memory usage report you need to install `psutil`. You can do that with `pip install psutil`.

Now when this method is run, you will see a report that will include:


```
init_mem_cpu_alloc_delta   =     1301MB
init_mem_cpu_peaked_delta  =      154MB
init_mem_gpu_alloc_delta   =      230MB
init_mem_gpu_peaked_delta  =        0MB
train_mem_cpu_alloc_delta  =     1345MB
train_mem_cpu_peaked_delta =        0MB
train_mem_gpu_alloc_delta  =      693MB
train_mem_gpu_peaked_delta =        7MB
```

**Understanding the reports:**

* the first segment, e.g., `train__`, tells you which stage the metrics are for. Reports starting with `init_`
  will be added to the first stage that gets run. So that if only evaluation is run, the memory usage for the
  `__init__` will be reported along with the `eval_` metrics.
* the third segment, is either `cpu` or `gpu`, tells you whether it’s the general RAM or the gpu0 memory
  metric.
* `*_alloc_delta` - is the difference in the used/allocated memory counter between the end and the start of the
  stage - it can be negative if a function released more memory than it allocated.
* `*_peaked_delta` - is any extra memory that was consumed and then freed - relative to the current allocated
  memory counter - it is never negative. When you look at the metrics of any stage you add up `alloc_delta` +
  `peaked_delta` and you know how much memory was needed to complete that stage.

The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the
main process does the bulk of work, but it could be not quite so if model parallel is used and then other GPUs may
use a different amount of gpu memory. This is also not the same under DataParallel where gpu0 may require much more
memory than the rest since it stores the gradient and optimizer states for all participating GPUs. Perhaps in the
future these reports will evolve to measure those too.

The CPU RAM metric measures RSS (Resident Set Size) includes both the memory which is unique to the process and the
memory shared with other processes. It is important to note that it does not include swapped out memory, so the
reports could be imprecise.

The CPU peak memory is measured using a sampling thread. Due to python’s GIL it may miss some of the peak memory if
that thread didn’t get a chance to run when the highest memory was used. Therefore this report can be less than
reality. Using `tracemalloc` would have reported the exact peak memory, but it doesn’t report memory allocations
outside of python. So if some C++ CUDA extension allocated its own memory it won’t be reported. And therefore it
was dropped in favor of the memory sampling approach, which reads the current process memory usage.

The GPU allocated and peak memory reporting is done with `torch.cuda.memory_allocated()` and
`torch.cuda.max_memory_allocated()`. This metric reports only “deltas” for pytorch-specific allocations, as
`torch.cuda` memory management system doesn’t track any memory allocated outside of pytorch. For example, the very
first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory.

Note that this tracker doesn’t account for memory allocations outside of [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer)’s `__init__`, `train`,
`evaluate` and `predict` calls.

Because `evaluation` calls may happen during `train`, we can’t handle nested invocations because
`torch.cuda.max_memory_allocated` is a single counter, so if it gets reset by a nested eval call, `train`’s tracker
will report incorrect info. If this [pytorch issue](https://github.com/pytorch/pytorch/issues/16266) gets resolved
it will be possible to change this class to be re-entrant. Until then we will only track the outer level of
`train`, `evaluate` and `predict` methods. Which means that if `eval` is called during `train`, it’s the latter
that will account for its memory usage and that of the former.

This also means that if any other tool that is used along the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) calls
`torch.cuda.reset_peak_memory_stats`, the gpu peak memory stats could be invalid. And the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) will disrupt
the normal behavior of any such tools that rely on calling `torch.cuda.reset_peak_memory_stats` themselves.

For best performance you may want to consider turning the memory profiling off for production runs.

#### metrics\_format

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_pt_utils.py#L932)

( metrics: dict  ) → metrics (`dict[str, float]`)

Parameters

* **metrics** (`dict[str, float]`) —
  The metrics returned from train/evaluate/predict

Returns

metrics (`dict[str, float]`)

The reformatted metrics

Reformat Trainer metrics values to a human-readable format.

#### num\_examples

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1865)

( dataloader: DataLoader  )

Helper to get number of samples in a `~torch.utils.data.DataLoader` by accessing its dataset. When
dataloader.dataset does not exist or has no length, estimates as best it can

#### num\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L1879)

( train\_dl: DataLoader max\_steps: typing.Optional[int] = None  )

Helper to get number of tokens in a `~torch.utils.data.DataLoader` by enumerating dataloader.

#### pop\_callback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L885)

( callback  ) → [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback)

Parameters

* **callback** (`type` or [`~transformers.TrainerCallback]`) —
  A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) class or an instance of a [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback). In the
  first case, will pop the first member of that class found in the list of callbacks.

Returns

[TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback)

The callback removed, if found.

Remove a callback from the current list of [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) and returns it.

If the callback is not found, returns `None` (and no error is raised).

#### predict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4505)

( test\_dataset: Dataset ignore\_keys: typing.Optional[list[str]] = None metric\_key\_prefix: str = 'test'  )

Parameters

* **test\_dataset** (`Dataset`) —
  Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
  `model.forward()` method are automatically removed. Has to implement the method `__len__`
* **ignore\_keys** (`list[str]`, *optional*) —
  A list of keys in the output of your model (if it is a dictionary) that should be ignored when
  gathering predictions.
* **metric\_key\_prefix** (`str`, *optional*, defaults to `"test"`) —
  An optional prefix to be used as the metrics key prefix. For example the metrics “bleu” will be named
  “test\_bleu” if the prefix is “test” (default)

Run prediction and returns predictions and potential metrics.

Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
will also return metrics, like in `evaluate()`.

If your predictions or labels have different sequence length (for instance because you’re doing dynamic padding
in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
one array. The padding index is -100.

Returns: *NamedTuple* A namedtuple with the following keys:

* predictions (`np.ndarray`): The predictions on `test_dataset`.
* label\_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
* metrics (`dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
  labels).

#### prediction\_loop

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L5178)

( dataloader: DataLoader description: str prediction\_loss\_only: typing.Optional[bool] = None ignore\_keys: typing.Optional[list[str]] = None metric\_key\_prefix: str = 'eval'  )

Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

Works both with or without labels.

#### prediction\_step

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4804)

( model: Module inputs: dict prediction\_loss\_only: bool ignore\_keys: typing.Optional[list[str]] = None  ) → tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]

Parameters

* **model** (`nn.Module`) —
  The model to evaluate.
* **inputs** (`dict[str, Union[torch.Tensor, Any]]`) —
  The inputs and targets of the model.

  The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
  argument `labels`. Check your model’s documentation for all accepted arguments.
* **prediction\_loss\_only** (`bool`) —
  Whether or not to return the loss only.
* **ignore\_keys** (`list[str]`, *optional*) —
  A list of keys in the output of your model (if it is a dictionary) that should be ignored when
  gathering predictions.

Returns

tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]

A tuple with the loss,
logits and labels (each being optional).

Perform an evaluation step on `model` using `inputs`.

Subclass and override to inject custom behavior.

#### propagate\_args\_to\_deepspeed

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L5542)

( auto\_find\_batch\_size = False  )

Sets values in the deepspeed plugin based on the Trainer args

#### push\_to\_hub

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L5094)

( commit\_message: typing.Optional[str] = 'End of training' blocking: bool = True token: typing.Optional[str] = None revision: typing.Optional[str] = None \*\*kwargs  )

Parameters

* **commit\_message** (`str`, *optional*, defaults to `"End of training"`) —
  Message to commit while pushing.
* **blocking** (`bool`, *optional*, defaults to `True`) —
  Whether the function should return only when the `git push` has finished.
* **token** (`str`, *optional*, defaults to `None`) —
  Token with write permission to overwrite Trainer’s original args.
* **revision** (`str`, *optional*) —
  The git revision to commit from. Defaults to the head of the “main” branch.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional keyword arguments passed along to [create\_model\_card()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.create_model_card).

Upload `self.model` and `self.processing_class` to the 🤗 model hub on the repo `self.args.hub_model_id`.

#### remove\_callback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L901)

( callback  )

Parameters

* **callback** (`type` or [`~transformers.TrainerCallback]`) —
  A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) class or an instance of a [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback). In the
  first case, will remove the first member of that class found in the list of callbacks.

Remove a callback from the current list of [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback).

#### save\_metrics

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_pt_utils.py#L1048)

( split metrics combined = True  )

Parameters

* **split** (`str`) —
  Mode/split name: one of `train`, `eval`, `test`, `all`
* **metrics** (`dict[str, float]`) —
  The metrics returned from train/evaluate/predict
* **combined** (`bool`, *optional*, defaults to `True`) —
  Creates combined metrics by updating `all_results.json` with metrics of this call

Save metrics into a json file for that split, e.g. `train_results.json`.

Under distributed environment this is done only for a process with rank 0.

To understand the metrics please read the docstring of [log\_metrics()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.log_metrics). The only difference is that raw
unformatted numbers are saved in the current method.

#### save\_model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L4155)

( output\_dir: typing.Optional[str] = None \_internal\_call: bool = False  )

Will save the model, so you can reload it using `from_pretrained()`.

Will only save from the main process.

#### save\_state

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_pt_utils.py#L1086)

( )

Saves the Trainer state, since Trainer.save\_model saves only the tokenizer with the model.

Under distributed environment this is done only for a process with rank 0.

#### set\_initial\_training\_values

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L5623)

( args: TrainingArguments dataloader: DataLoader total\_train\_batch\_size: int  )

Calculates and returns the following values:

* `num_train_epochs`
* `num_update_steps_per_epoch`
* `num_examples`
* `num_train_samples`
* `epoch_based`
* `len_dataloader`
* `max_steps`

#### train

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L2216)

( resume\_from\_checkpoint: typing.Union[bool, str, NoneType] = None trial: typing.Union[ForwardRef('optuna.Trial'), dict[str, typing.Any], NoneType] = None ignore\_keys\_for\_eval: typing.Optional[list[str]] = None \*\*kwargs  )

Parameters

* **resume\_from\_checkpoint** (`str` or `bool`, *optional*) —
  If a `str`, local path to a saved checkpoint as saved by a previous instance of [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer). If a
  `bool` and equals `True`, load the last checkpoint in *args.output\_dir* as saved by a previous instance
  of [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer). If present, training will resume from the model/optimizer/scheduler states loaded here.
* **trial** (`optuna.Trial` or `dict[str, Any]`, *optional*) —
  The trial run or the hyperparameter dictionary for hyperparameter search.
* **ignore\_keys\_for\_eval** (`list[str]`, *optional*) —
  A list of keys in the output of your model (if it is a dictionary) that should be ignored when
  gathering predictions for evaluation during the training.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional keyword arguments used to hide deprecated arguments

Main training entry point.

#### training\_step

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer.py#L3970)

( model: Module inputs: dict num\_items\_in\_batch: typing.Optional[torch.Tensor] = None  ) → `torch.Tensor`

Parameters

* **model** (`nn.Module`) —
  The model to train.
* **inputs** (`dict[str, Union[torch.Tensor, Any]]`) —
  The inputs and targets of the model.

  The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
  argument `labels`. Check your model’s documentation for all accepted arguments.

Returns

`torch.Tensor`

The tensor with training loss on this batch.

Perform a training step on a batch of inputs.

Subclass and override to inject custom behavior.

## Seq2SeqTrainer

### class transformers.Seq2SeqTrainer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_seq2seq.py#L53)

( model: typing.Union[ForwardRef('PreTrainedModel'), torch.nn.modules.module.Module] = None args: TrainingArguments = None data\_collator: typing.Optional[ForwardRef('DataCollator')] = None train\_dataset: typing.Union[torch.utils.data.dataset.Dataset, ForwardRef('IterableDataset'), ForwardRef('datasets.Dataset'), NoneType] = None eval\_dataset: typing.Union[torch.utils.data.dataset.Dataset, dict[str, torch.utils.data.dataset.Dataset], NoneType] = None processing\_class: typing.Union[ForwardRef('PreTrainedTokenizerBase'), ForwardRef('BaseImageProcessor'), ForwardRef('FeatureExtractionMixin'), ForwardRef('ProcessorMixin'), NoneType] = None model\_init: typing.Optional[typing.Callable[[], ForwardRef('PreTrainedModel')]] = None compute\_loss\_func: typing.Optional[typing.Callable] = None compute\_metrics: typing.Optional[typing.Callable[[ForwardRef('EvalPrediction')], dict]] = None callbacks: typing.Optional[list['TrainerCallback']] = None optimizers: tuple = (None, None) preprocess\_logits\_for\_metrics: typing.Optional[typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None  )

#### evaluate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_seq2seq.py#L137)

( eval\_dataset: typing.Optional[torch.utils.data.dataset.Dataset] = None ignore\_keys: typing.Optional[list[str]] = None metric\_key\_prefix: str = 'eval' \*\*gen\_kwargs  )

Parameters

* **eval\_dataset** (`Dataset`, *optional*) —
  Pass a dataset if you wish to override `self.eval_dataset`. If it is an [Dataset](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset), columns
  not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
  method.
* **ignore\_keys** (`list[str]`, *optional*) —
  A list of keys in the output of your model (if it is a dictionary) that should be ignored when
  gathering predictions.
* **metric\_key\_prefix** (`str`, *optional*, defaults to `"eval"`) —
  An optional prefix to be used as the metrics key prefix. For example the metrics “bleu” will be named
  “eval\_bleu” if the prefix is `"eval"` (default)
* **max\_length** (`int`, *optional*) —
  The maximum target length to use when predicting with the generate method.
* **num\_beams** (`int`, *optional*) —
  Number of beams for beam search that will be used when predicting with the generate method. 1 means no
  beam search.
* **gen\_kwargs** —
  Additional `generate` specific kwargs.

Run evaluation and returns metrics.

The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
(pass it to the init `compute_metrics` argument).

You can also subclass and override this method to inject custom behavior.

#### predict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_seq2seq.py#L193)

( test\_dataset: Dataset ignore\_keys: typing.Optional[list[str]] = None metric\_key\_prefix: str = 'test' \*\*gen\_kwargs  )

Parameters

* **test\_dataset** (`Dataset`) —
  Dataset to run the predictions on. If it is a [Dataset](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the
  `model.forward()` method are automatically removed. Has to implement the method `__len__`
* **ignore\_keys** (`list[str]`, *optional*) —
  A list of keys in the output of your model (if it is a dictionary) that should be ignored when
  gathering predictions.
* **metric\_key\_prefix** (`str`, *optional*, defaults to `"eval"`) —
  An optional prefix to be used as the metrics key prefix. For example the metrics “bleu” will be named
  “eval\_bleu” if the prefix is `"eval"` (default)
* **max\_length** (`int`, *optional*) —
  The maximum target length to use when predicting with the generate method.
* **num\_beams** (`int`, *optional*) —
  Number of beams for beam search that will be used when predicting with the generate method. 1 means no
  beam search.
* **gen\_kwargs** —
  Additional `generate` specific kwargs.

Run prediction and returns predictions and potential metrics.

Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
will also return metrics, like in `evaluate()`.

If your predictions or labels have different sequence lengths (for instance because you’re doing dynamic
padding in a token classification task) the predictions will be padded (on the right) to allow for
concatenation into one array. The padding index is -100.

Returns: *NamedTuple* A namedtuple with the following keys:

* predictions (`np.ndarray`): The predictions on `test_dataset`.
* label\_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
* metrics (`dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
  labels).

## TrainingArguments

### class transformers.TrainingArguments

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L215)

( output\_dir: typing.Optional[str] = None overwrite\_output\_dir: bool = False do\_train: bool = False do\_eval: bool = False do\_predict: bool = False eval\_strategy: typing.Union[transformers.trainer\_utils.IntervalStrategy, str] = 'no' prediction\_loss\_only: bool = False per\_device\_train\_batch\_size: int = 8 per\_device\_eval\_batch\_size: int = 8 per\_gpu\_train\_batch\_size: typing.Optional[int] = None per\_gpu\_eval\_batch\_size: typing.Optional[int] = None gradient\_accumulation\_steps: int = 1 eval\_accumulation\_steps: typing.Optional[int] = None eval\_delay: typing.Optional[float] = 0 torch\_empty\_cache\_steps: typing.Optional[int] = None learning\_rate: float = 5e-05 weight\_decay: float = 0.0 adam\_beta1: float = 0.9 adam\_beta2: float = 0.999 adam\_epsilon: float = 1e-08 max\_grad\_norm: float = 1.0 num\_train\_epochs: float = 3.0 max\_steps: int = -1 lr\_scheduler\_type: typing.Union[transformers.trainer\_utils.SchedulerType, str] = 'linear' lr\_scheduler\_kwargs: typing.Union[dict[str, typing.Any], str, NoneType] = <factory> warmup\_ratio: float = 0.0 warmup\_steps: int = 0 log\_level: str = 'passive' log\_level\_replica: str = 'warning' log\_on\_each\_node: bool = True logging\_dir: typing.Optional[str] = None logging\_strategy: typing.Union[transformers.trainer\_utils.IntervalStrategy, str] = 'steps' logging\_first\_step: bool = False logging\_steps: float = 500 logging\_nan\_inf\_filter: bool = True save\_strategy: typing.Union[transformers.trainer\_utils.SaveStrategy, str] = 'steps' save\_steps: float = 500 save\_total\_limit: typing.Optional[int] = None save\_safetensors: typing.Optional[bool] = True save\_on\_each\_node: bool = False save\_only\_model: bool = False restore\_callback\_states\_from\_checkpoint: bool = False no\_cuda: bool = False use\_cpu: bool = False use\_mps\_device: bool = False seed: int = 42 data\_seed: typing.Optional[int] = None jit\_mode\_eval: bool = False use\_ipex: bool = False bf16: bool = False fp16: bool = False fp16\_opt\_level: str = 'O1' half\_precision\_backend: str = 'auto' bf16\_full\_eval: bool = False fp16\_full\_eval: bool = False tf32: typing.Optional[bool] = None local\_rank: int = -1 ddp\_backend: typing.Optional[str] = None tpu\_num\_cores: typing.Optional[int] = None tpu\_metrics\_debug: bool = False debug: typing.Union[str, list[transformers.debug\_utils.DebugOption]] = '' dataloader\_drop\_last: bool = False eval\_steps: typing.Optional[float] = None dataloader\_num\_workers: int = 0 dataloader\_prefetch\_factor: typing.Optional[int] = None past\_index: int = -1 run\_name: typing.Optional[str] = None disable\_tqdm: typing.Optional[bool] = None remove\_unused\_columns: typing.Optional[bool] = True label\_names: typing.Optional[list[str]] = None load\_best\_model\_at\_end: typing.Optional[bool] = False metric\_for\_best\_model: typing.Optional[str] = None greater\_is\_better: typing.Optional[bool] = None ignore\_data\_skip: bool = False fsdp: typing.Union[list[transformers.trainer\_utils.FSDPOption], str, NoneType] = '' fsdp\_min\_num\_params: int = 0 fsdp\_config: typing.Union[dict[str, typing.Any], str, NoneType] = None fsdp\_transformer\_layer\_cls\_to\_wrap: typing.Optional[str] = None accelerator\_config: typing.Union[dict, str, NoneType] = None parallelism\_config: typing.Optional[ForwardRef('ParallelismConfig')] = None deepspeed: typing.Union[dict, str, NoneType] = None label\_smoothing\_factor: float = 0.0 optim: typing.Union[transformers.training\_args.OptimizerNames, str] = 'adamw\_torch\_fused' optim\_args: typing.Optional[str] = None adafactor: bool = False group\_by\_length: bool = False length\_column\_name: typing.Optional[str] = 'length' report\_to: typing.Union[NoneType, str, list[str]] = None ddp\_find\_unused\_parameters: typing.Optional[bool] = None ddp\_bucket\_cap\_mb: typing.Optional[int] = None ddp\_broadcast\_buffers: typing.Optional[bool] = None dataloader\_pin\_memory: bool = True dataloader\_persistent\_workers: bool = False skip\_memory\_metrics: bool = True use\_legacy\_prediction\_loop: bool = False push\_to\_hub: bool = False resume\_from\_checkpoint: typing.Optional[str] = None hub\_model\_id: typing.Optional[str] = None hub\_strategy: typing.Union[transformers.trainer\_utils.HubStrategy, str] = 'every\_save' hub\_token: typing.Optional[str] = None hub\_private\_repo: typing.Optional[bool] = None hub\_always\_push: bool = False hub\_revision: typing.Optional[str] = None gradient\_checkpointing: bool = False gradient\_checkpointing\_kwargs: typing.Union[dict[str, typing.Any], str, NoneType] = None include\_inputs\_for\_metrics: bool = False include\_for\_metrics: list = <factory> eval\_do\_concat\_batches: bool = True fp16\_backend: str = 'auto' push\_to\_hub\_model\_id: typing.Optional[str] = None push\_to\_hub\_organization: typing.Optional[str] = None push\_to\_hub\_token: typing.Optional[str] = None mp\_parameters: str = '' auto\_find\_batch\_size: bool = False full\_determinism: bool = False torchdynamo: typing.Optional[str] = None ray\_scope: typing.Optional[str] = 'last' ddp\_timeout: int = 1800 torch\_compile: bool = False torch\_compile\_backend: typing.Optional[str] = None torch\_compile\_mode: typing.Optional[str] = None include\_tokens\_per\_second: typing.Optional[bool] = False include\_num\_input\_tokens\_seen: typing.Optional[bool] = False neftune\_noise\_alpha: typing.Optional[float] = None optim\_target\_modules: typing.Union[NoneType, str, list[str]] = None batch\_eval\_metrics: bool = False eval\_on\_start: bool = False use\_liger\_kernel: typing.Optional[bool] = False liger\_kernel\_config: typing.Optional[dict[str, bool]] = None eval\_use\_gather\_object: typing.Optional[bool] = False average\_tokens\_across\_devices: typing.Optional[bool] = True  )

Parameters

* **output\_dir** (`str`, *optional*, defaults to `"trainer_output"`) —
  The output directory where the model predictions and checkpoints will be written.
* **overwrite\_output\_dir** (`bool`, *optional*, defaults to `False`) —
  If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir`
  points to a checkpoint directory.
* **do\_train** (`bool`, *optional*, defaults to `False`) —
  Whether to run training or not. This argument is not directly used by [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), it’s intended to be used
  by your training/evaluation scripts instead. See the [example
  scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
* **do\_eval** (`bool`, *optional*) —
  Whether to run evaluation on the validation set or not. Will be set to `True` if `eval_strategy` is
  different from `"no"`. This argument is not directly used by [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), it’s intended to be used by your
  training/evaluation scripts instead. See the [example
  scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
* **do\_predict** (`bool`, *optional*, defaults to `False`) —
  Whether to run predictions on the test set or not. This argument is not directly used by [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), it’s
  intended to be used by your training/evaluation scripts instead. See the [example
  scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
* **eval\_strategy** (`str` or [IntervalStrategy](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.IntervalStrategy), *optional*, defaults to `"no"`) —
  The evaluation strategy to adopt during training. Possible values are:
  + `"no"`: No evaluation is done during training.
  + `"steps"`: Evaluation is done (and logged) every `eval_steps`.
  + `"epoch"`: Evaluation is done at the end of each epoch.
* **prediction\_loss\_only** (`bool`, *optional*, defaults to `False`) —
  When performing evaluation and generating predictions, only returns the loss.
* **per\_device\_train\_batch\_size** (`int`, *optional*, defaults to 8) —
  The batch size *per device*. The **global batch size** is computed as:
  `per_device_train_batch_size * number_of_devices` in multi-GPU or distributed setups.
* **per\_device\_eval\_batch\_size** (`int`, *optional*, defaults to 8) —
  The batch size per device accelerator core/CPU for evaluation.
* **gradient\_accumulation\_steps** (`int`, *optional*, defaults to 1) —
  Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

  When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging,
  evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.
* **eval\_accumulation\_steps** (`int`, *optional*) —
  Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
  left unset, the whole predictions are accumulated on the device accelerator before being moved to the CPU (faster but
  requires more memory).
* **eval\_delay** (`float`, *optional*) —
  Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
  eval\_strategy.
* **torch\_empty\_cache\_steps** (`int`, *optional*) —
  Number of steps to wait before calling `torch.<device>.empty_cache()`. If left unset or set to None, cache will not be emptied.

  This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of about [10% slower performance](https://github.com/huggingface/transformers/issues/31372).
* **learning\_rate** (`float`, *optional*, defaults to 5e-5) —
  The initial learning rate for `AdamW` optimizer.
* **weight\_decay** (`float`, *optional*, defaults to 0) —
  The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in `AdamW`
  optimizer.
* **adam\_beta1** (`float`, *optional*, defaults to 0.9) —
  The beta1 hyperparameter for the `AdamW` optimizer.
* **adam\_beta2** (`float`, *optional*, defaults to 0.999) —
  The beta2 hyperparameter for the `AdamW` optimizer.
* **adam\_epsilon** (`float`, *optional*, defaults to 1e-8) —
  The epsilon hyperparameter for the `AdamW` optimizer.
* **max\_grad\_norm** (`float`, *optional*, defaults to 1.0) —
  Maximum gradient norm (for gradient clipping).
* **num\_train\_epochs(`float`,** *optional*, defaults to 3.0) —
  Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
  the last epoch before stopping training).
* **max\_steps** (`int`, *optional*, defaults to -1) —
  If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
  For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
  `max_steps` is reached.
* **lr\_scheduler\_type** (`str` or [SchedulerType](/docs/transformers/v4.56.2/en/main_classes/optimizer_schedules#transformers.SchedulerType), *optional*, defaults to `"linear"`) —
  The scheduler type to use. See the documentation of [SchedulerType](/docs/transformers/v4.56.2/en/main_classes/optimizer_schedules#transformers.SchedulerType) for all possible values.
* **lr\_scheduler\_kwargs** (‘dict’, *optional*, defaults to {}) —
  The extra arguments for the lr\_scheduler. See the documentation of each scheduler for possible values.
* **warmup\_ratio** (`float`, *optional*, defaults to 0.0) —
  Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
* **warmup\_steps** (`int`, *optional*, defaults to 0) —
  Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.
* **log\_level** (`str`, *optional*, defaults to `passive`) —
  Logger log level to use on the main process. Possible choices are the log levels as strings: ‘debug’,
  ‘info’, ‘warning’, ‘error’ and ‘critical’, plus a ‘passive’ level which doesn’t set anything and keeps the
  current log level for the Transformers library (which will be `"warning"` by default).
* **log\_level\_replica** (`str`, *optional*, defaults to `"warning"`) —
  Logger log level to use on replicas. Same choices as `log_level`”
* **log\_on\_each\_node** (`bool`, *optional*, defaults to `True`) —
  In multinode distributed training, whether to log using `log_level` once per node, or only on the main
  node.
* **logging\_dir** (`str`, *optional*) —
  [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
  \*output\_dir/runs/**CURRENT\_DATETIME\_HOSTNAME\***.
* **logging\_strategy** (`str` or [IntervalStrategy](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.IntervalStrategy), *optional*, defaults to `"steps"`) —
  The logging strategy to adopt during training. Possible values are:
  + `"no"`: No logging is done during training.
  + `"epoch"`: Logging is done at the end of each epoch.
  + `"steps"`: Logging is done every `logging_steps`.
* **logging\_first\_step** (`bool`, *optional*, defaults to `False`) —
  Whether to log the first `global_step` or not.
* **logging\_steps** (`int` or `float`, *optional*, defaults to 500) —
  Number of update steps between two logs if `logging_strategy="steps"`. Should be an integer or a float in
  range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
* **logging\_nan\_inf\_filter** (`bool`, *optional*, defaults to `True`) —
  Whether to filter `nan` and `inf` losses for logging. If set to `True` the loss of every step that is `nan`
  or `inf` is filtered and the average loss of the current logging window is taken instead.

  `logging_nan_inf_filter` only influences the logging of loss values, it does not change the behavior the
  gradient is computed or applied to the model.
* **save\_strategy** (`str` or `SaveStrategy`, *optional*, defaults to `"steps"`) —
  The checkpoint save strategy to adopt during training. Possible values are:
  + `"no"`: No save is done during training.
  + `"epoch"`: Save is done at the end of each epoch.
  + `"steps"`: Save is done every `save_steps`.
  + `"best"`: Save is done whenever a new `best_metric` is achieved.

  If `"epoch"` or `"steps"` is chosen, saving will also be performed at the
  very end of training, always.
* **save\_steps** (`int` or `float`, *optional*, defaults to 500) —
  Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Should be an integer or a
  float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
* **save\_total\_limit** (`int`, *optional*) —
  If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
  `output_dir`. When `load_best_model_at_end` is enabled, the “best” checkpoint according to
  `metric_for_best_model` will always be retained in addition to the most recent ones. For example, for
  `save_total_limit=5` and `load_best_model_at_end`, the four last checkpoints will always be retained
  alongside the best model. When `save_total_limit=1` and `load_best_model_at_end`, it is possible that two
  checkpoints are saved: the last one and the best one (if they are different).
* **save\_safetensors** (`bool`, *optional*, defaults to `True`) —
  Use [safetensors](https://huggingface.co/docs/safetensors) saving and loading for state dicts instead of
  default `torch.load` and `torch.save`.
* **save\_on\_each\_node** (`bool`, *optional*, defaults to `False`) —
  When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on
  the main one.

  This should not be activated when the different nodes use the same storage as the files will be saved with
  the same names for each node.
* **save\_only\_model** (`bool`, *optional*, defaults to `False`) —
  When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state.
  Note that when this is true, you won’t be able to resume training from checkpoint.
  This enables you to save storage by not storing the optimizer, scheduler & rng state.
  You can only load the model using `from_pretrained` with this option set to `True`.
* **restore\_callback\_states\_from\_checkpoint** (`bool`, *optional*, defaults to `False`) —
  Whether to restore the callback states from the checkpoint. If `True`, will override
  callbacks passed to the `Trainer` if they exist in the checkpoint.”
* **use\_cpu** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use cpu. If set to False, we will use cuda or mps device if available.
* **seed** (`int`, *optional*, defaults to 42) —
  Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
  `~Trainer.model_init` function to instantiate the model if it has some randomly initialized parameters.
* **data\_seed** (`int`, *optional*) —
  Random seed to be used with data samplers. If not set, random generators for data sampling will use the
  same seed as `seed`. This can be used to ensure reproducibility of data sampling, independent of the model
  seed.
* **jit\_mode\_eval** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use PyTorch jit trace for inference.
* **use\_ipex** (`bool`, *optional*, defaults to `False`) —
  Use Intel extension for PyTorch when it is available. [IPEX
  installation](https://github.com/intel/intel-extension-for-pytorch).
* **bf16** (`bool`, *optional*, defaults to `False`) —
  Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher
  NVIDIA architecture or Intel XPU or using CPU (use\_cpu) or Ascend NPU. This is an experimental API and it may change.
* **fp16** (`bool`, *optional*, defaults to `False`) —
  Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
* **fp16\_opt\_level** (`str`, *optional*, defaults to ‘O1’) —
  For `fp16` training, Apex AMP optimization level selected in [‘O0’, ‘O1’, ‘O2’, and ‘O3’]. See details on
  the [Apex documentation](https://nvidia.github.io/apex/amp).
* **fp16\_backend** (`str`, *optional*, defaults to `"auto"`) —
  This argument is deprecated. Use `half_precision_backend` instead.
* **half\_precision\_backend** (`str`, *optional*, defaults to `"auto"`) —
  The backend to use for mixed precision training. Must be one of `"auto", "apex", "cpu_amp"`. `"auto"` will
  use CPU/CUDA AMP or APEX depending on the PyTorch version detected, while the other choices will force the
  requested backend.
* **bf16\_full\_eval** (`bool`, *optional*, defaults to `False`) —
  Whether to use full bfloat16 evaluation instead of 32-bit. This will be faster and save memory but can harm
  metric values. This is an experimental API and it may change.
* **fp16\_full\_eval** (`bool`, *optional*, defaults to `False`) —
  Whether to use full float16 evaluation instead of 32-bit. This will be faster and save memory but can harm
  metric values.
* **tf32** (`bool`, *optional*) —
  Whether to enable the TF32 mode, available in Ampere and newer GPU architectures. The default value depends
  on PyTorch’s version default of `torch.backends.cuda.matmul.allow_tf32`. For more details please refer to
  the [TF32](https://huggingface.co/docs/transformers/perf_train_gpu_one#tf32) documentation. This is an
  experimental API and it may change.
* **local\_rank** (`int`, *optional*, defaults to -1) —
  Rank of the process during distributed training.
* **ddp\_backend** (`str`, *optional*) —
  The backend to use for distributed training. Must be one of `"nccl"`, `"mpi"`, `"ccl"`, `"gloo"`, `"hccl"`.
* **tpu\_num\_cores** (`int`, *optional*) —
  When training on TPU, the number of TPU cores (automatically passed by launcher script).
* **dataloader\_drop\_last** (`bool`, *optional*, defaults to `False`) —
  Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
  or not.
* **eval\_steps** (`int` or `float`, *optional*) —
  Number of update steps between two evaluations if `eval_strategy="steps"`. Will default to the same
  value as `logging_steps` if not set. Should be an integer or a float in range `[0,1)`. If smaller than 1,
  will be interpreted as ratio of total training steps.
* **dataloader\_num\_workers** (`int`, *optional*, defaults to 0) —
  Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
  main process.
* **past\_index** (`int`, *optional*, defaults to -1) —
  Some models like [TransformerXL](../model_doc/transformerxl) or [XLNet](../model_doc/xlnet) can make use of
  the past hidden states for their predictions. If this argument is set to a positive int, the `Trainer` will
  use the corresponding output (usually index 2) as the past state and feed it to the model at the next
  training step under the keyword argument `mems`.
* **run\_name** (`str`, *optional*, defaults to `output_dir`) —
  A descriptor for the run. Typically used for [trackio](https://github.com/gradio-app/trackio),
  [wandb](https://www.wandb.com/), [mlflow](https://www.mlflow.org/), [comet](https://www.comet.com/site) and
  [swanlab](https://swanlab.cn) logging. If not specified, will be the same as `output_dir`.
* **disable\_tqdm** (`bool`, *optional*) —
  Whether or not to disable the tqdm progress bars and table of metrics produced by
  `~notebook.NotebookTrainingTracker` in Jupyter Notebooks. Will default to `True` if the logging level is
  set to warn or lower (default), `False` otherwise.
* **remove\_unused\_columns** (`bool`, *optional*, defaults to `True`) —
  Whether or not to automatically remove the columns unused by the model forward method.
* **label\_names** (`list[str]`, *optional*) —
  The list of keys in your dictionary of inputs that correspond to the labels.

  Will eventually default to the list of argument names accepted by the model that contain the word “label”,
  except if the model used is one of the `XxxForQuestionAnswering` in which case it will also include the
  `["start_positions", "end_positions"]` keys.

  You should only specify `label_names` if you’re using custom label names or if your model’s `forward` consumes multiple label tensors (e.g., extractive QA).
* **load\_best\_model\_at\_end** (`bool`, *optional*, defaults to `False`) —
  Whether or not to load the best model found during training at the end of training. When this option is
  enabled, the best checkpoint will always be saved. See
  [`save_total_limit`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_total_limit)
  for more.

  When set to `True`, the parameters `save_strategy` needs to be the same as `eval_strategy`, and in
  the case it is “steps”, `save_steps` must be a round multiple of `eval_steps`.
* **metric\_for\_best\_model** (`str`, *optional*) —
  Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different
  models. Must be the name of a metric returned by the evaluation with or without the prefix `"eval_"`.

  If not specified, this will default to `"loss"` when either `load_best_model_at_end == True`
  or `lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU` (to use the evaluation loss).

  If you set this value, `greater_is_better` will default to `True` unless the name ends with “loss”.
  Don’t forget to set it to `False` if your metric is better when lower.
* **greater\_is\_better** (`bool`, *optional*) —
  Use in conjunction with `load_best_model_at_end` and `metric_for_best_model` to specify if better models
  should have a greater metric or not. Will default to:
  + `True` if `metric_for_best_model` is set to a value that doesn’t end in `"loss"`.
  + `False` if `metric_for_best_model` is not set, or set to a value that ends in `"loss"`.
* **ignore\_data\_skip** (`bool`, *optional*, defaults to `False`) —
  When resuming training, whether or not to skip the epochs and batches to get the data loading at the same
  stage as in the previous training. If set to `True`, the training will begin faster (as that skipping step
  can take a long time) but will not yield the same results as the interrupted training would have.
* **fsdp** (`bool`, `str` or list of `FSDPOption`, *optional*, defaults to `''`) —
  Use PyTorch Distributed Parallel Training (in distributed training only).

  A list of options along the following:

  + `"full_shard"`: Shard parameters, gradients and optimizer states.
  + `"shard_grad_op"`: Shard optimizer states and gradients.
  + `"hybrid_shard"`: Apply `FULL_SHARD` within a node, and replicate parameters across nodes.
  + `"hybrid_shard_zero2"`: Apply `SHARD_GRAD_OP` within a node, and replicate parameters across nodes.
  + `"offload"`: Offload parameters and gradients to CPUs (only compatible with `"full_shard"` and
    `"shard_grad_op"`).
  + `"auto_wrap"`: Automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`.
* **fsdp\_config** (`str` or `dict`, *optional*) —
  Config to be used with fsdp (Pytorch Distributed Parallel Training). The value is either a location of
  fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`.

  A List of config and its options:

  + min\_num\_params (`int`, *optional*, defaults to `0`):
    FSDP’s minimum number of parameters for Default Auto Wrapping. (useful only when `fsdp` field is
    passed).
  + transformer\_layer\_cls\_to\_wrap (`list[str]`, *optional*):
    List of transformer layer class names (case-sensitive) to wrap, e.g, `BertLayer`, `GPTJBlock`,
    `T5Block` … (useful only when `fsdp` flag is passed).
  + backward\_prefetch (`str`, *optional*)
    FSDP’s backward prefetch mode. Controls when to prefetch next set of parameters (useful only when
    `fsdp` field is passed).

    A list of options along the following:

    - `"backward_pre"` : Prefetches the next set of parameters before the current set of parameter’s
      gradient
      computation.
    - `"backward_post"` : This prefetches the next set of parameters after the current set of
      parameter’s
      gradient computation.
  + forward\_prefetch (`bool`, *optional*, defaults to `False`)
    FSDP’s forward prefetch mode (useful only when `fsdp` field is passed).
    If `"True"`, then FSDP explicitly prefetches the next upcoming all-gather while executing in the
    forward pass.
  + limit\_all\_gathers (`bool`, *optional*, defaults to `False`)
    FSDP’s limit\_all\_gathers (useful only when `fsdp` field is passed).
    If `"True"`, FSDP explicitly synchronizes the CPU thread to prevent too many in-flight
    all-gathers.
  + use\_orig\_params (`bool`, *optional*, defaults to `True`)
    If `"True"`, allows non-uniform `requires_grad` during init, which means support for interspersed
    frozen and trainable parameters. Useful in cases such as parameter-efficient fine-tuning. Please
    refer this
    [blog](<https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019>
  + sync\_module\_states (`bool`, *optional*, defaults to `True`)
    If `"True"`, each individually wrapped FSDP unit will broadcast module parameters from rank 0 to
    ensure they are the same across all ranks after initialization
  + cpu\_ram\_efficient\_loading (`bool`, *optional*, defaults to `False`)
    If `"True"`, only the first process loads the pretrained model checkpoint while all other processes
    have empty weights. When this setting as `"True"`, `sync_module_states` also must to be `"True"`,
    otherwise all the processes except the main process would have random weights leading to unexpected
    behaviour during training.
  + activation\_checkpointing (`bool`, *optional*, defaults to `False`):
    If `"True"`, activation checkpointing is a technique to reduce memory usage by clearing activations of
    certain layers and recomputing them during a backward pass. Effectively, this trades extra
    computation time for reduced memory usage.
  + xla (`bool`, *optional*, defaults to `False`):
    Whether to use PyTorch/XLA Fully Sharded Data Parallel Training. This is an experimental feature
    and its API may evolve in the future.
  + xla\_fsdp\_settings (`dict`, *optional*)
    The value is a dictionary which stores the XLA FSDP wrapping parameters.

    For a complete list of options, please see [here](https://github.com/pytorch/xla/blob/master/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py).
  + xla\_fsdp\_grad\_ckpt (`bool`, *optional*, defaults to `False`):
    Will use gradient checkpointing over each nested XLA FSDP wrapped layer. This setting can only be
    used when the xla flag is set to true, and an auto wrapping policy is specified through
    fsdp\_min\_num\_params or fsdp\_transformer\_layer\_cls\_to\_wrap.
* **deepspeed** (`str` or `dict`, *optional*) —
  Use [Deepspeed](https://github.com/deepspeedai/DeepSpeed). This is an experimental feature and its API may
  evolve in the future. The value is either the location of DeepSpeed json config file (e.g.,
  `ds_config.json`) or an already loaded json file as a `dict`”

  If enabling any Zero-init, make sure that your model is not initialized until
  \*after\* initializing the `TrainingArguments`, else it will not be applied.
* **accelerator\_config** (`str`, `dict`, or `AcceleratorConfig`, *optional*) —
  Config to be used with the internal `Accelerator` implementation. The value is either a location of
  accelerator json config file (e.g., `accelerator_config.json`), an already loaded json file as `dict`,
  or an instance of `AcceleratorConfig`.

  A list of config and its options:

  + split\_batches (`bool`, *optional*, defaults to `False`):
    Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
    `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
    round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
    in your script multiplied by the number of processes.
  + dispatch\_batches (`bool`, *optional*):
    If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
    and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
    underlying dataset is an `IterableDataset`, `False` otherwise.
  + even\_batches (`bool`, *optional*, defaults to `True`):
    If set to `True`, in cases where the total batch size across all processes does not exactly divide the
    dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
    all workers.
  + use\_seedable\_sampler (`bool`, *optional*, defaults to `True`):
    Whether or not use a fully seedable random sampler (`accelerate.data_loader.SeedableRandomSampler`). Ensures
    training results are fully reproducible using a different sampling technique. While seed-to-seed results
    may differ, on average the differences are negligible when using multiple different seeds to compare. Should
    also be ran with `~utils.set_seed` for the best results.
  + use\_configured\_state (`bool`, *optional*, defaults to `False`):
    Whether or not to use a pre-configured `AcceleratorState` or `PartialState` defined before calling `TrainingArguments`.
    If `True`, an `Accelerator` or `PartialState` must be initialized. Note that by doing so, this could lead to issues
    with hyperparameter tuning.
* **parallelism\_config** (`ParallelismConfig`, *optional*) —
  Parallelism configuration for the training run. Requires Accelerate `1.10.1`
* **label\_smoothing\_factor** (`float`, *optional*, defaults to 0.0) —
  The label smoothing factor to use. Zero means no label smoothing, otherwise the underlying onehot-encoded
  labels are changed from 0s and 1s to `label_smoothing_factor/num_labels` and `1 - label_smoothing_factor + label_smoothing_factor/num_labels` respectively.
* **debug** (`str` or list of `DebugOption`, *optional*, defaults to `""`) —
  Enable one or more debug features. This is an experimental feature.

  Possible options are:

  + `"underflow_overflow"`: detects overflow in model’s input/outputs and reports the last frames that led to
    the event
  + `"tpu_metrics_debug"`: print debug metrics on TPU

  The options should be separated by whitespaces.
* **optim** (`str` or `training_args.OptimizerNames`, *optional*, defaults to `"adamw_torch"` (for torch>=2.8 `"adamw_torch_fused"`)) —
  The optimizer to use, such as “adamw\_torch”, “adamw\_torch\_fused”, “adamw\_apex\_fused”, “adamw\_anyprecision”,
  “adafactor”. See `OptimizerNames` in [training\_args.py](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)
  for a full list of optimizers.
* **optim\_args** (`str`, *optional*) —
  Optional arguments that are supplied to optimizers such as AnyPrecisionAdamW, AdEMAMix, and GaLore.
* **group\_by\_length** (`bool`, *optional*, defaults to `False`) —
  Whether or not to group together samples of roughly the same length in the training dataset (to minimize
  padding applied and be more efficient). Only useful if applying dynamic padding.
* **length\_column\_name** (`str`, *optional*, defaults to `"length"`) —
  Column name for precomputed lengths. If the column exists, grouping by length will use these values rather
  than computing them on train startup. Ignored unless `group_by_length` is `True` and the dataset is an
  instance of `Dataset`.
* **report\_to** (`str` or `list[str]`, *optional*, defaults to `"all"`) —
  The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
  `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"neptune"`,
  `"swanlab"`, `"tensorboard"`, `"trackio"` and `"wandb"`. Use `"all"` to report to all integrations
  installed, `"none"` for no integrations.
* **ddp\_find\_unused\_parameters** (`bool`, *optional*) —
  When using distributed training, the value of the flag `find_unused_parameters` passed to
  `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
* **ddp\_bucket\_cap\_mb** (`int`, *optional*) —
  When using distributed training, the value of the flag `bucket_cap_mb` passed to `DistributedDataParallel`.
* **ddp\_broadcast\_buffers** (`bool`, *optional*) —
  When using distributed training, the value of the flag `broadcast_buffers` passed to
  `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
* **dataloader\_pin\_memory** (`bool`, *optional*, defaults to `True`) —
  Whether you want to pin memory in data loaders or not. Will default to `True`.
* **dataloader\_persistent\_workers** (`bool`, *optional*, defaults to `False`) —
  If True, the data loader will not shut down the worker processes after a dataset has been consumed once.
  This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will
  increase RAM usage. Will default to `False`.
* **dataloader\_prefetch\_factor** (`int`, *optional*) —
  Number of batches loaded in advance by each worker.
  2 means there will be a total of 2 \* num\_workers batches prefetched across all workers.
* **skip\_memory\_metrics** (`bool`, *optional*, defaults to `True`) —
  Whether to skip adding of memory profiler reports to metrics. This is skipped by default because it slows
  down the training and evaluation speed.
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push the model to the Hub every time the model is saved. If this is activated,
  `output_dir` will begin a git directory synced with the repo (determined by `hub_model_id`) and the content
  will be pushed each time a save is triggered (depending on your `save_strategy`). Calling
  [save\_model()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.save_model) will also trigger a push.

  If `output_dir` exists, it needs to be a local clone of the repository to which the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) will be
  pushed.
* **resume\_from\_checkpoint** (`str`, *optional*) —
  The path to a folder with a valid checkpoint for your model. This argument is not directly used by
  [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), it’s intended to be used by your training/evaluation scripts instead. See the [example
  scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
* **hub\_model\_id** (`str`, *optional*) —
  The name of the repository to keep in sync with the local *output\_dir*. It can be a simple model ID in
  which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
  for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
  `"organization_name/model"`. Will default to `user_name/output_dir_name` with *output\_dir\_name* being the
  name of `output_dir`.

  Will default to the name of `output_dir`.
* **hub\_strategy** (`str` or `HubStrategy`, *optional*, defaults to `"every_save"`) —
  Defines the scope of what is pushed to the Hub and when. Possible values are:
  + `"end"`: push the model, its configuration, the processing class e.g. tokenizer (if passed along to the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer)) and a
    draft of a model card when the [save\_model()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.save_model) method is called.
  + `"every_save"`: push the model, its configuration, the processing class e.g. tokenizer (if passed along to the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer)) and
    a draft of a model card each time there is a model save. The pushes are asynchronous to not block
    training, and in case the save are very frequent, a new push is only attempted if the previous one is
    finished. A last push is made with the final model at the end of training.
  + `"checkpoint"`: like `"every_save"` but the latest checkpoint is also pushed in a subfolder named
    last-checkpoint, allowing you to resume training easily with
    `trainer.train(resume_from_checkpoint="last-checkpoint")`.
  + `"all_checkpoints"`: like `"checkpoint"` but all checkpoints are pushed like they appear in the output
    folder (so you will get one checkpoint folder per folder in your final repository)
* **hub\_token** (`str`, *optional*) —
  The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
  `hf auth login`.
* **hub\_private\_repo** (`bool`, *optional*) —
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
* **hub\_always\_push** (`bool`, *optional*, defaults to `False`) —
  Unless this is `True`, the `Trainer` will skip pushing a checkpoint when the previous push is not finished.
* **hub\_revision** (`str`, *optional*) —
  The revision to use when pushing to the Hub. Can be a branch name, a tag, or a commit hash.
* **gradient\_checkpointing** (`bool`, *optional*, defaults to `False`) —
  If True, use gradient checkpointing to save memory at the expense of slower backward pass.
* **gradient\_checkpointing\_kwargs** (`dict`, *optional*, defaults to `None`) —
  Key word arguments to be passed to the `gradient_checkpointing_enable` method.
* **include\_inputs\_for\_metrics** (`bool`, *optional*, defaults to `False`) —
  This argument is deprecated. Use `include_for_metrics` instead, e.g, `include_for_metrics = ["inputs"]`.
* **include\_for\_metrics** (`list[str]`, *optional*, defaults to `[]`) —
  Include additional data in the `compute_metrics` function if needed for metrics computation.
  Possible options to add to `include_for_metrics` list:
  + `"inputs"`: Input data passed to the model, intended for calculating input dependent metrics.
  + `"loss"`: Loss values computed during evaluation, intended for calculating loss dependent metrics.
* **eval\_do\_concat\_batches** (`bool`, *optional*, defaults to `True`) —
  Whether to recursively concat inputs/losses/labels/predictions across batches. If `False`,
  will instead store them as lists, with each batch kept separate.
* **auto\_find\_batch\_size** (`bool`, *optional*, defaults to `False`) —
  Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding
  CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)
* **full\_determinism** (`bool`, *optional*, defaults to `False`) —
  If `True`, [enable\_full\_determinism()](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.enable_full_determinism) is called instead of [set\_seed()](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.set_seed) to ensure reproducible results in
  distributed training. Important: this will negatively impact the performance, so only use it for debugging.
* **torchdynamo** (`str`, *optional*) —
  If set, the backend compiler for TorchDynamo. Possible choices are `"eager"`, `"aot_eager"`, `"inductor"`,
  `"nvfuser"`, `"aot_nvfuser"`, `"aot_cudagraphs"`, `"ofi"`, `"fx2trt"`, `"onnxrt"` and `"ipex"`.
* **ray\_scope** (`str`, *optional*, defaults to `"last"`) —
  The scope to use when doing hyperparameter search with Ray. By default, `"last"` will be used. Ray will
  then use the last checkpoint of all trials, compare those, and select the best one. However, other options
  are also available. See the [Ray documentation](https://docs.ray.io/en/latest/tune/api_docs/analysis.html#ray.tune.ExperimentAnalysis.get_best_trial) for
  more options.
* **ddp\_timeout** (`int`, *optional*, defaults to 1800) —
  The timeout for `torch.distributed.init_process_group` calls, used to avoid GPU socket timeouts when
  performing slow operations in distributed runnings. Please refer the [PyTorch documentation]
  (<https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>) for more
  information.
* **use\_mps\_device** (`bool`, *optional*, defaults to `False`) —
  This argument is deprecated.`mps` device will be used if it is available similar to `cuda` device.
* **torch\_compile** (`bool`, *optional*, defaults to `False`) —
  Whether or not to compile the model using PyTorch 2.0
  [`torch.compile`](https://pytorch.org/get-started/pytorch-2.0/).

  This will use the best defaults for the [`torch.compile`
  API](https://pytorch.org/docs/stable/generated/torch.compile.html?highlight=torch+compile#torch.compile).
  You can customize the defaults with the argument `torch_compile_backend` and `torch_compile_mode` but we
  don’t guarantee any of them will work as the support is progressively rolled in in PyTorch.

  This flag and the whole compile API is experimental and subject to change in future releases.
* **torch\_compile\_backend** (`str`, *optional*) —
  The backend to use in `torch.compile`. If set to any value, `torch_compile` will be set to `True`.

  Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.

  This flag is experimental and subject to change in future releases.
* **torch\_compile\_mode** (`str`, *optional*) —
  The mode to use in `torch.compile`. If set to any value, `torch_compile` will be set to `True`.

  Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.

  This flag is experimental and subject to change in future releases.
* **include\_tokens\_per\_second** (`bool`, *optional*) —
  Whether or not to compute the number of tokens per second per device for training speed metrics.

  This will iterate over the entire training dataloader once beforehand,

  and will slow down the entire process.
* **include\_num\_input\_tokens\_seen** (`bool`, *optional*) —
  Whether or not to track the number of input tokens seen throughout training.

  May be slower in distributed training as gather operations must be called.
* **neftune\_noise\_alpha** (`Optional[float]`) —
  If not `None`, this will activate NEFTune noise embeddings. This can drastically improve model performance
  for instruction fine-tuning. Check out the [original paper](https://huggingface.co/papers/2310.05914) and the
  [original code](https://github.com/neelsjain/NEFTune). Support transformers `PreTrainedModel` and also
  `PeftModel` from peft. The original paper used values in the range [5.0, 15.0].
* **optim\_target\_modules** (`Union[str, list[str]]`, *optional*) —
  The target modules to optimize, i.e. the module names that you would like to train.
  Currently used for the GaLore algorithm (<https://huggingface.co/papers/2403.03507>) and APOLLO algorithm (<https://huggingface.co/papers/2412.05270>).
  See GaLore implementation (<https://github.com/jiaweizzhao/GaLore>) and APOLLO implementation (<https://github.com/zhuhanqing/APOLLO>) for more details.
  You need to make sure to pass a valid GaLore or APOLLO optimizer, e.g., one of: “apollo\_adamw”, “galore\_adamw”, “galore\_adamw\_8bit”, “galore\_adafactor” and make sure that the target modules are `nn.Linear` modules only.
* **batch\_eval\_metrics** (`Optional[bool]`, defaults to `False`) —
  If set to `True`, evaluation will call compute\_metrics at the end of each batch to accumulate statistics
  rather than saving all eval logits in memory. When set to `True`, you must pass a compute\_metrics function
  that takes a boolean argument `compute_result`, which when passed `True`, will trigger the final global
  summary statistics from the batch-level summary statistics you’ve accumulated over the evaluation set.
* **eval\_on\_start** (`bool`, *optional*, defaults to `False`) —
  Whether to perform a evaluation step (sanity check) before the training to ensure the validation steps works correctly.
* **eval\_use\_gather\_object** (`bool`, *optional*, defaults to `False`) —
  Whether to run recursively gather object in a nested list/tuple/dictionary of objects from all devices. This should only be enabled if users are not just returning tensors, and this is actively discouraged by PyTorch.
* **use\_liger\_kernel** (`bool`, *optional*, defaults to `False`) —
  Whether enable [Liger](https://github.com/linkedin/Liger-Kernel) Kernel for LLM model training.
  It can effectively increase multi-GPU training throughput by ~20% and reduces memory usage by ~60%, works out of the box with
  flash attention, PyTorch FSDP, and Microsoft DeepSpeed. Currently, it supports llama, mistral, mixtral and gemma models.
* **liger\_kernel\_config** (`Optional[dict]`, *optional*) —
  Configuration to be used for Liger Kernel. When use\_liger\_kernel=True, this dict is passed as keyword arguments to the
  `_apply_liger_kernel_to_instance` function, which specifies which kernels to apply. Available options vary by model but typically
  include: ‘rope’, ‘swiglu’, ‘cross\_entropy’, ‘fused\_linear\_cross\_entropy’, ‘rms\_norm’, etc. If `None`, use the default kernel configurations.
* **average\_tokens\_across\_devices** (`bool`, *optional*, defaults to `True`) —
  Whether or not to average tokens across devices. If enabled, will use all\_reduce to synchronize
  num\_tokens\_in\_batch for precise loss calculation. Reference:
  <https://github.com/huggingface/transformers/issues/34242>

TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
itself**.

Using [HfArgumentParser](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.HfArgumentParser) we can turn this class into
[argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
command line.

#### get\_process\_log\_level

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2439)

( )

Returns the log level to be used depending on whether this process is the main process of node 0, main process
of node non-0, or a non-main process.

For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn’t do
anything) unless overridden by `log_level` argument.

For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
argument.

The choice between the main and replica process settings is made according to the return value of `should_log`.

#### get\_warmup\_steps

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2528)

( num\_training\_steps: int  )

Get number of steps used for a linear warmup.

#### main\_process\_first

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2477)

( local = True desc = 'work'  )

Parameters

* **local** (`bool`, *optional*, defaults to `True`) —
  if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
  rank 0 In multi-node environment with a shared filesystem you most likely will want to use
  `local=False` so that only the main process of the first node will do the processing. If however, the
  filesystem is not shared, then the main process of each node will need to do the processing, which is
  the default behavior.
* **desc** (`str`, *optional*, defaults to `"work"`) —
  a work description to be used in debug logs

A context manager for torch distributed environment where on needs to do something on the main process, while
blocking replicas, and when it’s finished releasing the replicas.

One such use is for `datasets`’s `map` feature which to be efficient should be run once on the main process,
which upon completion saves a cached version of results and which then automatically gets loaded by the
replicas.

#### set\_dataloader

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L3069)

( train\_batch\_size: int = 8 eval\_batch\_size: int = 8 drop\_last: bool = False num\_workers: int = 0 pin\_memory: bool = True persistent\_workers: bool = False prefetch\_factor: typing.Optional[int] = None auto\_find\_batch\_size: bool = False ignore\_data\_skip: bool = False sampler\_seed: typing.Optional[int] = None  )

Parameters

* **drop\_last** (`bool`, *optional*, defaults to `False`) —
  Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch
  size) or not.
* **num\_workers** (`int`, *optional*, defaults to 0) —
  Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in
  the main process.
* **pin\_memory** (`bool`, *optional*, defaults to `True`) —
  Whether you want to pin memory in data loaders or not. Will default to `True`.
* **persistent\_workers** (`bool`, *optional*, defaults to `False`) —
  If True, the data loader will not shut down the worker processes after a dataset has been consumed
  once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training,
  but will increase RAM usage. Will default to `False`.
* **prefetch\_factor** (`int`, *optional*) —
  Number of batches loaded in advance by each worker.
  2 means there will be a total of 2 \* num\_workers batches prefetched across all workers.
* **auto\_find\_batch\_size** (`bool`, *optional*, defaults to `False`) —
  Whether to find a batch size that will fit into memory automatically through exponential decay,
  avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)
* **ignore\_data\_skip** (`bool`, *optional*, defaults to `False`) —
  When resuming training, whether or not to skip the epochs and batches to get the data loading at the
  same stage as in the previous training. If set to `True`, the training will begin faster (as that
  skipping step can take a long time) but will not yield the same results as the interrupted training
  would have.
* **sampler\_seed** (`int`, *optional*) —
  Random seed to be used with data samplers. If not set, random generators for data sampling will use the
  same seed as `self.seed`. This can be used to ensure reproducibility of data sampling, independent of
  the model seed.

A method that regroups all arguments linked to the dataloaders creation.

Example:


```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_dataloader(train_batch_size=16, eval_batch_size=64)
>>> args.per_device_train_batch_size
16
```

#### set\_evaluate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2674)

( strategy: typing.Union[str, transformers.trainer\_utils.IntervalStrategy] = 'no' steps: int = 500 batch\_size: int = 8 accumulation\_steps: typing.Optional[int] = None delay: typing.Optional[float] = None loss\_only: bool = False jit\_mode: bool = False  )

Parameters

* **strategy** (`str` or [IntervalStrategy](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.IntervalStrategy), *optional*, defaults to `"no"`) —
  The evaluation strategy to adopt during training. Possible values are:
  + `"no"`: No evaluation is done during training.
  + `"steps"`: Evaluation is done (and logged) every `steps`.
  + `"epoch"`: Evaluation is done at the end of each epoch.

  Setting a `strategy` different from `"no"` will set `self.do_eval` to `True`.
* **steps** (`int`, *optional*, defaults to 500) —
  Number of update steps between two evaluations if `strategy="steps"`.
* **batch\_size** (`int` *optional*, defaults to 8) —
  The batch size per device (GPU/TPU core/CPU…) used for evaluation.
* **accumulation\_steps** (`int`, *optional*) —
  Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU.
  If left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster
  but requires more memory).
* **delay** (`float`, *optional*) —
  Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
  eval\_strategy.
* **loss\_only** (`bool`, *optional*, defaults to `False`) —
  Ignores all outputs except the loss.
* **jit\_mode** (`bool`, *optional*) —
  Whether or not to use PyTorch jit trace for inference.

A method that regroups all arguments linked to evaluation.

Example:


```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_evaluate(strategy="steps", steps=100)
>>> args.eval_steps
100
```

#### set\_logging

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2824)

( strategy: typing.Union[str, transformers.trainer\_utils.IntervalStrategy] = 'steps' steps: int = 500 report\_to: typing.Union[str, list[str]] = 'none' level: str = 'passive' first\_step: bool = False nan\_inf\_filter: bool = False on\_each\_node: bool = False replica\_level: str = 'passive'  )

Parameters

* **strategy** (`str` or [IntervalStrategy](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.IntervalStrategy), *optional*, defaults to `"steps"`) —
  The logging strategy to adopt during training. Possible values are:
  + `"no"`: No logging is done during training.
  + `"epoch"`: Logging is done at the end of each epoch.
  + `"steps"`: Logging is done every `logging_steps`.
* **steps** (`int`, *optional*, defaults to 500) —
  Number of update steps between two logs if `strategy="steps"`.
* **level** (`str`, *optional*, defaults to `"passive"`) —
  Logger log level to use on the main process. Possible choices are the log levels as strings: `"debug"`,
  `"info"`, `"warning"`, `"error"` and `"critical"`, plus a `"passive"` level which doesn’t set anything
  and lets the application set the level.
* **report\_to** (`str` or `list[str]`, *optional*, defaults to `"all"`) —
  The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
  `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`,
  `"neptune"`, `"swanlab"`, `"tensorboard"`, `"trackio"` and `"wandb"`. Use `"all"` to report to all
  integrations installed, `"none"` for no integrations.
* **first\_step** (`bool`, *optional*, defaults to `False`) —
  Whether to log and evaluate the first `global_step` or not.
* **nan\_inf\_filter** (`bool`, *optional*, defaults to `True`) —
  Whether to filter `nan` and `inf` losses for logging. If set to `True` the loss of every step that is
  `nan` or `inf` is filtered and the average loss of the current logging window is taken instead.

  `nan_inf_filter` only influences the logging of loss values, it does not change the behavior the
  gradient is computed or applied to the model.
* **on\_each\_node** (`bool`, *optional*, defaults to `True`) —
  In multinode distributed training, whether to log using `log_level` once per node, or only on the main
  node.
* **replica\_level** (`str`, *optional*, defaults to `"passive"`) —
  Logger log level to use on replicas. Same choices as `log_level`

A method that regroups all arguments linked to logging.

Example:


```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_logging(strategy="steps", steps=100)
>>> args.logging_steps
100
```

#### set\_lr\_scheduler

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L3024)

( name: typing.Union[str, transformers.trainer\_utils.SchedulerType] = 'linear' num\_epochs: float = 3.0 max\_steps: int = -1 warmup\_ratio: float = 0 warmup\_steps: int = 0  )

Parameters

* **name** (`str` or [SchedulerType](/docs/transformers/v4.56.2/en/main_classes/optimizer_schedules#transformers.SchedulerType), *optional*, defaults to `"linear"`) —
  The scheduler type to use. See the documentation of [SchedulerType](/docs/transformers/v4.56.2/en/main_classes/optimizer_schedules#transformers.SchedulerType) for all possible values.
* **num\_epochs(`float`,** *optional*, defaults to 3.0) —
  Total number of training epochs to perform (if not an integer, will perform the decimal part percents
  of the last epoch before stopping training).
* **max\_steps** (`int`, *optional*, defaults to -1) —
  If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
  For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
  `max_steps` is reached.
* **warmup\_ratio** (`float`, *optional*, defaults to 0.0) —
  Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
* **warmup\_steps** (`int`, *optional*, defaults to 0) —
  Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of
  `warmup_ratio`.

A method that regroups all arguments linked to the learning rate scheduler and its hyperparameters.

Example:


```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_lr_scheduler(name="cosine", warmup_ratio=0.05)
>>> args.warmup_ratio
0.05
```

#### set\_optimizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2973)

( name: typing.Union[str, transformers.training\_args.OptimizerNames] = 'adamw\_torch' learning\_rate: float = 5e-05 weight\_decay: float = 0 beta1: float = 0.9 beta2: float = 0.999 epsilon: float = 1e-08 args: typing.Optional[str] = None  )

Parameters

* **name** (`str` or `training_args.OptimizerNames`, *optional*, defaults to `"adamw_torch"`) —
  The optimizer to use: `"adamw_torch"`, `"adamw_torch_fused"`, `"adamw_apex_fused"`,
  `"adamw_anyprecision"` or `"adafactor"`.
* **learning\_rate** (`float`, *optional*, defaults to 5e-5) —
  The initial learning rate.
* **weight\_decay** (`float`, *optional*, defaults to 0) —
  The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights.
* **beta1** (`float`, *optional*, defaults to 0.9) —
  The beta1 hyperparameter for the adam optimizer or its variants.
* **beta2** (`float`, *optional*, defaults to 0.999) —
  The beta2 hyperparameter for the adam optimizer or its variants.
* **epsilon** (`float`, *optional*, defaults to 1e-8) —
  The epsilon hyperparameter for the adam optimizer or its variants.
* **args** (`str`, *optional*) —
  Optional arguments that are supplied to AnyPrecisionAdamW (only useful when
  `optim="adamw_anyprecision"`).

A method that regroups all arguments linked to the optimizer and its hyperparameters.

Example:


```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_optimizer(name="adamw_torch", beta1=0.8)
>>> args.optim
'adamw_torch'
```

#### set\_push\_to\_hub

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2899)

( model\_id: str strategy: typing.Union[str, transformers.trainer\_utils.HubStrategy] = 'every\_save' token: typing.Optional[str] = None private\_repo: typing.Optional[bool] = None always\_push: bool = False revision: typing.Optional[str] = None  )

Parameters

* **model\_id** (`str`) —
  The name of the repository to keep in sync with the local *output\_dir*. It can be a simple model ID in
  which case the model will be pushed in your namespace. Otherwise it should be the whole repository
  name, for instance `"user_name/model"`, which allows you to push to an organization you are a member of
  with `"organization_name/model"`.
* **strategy** (`str` or `HubStrategy`, *optional*, defaults to `"every_save"`) —
  Defines the scope of what is pushed to the Hub and when. Possible values are:
  + `"end"`: push the model, its configuration, the processing\_class e.g. tokenizer (if passed along to the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer)) and a
    draft of a model card when the [save\_model()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.save_model) method is called.
  + `"every_save"`: push the model, its configuration, the processing\_class e.g. tokenizer (if passed along to the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer))
    and
    a draft of a model card each time there is a model save. The pushes are asynchronous to not block
    training, and in case the save are very frequent, a new push is only attempted if the previous one is
    finished. A last push is made with the final model at the end of training.
  + `"checkpoint"`: like `"every_save"` but the latest checkpoint is also pushed in a subfolder named
    last-checkpoint, allowing you to resume training easily with
    `trainer.train(resume_from_checkpoint="last-checkpoint")`.
  + `"all_checkpoints"`: like `"checkpoint"` but all checkpoints are pushed like they appear in the
    output
    folder (so you will get one checkpoint folder per folder in your final repository)
* **token** (`str`, *optional*) —
  The token to use to push the model to the Hub. Will default to the token in the cache folder obtained
  with `hf auth login`.
* **private\_repo** (`bool`, *optional*, defaults to `False`) —
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
* **always\_push** (`bool`, *optional*, defaults to `False`) —
  Unless this is `True`, the `Trainer` will skip pushing a checkpoint when the previous push is not
  finished.
* **revision** (`str`, *optional*) —
  The revision to use when pushing to the Hub. Can be a branch name, a tag, or a commit hash.

A method that regroups all arguments linked to synchronizing checkpoints with the Hub.

Calling this method will set `self.push_to_hub` to `True`, which means the `output_dir` will begin a git
directory synced with the repo (determined by `model_id`) and the content will be pushed each time a save is
triggered (depending on your `self.save_strategy`). Calling [save\_model()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.save_model) will also trigger a push.

Example:


```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_push_to_hub("me/awesome-model")
>>> args.hub_model_id
'me/awesome-model'
```

#### set\_save

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2775)

( strategy: typing.Union[str, transformers.trainer\_utils.IntervalStrategy] = 'steps' steps: int = 500 total\_limit: typing.Optional[int] = None on\_each\_node: bool = False  )

Parameters

* **strategy** (`str` or [IntervalStrategy](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.IntervalStrategy), *optional*, defaults to `"steps"`) —
  The checkpoint save strategy to adopt during training. Possible values are:
  + `"no"`: No save is done during training.
  + `"epoch"`: Save is done at the end of each epoch.
  + `"steps"`: Save is done every `save_steps`.
* **steps** (`int`, *optional*, defaults to 500) —
  Number of updates steps before two checkpoint saves if `strategy="steps"`.
* **total\_limit** (`int`, *optional*) —
  If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
  `output_dir`.
* **on\_each\_node** (`bool`, *optional*, defaults to `False`) —
  When doing multi-node distributed training, whether to save models and checkpoints on each node, or
  only on the main one.

  This should not be activated when the different nodes use the same storage as the files will be saved
  with the same names for each node.

A method that regroups all arguments linked to checkpoint saving.

Example:


```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_save(strategy="steps", steps=100)
>>> args.save_steps
100
```

#### set\_testing

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2735)

( batch\_size: int = 8 loss\_only: bool = False jit\_mode: bool = False  )

Parameters

* **batch\_size** (`int` *optional*, defaults to 8) —
  The batch size per device (GPU/TPU core/CPU…) used for testing.
* **loss\_only** (`bool`, *optional*, defaults to `False`) —
  Ignores all outputs except the loss.
* **jit\_mode** (`bool`, *optional*) —
  Whether or not to use PyTorch jit trace for inference.

A method that regroups all basic arguments linked to testing on a held-out dataset.

Calling this method will automatically set `self.do_predict` to `True`.

Example:


```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_testing(batch_size=32)
>>> args.per_device_eval_batch_size
32
```

#### set\_training

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2599)

( learning\_rate: float = 5e-05 batch\_size: int = 8 weight\_decay: float = 0 num\_epochs: float = 3 max\_steps: int = -1 gradient\_accumulation\_steps: int = 1 seed: int = 42 gradient\_checkpointing: bool = False  )

Parameters

* **learning\_rate** (`float`, *optional*, defaults to 5e-5) —
  The initial learning rate for the optimizer.
* **batch\_size** (`int` *optional*, defaults to 8) —
  The batch size per device (GPU/TPU core/CPU…) used for training.
* **weight\_decay** (`float`, *optional*, defaults to 0) —
  The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the
  optimizer.
* **num\_train\_epochs(`float`,** *optional*, defaults to 3.0) —
  Total number of training epochs to perform (if not an integer, will perform the decimal part percents
  of the last epoch before stopping training).
* **max\_steps** (`int`, *optional*, defaults to -1) —
  If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
  For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
  `max_steps` is reached.
* **gradient\_accumulation\_steps** (`int`, *optional*, defaults to 1) —
  Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

  When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
  logging, evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training
  examples.
* **seed** (`int`, *optional*, defaults to 42) —
  Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use
  the `~Trainer.model_init` function to instantiate the model if it has some randomly initialized
  parameters.
* **gradient\_checkpointing** (`bool`, *optional*, defaults to `False`) —
  If True, use gradient checkpointing to save memory at the expense of slower backward pass.

A method that regroups all basic arguments linked to the training.

Calling this method will automatically set `self.do_train` to `True`.

Example:


```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_training(learning_rate=1e-4, batch_size=32)
>>> args.learning_rate
1e-4
```

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2549)

( )

Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
the token values by removing their value.

#### to\_json\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2579)

( )

Serializes this instance to a JSON string.

#### to\_sanitized\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args.py#L2585)

( )

Sanitized serialization to use with TensorBoard’s hparams

## Seq2SeqTrainingArguments

### class transformers.Seq2SeqTrainingArguments

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args_seq2seq.py#L30)

( output\_dir: typing.Optional[str] = None overwrite\_output\_dir: bool = False do\_train: bool = False do\_eval: bool = False do\_predict: bool = False eval\_strategy: typing.Union[transformers.trainer\_utils.IntervalStrategy, str] = 'no' prediction\_loss\_only: bool = False per\_device\_train\_batch\_size: int = 8 per\_device\_eval\_batch\_size: int = 8 per\_gpu\_train\_batch\_size: typing.Optional[int] = None per\_gpu\_eval\_batch\_size: typing.Optional[int] = None gradient\_accumulation\_steps: int = 1 eval\_accumulation\_steps: typing.Optional[int] = None eval\_delay: typing.Optional[float] = 0 torch\_empty\_cache\_steps: typing.Optional[int] = None learning\_rate: float = 5e-05 weight\_decay: float = 0.0 adam\_beta1: float = 0.9 adam\_beta2: float = 0.999 adam\_epsilon: float = 1e-08 max\_grad\_norm: float = 1.0 num\_train\_epochs: float = 3.0 max\_steps: int = -1 lr\_scheduler\_type: typing.Union[transformers.trainer\_utils.SchedulerType, str] = 'linear' lr\_scheduler\_kwargs: typing.Union[dict[str, typing.Any], str, NoneType] = <factory> warmup\_ratio: float = 0.0 warmup\_steps: int = 0 log\_level: str = 'passive' log\_level\_replica: str = 'warning' log\_on\_each\_node: bool = True logging\_dir: typing.Optional[str] = None logging\_strategy: typing.Union[transformers.trainer\_utils.IntervalStrategy, str] = 'steps' logging\_first\_step: bool = False logging\_steps: float = 500 logging\_nan\_inf\_filter: bool = True save\_strategy: typing.Union[transformers.trainer\_utils.SaveStrategy, str] = 'steps' save\_steps: float = 500 save\_total\_limit: typing.Optional[int] = None save\_safetensors: typing.Optional[bool] = True save\_on\_each\_node: bool = False save\_only\_model: bool = False restore\_callback\_states\_from\_checkpoint: bool = False no\_cuda: bool = False use\_cpu: bool = False use\_mps\_device: bool = False seed: int = 42 data\_seed: typing.Optional[int] = None jit\_mode\_eval: bool = False use\_ipex: bool = False bf16: bool = False fp16: bool = False fp16\_opt\_level: str = 'O1' half\_precision\_backend: str = 'auto' bf16\_full\_eval: bool = False fp16\_full\_eval: bool = False tf32: typing.Optional[bool] = None local\_rank: int = -1 ddp\_backend: typing.Optional[str] = None tpu\_num\_cores: typing.Optional[int] = None tpu\_metrics\_debug: bool = False debug: typing.Union[str, list[transformers.debug\_utils.DebugOption]] = '' dataloader\_drop\_last: bool = False eval\_steps: typing.Optional[float] = None dataloader\_num\_workers: int = 0 dataloader\_prefetch\_factor: typing.Optional[int] = None past\_index: int = -1 run\_name: typing.Optional[str] = None disable\_tqdm: typing.Optional[bool] = None remove\_unused\_columns: typing.Optional[bool] = True label\_names: typing.Optional[list[str]] = None load\_best\_model\_at\_end: typing.Optional[bool] = False metric\_for\_best\_model: typing.Optional[str] = None greater\_is\_better: typing.Optional[bool] = None ignore\_data\_skip: bool = False fsdp: typing.Union[list[transformers.trainer\_utils.FSDPOption], str, NoneType] = '' fsdp\_min\_num\_params: int = 0 fsdp\_config: typing.Union[dict[str, typing.Any], str, NoneType] = None fsdp\_transformer\_layer\_cls\_to\_wrap: typing.Optional[str] = None accelerator\_config: typing.Union[dict, str, NoneType] = None parallelism\_config: typing.Optional[ForwardRef('ParallelismConfig')] = None deepspeed: typing.Union[dict, str, NoneType] = None label\_smoothing\_factor: float = 0.0 optim: typing.Union[transformers.training\_args.OptimizerNames, str] = 'adamw\_torch\_fused' optim\_args: typing.Optional[str] = None adafactor: bool = False group\_by\_length: bool = False length\_column\_name: typing.Optional[str] = 'length' report\_to: typing.Union[NoneType, str, list[str]] = None ddp\_find\_unused\_parameters: typing.Optional[bool] = None ddp\_bucket\_cap\_mb: typing.Optional[int] = None ddp\_broadcast\_buffers: typing.Optional[bool] = None dataloader\_pin\_memory: bool = True dataloader\_persistent\_workers: bool = False skip\_memory\_metrics: bool = True use\_legacy\_prediction\_loop: bool = False push\_to\_hub: bool = False resume\_from\_checkpoint: typing.Optional[str] = None hub\_model\_id: typing.Optional[str] = None hub\_strategy: typing.Union[transformers.trainer\_utils.HubStrategy, str] = 'every\_save' hub\_token: typing.Optional[str] = None hub\_private\_repo: typing.Optional[bool] = None hub\_always\_push: bool = False hub\_revision: typing.Optional[str] = None gradient\_checkpointing: bool = False gradient\_checkpointing\_kwargs: typing.Union[dict[str, typing.Any], str, NoneType] = None include\_inputs\_for\_metrics: bool = False include\_for\_metrics: list = <factory> eval\_do\_concat\_batches: bool = True fp16\_backend: str = 'auto' push\_to\_hub\_model\_id: typing.Optional[str] = None push\_to\_hub\_organization: typing.Optional[str] = None push\_to\_hub\_token: typing.Optional[str] = None mp\_parameters: str = '' auto\_find\_batch\_size: bool = False full\_determinism: bool = False torchdynamo: typing.Optional[str] = None ray\_scope: typing.Optional[str] = 'last' ddp\_timeout: int = 1800 torch\_compile: bool = False torch\_compile\_backend: typing.Optional[str] = None torch\_compile\_mode: typing.Optional[str] = None include\_tokens\_per\_second: typing.Optional[bool] = False include\_num\_input\_tokens\_seen: typing.Optional[bool] = False neftune\_noise\_alpha: typing.Optional[float] = None optim\_target\_modules: typing.Union[NoneType, str, list[str]] = None batch\_eval\_metrics: bool = False eval\_on\_start: bool = False use\_liger\_kernel: typing.Optional[bool] = False liger\_kernel\_config: typing.Optional[dict[str, bool]] = None eval\_use\_gather\_object: typing.Optional[bool] = False average\_tokens\_across\_devices: typing.Optional[bool] = True sortish\_sampler: bool = False predict\_with\_generate: bool = False generation\_max\_length: typing.Optional[int] = None generation\_num\_beams: typing.Optional[int] = None generation\_config: typing.Union[str, pathlib.Path, transformers.generation.configuration\_utils.GenerationConfig, NoneType] = None  )

Parameters

* **output\_dir** (`str`, *optional*, defaults to `"trainer_output"`) —
  The output directory where the model predictions and checkpoints will be written.
* **overwrite\_output\_dir** (`bool`, *optional*, defaults to `False`) —
  If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir`
  points to a checkpoint directory.
* **do\_train** (`bool`, *optional*, defaults to `False`) —
  Whether to run training or not. This argument is not directly used by [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), it’s intended to be used
  by your training/evaluation scripts instead. See the [example
  scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
* **do\_eval** (`bool`, *optional*) —
  Whether to run evaluation on the validation set or not. Will be set to `True` if `eval_strategy` is
  different from `"no"`. This argument is not directly used by [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), it’s intended to be used by your
  training/evaluation scripts instead. See the [example
  scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
* **do\_predict** (`bool`, *optional*, defaults to `False`) —
  Whether to run predictions on the test set or not. This argument is not directly used by [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), it’s
  intended to be used by your training/evaluation scripts instead. See the [example
  scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
* **eval\_strategy** (`str` or [IntervalStrategy](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.IntervalStrategy), *optional*, defaults to `"no"`) —
  The evaluation strategy to adopt during training. Possible values are:
  + `"no"`: No evaluation is done during training.
  + `"steps"`: Evaluation is done (and logged) every `eval_steps`.
  + `"epoch"`: Evaluation is done at the end of each epoch.
* **prediction\_loss\_only** (`bool`, *optional*, defaults to `False`) —
  When performing evaluation and generating predictions, only returns the loss.
* **per\_device\_train\_batch\_size** (`int`, *optional*, defaults to 8) —
  The batch size *per device*. The **global batch size** is computed as:
  `per_device_train_batch_size * number_of_devices` in multi-GPU or distributed setups.
* **per\_device\_eval\_batch\_size** (`int`, *optional*, defaults to 8) —
  The batch size per device accelerator core/CPU for evaluation.
* **gradient\_accumulation\_steps** (`int`, *optional*, defaults to 1) —
  Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

  When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging,
  evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.
* **eval\_accumulation\_steps** (`int`, *optional*) —
  Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
  left unset, the whole predictions are accumulated on the device accelerator before being moved to the CPU (faster but
  requires more memory).
* **eval\_delay** (`float`, *optional*) —
  Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
  eval\_strategy.
* **torch\_empty\_cache\_steps** (`int`, *optional*) —
  Number of steps to wait before calling `torch.<device>.empty_cache()`. If left unset or set to None, cache will not be emptied.

  This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of about [10% slower performance](https://github.com/huggingface/transformers/issues/31372).
* **learning\_rate** (`float`, *optional*, defaults to 5e-5) —
  The initial learning rate for `AdamW` optimizer.
* **weight\_decay** (`float`, *optional*, defaults to 0) —
  The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in `AdamW`
  optimizer.
* **adam\_beta1** (`float`, *optional*, defaults to 0.9) —
  The beta1 hyperparameter for the `AdamW` optimizer.
* **adam\_beta2** (`float`, *optional*, defaults to 0.999) —
  The beta2 hyperparameter for the `AdamW` optimizer.
* **adam\_epsilon** (`float`, *optional*, defaults to 1e-8) —
  The epsilon hyperparameter for the `AdamW` optimizer.
* **max\_grad\_norm** (`float`, *optional*, defaults to 1.0) —
  Maximum gradient norm (for gradient clipping).
* **num\_train\_epochs(`float`,** *optional*, defaults to 3.0) —
  Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
  the last epoch before stopping training).
* **max\_steps** (`int`, *optional*, defaults to -1) —
  If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
  For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
  `max_steps` is reached.
* **lr\_scheduler\_type** (`str` or [SchedulerType](/docs/transformers/v4.56.2/en/main_classes/optimizer_schedules#transformers.SchedulerType), *optional*, defaults to `"linear"`) —
  The scheduler type to use. See the documentation of [SchedulerType](/docs/transformers/v4.56.2/en/main_classes/optimizer_schedules#transformers.SchedulerType) for all possible values.
* **lr\_scheduler\_kwargs** (‘dict’, *optional*, defaults to {}) —
  The extra arguments for the lr\_scheduler. See the documentation of each scheduler for possible values.
* **warmup\_ratio** (`float`, *optional*, defaults to 0.0) —
  Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
* **warmup\_steps** (`int`, *optional*, defaults to 0) —
  Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.
* **log\_level** (`str`, *optional*, defaults to `passive`) —
  Logger log level to use on the main process. Possible choices are the log levels as strings: ‘debug’,
  ‘info’, ‘warning’, ‘error’ and ‘critical’, plus a ‘passive’ level which doesn’t set anything and keeps the
  current log level for the Transformers library (which will be `"warning"` by default).
* **log\_level\_replica** (`str`, *optional*, defaults to `"warning"`) —
  Logger log level to use on replicas. Same choices as `log_level`”
* **log\_on\_each\_node** (`bool`, *optional*, defaults to `True`) —
  In multinode distributed training, whether to log using `log_level` once per node, or only on the main
  node.
* **logging\_dir** (`str`, *optional*) —
  [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
  \*output\_dir/runs/**CURRENT\_DATETIME\_HOSTNAME\***.
* **logging\_strategy** (`str` or [IntervalStrategy](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.IntervalStrategy), *optional*, defaults to `"steps"`) —
  The logging strategy to adopt during training. Possible values are:
  + `"no"`: No logging is done during training.
  + `"epoch"`: Logging is done at the end of each epoch.
  + `"steps"`: Logging is done every `logging_steps`.
* **logging\_first\_step** (`bool`, *optional*, defaults to `False`) —
  Whether to log the first `global_step` or not.
* **logging\_steps** (`int` or `float`, *optional*, defaults to 500) —
  Number of update steps between two logs if `logging_strategy="steps"`. Should be an integer or a float in
  range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
* **logging\_nan\_inf\_filter** (`bool`, *optional*, defaults to `True`) —
  Whether to filter `nan` and `inf` losses for logging. If set to `True` the loss of every step that is `nan`
  or `inf` is filtered and the average loss of the current logging window is taken instead.

  `logging_nan_inf_filter` only influences the logging of loss values, it does not change the behavior the
  gradient is computed or applied to the model.
* **save\_strategy** (`str` or `SaveStrategy`, *optional*, defaults to `"steps"`) —
  The checkpoint save strategy to adopt during training. Possible values are:
  + `"no"`: No save is done during training.
  + `"epoch"`: Save is done at the end of each epoch.
  + `"steps"`: Save is done every `save_steps`.
  + `"best"`: Save is done whenever a new `best_metric` is achieved.

  If `"epoch"` or `"steps"` is chosen, saving will also be performed at the
  very end of training, always.
* **save\_steps** (`int` or `float`, *optional*, defaults to 500) —
  Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Should be an integer or a
  float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
* **save\_total\_limit** (`int`, *optional*) —
  If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
  `output_dir`. When `load_best_model_at_end` is enabled, the “best” checkpoint according to
  `metric_for_best_model` will always be retained in addition to the most recent ones. For example, for
  `save_total_limit=5` and `load_best_model_at_end`, the four last checkpoints will always be retained
  alongside the best model. When `save_total_limit=1` and `load_best_model_at_end`, it is possible that two
  checkpoints are saved: the last one and the best one (if they are different).
* **save\_safetensors** (`bool`, *optional*, defaults to `True`) —
  Use [safetensors](https://huggingface.co/docs/safetensors) saving and loading for state dicts instead of
  default `torch.load` and `torch.save`.
* **save\_on\_each\_node** (`bool`, *optional*, defaults to `False`) —
  When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on
  the main one.

  This should not be activated when the different nodes use the same storage as the files will be saved with
  the same names for each node.
* **save\_only\_model** (`bool`, *optional*, defaults to `False`) —
  When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state.
  Note that when this is true, you won’t be able to resume training from checkpoint.
  This enables you to save storage by not storing the optimizer, scheduler & rng state.
  You can only load the model using `from_pretrained` with this option set to `True`.
* **restore\_callback\_states\_from\_checkpoint** (`bool`, *optional*, defaults to `False`) —
  Whether to restore the callback states from the checkpoint. If `True`, will override
  callbacks passed to the `Trainer` if they exist in the checkpoint.”
* **use\_cpu** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use cpu. If set to False, we will use cuda or mps device if available.
* **seed** (`int`, *optional*, defaults to 42) —
  Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
  `~Trainer.model_init` function to instantiate the model if it has some randomly initialized parameters.
* **data\_seed** (`int`, *optional*) —
  Random seed to be used with data samplers. If not set, random generators for data sampling will use the
  same seed as `seed`. This can be used to ensure reproducibility of data sampling, independent of the model
  seed.
* **jit\_mode\_eval** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use PyTorch jit trace for inference.
* **use\_ipex** (`bool`, *optional*, defaults to `False`) —
  Use Intel extension for PyTorch when it is available. [IPEX
  installation](https://github.com/intel/intel-extension-for-pytorch).
* **bf16** (`bool`, *optional*, defaults to `False`) —
  Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher
  NVIDIA architecture or Intel XPU or using CPU (use\_cpu) or Ascend NPU. This is an experimental API and it may change.
* **fp16** (`bool`, *optional*, defaults to `False`) —
  Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
* **fp16\_opt\_level** (`str`, *optional*, defaults to ‘O1’) —
  For `fp16` training, Apex AMP optimization level selected in [‘O0’, ‘O1’, ‘O2’, and ‘O3’]. See details on
  the [Apex documentation](https://nvidia.github.io/apex/amp).
* **fp16\_backend** (`str`, *optional*, defaults to `"auto"`) —
  This argument is deprecated. Use `half_precision_backend` instead.
* **half\_precision\_backend** (`str`, *optional*, defaults to `"auto"`) —
  The backend to use for mixed precision training. Must be one of `"auto", "apex", "cpu_amp"`. `"auto"` will
  use CPU/CUDA AMP or APEX depending on the PyTorch version detected, while the other choices will force the
  requested backend.
* **bf16\_full\_eval** (`bool`, *optional*, defaults to `False`) —
  Whether to use full bfloat16 evaluation instead of 32-bit. This will be faster and save memory but can harm
  metric values. This is an experimental API and it may change.
* **fp16\_full\_eval** (`bool`, *optional*, defaults to `False`) —
  Whether to use full float16 evaluation instead of 32-bit. This will be faster and save memory but can harm
  metric values.
* **tf32** (`bool`, *optional*) —
  Whether to enable the TF32 mode, available in Ampere and newer GPU architectures. The default value depends
  on PyTorch’s version default of `torch.backends.cuda.matmul.allow_tf32`. For more details please refer to
  the [TF32](https://huggingface.co/docs/transformers/perf_train_gpu_one#tf32) documentation. This is an
  experimental API and it may change.
* **local\_rank** (`int`, *optional*, defaults to -1) —
  Rank of the process during distributed training.
* **ddp\_backend** (`str`, *optional*) —
  The backend to use for distributed training. Must be one of `"nccl"`, `"mpi"`, `"ccl"`, `"gloo"`, `"hccl"`.
* **tpu\_num\_cores** (`int`, *optional*) —
  When training on TPU, the number of TPU cores (automatically passed by launcher script).
* **dataloader\_drop\_last** (`bool`, *optional*, defaults to `False`) —
  Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
  or not.
* **eval\_steps** (`int` or `float`, *optional*) —
  Number of update steps between two evaluations if `eval_strategy="steps"`. Will default to the same
  value as `logging_steps` if not set. Should be an integer or a float in range `[0,1)`. If smaller than 1,
  will be interpreted as ratio of total training steps.
* **dataloader\_num\_workers** (`int`, *optional*, defaults to 0) —
  Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
  main process.
* **past\_index** (`int`, *optional*, defaults to -1) —
  Some models like [TransformerXL](../model_doc/transformerxl) or [XLNet](../model_doc/xlnet) can make use of
  the past hidden states for their predictions. If this argument is set to a positive int, the `Trainer` will
  use the corresponding output (usually index 2) as the past state and feed it to the model at the next
  training step under the keyword argument `mems`.
* **run\_name** (`str`, *optional*, defaults to `output_dir`) —
  A descriptor for the run. Typically used for [trackio](https://github.com/gradio-app/trackio),
  [wandb](https://www.wandb.com/), [mlflow](https://www.mlflow.org/), [comet](https://www.comet.com/site) and
  [swanlab](https://swanlab.cn) logging. If not specified, will be the same as `output_dir`.
* **disable\_tqdm** (`bool`, *optional*) —
  Whether or not to disable the tqdm progress bars and table of metrics produced by
  `~notebook.NotebookTrainingTracker` in Jupyter Notebooks. Will default to `True` if the logging level is
  set to warn or lower (default), `False` otherwise.
* **remove\_unused\_columns** (`bool`, *optional*, defaults to `True`) —
  Whether or not to automatically remove the columns unused by the model forward method.
* **label\_names** (`list[str]`, *optional*) —
  The list of keys in your dictionary of inputs that correspond to the labels.

  Will eventually default to the list of argument names accepted by the model that contain the word “label”,
  except if the model used is one of the `XxxForQuestionAnswering` in which case it will also include the
  `["start_positions", "end_positions"]` keys.

  You should only specify `label_names` if you’re using custom label names or if your model’s `forward` consumes multiple label tensors (e.g., extractive QA).
* **load\_best\_model\_at\_end** (`bool`, *optional*, defaults to `False`) —
  Whether or not to load the best model found during training at the end of training. When this option is
  enabled, the best checkpoint will always be saved. See
  [`save_total_limit`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_total_limit)
  for more.

  When set to `True`, the parameters `save_strategy` needs to be the same as `eval_strategy`, and in
  the case it is “steps”, `save_steps` must be a round multiple of `eval_steps`.
* **metric\_for\_best\_model** (`str`, *optional*) —
  Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different
  models. Must be the name of a metric returned by the evaluation with or without the prefix `"eval_"`.

  If not specified, this will default to `"loss"` when either `load_best_model_at_end == True`
  or `lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU` (to use the evaluation loss).

  If you set this value, `greater_is_better` will default to `True` unless the name ends with “loss”.
  Don’t forget to set it to `False` if your metric is better when lower.
* **greater\_is\_better** (`bool`, *optional*) —
  Use in conjunction with `load_best_model_at_end` and `metric_for_best_model` to specify if better models
  should have a greater metric or not. Will default to:
  + `True` if `metric_for_best_model` is set to a value that doesn’t end in `"loss"`.
  + `False` if `metric_for_best_model` is not set, or set to a value that ends in `"loss"`.
* **ignore\_data\_skip** (`bool`, *optional*, defaults to `False`) —
  When resuming training, whether or not to skip the epochs and batches to get the data loading at the same
  stage as in the previous training. If set to `True`, the training will begin faster (as that skipping step
  can take a long time) but will not yield the same results as the interrupted training would have.
* **fsdp** (`bool`, `str` or list of `FSDPOption`, *optional*, defaults to `''`) —
  Use PyTorch Distributed Parallel Training (in distributed training only).

  A list of options along the following:

  + `"full_shard"`: Shard parameters, gradients and optimizer states.
  + `"shard_grad_op"`: Shard optimizer states and gradients.
  + `"hybrid_shard"`: Apply `FULL_SHARD` within a node, and replicate parameters across nodes.
  + `"hybrid_shard_zero2"`: Apply `SHARD_GRAD_OP` within a node, and replicate parameters across nodes.
  + `"offload"`: Offload parameters and gradients to CPUs (only compatible with `"full_shard"` and
    `"shard_grad_op"`).
  + `"auto_wrap"`: Automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`.
* **fsdp\_config** (`str` or `dict`, *optional*) —
  Config to be used with fsdp (Pytorch Distributed Parallel Training). The value is either a location of
  fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`.

  A List of config and its options:

  + min\_num\_params (`int`, *optional*, defaults to `0`):
    FSDP’s minimum number of parameters for Default Auto Wrapping. (useful only when `fsdp` field is
    passed).
  + transformer\_layer\_cls\_to\_wrap (`list[str]`, *optional*):
    List of transformer layer class names (case-sensitive) to wrap, e.g, `BertLayer`, `GPTJBlock`,
    `T5Block` … (useful only when `fsdp` flag is passed).
  + backward\_prefetch (`str`, *optional*)
    FSDP’s backward prefetch mode. Controls when to prefetch next set of parameters (useful only when
    `fsdp` field is passed).

    A list of options along the following:

    - `"backward_pre"` : Prefetches the next set of parameters before the current set of parameter’s
      gradient
      computation.
    - `"backward_post"` : This prefetches the next set of parameters after the current set of
      parameter’s
      gradient computation.
  + forward\_prefetch (`bool`, *optional*, defaults to `False`)
    FSDP’s forward prefetch mode (useful only when `fsdp` field is passed).
    If `"True"`, then FSDP explicitly prefetches the next upcoming all-gather while executing in the
    forward pass.
  + limit\_all\_gathers (`bool`, *optional*, defaults to `False`)
    FSDP’s limit\_all\_gathers (useful only when `fsdp` field is passed).
    If `"True"`, FSDP explicitly synchronizes the CPU thread to prevent too many in-flight
    all-gathers.
  + use\_orig\_params (`bool`, *optional*, defaults to `True`)
    If `"True"`, allows non-uniform `requires_grad` during init, which means support for interspersed
    frozen and trainable parameters. Useful in cases such as parameter-efficient fine-tuning. Please
    refer this
    [blog](<https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019>
  + sync\_module\_states (`bool`, *optional*, defaults to `True`)
    If `"True"`, each individually wrapped FSDP unit will broadcast module parameters from rank 0 to
    ensure they are the same across all ranks after initialization
  + cpu\_ram\_efficient\_loading (`bool`, *optional*, defaults to `False`)
    If `"True"`, only the first process loads the pretrained model checkpoint while all other processes
    have empty weights. When this setting as `"True"`, `sync_module_states` also must to be `"True"`,
    otherwise all the processes except the main process would have random weights leading to unexpected
    behaviour during training.
  + activation\_checkpointing (`bool`, *optional*, defaults to `False`):
    If `"True"`, activation checkpointing is a technique to reduce memory usage by clearing activations of
    certain layers and recomputing them during a backward pass. Effectively, this trades extra
    computation time for reduced memory usage.
  + xla (`bool`, *optional*, defaults to `False`):
    Whether to use PyTorch/XLA Fully Sharded Data Parallel Training. This is an experimental feature
    and its API may evolve in the future.
  + xla\_fsdp\_settings (`dict`, *optional*)
    The value is a dictionary which stores the XLA FSDP wrapping parameters.

    For a complete list of options, please see [here](https://github.com/pytorch/xla/blob/master/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py).
  + xla\_fsdp\_grad\_ckpt (`bool`, *optional*, defaults to `False`):
    Will use gradient checkpointing over each nested XLA FSDP wrapped layer. This setting can only be
    used when the xla flag is set to true, and an auto wrapping policy is specified through
    fsdp\_min\_num\_params or fsdp\_transformer\_layer\_cls\_to\_wrap.
* **deepspeed** (`str` or `dict`, *optional*) —
  Use [Deepspeed](https://github.com/deepspeedai/DeepSpeed). This is an experimental feature and its API may
  evolve in the future. The value is either the location of DeepSpeed json config file (e.g.,
  `ds_config.json`) or an already loaded json file as a `dict`”

  If enabling any Zero-init, make sure that your model is not initialized until
  \*after\* initializing the `TrainingArguments`, else it will not be applied.
* **accelerator\_config** (`str`, `dict`, or `AcceleratorConfig`, *optional*) —
  Config to be used with the internal `Accelerator` implementation. The value is either a location of
  accelerator json config file (e.g., `accelerator_config.json`), an already loaded json file as `dict`,
  or an instance of `AcceleratorConfig`.

  A list of config and its options:

  + split\_batches (`bool`, *optional*, defaults to `False`):
    Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
    `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
    round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
    in your script multiplied by the number of processes.
  + dispatch\_batches (`bool`, *optional*):
    If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
    and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
    underlying dataset is an `IterableDataset`, `False` otherwise.
  + even\_batches (`bool`, *optional*, defaults to `True`):
    If set to `True`, in cases where the total batch size across all processes does not exactly divide the
    dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
    all workers.
  + use\_seedable\_sampler (`bool`, *optional*, defaults to `True`):
    Whether or not use a fully seedable random sampler (`accelerate.data_loader.SeedableRandomSampler`). Ensures
    training results are fully reproducible using a different sampling technique. While seed-to-seed results
    may differ, on average the differences are negligible when using multiple different seeds to compare. Should
    also be ran with `~utils.set_seed` for the best results.
  + use\_configured\_state (`bool`, *optional*, defaults to `False`):
    Whether or not to use a pre-configured `AcceleratorState` or `PartialState` defined before calling `TrainingArguments`.
    If `True`, an `Accelerator` or `PartialState` must be initialized. Note that by doing so, this could lead to issues
    with hyperparameter tuning.
* **parallelism\_config** (`ParallelismConfig`, *optional*) —
  Parallelism configuration for the training run. Requires Accelerate `1.10.1`
* **label\_smoothing\_factor** (`float`, *optional*, defaults to 0.0) —
  The label smoothing factor to use. Zero means no label smoothing, otherwise the underlying onehot-encoded
  labels are changed from 0s and 1s to `label_smoothing_factor/num_labels` and `1 - label_smoothing_factor + label_smoothing_factor/num_labels` respectively.
* **debug** (`str` or list of `DebugOption`, *optional*, defaults to `""`) —
  Enable one or more debug features. This is an experimental feature.

  Possible options are:

  + `"underflow_overflow"`: detects overflow in model’s input/outputs and reports the last frames that led to
    the event
  + `"tpu_metrics_debug"`: print debug metrics on TPU

  The options should be separated by whitespaces.
* **optim** (`str` or `training_args.OptimizerNames`, *optional*, defaults to `"adamw_torch"` (for torch>=2.8 `"adamw_torch_fused"`)) —
  The optimizer to use, such as “adamw\_torch”, “adamw\_torch\_fused”, “adamw\_apex\_fused”, “adamw\_anyprecision”,
  “adafactor”. See `OptimizerNames` in [training\_args.py](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)
  for a full list of optimizers.
* **optim\_args** (`str`, *optional*) —
  Optional arguments that are supplied to optimizers such as AnyPrecisionAdamW, AdEMAMix, and GaLore.
* **group\_by\_length** (`bool`, *optional*, defaults to `False`) —
  Whether or not to group together samples of roughly the same length in the training dataset (to minimize
  padding applied and be more efficient). Only useful if applying dynamic padding.
* **length\_column\_name** (`str`, *optional*, defaults to `"length"`) —
  Column name for precomputed lengths. If the column exists, grouping by length will use these values rather
  than computing them on train startup. Ignored unless `group_by_length` is `True` and the dataset is an
  instance of `Dataset`.
* **report\_to** (`str` or `list[str]`, *optional*, defaults to `"all"`) —
  The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
  `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"neptune"`,
  `"swanlab"`, `"tensorboard"`, `"trackio"` and `"wandb"`. Use `"all"` to report to all integrations
  installed, `"none"` for no integrations.
* **ddp\_find\_unused\_parameters** (`bool`, *optional*) —
  When using distributed training, the value of the flag `find_unused_parameters` passed to
  `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
* **ddp\_bucket\_cap\_mb** (`int`, *optional*) —
  When using distributed training, the value of the flag `bucket_cap_mb` passed to `DistributedDataParallel`.
* **ddp\_broadcast\_buffers** (`bool`, *optional*) —
  When using distributed training, the value of the flag `broadcast_buffers` passed to
  `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
* **dataloader\_pin\_memory** (`bool`, *optional*, defaults to `True`) —
  Whether you want to pin memory in data loaders or not. Will default to `True`.
* **dataloader\_persistent\_workers** (`bool`, *optional*, defaults to `False`) —
  If True, the data loader will not shut down the worker processes after a dataset has been consumed once.
  This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will
  increase RAM usage. Will default to `False`.
* **dataloader\_prefetch\_factor** (`int`, *optional*) —
  Number of batches loaded in advance by each worker.
  2 means there will be a total of 2 \* num\_workers batches prefetched across all workers.
* **skip\_memory\_metrics** (`bool`, *optional*, defaults to `True`) —
  Whether to skip adding of memory profiler reports to metrics. This is skipped by default because it slows
  down the training and evaluation speed.
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push the model to the Hub every time the model is saved. If this is activated,
  `output_dir` will begin a git directory synced with the repo (determined by `hub_model_id`) and the content
  will be pushed each time a save is triggered (depending on your `save_strategy`). Calling
  [save\_model()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.save_model) will also trigger a push.

  If `output_dir` exists, it needs to be a local clone of the repository to which the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) will be
  pushed.
* **resume\_from\_checkpoint** (`str`, *optional*) —
  The path to a folder with a valid checkpoint for your model. This argument is not directly used by
  [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), it’s intended to be used by your training/evaluation scripts instead. See the [example
  scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
* **hub\_model\_id** (`str`, *optional*) —
  The name of the repository to keep in sync with the local *output\_dir*. It can be a simple model ID in
  which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
  for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
  `"organization_name/model"`. Will default to `user_name/output_dir_name` with *output\_dir\_name* being the
  name of `output_dir`.

  Will default to the name of `output_dir`.
* **hub\_strategy** (`str` or `HubStrategy`, *optional*, defaults to `"every_save"`) —
  Defines the scope of what is pushed to the Hub and when. Possible values are:
  + `"end"`: push the model, its configuration, the processing class e.g. tokenizer (if passed along to the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer)) and a
    draft of a model card when the [save\_model()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.save_model) method is called.
  + `"every_save"`: push the model, its configuration, the processing class e.g. tokenizer (if passed along to the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer)) and
    a draft of a model card each time there is a model save. The pushes are asynchronous to not block
    training, and in case the save are very frequent, a new push is only attempted if the previous one is
    finished. A last push is made with the final model at the end of training.
  + `"checkpoint"`: like `"every_save"` but the latest checkpoint is also pushed in a subfolder named
    last-checkpoint, allowing you to resume training easily with
    `trainer.train(resume_from_checkpoint="last-checkpoint")`.
  + `"all_checkpoints"`: like `"checkpoint"` but all checkpoints are pushed like they appear in the output
    folder (so you will get one checkpoint folder per folder in your final repository)
* **hub\_token** (`str`, *optional*) —
  The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
  `hf auth login`.
* **hub\_private\_repo** (`bool`, *optional*) —
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
* **hub\_always\_push** (`bool`, *optional*, defaults to `False`) —
  Unless this is `True`, the `Trainer` will skip pushing a checkpoint when the previous push is not finished.
* **hub\_revision** (`str`, *optional*) —
  The revision to use when pushing to the Hub. Can be a branch name, a tag, or a commit hash.
* **gradient\_checkpointing** (`bool`, *optional*, defaults to `False`) —
  If True, use gradient checkpointing to save memory at the expense of slower backward pass.
* **gradient\_checkpointing\_kwargs** (`dict`, *optional*, defaults to `None`) —
  Key word arguments to be passed to the `gradient_checkpointing_enable` method.
* **include\_inputs\_for\_metrics** (`bool`, *optional*, defaults to `False`) —
  This argument is deprecated. Use `include_for_metrics` instead, e.g, `include_for_metrics = ["inputs"]`.
* **include\_for\_metrics** (`list[str]`, *optional*, defaults to `[]`) —
  Include additional data in the `compute_metrics` function if needed for metrics computation.
  Possible options to add to `include_for_metrics` list:
  + `"inputs"`: Input data passed to the model, intended for calculating input dependent metrics.
  + `"loss"`: Loss values computed during evaluation, intended for calculating loss dependent metrics.
* **eval\_do\_concat\_batches** (`bool`, *optional*, defaults to `True`) —
  Whether to recursively concat inputs/losses/labels/predictions across batches. If `False`,
  will instead store them as lists, with each batch kept separate.
* **auto\_find\_batch\_size** (`bool`, *optional*, defaults to `False`) —
  Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding
  CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)
* **full\_determinism** (`bool`, *optional*, defaults to `False`) —
  If `True`, [enable\_full\_determinism()](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.enable_full_determinism) is called instead of [set\_seed()](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.set_seed) to ensure reproducible results in
  distributed training. Important: this will negatively impact the performance, so only use it for debugging.
* **torchdynamo** (`str`, *optional*) —
  If set, the backend compiler for TorchDynamo. Possible choices are `"eager"`, `"aot_eager"`, `"inductor"`,
  `"nvfuser"`, `"aot_nvfuser"`, `"aot_cudagraphs"`, `"ofi"`, `"fx2trt"`, `"onnxrt"` and `"ipex"`.
* **ray\_scope** (`str`, *optional*, defaults to `"last"`) —
  The scope to use when doing hyperparameter search with Ray. By default, `"last"` will be used. Ray will
  then use the last checkpoint of all trials, compare those, and select the best one. However, other options
  are also available. See the [Ray documentation](https://docs.ray.io/en/latest/tune/api_docs/analysis.html#ray.tune.ExperimentAnalysis.get_best_trial) for
  more options.
* **ddp\_timeout** (`int`, *optional*, defaults to 1800) —
  The timeout for `torch.distributed.init_process_group` calls, used to avoid GPU socket timeouts when
  performing slow operations in distributed runnings. Please refer the [PyTorch documentation]
  (<https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>) for more
  information.
* **use\_mps\_device** (`bool`, *optional*, defaults to `False`) —
  This argument is deprecated.`mps` device will be used if it is available similar to `cuda` device.
* **torch\_compile** (`bool`, *optional*, defaults to `False`) —
  Whether or not to compile the model using PyTorch 2.0
  [`torch.compile`](https://pytorch.org/get-started/pytorch-2.0/).

  This will use the best defaults for the [`torch.compile`
  API](https://pytorch.org/docs/stable/generated/torch.compile.html?highlight=torch+compile#torch.compile).
  You can customize the defaults with the argument `torch_compile_backend` and `torch_compile_mode` but we
  don’t guarantee any of them will work as the support is progressively rolled in in PyTorch.

  This flag and the whole compile API is experimental and subject to change in future releases.
* **torch\_compile\_backend** (`str`, *optional*) —
  The backend to use in `torch.compile`. If set to any value, `torch_compile` will be set to `True`.

  Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.

  This flag is experimental and subject to change in future releases.
* **torch\_compile\_mode** (`str`, *optional*) —
  The mode to use in `torch.compile`. If set to any value, `torch_compile` will be set to `True`.

  Refer to the PyTorch doc for possible values and note that they may change across PyTorch versions.

  This flag is experimental and subject to change in future releases.
* **include\_tokens\_per\_second** (`bool`, *optional*) —
  Whether or not to compute the number of tokens per second per device for training speed metrics.

  This will iterate over the entire training dataloader once beforehand,

  and will slow down the entire process.
* **include\_num\_input\_tokens\_seen** (`bool`, *optional*) —
  Whether or not to track the number of input tokens seen throughout training.

  May be slower in distributed training as gather operations must be called.
* **neftune\_noise\_alpha** (`Optional[float]`) —
  If not `None`, this will activate NEFTune noise embeddings. This can drastically improve model performance
  for instruction fine-tuning. Check out the [original paper](https://huggingface.co/papers/2310.05914) and the
  [original code](https://github.com/neelsjain/NEFTune). Support transformers `PreTrainedModel` and also
  `PeftModel` from peft. The original paper used values in the range [5.0, 15.0].
* **optim\_target\_modules** (`Union[str, list[str]]`, *optional*) —
  The target modules to optimize, i.e. the module names that you would like to train.
  Currently used for the GaLore algorithm (<https://huggingface.co/papers/2403.03507>) and APOLLO algorithm (<https://huggingface.co/papers/2412.05270>).
  See GaLore implementation (<https://github.com/jiaweizzhao/GaLore>) and APOLLO implementation (<https://github.com/zhuhanqing/APOLLO>) for more details.
  You need to make sure to pass a valid GaLore or APOLLO optimizer, e.g., one of: “apollo\_adamw”, “galore\_adamw”, “galore\_adamw\_8bit”, “galore\_adafactor” and make sure that the target modules are `nn.Linear` modules only.
* **batch\_eval\_metrics** (`Optional[bool]`, defaults to `False`) —
  If set to `True`, evaluation will call compute\_metrics at the end of each batch to accumulate statistics
  rather than saving all eval logits in memory. When set to `True`, you must pass a compute\_metrics function
  that takes a boolean argument `compute_result`, which when passed `True`, will trigger the final global
  summary statistics from the batch-level summary statistics you’ve accumulated over the evaluation set.
* **eval\_on\_start** (`bool`, *optional*, defaults to `False`) —
  Whether to perform a evaluation step (sanity check) before the training to ensure the validation steps works correctly.
* **eval\_use\_gather\_object** (`bool`, *optional*, defaults to `False`) —
  Whether to run recursively gather object in a nested list/tuple/dictionary of objects from all devices. This should only be enabled if users are not just returning tensors, and this is actively discouraged by PyTorch.
* **use\_liger\_kernel** (`bool`, *optional*, defaults to `False`) —
  Whether enable [Liger](https://github.com/linkedin/Liger-Kernel) Kernel for LLM model training.
  It can effectively increase multi-GPU training throughput by ~20% and reduces memory usage by ~60%, works out of the box with
  flash attention, PyTorch FSDP, and Microsoft DeepSpeed. Currently, it supports llama, mistral, mixtral and gemma models.
* **liger\_kernel\_config** (`Optional[dict]`, *optional*) —
  Configuration to be used for Liger Kernel. When use\_liger\_kernel=True, this dict is passed as keyword arguments to the
  `_apply_liger_kernel_to_instance` function, which specifies which kernels to apply. Available options vary by model but typically
  include: ‘rope’, ‘swiglu’, ‘cross\_entropy’, ‘fused\_linear\_cross\_entropy’, ‘rms\_norm’, etc. If `None`, use the default kernel configurations.
* **average\_tokens\_across\_devices** (`bool`, *optional*, defaults to `True`) —
  Whether or not to average tokens across devices. If enabled, will use all\_reduce to synchronize
  num\_tokens\_in\_batch for precise loss calculation. Reference:
  <https://github.com/huggingface/transformers/issues/34242>
* **predict\_with\_generate** (`bool`, *optional*, defaults to `False`) —
  Whether to use generate to calculate generative metrics (ROUGE, BLEU).
* **generation\_max\_length** (`int`, *optional*) —
  The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
  `max_length` value of the model configuration.
* **generation\_num\_beams** (`int`, *optional*) —
  The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
  `num_beams` value of the model configuration.
* **generation\_config** (`str` or `Path` or [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig), *optional*) —
  Allows to load a [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig) from the `from_pretrained` method. This can be either:
  + a string, the *model id* of a pretrained model configuration hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a configuration file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig.save_pretrained) method, e.g., `./my_model_directory/`.
  + a [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig) object.

TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
itself**.

Using [HfArgumentParser](/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.HfArgumentParser) we can turn this class into
[argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
command line.

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/training_args_seq2seq.py#L80)

( )

Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
serialization support). It obfuscates the token values by removing their value.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/trainer.md)
