# Callbacks

Callbacks are objects that can customize the behavior of the training loop in the PyTorch
[Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) that can inspect the training loop state (for progress reporting, logging on TensorBoard or other ML
platforms…) and take decisions (like early stopping).

Callbacks are “read only” pieces of code, apart from the [TrainerControl](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerControl) object they return, they
cannot change anything in the training loop. For customizations that require changes in the training loop, you should
subclass [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) and override the methods you need (see <trainer> for examples).

By default, `TrainingArguments.report_to` is set to `"all"`, so a [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) will use the following callbacks.

* [DefaultFlowCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.DefaultFlowCallback) which handles the default behavior for logging, saving and evaluation.
* [PrinterCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.PrinterCallback) or [ProgressCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.ProgressCallback) to display progress and print the
  logs (the first one is used if you deactivate tqdm through the [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments), otherwise
  it’s the second one).
* [TensorBoardCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.TensorBoardCallback) if tensorboard is accessible (either through PyTorch >= 1.4
  or tensorboardX).
* [TrackioCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.TrackioCallback) if [trackio](https://github.com/gradio-app/trackio) is installed.
* [WandbCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.WandbCallback) if [wandb](https://www.wandb.com/) is installed.
* [CometCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.CometCallback) if [comet\_ml](https://www.comet.com/site/) is installed.
* [MLflowCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.MLflowCallback) if [mlflow](https://www.mlflow.org/) is installed.
* [NeptuneCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.NeptuneCallback) if [neptune](https://neptune.ai/) is installed.
* [AzureMLCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.AzureMLCallback) if [azureml-sdk](https://pypi.org/project/azureml-sdk/) is
  installed.
* [CodeCarbonCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.CodeCarbonCallback) if [codecarbon](https://pypi.org/project/codecarbon/) is
  installed.
* [ClearMLCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.ClearMLCallback) if [clearml](https://github.com/allegroai/clearml) is installed.
* [DagsHubCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.DagsHubCallback) if [dagshub](https://dagshub.com/) is installed.
* [FlyteCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.FlyteCallback) if [flyte](https://flyte.org/) is installed.
* [DVCLiveCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.DVCLiveCallback) if [dvclive](https://dvc.org/doc/dvclive) is installed.
* [SwanLabCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.integrations.SwanLabCallback) if [swanlab](http://swanlab.cn/) is installed.

If a package is installed but you don’t wish to use the accompanying integration, you can change `TrainingArguments.report_to` to a list of just those integrations you want to use (e.g. `["azure_ml", "wandb"]`).

The main class that implements callbacks is [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback). It gets the
[TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) used to instantiate the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), can access that
Trainer’s internal state via [TrainerState](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerState), and can take some actions on the training loop via
[TrainerControl](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerControl).

## Available Callbacks

Here is the list of the available [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) in the library:

### class transformers.integrations.CometCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1190)

( )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [Comet ML](https://www.comet.com/site/).

#### setup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1204)

( args state model  )

Setup the optional Comet integration.

Environment:

* **COMET\_MODE** (`str`, *optional*, default to `get_or_create`):
  Control whether to create and log to a new Comet experiment or append to an existing experiment.
  It accepts the following values:
  + `get_or_create`: Decides automatically depending if
    `COMET_EXPERIMENT_KEY` is set and whether an Experiment
    with that key already exists or not.
  + `create`: Always create a new Comet Experiment.
  + `get`: Always try to append to an Existing Comet Experiment.
    Requires `COMET_EXPERIMENT_KEY` to be set.
  + `ONLINE`: **deprecated**, used to create an online
    Experiment. Use `COMET_START_ONLINE=1` instead.
  + `OFFLINE`: **deprecated**, used to created an offline
    Experiment. Use `COMET_START_ONLINE=0` instead.
  + `DISABLED`: **deprecated**, used to disable Comet logging.
    Use the `--report_to` flag to control the integrations used
    for logging result instead.
* **COMET\_PROJECT\_NAME** (`str`, *optional*):
  Comet project name for experiments.
* **COMET\_LOG\_ASSETS** (`str`, *optional*, defaults to `TRUE`):
  Whether or not to log training assets (tf event logs, checkpoints, etc), to Comet. Can be `TRUE`, or
  `FALSE`.

For a number of configurable items in the environment, see
[here](https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#explore-comet-configuration-options).

### class transformers.DefaultFlowCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L574)

( )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that handles the default flow of the training loop for logs, evaluation and checkpoints.

### class transformers.PrinterCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L701)

( )

A bare [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that just prints the logs.

### class transformers.ProgressCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L627)

( max\_str\_len: int = 100  )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that displays the progress of training or evaluation.
You can modify `max_str_len` to control how long strings are truncated when logging.

### class transformers.EarlyStoppingCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L712)

( early\_stopping\_patience: int = 1 early\_stopping\_threshold: typing.Optional[float] = 0.0  )

Parameters

* **early\_stopping\_patience** (`int`) —
  Use with `metric_for_best_model` to stop training when the specified metric worsens for
  `early_stopping_patience` evaluation calls.
* **early\_stopping\_threshold(`float`,** *optional*) —
  Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
  specified metric must improve to satisfy early stopping conditions. `

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that handles early stopping.

This callback depends on [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) argument *load\_best\_model\_at\_end* functionality to set best\_metric
in [TrainerState](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerState). Note that if the [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) argument *save\_steps* differs from *eval\_steps*, the
early stopping will not occur until the next save step.

### class transformers.integrations.TensorBoardCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L671)

( tb\_writer = None  )

Parameters

* **tb\_writer** (`SummaryWriter`, *optional*) —
  The writer to use. Will instantiate one if not set.

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

### class transformers.integrations.TrackioCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1068)

( )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that logs metrics to Trackio.

It records training metrics, model (and PEFT) configuration, and GPU memory usage.
If `nvidia-ml-py` is installed, GPU power consumption is also tracked.

**Requires**:


```
pip install trackio
```

#### setup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1091)

( args state model \*\*kwargs  )

Setup the optional Trackio integration.

To customize the setup you can also override the following environment variables:

Environment:

* **TRACKIO\_PROJECT** (`str`, *optional*, defaults to `"huggingface"`):
  The name of the project (can be an existing project to continue tracking or a new project to start tracking
  from scratch).
* **TRACKIO\_SPACE\_ID** (`str`, *optional*, defaults to `None`):
  If set, the project will be logged to a Hugging Face Space instead of a local directory. Should be a
  complete Space name like `"username/reponame"` or `"orgname/reponame"`, or just `“reponame” in which case
  the Space will be created in the currently-logged-in Hugging Face user’s namespace. If the Space does not
  exist, it will be created. If the Space already exists, the project will be logged to it.

### class transformers.integrations.WandbCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L804)

( )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that logs metrics, media, model checkpoints to [Weight and Biases](https://www.wandb.com/).

#### setup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L833)

( args state model \*\*kwargs  )

Setup the optional Weights & Biases (*wandb*) integration.

One can subclass and override this method to customize the setup if needed. Find more information
[here](https://docs.wandb.ai/guides/integrations/huggingface). You can also override the following environment
variables:

Environment:

* **WANDB\_LOG\_MODEL** (`str`, *optional*, defaults to `"false"`):
  Whether to log model and checkpoints during training. Can be `"end"`, `"checkpoint"` or `"false"`. If set
  to `"end"`, the model will be uploaded at the end of training. If set to `"checkpoint"`, the checkpoint
  will be uploaded every `args.save_steps` . If set to `"false"`, the model will not be uploaded. Use along
  with `load_best_model_at_end()` to upload best model.

  Deprecated in 5.0

  Setting `WANDB_LOG_MODEL` as `bool` will be deprecated in version 5 of 🤗 Transformers.
* **WANDB\_WATCH** (`str`, *optional* defaults to `"false"`):
  Can be `"gradients"`, `"all"`, `"parameters"`, or `"false"`. Set to `"all"` to log gradients and
  parameters.
* **WANDB\_PROJECT** (`str`, *optional*, defaults to `"huggingface"`):
  Set this to a custom string to store results in a different project.
* **WANDB\_DISABLED** (`bool`, *optional*, defaults to `False`):
  Whether to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.

### class transformers.integrations.MLflowCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1353)

( )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [MLflow](https://www.mlflow.org/). Can be disabled by setting
environment variable `DISABLE_MLFLOW_INTEGRATION = TRUE`.

#### setup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1372)

( args state model  )

Setup the optional MLflow integration.

Environment:

* **HF\_MLFLOW\_LOG\_ARTIFACTS** (`str`, *optional*):
  Whether to use MLflow `.log_artifact()` facility to log artifacts. This only makes sense if logging to a
  remote server, e.g. s3 or GCS. If set to `True` or *1*, will copy each saved checkpoint on each save in
  [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments)’s `output_dir` to the local or remote artifact storage. Using it without a remote
  storage will just copy the files to your artifact location.
* **MLFLOW\_TRACKING\_URI** (`str`, *optional*):
  Whether to store runs at a specific path or remote server. Unset by default, which skips setting the
  tracking URI entirely.
* **MLFLOW\_EXPERIMENT\_NAME** (`str`, *optional*, defaults to `None`):
  Whether to use an MLflow experiment\_name under which to launch the run. Default to `None` which will point
  to the `Default` experiment in MLflow. Otherwise, it is a case sensitive name of the experiment to be
  activated. If an experiment with this name does not exist, a new experiment with this name is created.
* **MLFLOW\_TAGS** (`str`, *optional*):
  A string dump of a dictionary of key/value pair to be added to the MLflow run as tags. Example:
  `os.environ['MLFLOW_TAGS']='{"release.candidate": "RC1", "release.version": "2.2.0"}'`.
* **MLFLOW\_NESTED\_RUN** (`str`, *optional*):
  Whether to use MLflow nested runs. If set to `True` or *1*, will create a nested run inside the current
  run.
* **MLFLOW\_RUN\_ID** (`str`, *optional*):
  Allow to reattach to an existing run which can be useful when resuming training from a checkpoint. When
  `MLFLOW_RUN_ID` environment variable is set, `start_run` attempts to resume a run with the specified run ID
  and other parameters are ignored.
* **MLFLOW\_FLATTEN\_PARAMS** (`str`, *optional*, defaults to `False`):
  Whether to flatten the parameters dictionary before logging.
* **MLFLOW\_MAX\_LOG\_PARAMS** (`int`, *optional*):
  Set the maximum number of parameters to log in the run.

### class transformers.integrations.AzureMLCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1330)

( azureml\_run = None  )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [AzureML](https://pypi.org/project/azureml-sdk/).

### class transformers.integrations.CodeCarbonCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1867)

( )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that tracks the CO2 emission of training.

### class transformers.integrations.NeptuneCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1594)

( api\_token: typing.Optional[str] = None project: typing.Optional[str] = None name: typing.Optional[str] = None base\_namespace: str = 'finetuning' run = None log\_parameters: bool = True log\_checkpoints: typing.Optional[str] = None \*\*neptune\_run\_kwargs  )

Parameters

* **api\_token** (`str`, *optional*) — Neptune API token obtained upon registration.
  You can leave this argument out if you have saved your token to the `NEPTUNE_API_TOKEN` environment
  variable (strongly recommended). See full setup instructions in the
  [docs](https://docs.neptune.ai/setup/installation).
* **project** (`str`, *optional*) — Name of an existing Neptune project, in the form “workspace-name/project-name”.
  You can find and copy the name in Neptune from the project settings -> Properties. If None (default), the
  value of the `NEPTUNE_PROJECT` environment variable is used.
* **name** (`str`, *optional*) — Custom name for the run.
* **base\_namespace** (`str`, *optional*, defaults to “finetuning”) — In the Neptune run, the root namespace
  that will contain all of the metadata logged by the callback.
* **log\_parameters** (`bool`, *optional*, defaults to `True`) —
  If True, logs all Trainer arguments and model parameters provided by the Trainer.
* **log\_checkpoints** (`str`, *optional*) — If “same”, uploads checkpoints whenever they are saved by the Trainer.
  If “last”, uploads only the most recently saved checkpoint. If “best”, uploads the best checkpoint (among
  the ones saved by the Trainer). If `None`, does not upload checkpoints.
* **run** (`Run`, *optional*) — Pass a Neptune run object if you want to continue logging to an existing run.
  Read more about resuming runs in the [docs](https://docs.neptune.ai/logging/to_existing_object).
* \***\*neptune\_run\_kwargs** (*optional*) —
  Additional keyword arguments to be passed directly to the
  [`neptune.init_run()`](https://docs.neptune.ai/api/neptune#init_run) function when a new run is created.

TrainerCallback that sends the logs to [Neptune](https://app.neptune.ai).

For instructions and examples, see the [Transformers integration
guide](https://docs.neptune.ai/integrations/transformers) in the Neptune documentation.

### class transformers.integrations.ClearMLCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1901)

( )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [ClearML](https://clear.ml/).

Environment:

* **CLEARML\_PROJECT** (`str`, *optional*, defaults to `HuggingFace Transformers`):
  ClearML project name.
* **CLEARML\_TASK** (`str`, *optional*, defaults to `Trainer`):
  ClearML task name.
* **CLEARML\_LOG\_MODEL** (`bool`, *optional*, defaults to `False`):
  Whether to log models as artifacts during training.

### class transformers.integrations.DagsHubCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1534)

( )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that logs to [DagsHub](https://dagshub.com/). Extends `MLflowCallback`

#### setup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L1548)

( \*args \*\*kwargs  )

Setup the DagsHub’s Logging integration.

Environment:

* **HF\_DAGSHUB\_LOG\_ARTIFACTS** (`str`, *optional*):
  Whether to save the data and model artifacts for the experiment. Default to `False`.

### class transformers.integrations.FlyteCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L2152)

( save\_log\_history: bool = True sync\_checkpoints: bool = True  )

Parameters

* **save\_log\_history** (`bool`, *optional*, defaults to `True`) —
  When set to True, the training logs are saved as a Flyte Deck.
* **sync\_checkpoints** (`bool`, *optional*, defaults to `True`) —
  When set to True, checkpoints are synced with Flyte and can be used to resume training in the case of an
  interruption.

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [Flyte](https://flyte.org/).
NOTE: This callback only works within a Flyte task.

Example:


```
# Note: This example skips over some setup steps for brevity.
from flytekit import current_context, task


@task
def train_hf_transformer():
    cp = current_context().checkpoint
    trainer = Trainer(..., callbacks=[FlyteCallback()])
    output = trainer.train(resume_from_checkpoint=cp.restore())
```

### class transformers.integrations.DVCLiveCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L2215)

( live: typing.Optional[typing.Any] = None log\_model: typing.Union[typing.Literal['all'], bool, NoneType] = None \*\*kwargs  )

Parameters

* **live** (`dvclive.Live`, *optional*, defaults to `None`) —
  Optional Live instance. If None, a new instance will be created using \*\*kwargs.
* **log\_model** (Union[Literal[“all”], bool], *optional*, defaults to `None`) —
  Whether to use `dvclive.Live.log_artifact()` to log checkpoints created by [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer). If set to `True`,
  the final checkpoint is logged at the end of training. If set to `"all"`, the entire
  [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments)’s `output_dir` is logged at each checkpoint.

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [DVCLive](https://www.dvc.org/doc/dvclive).

Use the environment variables below in `setup` to configure the integration. To customize this callback beyond
those environment variables, see [here](https://dvc.org/doc/dvclive/ml-frameworks/huggingface).

#### setup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L2256)

( args state model  )

Setup the optional DVCLive integration. To customize this callback beyond the environment variables below, see
[here](https://dvc.org/doc/dvclive/ml-frameworks/huggingface).

Environment:

* **HF\_DVCLIVE\_LOG\_MODEL** (`str`, *optional*):
  Whether to use `dvclive.Live.log_artifact()` to log checkpoints created by [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer). If set to `True` or
  *1*, the final checkpoint is logged at the end of training. If set to `all`, the entire
  [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments)’s `output_dir` is logged at each checkpoint.

### class transformers.integrations.SwanLabCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L2320)

( )

A [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) that logs metrics, media, model checkpoints to [SwanLab](https://swanlab.cn/).

#### setup

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/integration_utils.py#L2334)

( args state model \*\*kwargs  )

Setup the optional SwanLab (*swanlab*) integration.

One can subclass and override this method to customize the setup if needed. Find more information
[here](https://docs.swanlab.cn/guide_cloud/integration/integration-huggingface-transformers.html).

You can also override the following environment variables. Find more information about environment
variables [here](https://docs.swanlab.cn/en/api/environment-variable.html#environment-variables)

Environment:

* **SWANLAB\_API\_KEY** (`str`, *optional*, defaults to `None`):
  Cloud API Key. During login, this environment variable is checked first. If it doesn’t exist, the system
  checks if the user is already logged in. If not, the login process is initiated.

  + If a string is passed to the login interface, this environment variable is ignored.
  + If the user is already logged in, this environment variable takes precedence over locally stored
    login information.
* **SWANLAB\_PROJECT** (`str`, *optional*, defaults to `None`):
  Set this to a custom string to store results in a different project. If not specified, the name of the current
  running directory is used.
* **SWANLAB\_LOG\_DIR** (`str`, *optional*, defaults to `swanlog`):
  This environment variable specifies the storage path for log files when running in local mode.
  By default, logs are saved in a folder named swanlog under the working directory.
* **SWANLAB\_MODE** (`Literal["local", "cloud", "disabled"]`, *optional*, defaults to `cloud`):
  SwanLab’s parsing mode, which involves callbacks registered by the operator. Currently, there are three modes:
  local, cloud, and disabled. Note: Case-sensitive. Find more information
  [here](https://docs.swanlab.cn/en/api/py-init.html#swanlab-init)
* **SWANLAB\_LOG\_MODEL** (`str`, *optional*, defaults to `None`):
  SwanLab does not currently support the save mode functionality.This feature will be available in a future
  release
* **SWANLAB\_WEB\_HOST** (`str`, *optional*, defaults to `None`):
  Web address for the SwanLab cloud environment for private version (its free)
* **SWANLAB\_API\_HOST** (`str`, *optional*, defaults to `None`):
  API address for the SwanLab cloud environment for private version (its free)

## TrainerCallback

### class transformers.TrainerCallback

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L297)

( )

Parameters

* **args** ([TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments)) —
  The training arguments used to instantiate the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer).
* **state** ([TrainerState](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerState)) —
  The current state of the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer).
* **control** ([TrainerControl](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerControl)) —
  The object that is returned to the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) and can be used to make some decisions.
* **model** ([PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel) or `torch.nn.Module`) —
  The model being trained.
* **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) —
  The tokenizer used for encoding the data. This is deprecated in favour of `processing_class`.
* **processing\_class** ([`PreTrainedTokenizer` or `BaseImageProcessor` or `ProcessorMixin` or `FeatureExtractionMixin`]) —
  The processing class used for encoding the data. Can be a tokenizer, a processor, an image processor or a feature extractor.
* **optimizer** (`torch.optim.Optimizer`) —
  The optimizer used for the training steps.
* **lr\_scheduler** (`torch.optim.lr_scheduler.LambdaLR`) —
  The scheduler used for setting the learning rate.
* **train\_dataloader** (`torch.utils.data.DataLoader`, *optional*) —
  The current dataloader used for training.
* **eval\_dataloader** (`torch.utils.data.DataLoader`, *optional*) —
  The current dataloader used for evaluation.
* **metrics** (`dict[str, float]`) —
  The metrics computed by the last evaluation phase.

  Those are only accessible in the event `on_evaluate`.
* **logs** (`dict[str, float]`) —
  The values to log.

  Those are only accessible in the event `on_log`.

A class for objects that will inspect the state of the training loop at some events and take some decisions. At
each of those events the following arguments are available:

The `control` object is the only one that can be changed by the callback, in which case the event that changes it
should return the modified version.

The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
simple [PrinterCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.PrinterCallback).

Example:


```
class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
```

#### on\_epoch\_begin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L368)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called at the beginning of an epoch.

#### on\_epoch\_end

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L374)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called at the end of an epoch.

#### on\_evaluate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L412)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called after an evaluation phase.

#### on\_init\_end

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L350)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called at the end of the initialization of the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer).

#### on\_log

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L430)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called after logging the last logs.

#### on\_optimizer\_step

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L393)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called after the optimizer step but before gradients are zeroed out. Useful for monitoring gradients.

#### on\_pre\_optimizer\_step

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L387)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called before the optimizer step but after gradient clipping. Useful for monitoring gradients.

#### on\_predict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L418)

( args: TrainingArguments state: TrainerState control: TrainerControl metrics \*\*kwargs  )

Event called after a successful prediction.

#### on\_prediction\_step

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L436)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called after a prediction step.

#### on\_save

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L424)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called after a checkpoint save.

#### on\_step\_begin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L380)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called at the beginning of a training step. If using gradient accumulation, one training step might take
several inputs.

#### on\_step\_end

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L405)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called at the end of a training step. If using gradient accumulation, one training step might take
several inputs.

#### on\_substep\_end

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L399)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called at the end of an substep during gradient accumulation.

#### on\_train\_begin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L356)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called at the beginning of training.

#### on\_train\_end

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L362)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs  )

Event called at the end of training.

Here is an example of how to register a custom callback with the PyTorch [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer):


```
class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MyCallback],  # We can either pass the callback class this way or an instance of it (MyCallback())
)
```

Another way to register a callback is to call `trainer.add_callback()` as follows:


```
trainer = Trainer(...)
trainer.add_callback(MyCallback)
# Alternatively, we can pass an instance of the callback class
trainer.add_callback(MyCallback())
```

## TrainerState

### class transformers.TrainerState

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L36)

( epoch: typing.Optional[float] = None global\_step: int = 0 max\_steps: int = 0 logging\_steps: int = 500 eval\_steps: int = 500 save\_steps: int = 500 train\_batch\_size: typing.Optional[int] = None num\_train\_epochs: int = 0 num\_input\_tokens\_seen: int = 0 total\_flos: float = 0 log\_history: list = None best\_metric: typing.Optional[float] = None best\_global\_step: typing.Optional[int] = None best\_model\_checkpoint: typing.Optional[str] = None is\_local\_process\_zero: bool = True is\_world\_process\_zero: bool = True is\_hyper\_param\_search: bool = False trial\_name: typing.Optional[str] = None trial\_params: dict = None stateful\_callbacks: list = None  )

Parameters

* **epoch** (`float`, *optional*) —
  Only set during training, will represent the epoch the training is at (the decimal part being the
  percentage of the current epoch completed).
* **global\_step** (`int`, *optional*, defaults to 0) —
  During training, represents the number of update steps completed.
* **max\_steps** (`int`, *optional*, defaults to 0) —
  The number of update steps to do during the current training.
* **logging\_steps** (`int`, *optional*, defaults to 500) —
  Log every X updates steps
* **eval\_steps** (`int`, *optional*) —
  Run an evaluation every X steps.
* **save\_steps** (`int`, *optional*, defaults to 500) —
  Save checkpoint every X updates steps.
* **train\_batch\_size** (`int`, *optional*) —
  The batch size for the training dataloader. Only needed when
  `auto_find_batch_size` has been used.
* **num\_input\_tokens\_seen** (`int`, *optional*, defaults to 0) —
  When tracking the inputs tokens, the number of tokens seen during training (number of input tokens, not the
  number of prediction tokens).
* **total\_flos** (`float`, *optional*, defaults to 0) —
  The total number of floating operations done by the model since the beginning of training (stored as floats
  to avoid overflow).
* **log\_history** (`list[dict[str, float]]`, *optional*) —
  The list of logs done since the beginning of training.
* **best\_metric** (`float`, *optional*) —
  When tracking the best model, the value of the best metric encountered so far.
* **best\_global\_step** (`int`, *optional*) —
  When tracking the best model, the step at which the best metric was encountered.
  Used for setting `best_model_checkpoint`.
* **best\_model\_checkpoint** (`str`, *optional*) —
  When tracking the best model, the value of the name of the checkpoint for the best model encountered so
  far.
* **is\_local\_process\_zero** (`bool`, *optional*, defaults to `True`) —
  Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
  several machines) main process.
* **is\_world\_process\_zero** (`bool`, *optional*, defaults to `True`) —
  Whether or not this process is the global main process (when training in a distributed fashion on several
  machines, this is only going to be `True` for one process).
* **is\_hyper\_param\_search** (`bool`, *optional*, defaults to `False`) —
  Whether we are in the process of a hyper parameter search using Trainer.hyperparameter\_search. This will
  impact the way data will be logged in TensorBoard.
* **stateful\_callbacks** (`list[StatefulTrainerCallback]`, *optional*) —
  Callbacks attached to the `Trainer` that should have their states be saved or restored.
  Relevant callbacks should implement a `state` and `from_state` function.

A class containing the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) inner state that will be saved along the model and optimizer when checkpointing
and passed to the [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback).

In all this class, one step is to be understood as one update step. When using gradient accumulation, one update
step may require several forward and backward passes: if you use `gradient_accumulation_steps=n`, then one update
step requires going through *n* batches.

#### compute\_steps

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L157)

( args max\_steps  )

Calculates and stores the absolute value for logging,
eval, and save steps based on if it was a proportion
or not.

#### init\_training\_references

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L170)

( trainer max\_steps num\_train\_epochs trial  )

Stores the initial training references needed in `self`

#### load\_from\_json

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L150)

( json\_path: str  )

Create an instance from the content of `json_path`.

#### save\_to\_json

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L144)

( json\_path: str  )

Save the content of this instance in JSON format inside `json_path`.

## TrainerControl

### class transformers.TrainerControl

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/trainer_callback.py#L236)

( should\_training\_stop: bool = False should\_epoch\_stop: bool = False should\_save: bool = False should\_evaluate: bool = False should\_log: bool = False  )

Parameters

* **should\_training\_stop** (`bool`, *optional*, defaults to `False`) —
  Whether or not the training should be interrupted.

  If `True`, this variable will not be set back to `False`. The training will just stop.
* **should\_epoch\_stop** (`bool`, *optional*, defaults to `False`) —
  Whether or not the current epoch should be interrupted.

  If `True`, this variable will be set back to `False` at the beginning of the next epoch.
* **should\_save** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should be saved at this step.

  If `True`, this variable will be set back to `False` at the beginning of the next step.
* **should\_evaluate** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should be evaluated at this step.

  If `True`, this variable will be set back to `False` at the beginning of the next step.
* **should\_log** (`bool`, *optional*, defaults to `False`) —
  Whether or not the logs should be reported at this step.

  If `True`, this variable will be set back to `False` at the beginning of the next step.

A class that handles the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) control flow. This class is used by the [TrainerCallback](/docs/transformers/v4.56.2/en/main_classes/callback#transformers.TrainerCallback) to activate some
switches in the training loop.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/callback.md)
