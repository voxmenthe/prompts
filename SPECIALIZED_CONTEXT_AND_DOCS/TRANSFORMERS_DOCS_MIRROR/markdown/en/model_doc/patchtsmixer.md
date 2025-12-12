*This model was released on 2023-06-14 and added to Hugging Face Transformers on 2023-12-05.*

# PatchTSMixer

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The PatchTSMixer model was proposed in [TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting](https://huggingface.co/papers/2306.09364) by Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong and Jayant Kalagnanam.

PatchTSMixer is a lightweight time-series modeling approach based on the MLP-Mixer architecture. In this HuggingFace implementation, we provide PatchTSMixer’s capabilities to effortlessly facilitate lightweight mixing across patches, channels, and hidden features for effective multivariate time-series modeling. It also supports various attention mechanisms starting from simple gated attention to more complex self-attention blocks that can be customized accordingly. The model can be pretrained and subsequently used for various downstream tasks such as forecasting, classification and regression.

The abstract from the paper is the following:

*TSMixer is a lightweight neural architecture exclusively composed of multi-layer perceptron (MLP) modules designed for multivariate forecasting and representation learning on patched time series. Our model draws inspiration from the success of MLP-Mixer models in computer vision. We demonstrate the challenges involved in adapting Vision MLP-Mixer for time series and introduce empirically validated components to enhance accuracy. This includes a novel design paradigm of attaching online reconciliation heads to the MLP-Mixer backbone, for explicitly modeling the time-series properties such as hierarchy and channel-correlations. We also propose a Hybrid channel modeling approach to effectively handle noisy channel interactions and generalization across diverse datasets, a common challenge in existing patch channel-mixing methods. Additionally, a simple gated attention mechanism is introduced in the backbone to prioritize important features. By incorporating these lightweight components, we significantly enhance the learning capability of simple MLP structures, outperforming complex Transformer models with minimal computing usage. Moreover, TSMixer’s modular design enables compatibility with both supervised and masked self-supervised learning methods, making it a promising building block for time-series Foundation Models. TSMixer outperforms state-of-the-art MLP and Transformer models in forecasting by a considerable margin of 8-60%. It also outperforms the latest strong benchmarks of Patch-Transformer models (by 1-2%) with a significant reduction in memory and runtime (2-3X).*

This model was contributed by [ajati](https://huggingface.co/ajati), [vijaye12](https://huggingface.co/vijaye12),
[gsinthong](https://huggingface.co/gsinthong), [namctin](https://huggingface.co/namctin),
[wmgifford](https://huggingface.co/wmgifford), [kashif](https://huggingface.co/kashif).

## Usage example

The code snippet below shows how to randomly initialize a PatchTSMixer model. The model is compatible with the [Trainer API](../trainer).


```
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
from transformers import Trainer, TrainingArguments,


config = PatchTSMixerConfig(context_length = 512, prediction_length = 96)
model = PatchTSMixerForPrediction(config)
trainer = Trainer(model=model, args=training_args, 
            train_dataset=train_dataset,
            eval_dataset=valid_dataset)
trainer.train()
results = trainer.evaluate(test_dataset)
```

## Usage tips

The model can also be used for time series classification and time series regression. See the respective [PatchTSMixerForTimeSeriesClassification](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerForTimeSeriesClassification) and [PatchTSMixerForRegression](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerForRegression) classes.

## Resources

* A blog post explaining PatchTSMixer in depth can be found [here](https://huggingface.co/blog/patchtsmixer). The blog can also be opened in Google Colab.

## PatchTSMixerConfig

### class transformers.PatchTSMixerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/configuration_patchtsmixer.py#L26)

( context\_length: int = 32 patch\_length: int = 8 num\_input\_channels: int = 1 patch\_stride: int = 8 num\_parallel\_samples: int = 100 d\_model: int = 8 expansion\_factor: int = 2 num\_layers: int = 3 dropout: float = 0.2 mode: str = 'common\_channel' gated\_attn: bool = True norm\_mlp: str = 'LayerNorm' self\_attn: bool = False self\_attn\_heads: int = 1 use\_positional\_encoding: bool = False positional\_encoding\_type: str = 'sincos' scaling: typing.Union[str, bool, NoneType] = 'std' loss: str = 'mse' init\_std: float = 0.02 post\_init: bool = False norm\_eps: float = 1e-05 mask\_type: str = 'random' random\_mask\_ratio: float = 0.5 num\_forecast\_mask\_patches: typing.Union[int, list[int], NoneType] = [2] mask\_value: int = 0 masked\_loss: bool = True channel\_consistent\_masking: bool = True unmasked\_channel\_indices: typing.Optional[list[int]] = None head\_dropout: float = 0.2 distribution\_output: str = 'student\_t' prediction\_length: int = 16 prediction\_channel\_indices: typing.Optional[list] = None num\_targets: int = 3 output\_range: typing.Optional[list] = None head\_aggregation: str = 'max\_pool' \*\*kwargs  )

Parameters

* **context\_length** (`int`, *optional*, defaults to 32) —
  The context/history length for the input sequence.
* **patch\_length** (`int`, *optional*, defaults to 8) —
  The patch length for the input sequence.
* **num\_input\_channels** (`int`, *optional*, defaults to 1) —
  Number of input variates. For Univariate, set it to 1.
* **patch\_stride** (`int`, *optional*, defaults to 8) —
  Determines the overlap between two consecutive patches. Set it to patch\_length (or greater), if we want
  non-overlapping patches.
* **num\_parallel\_samples** (`int`, *optional*, defaults to 100) —
  The number of samples to generate in parallel for probabilistic forecast.
* **d\_model** (`int`, *optional*, defaults to 8) —
  Hidden dimension of the model. Recommended to set it as a multiple of patch\_length (i.e. 2-5X of
  patch\_length). Larger value indicates more complex model.
* **expansion\_factor** (`int`, *optional*, defaults to 2) —
  Expansion factor to use inside MLP. Recommended range is 2-5. Larger value indicates more complex model.
* **num\_layers** (`int`, *optional*, defaults to 3) —
  Number of layers to use. Recommended range is 3-15. Larger value indicates more complex model.
* **dropout** (`float`, *optional*, defaults to 0.2) —
  The dropout probability the `PatchTSMixer` backbone. Recommended range is 0.2-0.7
* **mode** (`str`, *optional*, defaults to `"common_channel"`) —
  Mixer Mode. Determines how to process the channels. Allowed values: “common\_channel”, “mix\_channel”. In
  “common\_channel” mode, we follow Channel-independent modelling with no explicit channel-mixing. Channel
  mixing happens in an implicit manner via shared weights across channels. (preferred first approach) In
  “mix\_channel” mode, we follow explicit channel-mixing in addition to patch and feature mixer. (preferred
  approach when channel correlations are very important to model)
* **gated\_attn** (`bool`, *optional*, defaults to `True`) —
  Enable Gated Attention.
* **norm\_mlp** (`str`, *optional*, defaults to `"LayerNorm"`) —
  Normalization layer (BatchNorm or LayerNorm).
* **self\_attn** (`bool`, *optional*, defaults to `False`) —
  Enable Tiny self attention across patches. This can be enabled when the output of Vanilla PatchTSMixer with
  gated attention is not satisfactory. Enabling this leads to explicit pair-wise attention and modelling
  across patches.
* **self\_attn\_heads** (`int`, *optional*, defaults to 1) —
  Number of self-attention heads. Works only when `self_attn` is set to `True`.
* **use\_positional\_encoding** (`bool`, *optional*, defaults to `False`) —
  Enable the use of positional embedding for the tiny self-attention layers. Works only when `self_attn` is
  set to `True`.
* **positional\_encoding\_type** (`str`, *optional*, defaults to `"sincos"`) —
  Positional encodings. Options `"random"` and `"sincos"` are supported. Works only when
  `use_positional_encoding` is set to `True`
* **scaling** (`string` or `bool`, *optional*, defaults to `"std"`) —
  Whether to scale the input targets via “mean” scaler, “std” scaler or no scaler if `None`. If `True`, the
  scaler is set to “mean”.
* **loss** (`string`, *optional*, defaults to `"mse"`) —
  The loss function for the model corresponding to the `distribution_output` head. For parametric
  distributions it is the negative log likelihood (“nll”) and for point estimates it is the mean squared
  error “mse”.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated normal weight initialization distribution.
* **post\_init** (`bool`, *optional*, defaults to `False`) —
  Whether to use custom weight initialization from `transformers` library, or the default initialization in
  `PyTorch`. Setting it to `False` performs `PyTorch` weight initialization.
* **norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  A value added to the denominator for numerical stability of normalization.
* **mask\_type** (`str`, *optional*, defaults to `"random"`) —
  Type of masking to use for Masked Pretraining mode. Allowed values are “random”, “forecast”. In Random
  masking, points are masked randomly. In Forecast masking, points are masked towards the end.
* **random\_mask\_ratio** (`float`, *optional*, defaults to 0.5) —
  Masking ratio to use when `mask_type` is `random`. Higher value indicates more masking.
* **num\_forecast\_mask\_patches** (`int` or `list`, *optional*, defaults to `[2]`) —
  Number of patches to be masked at the end of each batch sample. If it is an integer, all the samples in the
  batch will have the same number of masked patches. If it is a list, samples in the batch will be randomly
  masked by numbers defined in the list. This argument is only used for forecast pretraining.
* **mask\_value** (`float`, *optional*, defaults to `0.0`) —
  Mask value to use.
* **masked\_loss** (`bool`, *optional*, defaults to `True`) —
  Whether to compute pretraining loss only at the masked portions, or on the entire output.
* **channel\_consistent\_masking** (`bool`, *optional*, defaults to `True`) —
  When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
  across channels.
* **unmasked\_channel\_indices** (`list`, *optional*) —
  Channels that are not masked during pretraining.
* **head\_dropout** (`float`, *optional*, defaults to 0.2) —
  The dropout probability the `PatchTSMixer` head.
* **distribution\_output** (`string`, *optional*, defaults to `"student_t"`) —
  The distribution emission head for the model when loss is “nll”. Could be either “student\_t”, “normal” or
  “negative\_binomial”.
* **prediction\_length** (`int`, *optional*, defaults to 16) —
  Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
* **prediction\_channel\_indices** (`list`, *optional*) —
  List of channel indices to forecast. If None, forecast all channels. Target data is expected to have all
  channels and we explicitly filter the channels in prediction and target before loss computation.
* **num\_targets** (`int`, *optional*, defaults to 3) —
  Number of targets (dimensionality of the regressed variable) for a regression task.
* **output\_range** (`list`, *optional*) —
  Output range to restrict for the regression task. Defaults to None.
* **head\_aggregation** (`str`, *optional*, defaults to `"max_pool"`) —
  Aggregation mode to enable for classification or regression task. Allowed values are `None`, “use\_last”,
  “max\_pool”, “avg\_pool”.

This is the configuration class to store the configuration of a [PatchTSMixerModel](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerModel). It is used to instantiate a
PatchTSMixer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the PatchTSMixer
[ibm/patchtsmixer-etth1-pretrain](https://huggingface.co/ibm/patchtsmixer-etth1-pretrain) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import PatchTSMixerConfig, PatchTSMixerModel

>>> # Initializing a default PatchTSMixer configuration
>>> configuration = PatchTSMixerConfig()

>>> # Randomly initializing a model (with random weights) from the configuration
>>> model = PatchTSMixerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## PatchTSMixerModel

### class transformers.PatchTSMixerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py#L1218)

( config: PatchTSMixerConfig mask\_input: bool = False  )

Parameters

* **config** ([PatchTSMixerConfig](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **mask\_input** (`bool`, *optional*, defaults to `False`) —
  Whether to mask the input using the `PatchTSMixerMasking` module.

The PatchTSMixer Model for time-series forecasting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py#L1246)

( past\_values: Tensor observed\_mask: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = False return\_dict: typing.Optional[bool] = None  ) → `transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`) —
  Context values of the time series. For a pretraining task, this denotes the input time series to predict
  the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
  for classification or regression tasks, it denotes the appropriate context values of the time series.

  For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
  greater than 1.
* **observed\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*) —
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
  in `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
* **output\_hidden\_states** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PatchTSMixerConfig](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`) — Hidden-state at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Hidden-states of the model at the output of each layer.
* **patch\_input** (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`) — Patched input data to the model.
* **mask** (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`, *optional*) — Bool Tensor indicating True in masked patches and False otherwise.
* **loc** (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*) — Gives the mean of the context window per channel. Used for revin denorm outside the model, if revin
  enabled.
* **scale** (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*) — Gives the std dev of the context window per channel. Used for revin denorm outside the model, if revin
  enabled.

The [PatchTSMixerModel](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## PatchTSMixerForPrediction

### class transformers.PatchTSMixerForPrediction

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py#L1523)

( config: PatchTSMixerConfig  )

Parameters

* **config** (`PatchTSMixerConfig`) —
  Configuration.

`PatchTSMixer` for forecasting application.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py#L1567)

( past\_values: Tensor observed\_mask: typing.Optional[torch.Tensor] = None future\_values: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = False return\_loss: bool = True return\_dict: typing.Optional[bool] = None  ) → `transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForPredictionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`) —
  Context values of the time series. For a pretraining task, this denotes the input time series to predict
  the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
  for classification or regression tasks, it denotes the appropriate context values of the time series.

  For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
  greater than 1.
* **observed\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*) —
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
  in `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
* **future\_values** (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting, —
  `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*):
  Target values of the time series, that serve as labels for the model. The `future_values` is what the
  Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
  required for a pretraining task.

  For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
  to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
  pass the target data with all channels, as channel Filtering for both prediction and target will be
  manually applied before the loss computation.
* **output\_hidden\_states** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_loss** (`bool`, *optional*) —
  Whether to return the loss in the `forward` call.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForPredictionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForPredictionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PatchTSMixerConfig](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig)) and inputs.

* **loss** (`*optional*`, returned when `y` is provided, `torch.FloatTensor` of shape `()`) — Total loss.
* **prediction\_outputs** (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_input_channels)`) — Prediction output from the forecast head.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`) — Backbone embeddings before passing through the head.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **loc** (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`) — Input mean
* **scale** (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`) — Input std dev

The [PatchTSMixerForPrediction](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerForPrediction) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## PatchTSMixerForTimeSeriesClassification

### class transformers.PatchTSMixerForTimeSeriesClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py#L1762)

( config: PatchTSMixerConfig  )

Parameters

* **config** (`PatchTSMixerConfig`) —
  Configuration.

`PatchTSMixer` for classification application.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py#L1791)

( past\_values: Tensor target\_values: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = False return\_loss: bool = True return\_dict: typing.Optional[bool] = None  ) → `transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForTimeSeriesClassificationOutput` or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`) —
  Context values of the time series. For a pretraining task, this denotes the input time series to predict
  the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
  for classification or regression tasks, it denotes the appropriate context values of the time series.

  For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
  greater than 1.
* **target\_values** (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting, —
  `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*):
  Target
  values of the time series, that serve as labels for the model. The `target_values` is what the
  Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
  required for a pretraining task.

  For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
  to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
  pass the target data with all channels, as channel Filtering for both prediction and target will be
  manually applied before the loss computation.

  For a classification task, it has a shape of `(batch_size,)`.

  For a regression task, it has a shape of `(batch_size, num_targets)`.
* **output\_hidden\_states** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_loss** (`bool`, *optional*) —
  Whether to return the loss in the `forward` call.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForTimeSeriesClassificationOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForTimeSeriesClassificationOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PatchTSMixerConfig](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig)) and inputs.

* **loss** (`*optional*`, returned when `y` is provided, `torch.FloatTensor` of shape `()`) — Total loss.
* **prediction\_outputs** (`torch.FloatTensor` of shape `(batch_size, num_labels)`) — Prediction output from the classification head.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`) — Backbone embeddings before passing through the head.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [PatchTSMixerForTimeSeriesClassification](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerForTimeSeriesClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## PatchTSMixerForPretraining

### class transformers.PatchTSMixerForPretraining

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py#L1344)

( config: PatchTSMixerConfig  )

Parameters

* **config** ([PatchTSMixerConfig](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

`PatchTSMixer` for mask pretraining.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py#L1356)

( past\_values: Tensor observed\_mask: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = False return\_loss: bool = True return\_dict: typing.Optional[bool] = None  ) → `transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForPreTrainingOutput` or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`) —
  Context values of the time series. For a pretraining task, this denotes the input time series to predict
  the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
  for classification or regression tasks, it denotes the appropriate context values of the time series.

  For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
  greater than 1.
* **observed\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*) —
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
  in `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
* **output\_hidden\_states** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_loss** (`bool`, *optional*) —
  Whether to return the loss in the `forward` call.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForPreTrainingOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForPreTrainingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PatchTSMixerConfig](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig)) and inputs.

* **loss** (`*optional*`, returned when `y` is provided, `torch.FloatTensor` of shape `()`) — Total loss
* **prediction\_outputs** (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, patch_length)`) — Prediction output from the pretrain head.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`) — Backbone embeddings before passing through the head.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Hidden-states of the model at the output of each layer.

The [PatchTSMixerForPretraining](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerForPretraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## PatchTSMixerForRegression

### class transformers.PatchTSMixerForRegression

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py#L1941)

( config: PatchTSMixerConfig  )

Parameters

* **config** ([PatchTSMixerConfig](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

`PatchTSMixer` for regression application.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py#L1981)

( past\_values: Tensor target\_values: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = False return\_loss: bool = True return\_dict: typing.Optional[bool] = None  ) → `transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForRegressionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`) —
  Context values of the time series. For a pretraining task, this denotes the input time series to predict
  the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
  for classification or regression tasks, it denotes the appropriate context values of the time series.

  For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
  greater than 1.
* **target\_values** (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting, —
  `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*):
  Target values of the time series, that serve as labels for the model. The `target_values` is what the
  Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
  required for a pretraining task.

  For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
  to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
  pass the target data with all channels, as channel Filtering for both prediction and target will be
  manually applied before the loss computation.

  For a classification task, it has a shape of `(batch_size,)`.

  For a regression task, it has a shape of `(batch_size, num_targets)`.
* **output\_hidden\_states** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_loss** (`bool`, *optional*) —
  Whether to return the loss in the `forward` call.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForRegressionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.patchtsmixer.modeling_patchtsmixer.PatchTSMixerForRegressionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PatchTSMixerConfig](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig)) and inputs.

* **loss** (`*optional*`, returned when `y` is provided, `torch.FloatTensor` of shape `()`) — Total loss.
* **regression\_outputs** (`torch.FloatTensor` of shape `(batch_size, num_targets)`) — Prediction output from the regression head.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`) — Backbone embeddings before passing through the head.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [PatchTSMixerForRegression](/docs/transformers/v4.56.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerForRegression) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/patchtsmixer.md)
