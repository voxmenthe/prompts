*This model was released on 2022-11-27 and added to Hugging Face Transformers on 2023-11-13.*

# PatchTST

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The PatchTST model was proposed in [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://huggingface.co/papers/2211.14730) by Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong and Jayant Kalagnanam.

At a high level the model vectorizes time series into patches of a given size and encodes the resulting sequence of vectors via a Transformer that then outputs the prediction length forecast via an appropriate head. The model is illustrated in the following figure:

![model](https://github.com/namctin/transformers/assets/8100/150af169-29de-419a-8d98-eb78251c21fa)

The abstract from the paper is the following:

*We propose an efficient design of Transformer-based models for multivariate time series forecasting and self-supervised representation learning. It is based on two key components: (i) segmentation of time series into subseries-level patches which are served as input tokens to Transformer; (ii) channel-independence where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series. Patching design naturally has three-fold benefit: local semantic information is retained in the embedding; computation and memory usage of the attention maps are quadratically reduced given the same look-back window; and the model can attend longer history. Our channel-independent patch time series Transformer (PatchTST) can improve the long-term forecasting accuracy significantly when compared with that of SOTA Transformer-based models. We also apply our model to self-supervised pre-training tasks and attain excellent fine-tuning performance, which outperforms supervised training on large datasets. Transferring of masked pre-trained representation on one dataset to others also produces SOTA forecasting accuracy.*

This model was contributed by [namctin](https://huggingface.co/namctin), [gsinthong](https://huggingface.co/gsinthong), [diepi](https://huggingface.co/diepi), [vijaye12](https://huggingface.co/vijaye12), [wmgifford](https://huggingface.co/wmgifford), and [kashif](https://huggingface.co/kashif). The original code can be found [here](https://github.com/yuqinie98/PatchTST).

## Usage tips

The model can also be used for time series classification and time series regression. See the respective [PatchTSTForClassification](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTForClassification) and [PatchTSTForRegression](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTForRegression) classes.

## Resources

* A blog post explaining PatchTST in depth can be found [here](https://huggingface.co/blog/patchtst). The blog can also be opened in Google Colab.

## PatchTSTConfig

### class transformers.PatchTSTConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/configuration_patchtst.py#L26)

( num\_input\_channels: int = 1 context\_length: int = 32 distribution\_output: str = 'student\_t' loss: str = 'mse' patch\_length: int = 1 patch\_stride: int = 1 num\_hidden\_layers: int = 3 d\_model: int = 128 num\_attention\_heads: int = 4 share\_embedding: bool = True channel\_attention: bool = False ffn\_dim: int = 512 norm\_type: str = 'batchnorm' norm\_eps: float = 1e-05 attention\_dropout: float = 0.0 positional\_dropout: float = 0.0 path\_dropout: float = 0.0 ff\_dropout: float = 0.0 bias: bool = True activation\_function: str = 'gelu' pre\_norm: bool = True positional\_encoding\_type: str = 'sincos' use\_cls\_token: bool = False init\_std: float = 0.02 share\_projection: bool = True scaling: typing.Union[bool, str, NoneType] = 'std' do\_mask\_input: typing.Optional[bool] = None mask\_type: str = 'random' random\_mask\_ratio: float = 0.5 num\_forecast\_mask\_patches: typing.Union[list[int], int, NoneType] = [2] channel\_consistent\_masking: typing.Optional[bool] = False unmasked\_channel\_indices: typing.Optional[list[int]] = None mask\_value: int = 0 pooling\_type: str = 'mean' head\_dropout: float = 0.0 prediction\_length: int = 24 num\_targets: int = 1 output\_range: typing.Optional[list] = None num\_parallel\_samples: int = 100 \*\*kwargs  )

Parameters

* **num\_input\_channels** (`int`, *optional*, defaults to 1) —
  The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of
  multivariate targets.
* **context\_length** (`int`, *optional*, defaults to 32) —
  The context length of the input sequence.
* **distribution\_output** (`str`, *optional*, defaults to `"student_t"`) —
  The distribution emission head for the model when loss is “nll”. Could be either “student\_t”, “normal” or
  “negative\_binomial”.
* **loss** (`str`, *optional*, defaults to `"mse"`) —
  The loss function for the model corresponding to the `distribution_output` head. For parametric
  distributions it is the negative log likelihood (“nll”) and for point estimates it is the mean squared
  error “mse”.
* **patch\_length** (`int`, *optional*, defaults to 1) —
  Define the patch length of the patchification process.
* **patch\_stride** (`int`, *optional*, defaults to 1) —
  Define the stride of the patchification process.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 3) —
  Number of hidden layers.
* **d\_model** (`int`, *optional*, defaults to 128) —
  Dimensionality of the transformer layers.
* **num\_attention\_heads** (`int`, *optional*, defaults to 4) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **share\_embedding** (`bool`, *optional*, defaults to `True`) —
  Sharing the input embedding across all channels.
* **channel\_attention** (`bool`, *optional*, defaults to `False`) —
  Activate channel attention block in the Transformer to allow channels to attend each other.
* **ffn\_dim** (`int`, *optional*, defaults to 512) —
  Dimension of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **norm\_type** (`str` , *optional*, defaults to `"batchnorm"`) —
  Normalization at each Transformer layer. Can be `"batchnorm"` or `"layernorm"`.
* **norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  A value added to the denominator for numerical stability of normalization.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for the attention probabilities.
* **positional\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability in the positional embedding layer.
* **path\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout path in the residual block.
* **ff\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability used between the two layers of the feed-forward networks.
* **bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add bias in the feed-forward networks.
* **activation\_function** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (string) in the Transformer.`"gelu"` and `"relu"` are supported.
* **pre\_norm** (`bool`, *optional*, defaults to `True`) —
  Normalization is applied before self-attention if pre\_norm is set to `True`. Otherwise, normalization is
  applied after residual block.
* **positional\_encoding\_type** (`str`, *optional*, defaults to `"sincos"`) —
  Positional encodings. Options `"random"` and `"sincos"` are supported.
* **use\_cls\_token** (`bool`, *optional*, defaults to `False`) —
  Whether cls token is used.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated normal weight initialization distribution.
* **share\_projection** (`bool`, *optional*, defaults to `True`) —
  Sharing the projection layer across different channels in the forecast head.
* **scaling** (`Union`, *optional*, defaults to `"std"`) —
  Whether to scale the input targets via “mean” scaler, “std” scaler or no scaler if `None`. If `True`, the
  scaler is set to “mean”.
* **do\_mask\_input** (`bool`, *optional*) —
  Apply masking during the pretraining.
* **mask\_type** (`str`, *optional*, defaults to `"random"`) —
  Masking type. Only `"random"` and `"forecast"` are currently supported.
* **random\_mask\_ratio** (`float`, *optional*, defaults to 0.5) —
  Masking ratio applied to mask the input data during random pretraining.
* **num\_forecast\_mask\_patches** (`int` or `list`, *optional*, defaults to `[2]`) —
  Number of patches to be masked at the end of each batch sample. If it is an integer,
  all the samples in the batch will have the same number of masked patches. If it is a list,
  samples in the batch will be randomly masked by numbers defined in the list. This argument is only used
  for forecast pretraining.
* **channel\_consistent\_masking** (`bool`, *optional*, defaults to `False`) —
  If channel consistent masking is True, all the channels will have the same masking pattern.
* **unmasked\_channel\_indices** (`list`, *optional*) —
  Indices of channels that are not masked during pretraining. Values in the list are number between 1 and
  `num_input_channels`
* **mask\_value** (`int`, *optional*, defaults to 0) —
  Values in the masked patches will be filled by `mask_value`.
* **pooling\_type** (`str`, *optional*, defaults to `"mean"`) —
  Pooling of the embedding. `"mean"`, `"max"` and `None` are supported.
* **head\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for head.
* **prediction\_length** (`int`, *optional*, defaults to 24) —
  The prediction horizon that the model will output.
* **num\_targets** (`int`, *optional*, defaults to 1) —
  Number of targets for regression and classification tasks. For classification, it is the number of
  classes.
* **output\_range** (`list`, *optional*) —
  Output range for regression task. The range of output values can be set to enforce the model to produce
  values within a range.
* **num\_parallel\_samples** (`int`, *optional*, defaults to 100) —
  The number of samples is generated in parallel for probabilistic prediction.

This is the configuration class to store the configuration of an [PatchTSTModel](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTModel). It is used to instantiate an
PatchTST model according to the specified arguments, defining the model architecture.
[ibm/patchtst](https://huggingface.co/ibm/patchtst) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import PatchTSTConfig, PatchTSTModel

>>> # Initializing an PatchTST configuration with 12 time steps for prediction
>>> configuration = PatchTSTConfig(prediction_length=12)

>>> # Randomly initializing a model (with random weights) from the configuration
>>> model = PatchTSTModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## PatchTSTModel

### class transformers.PatchTSTModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/modeling_patchtst.py#L1067)

( config: PatchTSTConfig  )

Parameters

* **config** ([PatchTSTConfig](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Patchtst Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/modeling_patchtst.py#L1086)

( past\_values: Tensor past\_observed\_mask: typing.Optional[torch.Tensor] = None future\_values: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  )

Parameters

* **past\_values** (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*) —
  Input sequence to the model
* **past\_observed\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*) —
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
  in `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
* **future\_values** (`torch.BoolTensor` of shape `(batch_size, prediction_length, num_input_channels)`, *optional*) —
  Future target values associated with the `past_values`
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the output attention of all layers
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a `ModelOutput` instead of a plain tuple.

Examples:


```
>>> from huggingface_hub import hf_hub_download
>>> import torch
>>> from transformers import PatchTSTModel

>>> file = hf_hub_download(
...     repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
... )
>>> batch = torch.load(file)

>>> model = PatchTSTModel.from_pretrained("namctin/patchtst_etth1_pretrain")

>>> # during training, one provides both past and future values
>>> outputs = model(
...     past_values=batch["past_values"],
...     future_values=batch["future_values"],
... )

>>> last_hidden_state = outputs.last_hidden_state
```

## PatchTSTForPrediction

### class transformers.PatchTSTForPrediction

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/modeling_patchtst.py#L1558)

( config: PatchTSTConfig  )

Parameters

* **config** ([PatchTSTConfig](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The PatchTST for prediction model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/modeling_patchtst.py#L1588)

( past\_values: Tensor past\_observed\_mask: typing.Optional[torch.Tensor] = None future\_values: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  )

Parameters

* **past\_values** (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*) —
  Input sequence to the model
* **past\_observed\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*) —
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
  in `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
* **future\_values** (`torch.Tensor` of shape `(bs, forecast_len, num_input_channels)`, *optional*) —
  Future target values associated with the `past_values`
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the output attention of all layers
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a `ModelOutput` instead of a plain tuple.

Examples:


```
>>> from huggingface_hub import hf_hub_download
>>> import torch
>>> from transformers import PatchTSTConfig, PatchTSTForPrediction

>>> file = hf_hub_download(
...     repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
... )
>>> batch = torch.load(file)

>>> # Prediction task with 7 input channels and prediction length is 96
>>> model = PatchTSTForPrediction.from_pretrained("namctin/patchtst_etth1_forecast")

>>> # during training, one provides both past and future values
>>> outputs = model(
...     past_values=batch["past_values"],
...     future_values=batch["future_values"],
... )

>>> loss = outputs.loss
>>> loss.backward()

>>> # during inference, one only provides past values, the model outputs future values
>>> outputs = model(past_values=batch["past_values"])
>>> prediction_outputs = outputs.prediction_outputs
```

## PatchTSTForClassification

### class transformers.PatchTSTForClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/modeling_patchtst.py#L1365)

( config: PatchTSTConfig  )

Parameters

* **config** ([PatchTSTConfig](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The PatchTST for classification model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/modeling_patchtst.py#L1380)

( past\_values: Tensor target\_values: typing.Optional[torch.Tensor] = None past\_observed\_mask: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.patchtst.modeling_patchtst.PatchTSTForClassificationOutput` or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*) —
  Input sequence to the model
* **target\_values** (`torch.Tensor`, *optional*) —
  Labels associates with the `past_values`
* **past\_observed\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*) —
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
  in `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.patchtst.modeling_patchtst.PatchTSTForClassificationOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.patchtst.modeling_patchtst.PatchTSTForClassificationOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PatchTSTConfig](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTConfig)) and inputs.

* **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction
  (classification) loss.
* **prediction\_logits** (`torch.FloatTensor` of shape `(batch_size, num_targets)`) — Prediction scores of the PatchTST modeling head (scores before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [PatchTSTForClassification](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTForClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import PatchTSTConfig, PatchTSTForClassification

>>> # classification task with two input channel2 and 3 classes
>>> config = PatchTSTConfig(
...     num_input_channels=2,
...     num_targets=3,
...     context_length=512,
...     patch_length=12,
...     stride=12,
...     use_cls_token=True,
... )
>>> model = PatchTSTForClassification(config=config)

>>> # during inference, one only provides past values
>>> past_values = torch.randn(20, 512, 2)
>>> outputs = model(past_values=past_values)
>>> labels = outputs.prediction_logits
```

## PatchTSTForPretraining

### class transformers.PatchTSTForPretraining

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/modeling_patchtst.py#L1212)

( config: PatchTSTConfig  )

Parameters

* **config** ([PatchTSTConfig](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The PatchTST for pretrain model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/modeling_patchtst.py#L1223)

( past\_values: Tensor past\_observed\_mask: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  )

Parameters

* **past\_values** (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*) —
  Input sequence to the model
* **past\_observed\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*) —
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
  in `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the output attention of all layers
* **return\_dict** (`bool`, *optional*) — Whether or not to return a `ModelOutput` instead of a plain tuple.

Examples:


```
>>> from huggingface_hub import hf_hub_download
>>> import torch
>>> from transformers import PatchTSTConfig, PatchTSTForPretraining

>>> file = hf_hub_download(
...     repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
... )
>>> batch = torch.load(file)

>>> # Config for random mask pretraining
>>> config = PatchTSTConfig(
...     num_input_channels=7,
...     context_length=512,
...     patch_length=12,
...     stride=12,
...     mask_type='random',
...     random_mask_ratio=0.4,
...     use_cls_token=True,
... )
>>> # Config for forecast mask pretraining
>>> config = PatchTSTConfig(
...     num_input_channels=7,
...     context_length=512,
...     patch_length=12,
...     stride=12,
...     mask_type='forecast',
...     num_forecast_mask_patches=5,
...     use_cls_token=True,
... )
>>> model = PatchTSTForPretraining(config)

>>> # during training, one provides both past and future values
>>> outputs = model(past_values=batch["past_values"])

>>> loss = outputs.loss
>>> loss.backward()
```

## PatchTSTForRegression

### class transformers.PatchTSTForRegression

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/modeling_patchtst.py#L1806)

( config: PatchTSTConfig  )

Parameters

* **config** ([PatchTSTConfig](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The PatchTST for regression model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/patchtst/modeling_patchtst.py#L1833)

( past\_values: Tensor target\_values: typing.Optional[torch.Tensor] = None past\_observed\_mask: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.patchtst.modeling_patchtst.PatchTSTForRegressionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*) —
  Input sequence to the model
* **target\_values** (`torch.Tensor` of shape `(bs, num_input_channels)`) —
  Target values associates with the `past_values`
* **past\_observed\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*) —
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
  in `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
    Whether or not to return a `ModelOutput` instead of a plain tuple.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.patchtst.modeling_patchtst.PatchTSTForRegressionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.patchtst.modeling_patchtst.PatchTSTForRegressionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PatchTSTConfig](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTConfig)) and inputs.

* **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — MSE loss.
* **regression\_outputs** (`torch.FloatTensor` of shape `(batch_size, num_targets)`) — Regression outputs of the time series modeling heads.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [PatchTSTForRegression](/docs/transformers/v4.56.2/en/model_doc/patchtst#transformers.PatchTSTForRegression) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import PatchTSTConfig, PatchTSTForRegression

>>> # Regression task with 6 input channels and regress 2 targets
>>> model = PatchTSTForRegression.from_pretrained("namctin/patchtst_etth1_regression")

>>> # during inference, one only provides past values, the model outputs future values
>>> past_values = torch.randn(20, 512, 6)
>>> outputs = model(past_values=past_values)
>>> regression_outputs = outputs.regression_outputs
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/patchtst.md)
