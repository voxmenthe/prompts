*This model was released on 2023-10-14 and added to Hugging Face Transformers on 2025-04-16.*

# TimesFM

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model proposed in [A decoder-only foundation model for time-series forecasting](https://huggingface.co/papers/2310.10688) by Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. It is a decoder only model that uses non-overlapping patches of time-series data as input and outputs some output patch length prediction in an autoregressive fashion.

The abstract from the paper is the following:

*Motivated by recent advances in large language models for Natural Language Processing (NLP), we design a time-series foundation model for forecasting whose out-of-the-box zero-shot performance on a variety of public datasets comes close to the accuracy of state-of-the-art supervised forecasting models for each individual dataset. Our model is based on pretraining a patched-decoder style attention model on a large time-series corpus, and can work well across different forecasting history lengths, prediction lengths and temporal granularities.*

This model was contributed by [kashif](https://huggingface.co/kashif).
The original code can be found [here](https://github.com/google-research/timesfm).

To use the model:


```
import numpy as np
import torch
from transformers import TimesFmModelForPrediction


model = TimesFmModelForPrediction.from_pretrained(
    "google/timesfm-2.0-500m-pytorch",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto"
)


 # Create dummy inputs
forecast_input = [
    np.sin(np.linspace(0, 20, 100)),
    np.sin(np.linspace(0, 20, 200)),
    np.sin(np.linspace(0, 20, 400)),
]
frequency_input = [0, 1, 2]

# Convert inputs to sequence of tensors
forecast_input_tensor = [
    torch.tensor(ts, dtype=torch.bfloat16).to(model.device)
    for ts in forecast_input
]
frequency_input_tensor = torch.tensor(frequency_input, dtype=torch.long).to(model.device)

# Get predictions from the pre-trained model
with torch.no_grad():
    outputs = model(past_values=forecast_input_tensor, freq=frequency_input_tensor, return_dict=True)
    point_forecast_conv = outputs.mean_predictions.float().cpu().numpy()
    quantile_forecast_conv = outputs.full_predictions.float().cpu().numpy()
```

## TimesFmConfig

### class transformers.TimesFmConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timesfm/configuration_timesfm.py#L24)

( patch\_length: int = 32 context\_length: int = 512 horizon\_length: int = 128 freq\_size: int = 3 num\_hidden\_layers: int = 50 hidden\_size: int = 1280 intermediate\_size: int = 1280 head\_dim: int = 80 num\_attention\_heads: int = 16 tolerance: float = 1e-06 rms\_norm\_eps: float = 1e-06 quantiles: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] pad\_val: float = 1123581321.0 attention\_dropout: float = 0.0 use\_positional\_embedding: bool = False initializer\_range: float = 0.02 min\_timescale: int = 1 max\_timescale: int = 10000 \*\*kwargs  )

Parameters

* **patch\_length** (`int`, *optional*, defaults to 32) —
  The length of one patch in the input sequence.
* **context\_length** (`int`, *optional*, defaults to 512) —
  The length of the input context.
* **horizon\_length** (`int`, *optional*, defaults to 128) —
  The length of the prediction horizon.
* **freq\_size** (`int`, *optional*, defaults to 3) —
  The number of frequency embeddings.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 50) —
  Number of Transformer layers.
* **hidden\_size** (`int`, *optional*, defaults to 1280) —
  Size of the hidden layers in the feed-forward networks.
* **intermediate\_size** (`int`, *optional*, defaults to 1280) —
  Dimension of the MLP representations.
* **head\_dim** (`int`, *optional*, defaults to 80) —
  Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
  be defined as `num_attention_heads * head_dim`.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **tolerance** (`float`, *optional*, defaults to 1e-06) —
  The tolerance for the quantile loss.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the RMS normalization layers.
* **quantiles** (`list[float]`, *optional*, defaults to `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`) —
  The quantiles to predict.
* **pad\_val** (`float`, *optional*, defaults to 1123581321.0) —
  The value used to pad the predictions.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for the attention scores.
* **use\_positional\_embedding** (`bool`, *optional*, defaults to `False`) —
  Whether to add positional embeddings.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **min\_timescale** (`int`, *optional*, defaults to 1) —
  The start of the geometric positional index. Determines the periodicity of
  the added signal.
* **max\_timescale** (`int`, *optional*, defaults to 10000) —
  The end of the geometric positional index. Determines the frequency of the
  added signal.

This is the configuration class to store the configuration of a [TimesFmModelForPrediction](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmModelForPrediction) or a `TFTimesFmModel`. It is used to
instantiate a TimesFM model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the TimesFM
[google/timesfm-2.0-500m-pytorch](https://huggingface.co/google/timesfm-2.0-500m-pytorch) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## TimesFmModel

### class transformers.TimesFmModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timesfm/modeling_timesfm.py#L316)

( config: TimesFmConfig  )

Parameters

* **config** ([TimesFmConfig](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Timesfm Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timesfm/modeling_timesfm.py#L356)

( past\_values: Tensor past\_values\_padding: LongTensor freq: Tensor output\_attentions: bool = False output\_hidden\_states: bool = False  ) → `transformers.models.timesfm.modeling_timesfm.TimesFmOutput` or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Past values of the time series that serves as input to the model.
* **past\_values\_padding** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  The padding indicator of the time series.
* **freq** (`torch.LongTensor` of shape `(batch_size,)`) —
  Frequency indices for the time series data.
* **output\_attentions** (`bool`, defaults to `False`) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, defaults to `False`) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.models.timesfm.modeling_timesfm.TimesFmOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.timesfm.modeling_timesfm.TimesFmOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TimesFmConfig](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **loc** (`torch.Tensor` of shape `(batch_size, )`) — The mean of the time series inputs.
* **scale** (`torch.Tensor` of shape `(batch_size,)`) — The scale of the time series inputs.

The [TimesFmModel](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## TimesFmModelForPrediction

### class transformers.TimesFmModelForPrediction

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timesfm/modeling_timesfm.py#L579)

( config: TimesFmConfig  )

TimesFM model for quantile and mean prediction.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timesfm/modeling_timesfm.py#L667)

( past\_values: Sequence freq: typing.Optional[collections.abc.Sequence[typing.Union[torch.Tensor, int]]] = None window\_size: typing.Optional[int] = None future\_values: typing.Optional[torch.Tensor] = None forecast\_context\_len: typing.Optional[int] = None return\_forecast\_on\_context: bool = False truncate\_negative: bool = False output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → `transformers.models.timesfm.modeling_timesfm.TimesFmOutputForPrediction` or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Past values of the time series that serves as input to the model.
* **freq** (`torch.LongTensor` of shape `(batch_size,)`) —
  Frequency indices for the time series data.
* **window\_size** (`int`, *optional*) —
  Window size of trend + residual decomposition. If None then we do not do decomposition.
* **future\_values** (`torch.Tensor`, *optional*) —
  Optional future time series values to be used for loss computation.
* **forecast\_context\_len** (`int`, *optional*) —
  Optional max context length.
* **return\_forecast\_on\_context** (`bool`, *optional*) —
  True to return the forecast on the context when available, i.e. after the first input patch.
* **truncate\_negative** (`bool`, *optional*) —
  Truncate to only non-negative values if any of the contexts have non-negative values,
  otherwise do nothing.
* **output\_attentions** (`bool`, *optional*) —
  Whether to output the attentions.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether to output the hidden states.

Returns

`transformers.models.timesfm.modeling_timesfm.TimesFmOutputForPrediction` or `tuple(torch.FloatTensor)`

A `transformers.models.timesfm.modeling_timesfm.TimesFmOutputForPrediction` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TimesFmConfig](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **mean\_predictions** (`torch.Tensor` of shape `(batch_size, sequence_length)`) — The mean predictions of the time series.
* **full\_predictions** (`torch.Tensor` of shape `(batch_size, sequence_length)`) — The full predictions of the time series including the mean and the quantiles.
* **loss** (`torch.Tensor` of shape `(1,)`, *optional*, returned when `future_values` is provided) — The loss of the TimesFM model.

The [TimesFmModelForPrediction](/docs/transformers/v4.56.2/en/model_doc/timesfm#transformers.TimesFmModelForPrediction) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import TimesFmModelForPrediction

>>> model = TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch")

>>> forecast_input = [torch.linspace(0, 20, 100).sin(), torch.linspace(0, 20, 200).sin(), torch.linspace(0, 20, 400).sin()]
>>> frequency_input = torch.tensor([0, 1, 2], dtype=torch.long)

>>> # Generate
>>> with torch.no_grad():
>>>     outputs = model(past_values=forecast_input, freq=frequency_input, return_dict=True)
>>>     point_forecast_conv = outputs.mean_predictions
>>>     quantile_forecast_conv = outputs.full_predictions
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/timesfm.md)
