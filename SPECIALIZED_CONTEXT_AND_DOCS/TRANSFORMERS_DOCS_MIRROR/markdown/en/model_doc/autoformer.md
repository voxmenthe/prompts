*This model was released on 2021-06-24 and added to Hugging Face Transformers on 2023-05-30.*

# Autoformer

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Autoformer model was proposed in [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://huggingface.co/papers/2106.13008) by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.

This model augments the Transformer as a deep decomposition architecture, which can progressively decompose the trend and seasonal components during the forecasting process.

The abstract from the paper is the following:

*Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning. This paper studies the long-term forecasting problem of time series. Prior Transformer-based models adopt various self-attention mechanisms to discover the long-range dependencies. However, intricate temporal patterns of the long-term future prohibit the model from finding reliable dependencies. Also, Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, resulting in the information utilization bottleneck. Going beyond Transformers, we design Autoformer as a novel decomposition architecture with an Auto-Correlation mechanism. We break with the pre-processing convention of series decomposition and renovate it as a basic inner block of deep models. This design empowers Autoformer with progressive decomposition capacities for complex time series. Further, inspired by the stochastic process theory, we design the Auto-Correlation mechanism based on the series periodicity, which conducts the dependencies discovery and representation aggregation at the sub-series level. Auto-Correlation outperforms self-attention in both efficiency and accuracy. In long-term forecasting, Autoformer yields state-of-the-art accuracy, with a 38% relative improvement on six benchmarks, covering five practical applications: energy, traffic, economics, weather and disease.*

This model was contributed by [elisim](https://huggingface.co/elisim) and [kashif](https://huggingface.co/kashif).
The original code can be found [here](https://github.com/thuml/Autoformer).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

* Check out the Autoformer blog-post in HuggingFace blog: [Yes, Transformers are Effective for Time Series Forecasting (+ Autoformer)](https://huggingface.co/blog/autoformer)

## AutoformerConfig

### class transformers.AutoformerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/autoformer/configuration_autoformer.py#L26)

( prediction\_length: typing.Optional[int] = None context\_length: typing.Optional[int] = None distribution\_output: str = 'student\_t' loss: str = 'nll' input\_size: int = 1 lags\_sequence: list = [1, 2, 3, 4, 5, 6, 7] scaling: bool = True num\_time\_features: int = 0 num\_dynamic\_real\_features: int = 0 num\_static\_categorical\_features: int = 0 num\_static\_real\_features: int = 0 cardinality: typing.Optional[list[int]] = None embedding\_dimension: typing.Optional[list[int]] = None d\_model: int = 64 encoder\_attention\_heads: int = 2 decoder\_attention\_heads: int = 2 encoder\_layers: int = 2 decoder\_layers: int = 2 encoder\_ffn\_dim: int = 32 decoder\_ffn\_dim: int = 32 activation\_function: str = 'gelu' dropout: float = 0.1 encoder\_layerdrop: float = 0.1 decoder\_layerdrop: float = 0.1 attention\_dropout: float = 0.1 activation\_dropout: float = 0.1 num\_parallel\_samples: int = 100 init\_std: float = 0.02 use\_cache: bool = True is\_encoder\_decoder = True label\_length: int = 10 moving\_average: int = 25 autocorrelation\_factor: int = 3 \*\*kwargs  )

Parameters

* **prediction\_length** (`int`) —
  The prediction length for the decoder. In other words, the prediction horizon of the model.
* **context\_length** (`int`, *optional*, defaults to `prediction_length`) —
  The context length for the encoder. If unset, the context length will be the same as the
  `prediction_length`.
* **distribution\_output** (`string`, *optional*, defaults to `"student_t"`) —
  The distribution emission head for the model. Could be either “student\_t”, “normal” or “negative\_binomial”.
* **loss** (`string`, *optional*, defaults to `"nll"`) —
  The loss function for the model corresponding to the `distribution_output` head. For parametric
  distributions it is the negative log likelihood (nll) - which currently is the only supported one.
* **input\_size** (`int`, *optional*, defaults to 1) —
  The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of
  multivariate targets.
* **lags\_sequence** (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 5, 6, 7]`) —
  The lags of the input time series as covariates often dictated by the frequency. Default is `[1, 2, 3, 4, 5, 6, 7]`.
* **scaling** (`bool`, *optional* defaults to `True`) —
  Whether to scale the input targets.
* **num\_time\_features** (`int`, *optional*, defaults to 0) —
  The number of time features in the input time series.
* **num\_dynamic\_real\_features** (`int`, *optional*, defaults to 0) —
  The number of dynamic real valued features.
* **num\_static\_categorical\_features** (`int`, *optional*, defaults to 0) —
  The number of static categorical features.
* **num\_static\_real\_features** (`int`, *optional*, defaults to 0) —
  The number of static real valued features.
* **cardinality** (`list[int]`, *optional*) —
  The cardinality (number of different values) for each of the static categorical features. Should be a list
  of integers, having the same length as `num_static_categorical_features`. Cannot be `None` if
  `num_static_categorical_features` is > 0.
* **embedding\_dimension** (`list[int]`, *optional*) —
  The dimension of the embedding for each of the static categorical features. Should be a list of integers,
  having the same length as `num_static_categorical_features`. Cannot be `None` if
  `num_static_categorical_features` is > 0.
* **d\_model** (`int`, *optional*, defaults to 64) —
  Dimensionality of the transformer layers.
* **encoder\_layers** (`int`, *optional*, defaults to 2) —
  Number of encoder layers.
* **decoder\_layers** (`int`, *optional*, defaults to 2) —
  Number of decoder layers.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 2) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **decoder\_attention\_heads** (`int`, *optional*, defaults to 2) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 32) —
  Dimension of the “intermediate” (often named feed-forward) layer in encoder.
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 32) —
  Dimension of the “intermediate” (often named feed-forward) layer in decoder.
* **activation\_function** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and decoder. If string, `"gelu"` and
  `"relu"` are supported.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the encoder, and decoder.
* **encoder\_layerdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for the attention and fully connected layers for each encoder layer.
* **decoder\_layerdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for the attention and fully connected layers for each decoder layer.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for the attention probabilities.
* **activation\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability used between the two layers of the feed-forward networks.
* **num\_parallel\_samples** (`int`, *optional*, defaults to 100) —
  The number of samples to generate in parallel for each time step of inference.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated normal weight initialization distribution.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.
* **label\_length** (`int`, *optional*, defaults to 10) —
  Start token length of the Autoformer decoder, which is used for direct multi-step prediction (i.e.
  non-autoregressive generation).
* **moving\_average** (`int`, *optional*, defaults to 25) —
  The window size of the moving average. In practice, it’s the kernel size in AvgPool1d of the Decomposition
  Layer.
* **autocorrelation\_factor** (`int`, *optional*, defaults to 3) —
  “Attention” (i.e. AutoCorrelation mechanism) factor which is used to find top k autocorrelations delays.
  It’s recommended in the paper to set it to a number between 1 and 5.

This is the configuration class to store the configuration of an [AutoformerModel](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerModel). It is used to instantiate an
Autoformer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Autoformer
[huggingface/autoformer-tourism-monthly](https://huggingface.co/huggingface/autoformer-tourism-monthly)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import AutoformerConfig, AutoformerModel

>>> # Initializing a default Autoformer configuration
>>> configuration = AutoformerConfig()

>>> # Randomly initializing a model (with random weights) from the configuration
>>> model = AutoformerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## AutoformerModel

### class transformers.AutoformerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/autoformer/modeling_autoformer.py#L1256)

( config: AutoformerConfig  )

Parameters

* **config** ([AutoformerConfig](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Autoformer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/autoformer/modeling_autoformer.py#L1427)

( past\_values: Tensor past\_time\_features: Tensor past\_observed\_mask: Tensor static\_categorical\_features: typing.Optional[torch.Tensor] = None static\_real\_features: typing.Optional[torch.Tensor] = None future\_values: typing.Optional[torch.Tensor] = None future\_time\_features: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[list[torch.FloatTensor]] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None use\_cache: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → `transformers.models.autoformer.modeling_autoformer.AutoformerModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Past values of the time series, that serve as context in order to predict the future. These values may
  contain lags, i.e. additional values from the past which are added in order to serve as “extra context”.
  The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
  `static_categorical_features`, `static_real_features`, `past_time_features`).

  The sequence length here is equal to `context_length` + `max(config.lags_sequence)`.

  Missing values need to be replaced with zeros.
* **past\_time\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`, *optional*) —
  Optional time features, which the model internally will add to `past_values`. These could be things like
  “month of year”, “day of the month”, etc. encoded as vectors (for instance as Fourier features). These
  could also be so-called “age” features, which basically help the model know “at which point in life” a
  time-series is. Age features have small values for distant past time steps and increase monotonically the
  more we approach the current time step.

  These features serve as the “positional encodings” of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional time features.

  The Autoformer only learns additional embeddings for `static_categorical_features`.
* **past\_observed\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected in
  `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
* **static\_categorical\_features** (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*) —
  Optional static categorical features for which the model will learn an embedding, which it will add to the
  values of the time series.

  Static categorical features are features which have the same value for all time steps (static over time).

  A typical example of a static categorical feature is a time series ID.
* **static\_real\_features** (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*) —
  Optional static real features which the model will add to the values of the time series.

  Static real features are features which have the same value for all time steps (static over time).

  A typical example of a static real feature is promotion information.
* **future\_values** (`torch.FloatTensor` of shape `(batch_size, prediction_length)`) —
  Future values of the time series, that serve as labels for the model. The `future_values` is what the
  Transformer needs to learn to output, given the `past_values`.

  See the demo notebook and code snippets for details.

  Missing values need to be replaced with zeros.
* **future\_time\_features** (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`, *optional*) —
  Optional time features, which the model internally will add to `future_values`. These could be things like
  “month of year”, “day of the month”, etc. encoded as vectors (for instance as Fourier features). These
  could also be so-called “age” features, which basically help the model know “at which point in life” a
  time-series is. Age features have small values for distant past time steps and increase monotonically the
  more we approach the current time step.

  These features serve as the “positional encodings” of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional features.

  The Autoformer only learns additional embeddings for `static_categorical_features`.
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) —
  Tuple consists of `last_hidden_state`, `hidden_states` (*optional*) and `attentions` (*optional*)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` (*optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.models.autoformer.modeling_autoformer.AutoformerModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.autoformer.modeling_autoformer.AutoformerModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AutoformerConfig](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **trend** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Trend tensor for each time series.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
  `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **loc** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) — Shift values of each time series’ context window which is used to give the model inputs of the same
  magnitude and then used to shift back to the original magnitude.
* **scale** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) — Scaling values of each time series’ context window which is used to give the model inputs of the same
  magnitude and then used to rescale back to the original magnitude.
* **static\_features:** (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*) — Static features of each time series’ in a batch which are copied to the covariates at inference time.
* **static\_features** (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*, defaults to `None`) — Static features of each time series’ in a batch which are copied to the covariates at inference time.

The [AutoformerModel](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from huggingface_hub import hf_hub_download
>>> import torch
>>> from transformers import AutoformerModel

>>> file = hf_hub_download(
...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
... )
>>> batch = torch.load(file)

>>> model = AutoformerModel.from_pretrained("huggingface/autoformer-tourism-monthly")

>>> # during training, one provides both past and future values
>>> # as well as possible additional features
>>> outputs = model(
...     past_values=batch["past_values"],
...     past_time_features=batch["past_time_features"],
...     past_observed_mask=batch["past_observed_mask"],
...     static_categorical_features=batch["static_categorical_features"],
...     future_values=batch["future_values"],
...     future_time_features=batch["future_time_features"],
... )

>>> last_hidden_state = outputs.last_hidden_state
```

## AutoformerForPrediction

### class transformers.AutoformerForPrediction

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/autoformer/modeling_autoformer.py#L1654)

( config: AutoformerConfig  )

Parameters

* **config** ([AutoformerConfig](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Autoformer Model with a distribution head on top for time-series forecasting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/autoformer/modeling_autoformer.py#L1694)

( past\_values: Tensor past\_time\_features: Tensor past\_observed\_mask: Tensor static\_categorical\_features: typing.Optional[torch.Tensor] = None static\_real\_features: typing.Optional[torch.Tensor] = None future\_values: typing.Optional[torch.Tensor] = None future\_time\_features: typing.Optional[torch.Tensor] = None future\_observed\_mask: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[list[torch.FloatTensor]] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None use\_cache: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.Seq2SeqTSPredictionOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSPredictionOutput) or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Past values of the time series, that serve as context in order to predict the future. These values may
  contain lags, i.e. additional values from the past which are added in order to serve as “extra context”.
  The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
  `static_categorical_features`, `static_real_features`, `past_time_features`).

  The sequence length here is equal to `context_length` + `max(config.lags_sequence)`.

  Missing values need to be replaced with zeros.
* **past\_time\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`, *optional*) —
  Optional time features, which the model internally will add to `past_values`. These could be things like
  “month of year”, “day of the month”, etc. encoded as vectors (for instance as Fourier features). These
  could also be so-called “age” features, which basically help the model know “at which point in life” a
  time-series is. Age features have small values for distant past time steps and increase monotonically the
  more we approach the current time step.

  These features serve as the “positional encodings” of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional time features.

  The Autoformer only learns additional embeddings for `static_categorical_features`.
* **past\_observed\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected in
  `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
* **static\_categorical\_features** (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*) —
  Optional static categorical features for which the model will learn an embedding, which it will add to the
  values of the time series.

  Static categorical features are features which have the same value for all time steps (static over time).

  A typical example of a static categorical feature is a time series ID.
* **static\_real\_features** (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*) —
  Optional static real features which the model will add to the values of the time series.

  Static real features are features which have the same value for all time steps (static over time).

  A typical example of a static real feature is promotion information.
* **future\_values** (`torch.FloatTensor` of shape `(batch_size, prediction_length)`) —
  Future values of the time series, that serve as labels for the model. The `future_values` is what the
  Transformer needs to learn to output, given the `past_values`.

  See the demo notebook and code snippets for details.

  Missing values need to be replaced with zeros.
* **future\_time\_features** (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`, *optional*) —
  Optional time features, which the model internally will add to `future_values`. These could be things like
  “month of year”, “day of the month”, etc. encoded as vectors (for instance as Fourier features). These
  could also be so-called “age” features, which basically help the model know “at which point in life” a
  time-series is. Age features have small values for distant past time steps and increase monotonically the
  more we approach the current time step.

  These features serve as the “positional encodings” of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional features.

  The Autoformer only learns additional embeddings for `static_categorical_features`.
* **future\_observed\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*) —
  Boolean mask to indicate which `future_values` were observed and which were missing. Mask values selected
  in `[0, 1]`:
  + 1 for values that are **observed**,
  + 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

  This mask is used to filter out missing values for the final loss calculation.
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) —
  Tuple consists of `last_hidden_state`, `hidden_states` (*optional*) and `attentions` (*optional*)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` (*optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.Seq2SeqTSPredictionOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSPredictionOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqTSPredictionOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSPredictionOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AutoformerConfig](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when a `future_values` is provided) — Distributional loss.
* **params** (`torch.FloatTensor` of shape `(batch_size, num_samples, num_params)`) — Parameters of the chosen distribution.
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **loc** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) — Shift values of each time series’ context window which is used to give the model inputs of the same
  magnitude and then used to shift back to the original magnitude.
* **scale** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) — Scaling values of each time series’ context window which is used to give the model inputs of the same
  magnitude and then used to rescale back to the original magnitude.
* **static\_features** (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*) — Static features of each time series’ in a batch which are copied to the covariates at inference time.

The [AutoformerForPrediction](/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerForPrediction) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from huggingface_hub import hf_hub_download
>>> import torch
>>> from transformers import AutoformerForPrediction

>>> file = hf_hub_download(
...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
... )
>>> batch = torch.load(file)

>>> model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")

>>> # during training, one provides both past and future values
>>> # as well as possible additional features
>>> outputs = model(
...     past_values=batch["past_values"],
...     past_time_features=batch["past_time_features"],
...     past_observed_mask=batch["past_observed_mask"],
...     static_categorical_features=batch["static_categorical_features"],
...     future_values=batch["future_values"],
...     future_time_features=batch["future_time_features"],
... )

>>> loss = outputs.loss
>>> loss.backward()

>>> # during inference, one only provides past values
>>> # as well as possible additional features
>>> # the model autoregressively generates future values
>>> outputs = model.generate(
...     past_values=batch["past_values"],
...     past_time_features=batch["past_time_features"],
...     past_observed_mask=batch["past_observed_mask"],
...     static_categorical_features=batch["static_categorical_features"],
...     future_time_features=batch["future_time_features"],
... )

>>> mean_prediction = outputs.sequences.mean(dim=1)
```

The AutoformerForPrediction can also use static\_real\_features. To do so, set num\_static\_real\_features in
AutoformerConfig based on number of such features in the dataset (in case of tourism\_monthly dataset it

is equal to 1), initialize the model and call as shown below:


```
>>> from huggingface_hub import hf_hub_download
>>> import torch
>>> from transformers import AutoformerConfig, AutoformerForPrediction

>>> file = hf_hub_download(
...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
... )
>>> batch = torch.load(file)

>>> # check number of static real features
>>> num_static_real_features = batch["static_real_features"].shape[-1]

>>> # load configuration of pretrained model and override num_static_real_features
>>> configuration = AutoformerConfig.from_pretrained(
...     "huggingface/autoformer-tourism-monthly",
...     num_static_real_features=num_static_real_features,
... )
>>> # we also need to update feature_size as it is not recalculated
>>> configuration.feature_size += num_static_real_features

>>> model = AutoformerForPrediction(configuration)

>>> outputs = model(
...     past_values=batch["past_values"],
...     past_time_features=batch["past_time_features"],
...     past_observed_mask=batch["past_observed_mask"],
...     static_categorical_features=batch["static_categorical_features"],
...     static_real_features=batch["static_real_features"],
...     future_values=batch["future_values"],
...     future_time_features=batch["future_time_features"],
... )
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/autoformer.md)
