*This model was released on 2020-12-14 and added to Hugging Face Transformers on 2023-03-08.*

# Informer

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Informer model was proposed in [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://huggingface.co/papers/2012.07436) by Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.

This method introduces a Probabilistic Attention mechanism to select the “active” queries rather than the “lazy” queries and provides a sparse Transformer thus mitigating the quadratic compute and memory requirements of vanilla attention.

The abstract from the paper is the following:

*Many real-world applications require the prediction of long sequence time-series, such as electricity consumption planning. Long sequence time-series forecasting (LSTF) demands a high prediction capacity of the model, which is the ability to capture precise long-range dependency coupling between output and input efficiently. Recent studies have shown the potential of Transformer to increase the prediction capacity. However, there are several severe issues with Transformer that prevent it from being directly applicable to LSTF, including quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. To address these issues, we design an efficient transformer-based model for LSTF, named Informer, with three distinctive characteristics: (i) a ProbSparse self-attention mechanism, which achieves O(L logL) in time complexity and memory usage, and has comparable performance on sequences’ dependency alignment. (ii) the self-attention distilling highlights dominating attention by halving cascading layer input, and efficiently handles extreme long input sequences. (iii) the generative style decoder, while conceptually simple, predicts the long time-series sequences at one forward operation rather than a step-by-step way, which drastically improves the inference speed of long-sequence predictions. Extensive experiments on four large-scale datasets demonstrate that Informer significantly outperforms existing methods and provides a new solution to the LSTF problem.*

This model was contributed by [elisim](https://huggingface.co/elisim) and [kashif](https://huggingface.co/kashif).
The original code can be found [here](https://github.com/zhouhaoyi/Informer2020).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

* Check out the Informer blog-post in HuggingFace blog: [Multivariate Probabilistic Time Series Forecasting with Informer](https://huggingface.co/blog/informer)

## InformerConfig

### class transformers.InformerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/informer/configuration_informer.py#L26)

( prediction\_length: typing.Optional[int] = None context\_length: typing.Optional[int] = None distribution\_output: str = 'student\_t' loss: str = 'nll' input\_size: int = 1 lags\_sequence: typing.Optional[list[int]] = None scaling: typing.Union[str, bool, NoneType] = 'mean' num\_dynamic\_real\_features: int = 0 num\_static\_real\_features: int = 0 num\_static\_categorical\_features: int = 0 num\_time\_features: int = 0 cardinality: typing.Optional[list[int]] = None embedding\_dimension: typing.Optional[list[int]] = None d\_model: int = 64 encoder\_ffn\_dim: int = 32 decoder\_ffn\_dim: int = 32 encoder\_attention\_heads: int = 2 decoder\_attention\_heads: int = 2 encoder\_layers: int = 2 decoder\_layers: int = 2 is\_encoder\_decoder: bool = True activation\_function: str = 'gelu' dropout: float = 0.05 encoder\_layerdrop: float = 0.1 decoder\_layerdrop: float = 0.1 attention\_dropout: float = 0.1 activation\_dropout: float = 0.1 num\_parallel\_samples: int = 100 init\_std: float = 0.02 use\_cache = True attention\_type: str = 'prob' sampling\_factor: int = 5 distil: bool = True \*\*kwargs  )

Parameters

* **prediction\_length** (`int`) —
  The prediction length for the decoder. In other words, the prediction horizon of the model. This value is
  typically dictated by the dataset and we recommend to set it appropriately.
* **context\_length** (`int`, *optional*, defaults to `prediction_length`) —
  The context length for the encoder. If `None`, the context length will be the same as the
  `prediction_length`.
* **distribution\_output** (`string`, *optional*, defaults to `"student_t"`) —
  The distribution emission head for the model. Could be either “student\_t”, “normal” or “negative\_binomial”.
* **loss** (`string`, *optional*, defaults to `"nll"`) —
  The loss function for the model corresponding to the `distribution_output` head. For parametric
  distributions it is the negative log likelihood (nll) - which currently is the only supported one.
* **input\_size** (`int`, *optional*, defaults to 1) —
  The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of
  multivariate targets.
* **scaling** (`string` or `bool`, *optional* defaults to `"mean"`) —
  Whether to scale the input targets via “mean” scaler, “std” scaler or no scaler if `None`. If `True`, the
  scaler is set to “mean”.
* **lags\_sequence** (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 5, 6, 7]`) —
  The lags of the input time series as covariates often dictated by the frequency of the data. Default is
  `[1, 2, 3, 4, 5, 6, 7]` but we recommend to change it based on the dataset appropriately.
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
* **attention\_type** (`str`, *optional*, defaults to “prob”) —
  Attention used in encoder. This can be set to “prob” (Informer’s ProbAttention) or “full” (vanilla
  transformer’s canonical self-attention).
* **sampling\_factor** (`int`, *optional*, defaults to 5) —
  ProbSparse sampling factor (only makes affect when `attention_type`=“prob”). It is used to control the
  reduced query matrix (Q\_reduce) input length.
* **distil** (`bool`, *optional*, defaults to `True`) —
  Whether to use distilling in encoder.

This is the configuration class to store the configuration of an [InformerModel](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerModel). It is used to instantiate an
Informer model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Informer
[huggingface/informer-tourism-monthly](https://huggingface.co/huggingface/informer-tourism-monthly) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import InformerConfig, InformerModel

>>> # Initializing an Informer configuration with 12 time steps for prediction
>>> configuration = InformerConfig(prediction_length=12)

>>> # Randomly initializing a model (with random weights) from the configuration
>>> model = InformerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## InformerModel

### class transformers.InformerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/informer/modeling_informer.py#L1323)

( config: InformerConfig  )

Parameters

* **config** ([InformerConfig](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Informer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/informer/modeling_informer.py#L1464)

( past\_values: Tensor past\_time\_features: Tensor past\_observed\_mask: Tensor static\_categorical\_features: typing.Optional[torch.Tensor] = None static\_real\_features: typing.Optional[torch.Tensor] = None future\_values: typing.Optional[torch.Tensor] = None future\_time\_features: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[list[torch.FloatTensor]] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None use\_cache: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqTSModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`) —
  Past values of the time series, that serve as context in order to predict the future. The sequence size of
  this tensor must be larger than the `context_length` of the model, since the model will use the larger size
  to construct lag features, i.e. additional values from the past which are added in order to serve as “extra
  context”.

  The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if no
  `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest
  look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length of
  the past.

  The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
  `static_categorical_features`, `static_real_features`, `past_time_features` and lags).

  Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.

  For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
  variates in the time series per time step.
* **past\_time\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`) —
  Required time features, which the model internally will add to `past_values`. These could be things like
  “month of year”, “day of the month”, etc. encoded as vectors (for instance as Fourier features). These
  could also be so-called “age” features, which basically help the model know “at which point in life” a
  time-series is. Age features have small values for distant past time steps and increase monotonically the
  more we approach the current time step. Holiday features are also a good example of time features.

  These features serve as the “positional encodings” of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional time features. The Time Series Transformer only learns
  additional embeddings for `static_categorical_features`.

  Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
  must but known at prediction time.

  The `num_features` here is equal to `config.`num\_time\_features`+`config.num\_dynamic\_real\_features`.
* **past\_observed\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*) —
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
* **future\_values** (`torch.FloatTensor` of shape `(batch_size, prediction_length)` or `(batch_size, prediction_length, input_size)`, *optional*) —
  Future values of the time series, that serve as labels for the model. The `future_values` is what the
  Transformer needs during training to learn to output, given the `past_values`.

  The sequence length here is equal to `prediction_length`.

  See the demo notebook and code snippets for details.

  Optionally, during training any missing values need to be replaced with zeros and indicated via the
  `future_observed_mask`.

  For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
  variates in the time series per time step.
* **future\_time\_features** (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`) —
  Required time features for the prediction window, which the model internally will add to `future_values`.
  These could be things like “month of year”, “day of the month”, etc. encoded as vectors (for instance as
  Fourier features). These could also be so-called “age” features, which basically help the model know “at
  which point in life” a time-series is. Age features have small values for distant past time steps and
  increase monotonically the more we approach the current time step. Holiday features are also a good example
  of time features.

  These features serve as the “positional encodings” of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional time features. The Time Series Transformer only learns
  additional embeddings for `static_categorical_features`.

  Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
  must but known at prediction time.

  The `num_features` here is equal to `config.`num\_time\_features`+`config.num\_dynamic\_real\_features`.
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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.Seq2SeqTSModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqTSModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InformerConfig](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **loc** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) — Shift values of each time series’ context window which is used to give the model inputs of the same
  magnitude and then used to shift back to the original magnitude.
* **scale** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) — Scaling values of each time series’ context window which is used to give the model inputs of the same
  magnitude and then used to rescale back to the original magnitude.
* **static\_features** (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*) — Static features of each time series’ in a batch which are copied to the covariates at inference time.

The [InformerModel](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from huggingface_hub import hf_hub_download
>>> import torch
>>> from transformers import InformerModel

>>> file = hf_hub_download(
...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
... )
>>> batch = torch.load(file)

>>> model = InformerModel.from_pretrained("huggingface/informer-tourism-monthly")

>>> # during training, one provides both past and future values
>>> # as well as possible additional features
>>> outputs = model(
...     past_values=batch["past_values"],
...     past_time_features=batch["past_time_features"],
...     past_observed_mask=batch["past_observed_mask"],
...     static_categorical_features=batch["static_categorical_features"],
...     static_real_features=batch["static_real_features"],
...     future_values=batch["future_values"],
...     future_time_features=batch["future_time_features"],
... )

>>> last_hidden_state = outputs.last_hidden_state
```

## InformerForPrediction

### class transformers.InformerForPrediction

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/informer/modeling_informer.py#L1716)

( config: InformerConfig  )

Parameters

* **config** ([InformerConfig](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Informer Model with a distribution head on top for time-series forecasting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/informer/modeling_informer.py#L1757)

( past\_values: Tensor past\_time\_features: Tensor past\_observed\_mask: Tensor static\_categorical\_features: typing.Optional[torch.Tensor] = None static\_real\_features: typing.Optional[torch.Tensor] = None future\_values: typing.Optional[torch.Tensor] = None future\_time\_features: typing.Optional[torch.Tensor] = None future\_observed\_mask: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[list[torch.FloatTensor]] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None use\_cache: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqTSModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **past\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`) —
  Past values of the time series, that serve as context in order to predict the future. The sequence size of
  this tensor must be larger than the `context_length` of the model, since the model will use the larger size
  to construct lag features, i.e. additional values from the past which are added in order to serve as “extra
  context”.

  The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if no
  `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest
  look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length of
  the past.

  The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
  `static_categorical_features`, `static_real_features`, `past_time_features` and lags).

  Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.

  For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
  variates in the time series per time step.
* **past\_time\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`) —
  Required time features, which the model internally will add to `past_values`. These could be things like
  “month of year”, “day of the month”, etc. encoded as vectors (for instance as Fourier features). These
  could also be so-called “age” features, which basically help the model know “at which point in life” a
  time-series is. Age features have small values for distant past time steps and increase monotonically the
  more we approach the current time step. Holiday features are also a good example of time features.

  These features serve as the “positional encodings” of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional time features. The Time Series Transformer only learns
  additional embeddings for `static_categorical_features`.

  Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
  must but known at prediction time.

  The `num_features` here is equal to `config.`num\_time\_features`+`config.num\_dynamic\_real\_features`.
* **past\_observed\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*) —
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
* **future\_values** (`torch.FloatTensor` of shape `(batch_size, prediction_length)` or `(batch_size, prediction_length, input_size)`, *optional*) —
  Future values of the time series, that serve as labels for the model. The `future_values` is what the
  Transformer needs during training to learn to output, given the `past_values`.

  The sequence length here is equal to `prediction_length`.

  See the demo notebook and code snippets for details.

  Optionally, during training any missing values need to be replaced with zeros and indicated via the
  `future_observed_mask`.

  For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
  variates in the time series per time step.
* **future\_time\_features** (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`) —
  Required time features for the prediction window, which the model internally will add to `future_values`.
  These could be things like “month of year”, “day of the month”, etc. encoded as vectors (for instance as
  Fourier features). These could also be so-called “age” features, which basically help the model know “at
  which point in life” a time-series is. Age features have small values for distant past time steps and
  increase monotonically the more we approach the current time step. Holiday features are also a good example
  of time features.

  These features serve as the “positional encodings” of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional time features. The Time Series Transformer only learns
  additional embeddings for `static_categorical_features`.

  Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
  must but known at prediction time.

  The `num_features` here is equal to `config.`num\_time\_features`+`config.num\_dynamic\_real\_features`.
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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.Seq2SeqTSModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqTSModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InformerConfig](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **loc** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) — Shift values of each time series’ context window which is used to give the model inputs of the same
  magnitude and then used to shift back to the original magnitude.
* **scale** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) — Scaling values of each time series’ context window which is used to give the model inputs of the same
  magnitude and then used to rescale back to the original magnitude.
* **static\_features** (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*) — Static features of each time series’ in a batch which are copied to the covariates at inference time.

The [InformerForPrediction](/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerForPrediction) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from huggingface_hub import hf_hub_download
>>> import torch
>>> from transformers import InformerForPrediction

>>> file = hf_hub_download(
...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
... )
>>> batch = torch.load(file)

>>> model = InformerForPrediction.from_pretrained(
...     "huggingface/informer-tourism-monthly"
... )

>>> # during training, one provides both past and future values
>>> # as well as possible additional features
>>> outputs = model(
...     past_values=batch["past_values"],
...     past_time_features=batch["past_time_features"],
...     past_observed_mask=batch["past_observed_mask"],
...     static_categorical_features=batch["static_categorical_features"],
...     static_real_features=batch["static_real_features"],
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
...     static_real_features=batch["static_real_features"],
...     future_time_features=batch["future_time_features"],
... )

>>> mean_prediction = outputs.sequences.mean(dim=1)
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/informer.md)
