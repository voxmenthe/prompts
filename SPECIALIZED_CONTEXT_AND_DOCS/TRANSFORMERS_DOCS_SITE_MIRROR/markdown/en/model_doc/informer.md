# Informer

## Overview

The Informer model was proposed in [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://huggingface.co/papers/2012.07436) by Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.

This method introduces a Probabilistic Attention mechanism to select the "active" queries rather than the "lazy" queries and provides a sparse Transformer thus mitigating the quadratic compute and memory requirements of vanilla attention.

The abstract from the paper is the following:

*Many real-world applications require the prediction of long sequence time-series, such as electricity consumption planning. Long sequence time-series forecasting (LSTF) demands a high prediction capacity of the model, which is the ability to capture precise long-range dependency coupling between output and input efficiently. Recent studies have shown the potential of Transformer to increase the prediction capacity. However, there are several severe issues with Transformer that prevent it from being directly applicable to LSTF, including quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. To address these issues, we design an efficient transformer-based model for LSTF, named Informer, with three distinctive characteristics: (i) a ProbSparse self-attention mechanism, which achieves O(L logL) in time complexity and memory usage, and has comparable performance on sequences' dependency alignment. (ii) the self-attention distilling highlights dominating attention by halving cascading layer input, and efficiently handles extreme long input sequences. (iii) the generative style decoder, while conceptually simple, predicts the long time-series sequences at one forward operation rather than a step-by-step way, which drastically improves the inference speed of long-sequence predictions. Extensive experiments on four large-scale datasets demonstrate that Informer significantly outperforms existing methods and provides a new solution to the LSTF problem.*

This model was contributed by [elisim](https://huggingface.co/elisim) and [kashif](https://huggingface.co/kashif).
The original code can be found [here](https://github.com/zhouhaoyi/Informer2020).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- Check out the Informer blog-post in HuggingFace blog: [Multivariate Probabilistic Time Series Forecasting with Informer](https://huggingface.co/blog/informer)

## InformerConfig[[transformers.InformerConfig]]

#### transformers.InformerConfig[[transformers.InformerConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/informer/configuration_informer.py#L26)

This is the configuration class to store the configuration of an [InformerModel](/docs/transformers/main/en/model_doc/informer#transformers.InformerModel). It is used to instantiate an
Informer model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Informer
[huggingface/informer-tourism-monthly](https://huggingface.co/huggingface/informer-tourism-monthly) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import InformerConfig, InformerModel

>>> # Initializing an Informer configuration with 12 time steps for prediction
>>> configuration = InformerConfig(prediction_length=12)

>>> # Randomly initializing a model (with random weights) from the configuration
>>> model = InformerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

prediction_length (`int`) : The prediction length for the decoder. In other words, the prediction horizon of the model. This value is typically dictated by the dataset and we recommend to set it appropriately.

context_length (`int`, *optional*, defaults to `prediction_length`) : The context length for the encoder. If `None`, the context length will be the same as the `prediction_length`.

distribution_output (`string`, *optional*, defaults to `"student_t"`) : The distribution emission head for the model. Could be either "student_t", "normal" or "negative_binomial".

loss (`string`, *optional*, defaults to `"nll"`) : The loss function for the model corresponding to the `distribution_output` head. For parametric distributions it is the negative log likelihood (nll) - which currently is the only supported one.

input_size (`int`, *optional*, defaults to 1) : The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of multivariate targets.

scaling (`string` or `bool`, *optional* defaults to `"mean"`) : Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the scaler is set to "mean".

lags_sequence (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 5, 6, 7]`) : The lags of the input time series as covariates often dictated by the frequency of the data. Default is `[1, 2, 3, 4, 5, 6, 7]` but we recommend to change it based on the dataset appropriately.

num_time_features (`int`, *optional*, defaults to 0) : The number of time features in the input time series.

num_dynamic_real_features (`int`, *optional*, defaults to 0) : The number of dynamic real valued features.

num_static_categorical_features (`int`, *optional*, defaults to 0) : The number of static categorical features.

num_static_real_features (`int`, *optional*, defaults to 0) : The number of static real valued features.

cardinality (`list[int]`, *optional*) : The cardinality (number of different values) for each of the static categorical features. Should be a list of integers, having the same length as `num_static_categorical_features`. Cannot be `None` if `num_static_categorical_features` is > 0.

embedding_dimension (`list[int]`, *optional*) : The dimension of the embedding for each of the static categorical features. Should be a list of integers, having the same length as `num_static_categorical_features`. Cannot be `None` if `num_static_categorical_features` is > 0.

d_model (`int`, *optional*, defaults to 64) : Dimensionality of the transformer layers.

encoder_layers (`int`, *optional*, defaults to 2) : Number of encoder layers.

decoder_layers (`int`, *optional*, defaults to 2) : Number of decoder layers.

encoder_attention_heads (`int`, *optional*, defaults to 2) : Number of attention heads for each attention layer in the Transformer encoder.

decoder_attention_heads (`int`, *optional*, defaults to 2) : Number of attention heads for each attention layer in the Transformer decoder.

encoder_ffn_dim (`int`, *optional*, defaults to 32) : Dimension of the "intermediate" (often named feed-forward) layer in encoder.

decoder_ffn_dim (`int`, *optional*, defaults to 32) : Dimension of the "intermediate" (often named feed-forward) layer in decoder.

activation_function (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and decoder. If string, `"gelu"` and `"relu"` are supported.

dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the encoder, and decoder.

encoder_layerdrop (`float`, *optional*, defaults to 0.1) : The dropout probability for the attention and fully connected layers for each encoder layer.

decoder_layerdrop (`float`, *optional*, defaults to 0.1) : The dropout probability for the attention and fully connected layers for each decoder layer.

attention_dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for the attention probabilities.

activation_dropout (`float`, *optional*, defaults to 0.1) : The dropout probability used between the two layers of the feed-forward networks.

num_parallel_samples (`int`, *optional*, defaults to 100) : The number of samples to generate in parallel for each time step of inference.

init_std (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated normal weight initialization distribution.

use_cache (`bool`, *optional*, defaults to `True`) : Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.

attention_type (`str`, *optional*, defaults to "prob") : Attention used in encoder. This can be set to "prob" (Informer's ProbAttention) or "full" (vanilla transformer's canonical self-attention).

sampling_factor (`int`, *optional*, defaults to 5) : ProbSparse sampling factor (only makes affect when `attention_type`="prob"). It is used to control the reduced query matrix (Q_reduce) input length.

distil (`bool`, *optional*, defaults to `True`) : Whether to use distilling in encoder.

## InformerModel[[transformers.InformerModel]]

#### transformers.InformerModel[[transformers.InformerModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/informer/modeling_informer.py#L1145)

The bare Informer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.InformerModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/informer/modeling_informer.py#L1283[{"name": "past_values", "val": ": Tensor"}, {"name": "past_time_features", "val": ": Tensor"}, {"name": "past_observed_mask", "val": ": Tensor"}, {"name": "static_categorical_features", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "static_real_features", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "future_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "future_time_features", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "encoder_outputs", "val": ": typing.Optional[list[torch.FloatTensor]] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ""}]- **past_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`) --
  Past values of the time series, that serve as context in order to predict the future. The sequence size of
  this tensor must be larger than the `context_length` of the model, since the model will use the larger size
  to construct lag features, i.e. additional values from the past which are added in order to serve as "extra
  context".

  The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if no
  `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest
  look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length of
  the past.

  The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
  `static_categorical_features`, `static_real_features`, `past_time_features` and lags).

  Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.

  For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
  variates in the time series per time step.
- **past_time_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`) --
  Required time features, which the model internally will add to `past_values`. These could be things like
  "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These
  could also be so-called "age" features, which basically help the model know "at which point in life" a
  time-series is. Age features have small values for distant past time steps and increase monotonically the
  more we approach the current time step. Holiday features are also a good example of time features.

  These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional time features. The Time Series Transformer only learns
  additional embeddings for `static_categorical_features`.

  Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
  must but known at prediction time.

  The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
- **past_observed_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*) --
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected in
  `[0, 1]`:

  - 1 for values that are **observed**,
  - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
- **static_categorical_features** (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*) --
  Optional static categorical features for which the model will learn an embedding, which it will add to the
  values of the time series.

  Static categorical features are features which have the same value for all time steps (static over time).

  A typical example of a static categorical feature is a time series ID.
- **static_real_features** (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*) --
  Optional static real features which the model will add to the values of the time series.

  Static real features are features which have the same value for all time steps (static over time).

  A typical example of a static real feature is promotion information.
- **future_values** (`torch.FloatTensor` of shape `(batch_size, prediction_length)` or `(batch_size, prediction_length, input_size)`, *optional*) --
  Future values of the time series, that serve as labels for the model. The `future_values` is what the
  Transformer needs during training to learn to output, given the `past_values`.

  The sequence length here is equal to `prediction_length`.

  See the demo notebook and code snippets for details.

  Optionally, during training any missing values need to be replaced with zeros and indicated via the
  `future_observed_mask`.

  For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
  variates in the time series per time step.
- **future_time_features** (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`) --
  Required time features for the prediction window, which the model internally will add to `future_values`.
  These could be things like "month of year", "day of the month", etc. encoded as vectors (for instance as
  Fourier features). These could also be so-called "age" features, which basically help the model know "at
  which point in life" a time-series is. Age features have small values for distant past time steps and
  increase monotonically the more we approach the current time step. Holiday features are also a good example
  of time features.

  These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional time features. The Time Series Transformer only learns
  additional embeddings for `static_categorical_features`.

  Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
  must but known at prediction time.

  The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
- **decoder_attention_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
- **encoder_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) --
  Tuple consists of `last_hidden_state`, `hidden_states` (*optional*) and `attentions` (*optional*)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` (*optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.0[transformers.modeling_outputs.Seq2SeqTSModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.Seq2SeqTSModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InformerConfig](/docs/transformers/main/en/model_doc/informer#transformers.InformerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **loc** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) -- Shift values of each time series' context window which is used to give the model inputs of the same
  magnitude and then used to shift back to the original magnitude.
- **scale** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) -- Scaling values of each time series' context window which is used to give the model inputs of the same
  magnitude and then used to rescale back to the original magnitude.
- **static_features** (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*) -- Static features of each time series' in a batch which are copied to the covariates at inference time.
The [InformerModel](/docs/transformers/main/en/model_doc/informer#transformers.InformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
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

**Parameters:**

config ([InformerConfig](/docs/transformers/main/en/model_doc/informer#transformers.InformerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.Seq2SeqTSModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.Seq2SeqTSModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InformerConfig](/docs/transformers/main/en/model_doc/informer#transformers.InformerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **loc** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) -- Shift values of each time series' context window which is used to give the model inputs of the same
  magnitude and then used to shift back to the original magnitude.
- **scale** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) -- Scaling values of each time series' context window which is used to give the model inputs of the same
  magnitude and then used to rescale back to the original magnitude.
- **static_features** (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*) -- Static features of each time series' in a batch which are copied to the covariates at inference time.

## InformerForPrediction[[transformers.InformerForPrediction]]

#### transformers.InformerForPrediction[[transformers.InformerForPrediction]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/informer/modeling_informer.py#L1525)

The Informer Model with a distribution head on top for time-series forecasting.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.InformerForPrediction.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/informer/modeling_informer.py#L1560[{"name": "past_values", "val": ": Tensor"}, {"name": "past_time_features", "val": ": Tensor"}, {"name": "past_observed_mask", "val": ": Tensor"}, {"name": "static_categorical_features", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "static_real_features", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "future_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "future_time_features", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "future_observed_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "encoder_outputs", "val": ": typing.Optional[list[torch.FloatTensor]] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ""}]- **past_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`) --
  Past values of the time series, that serve as context in order to predict the future. The sequence size of
  this tensor must be larger than the `context_length` of the model, since the model will use the larger size
  to construct lag features, i.e. additional values from the past which are added in order to serve as "extra
  context".

  The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if no
  `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest
  look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length of
  the past.

  The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
  `static_categorical_features`, `static_real_features`, `past_time_features` and lags).

  Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.

  For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
  variates in the time series per time step.
- **past_time_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`) --
  Required time features, which the model internally will add to `past_values`. These could be things like
  "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These
  could also be so-called "age" features, which basically help the model know "at which point in life" a
  time-series is. Age features have small values for distant past time steps and increase monotonically the
  more we approach the current time step. Holiday features are also a good example of time features.

  These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional time features. The Time Series Transformer only learns
  additional embeddings for `static_categorical_features`.

  Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
  must but known at prediction time.

  The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
- **past_observed_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*) --
  Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected in
  `[0, 1]`:

  - 1 for values that are **observed**,
  - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
- **static_categorical_features** (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*) --
  Optional static categorical features for which the model will learn an embedding, which it will add to the
  values of the time series.

  Static categorical features are features which have the same value for all time steps (static over time).

  A typical example of a static categorical feature is a time series ID.
- **static_real_features** (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*) --
  Optional static real features which the model will add to the values of the time series.

  Static real features are features which have the same value for all time steps (static over time).

  A typical example of a static real feature is promotion information.
- **future_values** (`torch.FloatTensor` of shape `(batch_size, prediction_length)` or `(batch_size, prediction_length, input_size)`, *optional*) --
  Future values of the time series, that serve as labels for the model. The `future_values` is what the
  Transformer needs during training to learn to output, given the `past_values`.

  The sequence length here is equal to `prediction_length`.

  See the demo notebook and code snippets for details.

  Optionally, during training any missing values need to be replaced with zeros and indicated via the
  `future_observed_mask`.

  For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
  variates in the time series per time step.
- **future_time_features** (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`) --
  Required time features for the prediction window, which the model internally will add to `future_values`.
  These could be things like "month of year", "day of the month", etc. encoded as vectors (for instance as
  Fourier features). These could also be so-called "age" features, which basically help the model know "at
  which point in life" a time-series is. Age features have small values for distant past time steps and
  increase monotonically the more we approach the current time step. Holiday features are also a good example
  of time features.

  These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
  the position encodings are learned from scratch internally as parameters of the model, the Time Series
  Transformer requires to provide additional time features. The Time Series Transformer only learns
  additional embeddings for `static_categorical_features`.

  Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
  must but known at prediction time.

  The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
- **future_observed_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*) --
  Boolean mask to indicate which `future_values` were observed and which were missing. Mask values selected
  in `[0, 1]`:

  - 1 for values that are **observed**,
  - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

  This mask is used to filter out missing values for the final loss calculation.
- **decoder_attention_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
- **encoder_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) --
  Tuple consists of `last_hidden_state`, `hidden_states` (*optional*) and `attentions` (*optional*)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` (*optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.0[transformers.modeling_outputs.Seq2SeqTSModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.Seq2SeqTSModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InformerConfig](/docs/transformers/main/en/model_doc/informer#transformers.InformerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **loc** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) -- Shift values of each time series' context window which is used to give the model inputs of the same
  magnitude and then used to shift back to the original magnitude.
- **scale** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) -- Scaling values of each time series' context window which is used to give the model inputs of the same
  magnitude and then used to rescale back to the original magnitude.
- **static_features** (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*) -- Static features of each time series' in a batch which are copied to the covariates at inference time.
The [InformerForPrediction](/docs/transformers/main/en/model_doc/informer#transformers.InformerForPrediction) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
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

**Parameters:**

config ([InformerConfig](/docs/transformers/main/en/model_doc/informer#transformers.InformerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.Seq2SeqTSModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.Seq2SeqTSModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InformerConfig](/docs/transformers/main/en/model_doc/informer#transformers.InformerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [EncoderDecoderCache](/docs/transformers/main/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **loc** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) -- Shift values of each time series' context window which is used to give the model inputs of the same
  magnitude and then used to shift back to the original magnitude.
- **scale** (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*) -- Scaling values of each time series' context window which is used to give the model inputs of the same
  magnitude and then used to rescale back to the original magnitude.
- **static_features** (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*) -- Static features of each time series' in a batch which are copied to the covariates at inference time.
