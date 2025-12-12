*This model was released on 2024-10-21 and added to Hugging Face Transformers on 2025-01-10.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# Moonshine

[Moonshine](https://huggingface.co/papers/2410.15608) is an encoder-decoder speech recognition model optimized for real-time transcription and recognizing voice command. Instead of using traditional absolute position embeddings, Moonshine uses Rotary Position Embedding (RoPE) to handle speech with varying lengths without using padding. This improves efficiency during inference, making it ideal for resource-constrained devices.

You can find all the original Moonshine checkpoints under the [Useful Sensors](https://huggingface.co/UsefulSensors) organization.

Click on the Moonshine models in the right sidebar for more examples of how to apply Moonshine to different speech recognition tasks.

The example below demonstrates how to transcribe speech into text with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="automatic-speech-recognition",
    model="UsefulSensors/moonshine-base",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
```

## MoonshineConfig

### class transformers.MoonshineConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/configuration_moonshine.py#L25)

( vocab\_size = 32768 hidden\_size = 288 intermediate\_size = 1152 encoder\_num\_hidden\_layers = 6 decoder\_num\_hidden\_layers = 6 encoder\_num\_attention\_heads = 8 decoder\_num\_attention\_heads = 8 encoder\_num\_key\_value\_heads = None decoder\_num\_key\_value\_heads = None pad\_head\_dim\_to\_multiple\_of = None encoder\_hidden\_act = 'gelu' decoder\_hidden\_act = 'silu' max\_position\_embeddings = 512 initializer\_range = 0.02 decoder\_start\_token\_id = 1 use\_cache = True rope\_theta = 10000.0 rope\_scaling = None partial\_rotary\_factor = 0.9 is\_encoder\_decoder = True attention\_bias = False attention\_dropout = 0.0 bos\_token\_id = 1 eos\_token\_id = 2 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32768) —
  Vocabulary size of the Moonshine model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [MoonshineModel](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineModel).
* **hidden\_size** (`int`, *optional*, defaults to 288) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 1152) —
  Dimension of the MLP representations.
* **encoder\_num\_hidden\_layers** (`int`, *optional*, defaults to 6) —
  Number of hidden layers in the Transformer encoder.
* **decoder\_num\_hidden\_layers** (`int`, *optional*, defaults to 6) —
  Number of hidden layers in the Transformer decoder.
* **encoder\_num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **decoder\_num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **encoder\_num\_key\_value\_heads** (`int`, *optional*) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `encoder_num_key_value_heads=encoder_num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `encoder_num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
  `num_attention_heads`.
* **decoder\_num\_key\_value\_heads** (`int`, *optional*) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `decoder_num_key_value_heads=decoder_num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `decoder_num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
  `decoder_num_attention_heads`.
* **pad\_head\_dim\_to\_multiple\_of** (`int`, *optional*) —
  Pad head dimension in encoder and decoder to the next multiple of this value. Necessary for using certain
  optimized attention implementations.
* **encoder\_hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder.
* **decoder\_hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **decoder\_start\_token\_id** (`int`, *optional*, defaults to 1) —
  Corresponds to the ”<|startoftranscript|>” token, which is automatically used when no `decoder_input_ids`
  are provided to the `generate` function. It is used to guide the model`s generation process depending on
  the task.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (`Dict`, *optional*) —
  Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
  and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
  accordingly.
  Expected contents:
  `rope_type` (`str`):
  The sub-variant of RoPE to use. Can be one of [‘default’, ‘linear’, ‘dynamic’, ‘yarn’, ‘longrope’,
  ‘llama3’], with ‘default’ being the original RoPE implementation.
  `factor` (`float`, *optional*):
  Used with all rope types except ‘default’. The scaling factor to apply to the RoPE embeddings. In
  most scaling types, a `factor` of x will enable the model to handle sequences of length x *original maximum pre-trained length.
  `original_max_position_embeddings` (`int`,* optional*):
  Used with ‘dynamic’, ‘longrope’ and ‘llama3’. The original max position embeddings used during
  pretraining.
  `attention_factor` (`float`,* optional*):
  Used with ‘yarn’ and ‘longrope’. The scaling factor to be applied on the attention
  computation. If unspecified, it defaults to value recommended by the implementation, using the
  `factor` field to infer the suggested value.
  `beta_fast` (`float`,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for extrapolation (only) in the linear
  ramp function. If unspecified, it defaults to 32.
  `beta_slow` (`float`,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for interpolation (only) in the linear
  ramp function. If unspecified, it defaults to 1.
  `short_factor` (`list[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to short contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `long_factor` (`list[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to long contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `low_freq_factor` (`float`,* optional*):
  Only used with ‘llama3’. Scaling factor applied to low frequency components of the RoPE
  `high_freq_factor` (`float`,* optional\*):
  Only used with ‘llama3’. Scaling factor applied to high frequency components of the RoPE
* **partial\_rotary\_factor** (`float`, *optional*, defaults to 0.9) —
  Percentage of the query and keys which will have rotary embedding.
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Whether the model is used as an encoder/decoder or not.
* **attention\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **bos\_token\_id** (`int`, *optional*, defaults to 1) —
  Denotes beginning of sequences token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  Denotes end of sequences token id.

This is the configuration class to store the configuration of a [MoonshineModel](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineModel). It is used to instantiate a Moonshine
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Moonshine
[UsefulSensors/moonshine-tiny](https://huggingface.co/UsefulSensors/moonshine-tiny).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MoonshineModel, MoonshineConfig

>>> # Initializing a Moonshine style configuration
>>> configuration = MoonshineConfig().from_pretrained("UsefulSensors/moonshine-tiny")

>>> # Initializing a model from the configuration
>>> model = MoonshineModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MoonshineModel

### class transformers.MoonshineModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/modeling_moonshine.py#L819)

( config: MoonshineConfig  )

Parameters

* **config** ([MoonshineConfig](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Moonshine Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/modeling_moonshine.py#L887)

( input\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Union[transformers.cache\_utils.EncoderDecoderCache, tuple[torch.FloatTensor], NoneType] = None decoder\_inputs\_embeds: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_position\_ids: typing.Optional[tuple[torch.LongTensor]] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, audio_length)`) —
  Float values of the raw speech waveform. Raw speech waveform can be
  obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
  `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
  the soundfile library (`pip install soundfile`). To prepare the array into
  `input_values`, the [AutoFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoFeatureExtractor) should be used for padding
  and conversion into a tensor of type `torch.FloatTensor`.
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`Union[~cache_utils.EncoderDecoderCache, tuple[torch.FloatTensor], NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **decoder\_inputs\_embeds** (`tuple[torch.FloatTensor]` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **decoder\_position\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`) —
  Indices of positions of each input sequence tokens in the position embeddings.
  Used to calculate the position embeddings up to `config.decoder_config.max_position_embeddings`
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MoonshineConfig](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig)) and inputs.

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

The [MoonshineModel](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoFeatureExtractor, MoonshineModel
>>> from datasets import load_dataset

>>> model = MoonshineModel.from_pretrained("UsefulSensors/moonshine-tiny")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("UsefulSensors/moonshine-tiny")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
>>> input_values = inputs.input_values
>>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
>>> last_hidden_state = model(input_values, decoder_input_ids=decoder_input_ids).last_hidden_state
>>> list(last_hidden_state.shape)
[1, 2, 288]
```

#### \_mask\_input\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/modeling_moonshine.py#L844)

( input\_features: FloatTensor attention\_mask: typing.Optional[torch.LongTensor] = None  )

Masks extracted features along time axis and/or along feature axis according to
[SpecAugment](https://huggingface.co/papers/1904.08779).

## MoonshineForConditionalGeneration

### class transformers.MoonshineForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/modeling_moonshine.py#L982)

( config: MoonshineConfig  )

Parameters

* **config** ([MoonshineConfig](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Moonshine Model with a language modeling head. Can be used for automatic speech recognition.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/modeling_moonshine.py#L1008)

( input\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Union[transformers.cache\_utils.EncoderDecoderCache, tuple[torch.FloatTensor], NoneType] = None decoder\_inputs\_embeds: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_position\_ids: typing.Optional[tuple[torch.LongTensor]] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None labels: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, audio_length)`) —
  Float values of the raw speech waveform. Raw speech waveform can be
  obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
  `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
  the soundfile library (`pip install soundfile`). To prepare the array into
  `input_values`, the [AutoFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoFeatureExtractor) should be used for padding
  and conversion into a tensor of type `torch.FloatTensor`.
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`Union[~cache_utils.EncoderDecoderCache, tuple[torch.FloatTensor], NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **decoder\_inputs\_embeds** (`tuple[torch.FloatTensor]` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **decoder\_position\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`) —
  Indices of positions of each input sequence tokens in the position embeddings.
  Used to calculate the position embeddings up to `config.decoder_config.max_position_embeddings`
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

Returns

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MoonshineConfig](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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

The [MoonshineForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoProcessor, MoonshineForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
>>> model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
>>> input_values = inputs.input_values

>>> generated_ids = model.generate(input_values, max_new_tokens=100)

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> transcription
'Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L2140)

( inputs: typing.Optional[torch.Tensor] = None generation\_config: typing.Optional[transformers.generation.configuration\_utils.GenerationConfig] = None logits\_processor: typing.Optional[transformers.generation.logits\_process.LogitsProcessorList] = None stopping\_criteria: typing.Optional[transformers.generation.stopping\_criteria.StoppingCriteriaList] = None prefix\_allowed\_tokens\_fn: typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None synced\_gpus: typing.Optional[bool] = None assistant\_model: typing.Optional[ForwardRef('PreTrainedModel')] = None streamer: typing.Optional[ForwardRef('BaseStreamer')] = None negative\_prompt\_ids: typing.Optional[torch.Tensor] = None negative\_prompt\_attention\_mask: typing.Optional[torch.Tensor] = None use\_model\_defaults: typing.Optional[bool] = None custom\_generate: typing.Union[str, typing.Callable, NoneType] = None \*\*kwargs  ) → [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

Parameters

* **inputs** (`torch.Tensor` of varying shape depending on the modality, *optional*) —
  The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
  method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
  should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
  `input_ids`, `input_values`, `input_features`, or `pixel_values`.
* **generation\_config** ([GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig), *optional*) —
  The generation configuration to be used as base parametrization for the generation call. `**kwargs`
  passed to generate matching the attributes of `generation_config` will override them. If
  `generation_config` is not provided, the default will be used, which has the following loading
  priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
  configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)’s
  default values, whose documentation should be checked to parameterize generation.
* **logits\_processor** (`LogitsProcessorList`, *optional*) —
  Custom logits processors that complement the default logits processors built from arguments and
  generation config. If a logit processor is passed that is already created with the arguments or a
  generation config an error is thrown. This feature is intended for advanced users.
* **stopping\_criteria** (`StoppingCriteriaList`, *optional*) —
  Custom stopping criteria that complements the default stopping criteria built from arguments and a
  generation config. If a stopping criteria is passed that is already created with the arguments or a
  generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
  sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
  intended for advanced users.
* **prefix\_allowed\_tokens\_fn** (`Callable[[int, torch.Tensor], list[int]]`, *optional*) —
  If provided, this function constraints the beam search to allowed tokens only at each step. If not
  provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
  `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
  on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
  for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
  Retrieval](https://huggingface.co/papers/2010.00904).
* **synced\_gpus** (`bool`, *optional*) —
  Whether to continue running the while loop until max\_length. Unless overridden, this flag will be set
  to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
  deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
* **assistant\_model** (`PreTrainedModel`, *optional*) —
  An assistant model that can be used to accelerate generation. The assistant model must have the exact
  same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
  is much faster than running generation with the model you’re calling generate from. As such, the
  assistant model should be much smaller.
* **streamer** (`BaseStreamer`, *optional*) —
  Streamer object that will be used to stream the generated sequences. Generated tokens are passed
  through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
* **negative\_prompt\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  The negative prompt needed for some processors such as CFG. The batch size must match the input batch
  size. This is an experimental feature, subject to breaking API changes in future versions.
* **negative\_prompt\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Attention\_mask for `negative_prompt_ids`.
* **use\_model\_defaults** (`bool`, *optional*) —
  When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
  generation configuration (`model.generation_config`), as opposed to the global defaults
  (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
  `True`.
* **custom\_generate** (`str` or `Callable`, *optional*) —
  One of the following:
  + `str` (Hugging Face Hub repository name): runs the custom `generate` function defined at
    `custom_generate/generate.py` in that repository instead of the standard `generate` method. The
    repository fully replaces the generation logic, and the return type may differ.
  + `str` (local repository path): same as above but from a local path, `trust_remote_code` not required.
  + `Callable`: `generate` will perform the usual input preparation steps, then call the provided callable to
    run the decoding loop.
    For more information, see [the docs](../../generation_strategies#custom-generation-methods).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
  forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
  specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder\_*.

Returns

[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

A [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) (if `return_dict_in_generate=True`
or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) types are:

* [GenerateDecoderOnlyOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput),
* [GenerateBeamDecoderOnlyOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput)

If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) types are:

* [GenerateEncoderDecoderOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput),
* [GenerateBeamEncoderDecoderOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput)

Generates sequences of token ids for models with a language modeling head.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
model’s default generation configuration. You can override any `generation_config` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

For an overview of generation strategies and code examples, check out the [following
guide](../generation_strategies).

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/moonshine.md)
