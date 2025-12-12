*This model was released on 2024-09-17 and added to Hugging Face Transformers on 2024-10-16.*

# Moshi

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Moshi model was proposed in [Moshi: a speech-text foundation model for real-time dialogue](https://huggingface.co/papers/2410.00037) by Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave and Neil Zeghidour.

Moshi is a speech-text foundation model that casts spoken dialogue as speech-to-speech generation. Starting from a text language model backbone, Moshi generates speech as tokens from the residual quantizer of a neural audio codec, while modeling separately its own speech and that of the user into parallel streams. This allows for the removal of explicit speaker turns, and the modeling of arbitrary conversational dynamics. Moshi also predicts time-aligned text tokens as a prefix to audio tokens. This “Inner Monologue” method significantly improves the linguistic quality of generated speech and provides streaming speech recognition and text-to-speech. As a result, Moshi is the first real-time full-duplex spoken large language model, with a theoretical latency of 160ms, 200ms in practice.

![](https://huggingface.co/datasets/ylacombe/benchmark-comparison/resolve/main/moshi_architecture.png)

The abstract from the paper is the following:

*We introduce Moshi, a speech-text foundation model and full-duplex spoken dialogue framework. Current systems for spoken dialogue rely on pipelines of independent components, namely voice activity detection, speech recognition, textual dialogue and text-to-speech. Such frameworks cannot emulate the experience of real conversations. First, their complexity induces a latency of several seconds between interactions. Second, text being the intermediate modality for dialogue, non-linguistic information that modifies meaning— such as emotion or non-speech sounds— is lost in the interaction. Finally, they rely on a segmentation into speaker turns, which does not take into account overlapping speech, interruptions and interjections. Moshi solves these independent issues altogether by casting spoken dialogue as speech-to-speech generation. Starting from a text language model backbone, Moshi generates speech as tokens from the residual quantizer of a neural audio codec, while modeling separately its own speech and that of the user into parallel streams. This allows for the removal of explicit speaker turns, and the modeling of arbitrary conversational dynamics. We moreover extend the hierarchical semantic-to-acoustic token generation of previous work to first predict time-aligned text tokens as a prefix to audio tokens. Not only this “Inner Monologue” method significantly improves the linguistic quality of generated speech, but we also illustrate how it can provide streaming speech recognition and text-to-speech. Our resulting model is the first real-time full-duplex spoken large language model, with a theoretical latency of 160ms, 200ms in practice, and is available at github.com/kyutai-labs/moshi.*

Moshi deals with 3 streams of information:

1. The user’s audio
2. Moshi’s audio
3. Moshi’s textual output

Similarly to [~MusicgenModel](/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenModel), audio is represented with audio codebooks, which can be interpreted like tokens. The main difference between text tokens and audio codebooks is that audio codebooks introduce an additional dimension of information.
Text tokens are typically of dim `(batch_size, sequence_length)` but audio tokens are of dim `(batch_size, num_codebooks, sequence_length)`.

Moshi’s made of 3 components:

**1. The main decoder (Helium in the paper)**

It corresponds to [MoshiForCausalLM](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiForCausalLM). It is strictly a classic text LLM, that uses an architecture similar to `~GemmaForCausalLM`. In other words, it takes text tokens, embeds them, pass them through the decoder and a language head, to get text logits.

**2. The depth decoder**

On its own, it’s also a classic LLM, but this time, instead of generating over the time dimension, it generates over the codebook dimension.

It also means that its context length is `num_codebooks`, thus it can’t generate more than `num_codebooks`.

Note that each timestamp - i.e each codebook - gets its own set of Linear Layers and Embeddings.

**3. [MimiModel](/docs/transformers/v4.56.2/en/model_doc/mimi#transformers.MimiModel)**

It’s the audio encoder from Kyutai, that has recently been integrated to transformers, which is used to “tokenize” audio. It has the same use that [~EncodecModel](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel) has in [~MusicgenModel](/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenModel).

## Tips:

The original checkpoints can be converted using the conversion script `src/transformers/models/moshi/convert_moshi_transformers.py`

### How to use the model:

This implementation has two main aims:

1. quickly test model generation by simplifying the original API
2. simplify training. A training guide will come soon, but user contributions are welcomed!

It is designed for intermediate use. We strongly recommend using the original [implementation](https://github.com/kyutai-labs/moshi) to infer the model in real-time streaming.

**1. Model generation**

Moshi is a streaming auto-regressive model with two streams of audio. To put it differently, one audio stream corresponds to what the model said/will say and the other audio stream corresponds to what the user said/will say.

[MoshiForConditionalGeneration.generate()](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiForConditionalGeneration.generate) thus needs 3 inputs:

1. `input_ids` - corresponding to the text token history
2. `moshi_input_values` or `moshi_audio_codes`- corresponding to the model audio history
3. `user_input_values` or `user_audio_codes` - corresponding to the user audio history

These three inputs must be synchronized. Meaning that their lengths must correspond to the same number of tokens.

You can dynamically use the 3 inputs depending on what you want to test:

1. Simply check the model response to an user prompt - in that case, `input_ids` can be filled with pad tokens and `user_input_values` can be a zero tensor of the same shape than the user prompt.
2. Test more complex behaviour - in that case, you must be careful about how the input tokens are synchronized with the audios.

The original model is synchronized text with audio by padding the text in between each token enunciation.

To follow the example of the following image, `"Hello, I'm Moshi"` could be transformed to `"Hello,<pad><unk>I'm Moshi"`.

![](https://huggingface.co/datasets/ylacombe/benchmark-comparison/resolve/main/moshi_text_sync.png)

[MoshiForConditionalGeneration.generate()](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiForConditionalGeneration.generate) then auto-regressively feeds to itself its own audio stream, but since it doesn’t have access to the user input stream while using `transformers`, it will thus **assume that the user is producing blank audio**.


```
>>> from datasets import load_dataset, Audio
>>> import torch, math
>>> from transformers import MoshiForConditionalGeneration, AutoFeatureExtractor, AutoTokenizer, infer_device


>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/moshiko-pytorch-bf16")
>>> tokenizer = AutoTokenizer.from_pretrained("kyutai/moshiko-pytorch-bf16")
>>> device = infer_device()
>>> dtype = torch.bfloat16

>>> # prepare user input audio 
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> user_input_values = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(device=device, dtype=dtype)

>>> # prepare moshi input values - we suppose moshi didn't say anything while the user spoke
>>> moshi_input_values = torch.zeros_like(user_input_values.input_values)

>>> # prepare moshi input ids - we suppose moshi didn't say anything while the user spoke
>>> num_tokens = math.ceil(moshi_input_values.shape[-1] * waveform_to_token_ratio)
>>> input_ids = torch.ones((1, num_tokens), device=device, dtype=torch.int64) * tokenizer.encode("<pad>")[0]

>>> # generate 25 new tokens (around 2s of audio)
>>> output = model.generate(input_ids=input_ids, user_input_values=user_input_values.input_values, moshi_input_values=moshi_input_values, max_new_tokens=25)

>>> text_tokens = output.sequences
>>> audio_waveforms = output.audio_sequences
```

**2. Model training**

Most of the work has to be done during data creation/pre-processing, because of the need to align/synchronize streams.

Once it’s done, you can simply forward `text_labels` and `audio_labels` to [MoshiForConditionalGeneration.forward()](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiForConditionalGeneration.forward), alongside the usual inputs, to get the model loss.

A training guide will come soon, but user contributions are welcomed!

### How does the model forward the inputs / generate:

1. The input streams are embedded and combined into `inputs_embeds`.
2. `inputs_embeds` is passed through the main decoder, which processes it like a normal LLM would.
3. The main decoder outputs `text logits` but also its `last hidden state` which is called `temporal context` in the paper.
4. The depth decoder switches the dimension on which we forward / generate (codebooks instead of time). It uses the token generated from `text logits` and the `temporal context` to auto-regressively generate audio codebooks.

This model was contributed by [Yoach Lacombe (ylacombe)](https://huggingface.co/ylacombe).

The original code can be found [here](https://github.com/kyutai-labs/moshi).

## MoshiConfig

### class transformers.MoshiConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/configuration_moshi.py#L149)

( vocab\_size = 32000 hidden\_size = 4096 num\_hidden\_layers = 32 num\_attention\_heads = 32 num\_key\_value\_heads = None audio\_vocab\_size = None max\_position\_embeddings = 3000 rope\_theta = 10000.0 hidden\_act = 'silu' head\_dim = None initializer\_range = 0.02 use\_cache = True sliding\_window = 3000 attention\_dropout = 0.0 ffn\_dim = 22528 rms\_norm\_eps = 1e-08 num\_codebooks = 8 tie\_word\_embeddings = False \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32000) —
  Vocabulary size of the MoshiDecoder model. Defines the number of different tokens that can be
  represented by the `inputs_ids` passed when calling `MoshiDecoder`.
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the layers and the pooler layer of the main decoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 32) —
  Number of decoder layers.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the main decoder block.
* **num\_key\_value\_heads** (`int`, *optional*) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.
* **audio\_vocab\_size** (`int`, *optional*) —
  Vocabulary size of the audio part of model. Defines the number of different tokens that can be
  represented by the `audio_codes` passed when calling the Moshi models.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 3000) —
  The maximum sequence length that this model might ever be used with. Typically, set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **head\_dim** (`int`, *optional*, defaults to `hidden_size // num_attention_heads`) —
  The attention head dimension.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **sliding\_window** (`int`, *optional*, defaults to 3000) —
  Sliding window attention window size. If not specified, will default to `3000`.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **ffn\_dim** (`int`, *optional*, defaults to 22528) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the main decoder block. Must be even.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-08) —
  The epsilon used by the rms normalization layers.
* **num\_codebooks** (`int`, *optional*, defaults to 8) —
  The number of audio codebooks for each audio channels.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie weight embeddings
* **kwargs** (*optional*) —
  Dictionary of keyword arguments. Notably:
  + **audio\_encoder\_config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) — An instance of a configuration object that
    defines the audio encoder config.
  + **depth\_\_config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) — An instance of a configuration object that
    defines the depth decoder config.

This is the configuration class to store the configuration of a [MoshiModel](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiModel). It is used to instantiate a
Moshi model according to the specified arguments, defining the audio encoder, Moshi depth decoder and Moshi decoder
configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the Moshiko model,
e.g. [kmhf/hf-moshiko](https://huggingface.co/kmhf/hf-moshiko)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import (
...     MoshiConfig,
...     MoshiForConditionalGeneration,
... )

>>> configuration = MoshiConfig()

>>> # Initializing a MoshiForConditionalGeneration (with random weights) from the kmhf/hf-moshiko style configuration
>>> model = MoshiForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # Saving the model, including its configuration
>>> model.save_pretrained("kmhf/hf-moshiko")

>>> # loading model and config from pretrained folder
>>> moshi_config = MoshiConfig.from_pretrained("kmhf/hf-moshiko")
>>> model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko", config=moshi_config)
```

#### from\_audio\_encoder\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/configuration_moshi.py#L314)

( audio\_encoder\_config: PretrainedConfig \*\*kwargs  ) → [MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig)

Returns

[MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig)

An instance of a configuration object

Instantiate a [MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig) (or a derived class) from an audio encoder configuration.

## MoshiDepthConfig

### class transformers.MoshiDepthConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/configuration_moshi.py#L25)

( vocab\_size = 32000 hidden\_size = 1024 input\_size = 4096 num\_hidden\_layers = 6 num\_attention\_heads = 16 num\_key\_value\_heads = None audio\_vocab\_size = 2048 max\_position\_embeddings = 9 hidden\_act = 'silu' head\_dim = None initializer\_range = 0.02 use\_cache = True sliding\_window = 8 attention\_dropout = 0.0 ffn\_dim = 5632 rms\_norm\_eps = 1e-08 num\_codebooks = 8 tie\_word\_embeddings = False \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32000) —
  Vocabulary size of the MoshiDepthDecoder model. Defines the number of different tokens that can be
  represented by the `inputs_ids` passed when calling `MoshiDepthDecoder`.
* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the layers and the pooler layer of the depth decoder.
* **input\_size** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the input hidden states. Used to connect the main decoder to the depth decoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 6) —
  Number of depth decoder layers.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the depth decoder block.
* **num\_key\_value\_heads** (`int`, *optional*) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.
* **audio\_vocab\_size** (`int`, *optional*, defaults to 2048) —
  Vocabulary size of the audio part of model. Defines the number of different tokens that can be
  represented by the `audio_codes` passed when calling the Moshi models.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 9) —
  The maximum sequence length that this model might ever be used with. Typically, set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the depth decoder.
* **head\_dim** (`int`, *optional*, defaults to `hidden_size // num_attention_heads`) —
  The attention head dimension.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **sliding\_window** (`int`, *optional*, defaults to 8) —
  Sliding window attention window size. If not specified, will default to `8`.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **ffn\_dim** (`int`, *optional*, defaults to 5632) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the depth decoder block. Must be even.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-08) —
  The epsilon used by the rms normalization layers.
* **num\_codebooks** (`int`, *optional*, defaults to 8) —
  The number of audio codebooks for each audio channels.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie weight embeddings
* **kwargs** (*optional*) —
  Dictionary of keyword arguments. Notably:
  + **audio\_encoder\_config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) — An instance of a configuration object that
    defines the audio encoder config.

This is the configuration class to store the configuration of a `MoshiDepthDecoder`. It is used to instantiate a
Moshi depth decoder model according to the specified arguments, defining the Moshi depth decoder config.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import (
...     MoshiDepthConfig,
...     MoshiDepthDecoder,
... )

>>> configuration = MoshiDepthConfig()

>>> # Initializing a MoshiDepthDecoder (with random weights) from the kmhf/hf-moshiko style configuration
>>> model = MoshiDepthDecoder(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MoshiModel

### class transformers.MoshiModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/modeling_moshi.py#L1211)

( config: MoshiConfig  )

Parameters

* **config** ([MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Moshi Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/modeling_moshi.py#L1230)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MoshiModel](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## MoshiForCausalLM

### class transformers.MoshiForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/modeling_moshi.py#L1484)

( config  )

Parameters

* **config** ([MoshiForCausalLM](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Moshi decoder model with a text language modelling head on top. Only usable for text.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/modeling_moshi.py#L1497)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None labels: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs  ) → `transformers.models.moshi.modeling_moshi.MoshiCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.moshi.modeling_moshi.MoshiCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.moshi.modeling_moshi.MoshiCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MoshiForCausalLM](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, MoshiForCausalLM

>>> model = MoshiForCausalLM.from_pretrained("kmhf/hf-moshiko")
>>> tokenizer = AutoTokenizer.from_pretrained("kmhf/hf-moshiko")

>>> prompt = "What is your favorite condiment?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"What is your favorite condiment?"
```

## MoshiForConditionalGeneration

### class transformers.MoshiForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/modeling_moshi.py#L1602)

( config: MoshiConfig  )

Parameters

* **config** ([MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The original Moshi model with an audio encoder, a Moshi depth decoder and a Moshi decoder, for speech-to-speech.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/modeling_moshi.py#L1630)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.BoolTensor] = None user\_input\_values: typing.Optional[torch.FloatTensor] = None user\_audio\_codes: typing.Optional[torch.Tensor] = None moshi\_input\_values: typing.Optional[torch.FloatTensor] = None moshi\_audio\_codes: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None text\_labels: typing.Optional[torch.LongTensor] = None audio\_labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **user\_input\_values** (`torch.Tensor` of shape `(batch\_size, 1, audio\_sequence\_length), *optional*) —
  The audio waveforms used as audio user prompt for the generation.
* **user\_audio\_codes** (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length), *optional*) -- The audio codes used as audio user prompt for the generation. Has priority over` user\_input\_values`and represents the audio "tokens" of`user\_input\_values` once passed through the audio encoder.
* **moshi\_input\_values** (`torch.Tensor` of shape `(batch\_size, 1, audio\_sequence\_length), *optional*) —
  The audio waveforms used as audio Moshi prompt for the generation.
* **moshi\_audio\_codes** (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length), *optional*) -- The audio codes used as audio Moshi prompt for the generation. Has priority over` moshi\_input\_values`and represents the audio "tokens" of`moshi\_input\_values` once passed through the audio encoder.
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `input_ids` and `inputs_embeds` are both unset, `inputs_embeds` takes the value
  of `inputs_embeds`.
* **text\_labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for text language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
  `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
  are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
* **audio\_labels** (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`, *optional*) —
  Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
  `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
  are ignored (masked), the loss is only computed for labels in `[0, ..., config.audio_vocab_size]`
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MoshiConfig](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiConfig)) and inputs.

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

The [MoshiForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/moshi#transformers.MoshiForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import MoshiForConditionalGeneration
>>> import torch

>>> model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko")
>>> inputs = moshi.get_unconditional_inputs()

>>> logits = model(**inputs, ).logits
>>> logits.shape  # (bsz, seq_len, text_vocab_size)
torch.Size([1, 1, 32000])
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/modeling_moshi.py#L1935)

( input\_ids: typing.Optional[torch.LongTensor] = None user\_input\_values: typing.Optional[torch.FloatTensor] = None user\_audio\_codes: typing.Optional[torch.Tensor] = None moshi\_input\_values: typing.Optional[torch.FloatTensor] = None moshi\_audio\_codes: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None return\_audio\_waveforms: typing.Optional[bool] = True return\_audio\_codes: typing.Optional[bool] = None concat\_unconditional\_inputs: typing.Optional[bool] = True \*\*kwargs  )

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch\_size, sequence\_length), *optional*) —
  The sequence used as a text prompt for the generation.
* **user\_input\_values** (`torch.Tensor` of shape `(batch\_size, 1, audio\_sequence\_length), *optional*) —
  The audio waveforms used as audio user prompt for the generation.
* **user\_audio\_codes** (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length), *optional*) -- The audio codes used as audio user prompt for the generation. Has priority over` user\_input\_values`and represents the audio "tokens" of`user\_input\_values` once passed through the audio encoder.
* **moshi\_input\_values** (`torch.Tensor` of shape `(batch\_size, 1, audio\_sequence\_length), *optional*) —
  The audio waveforms used as audio Moshi prompt for the generation.
* **moshi\_audio\_codes** (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length), *optional*) -- The audio codes used as audio Moshi prompt for the generation. Has priority over` moshi\_input\_values`and represents the audio "tokens" of`moshi\_input\_values` once passed through the audio encoder.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` and the audio inputs you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert the inputs into associated vectors than the
  model’s internal embedding lookup matrix.
* **return\_audio\_waveforms** (`bool`, *optional*, defaults to `True`) —
  If `False`, won’t generate the audio waveforms.
* **return\_audio\_codes** (`bool`, *optional*) —
  If `True`, will also returns the generated audio codes, i.e the intermediate audio “tokens” which transforms to `audio_sequences` once passed through the audio decoder.
* **concat\_unconditional\_inputs** (`bool`, *optional*, defaults to `True`) —
  If `False`, won’t concatenate initial audio and text tokens.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Remaining dictionary of keyword arguments that are passed to the `generate` method. Refers to the
  original [`generate` docstrings](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate)
  for more information on how to use them.
  Note that keywords with a *depth\_* prefix will be input for the `generate` method of the
  depth decoder. Otherwise, the latter will use its default generation config.

Generates sequences of text token ids and audio tokens ids.

#### get\_unconditional\_inputs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moshi/modeling_moshi.py#L2397)

( num\_samples = 1  )

Parameters

* **num\_samples** (int, *optional*) —
  Number of audio samples to unconditionally generate.
* **max\_new\_tokens** (int, *optional*) —
  Number of tokens to generate for each sample. More tokens means longer audio samples, at the expense of
  longer inference (since more audio tokens need to be generated per sample).

Helper function to get null inputs for unconditional generation, enabling the model to be used without the
feature extractor or tokenizer.

Example:


```
>>> from transformers import MoshiForConditionalGeneration

>>> model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko-pytorch-bf16")

>>> # get the unconditional (or 'null') inputs for the model
>>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
>>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/moshi.md)
