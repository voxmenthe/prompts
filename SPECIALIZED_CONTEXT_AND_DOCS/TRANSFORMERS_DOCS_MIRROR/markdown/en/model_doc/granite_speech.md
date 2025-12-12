*This model was released on 2025-04-16 and added to Hugging Face Transformers on 2025-04-11.*

# Granite Speech

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The [Granite Speech](https://huggingface.co/papers/2505.08699) model ([blog post](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras)) is a multimodal language model, consisting of a speech encoder, speech projector, large language model, and LoRA adapter(s). More details regarding each component for the current (Granite 3.2 Speech) model architecture may be found below.

1. Speech Encoder: A [Conformer](https://huggingface.co/papers/2005.08100) encoder trained with Connectionist Temporal Classification (CTC) on character-level targets on ASR corpora. The encoder uses block-attention and self-conditioned CTC from the middle layer.
2. Speech Projector: A query transformer (q-former) operating on the outputs of the last encoder block. The encoder and projector temporally downsample the audio features to be merged into the multimodal embeddings to be processed by the llm.
3. Large Language Model: The Granite Speech model leverages Granite LLMs, which were originally proposed in [this paper](https://huggingface.co/papers/2408.13359).
4. LoRA adapter(s): The Granite Speech model contains a modality specific LoRA, which will be enabled when audio features are provided, and disabled otherwise.

Note that most of the aforementioned components are implemented generically to enable compatibility and potential integration with other model architectures in transformers.

This model was contributed by [Alexander Brooks](https://huggingface.co/abrooks9944), [Avihu Dekel](https://huggingface.co/Avihu), and [George Saon](https://huggingface.co/gsaon).

## Usage tips

* This model bundles its own LoRA adapter, which will be automatically loaded and enabled/disabled as needed during inference calls. Be sure to install [PEFT](https://github.com/huggingface/peft) to ensure the LoRA is correctly applied!

## GraniteSpeechConfig

### class transformers.GraniteSpeechConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granite_speech/configuration_granite_speech.py#L107)

( text\_config = None encoder\_config = None projector\_config = None audio\_token\_index = 49155 initializer\_range = 0.02 has\_lora\_adapter = True downsample\_rate = 5 window\_size = 15 \*\*kwargs  )

Parameters

* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `GraniteConfig`) —
  The config object or dictionary of the text backbone.
* **encoder\_config** (`GraniteSpeechEncoderConfig`, *optional*) —
  The config object or dictionary of the Granite Speech CTC Encoder.
* **projector\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `Blip2QFormerConfig`) —
  The config object or dictionary of the audio projector.
* **audio\_token\_index** (`int`, *optional*, defaults to 49155) —
  The audio token index to encode the audio prompt.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **has\_lora\_adapter** (`bool`, *optional*, defaults to `True`) —
  Indicates whether or not the model has a lora adapter that should only
  be activate when processing audio inputs.
* **downsample\_rate** (`int`, *optional*, defaults to 5) —
  Downsample rate for the audio feature extractor.
* **window\_size** (`int`, *optional*, defaults to 15) —
  Window size for the audio feature projector.

This is the configuration class to store the configuration of a [GraniteSpeechForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechForConditionalGeneration). It is used to instantiate an
Granite Speech model according to the specified arguments, defining the model architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import GraniteSpeechConfig, GraniteSpeechForConditionalGeneration

>>> # Initializing a GraniteSpeechConfig
>>> configuration = GraniteSpeechConfig()

>>> # Initializing a GraniteSpeechForConditionalGeneration (with random weights)
>>> model = GraniteSpeechForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## GraniteSpeechEncoderConfig

### class transformers.GraniteSpeechEncoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granite_speech/configuration_granite_speech.py#L21)

( input\_dim = 160 num\_layers = 10 hidden\_dim = 1024 feedforward\_mult = 4 num\_heads = 8 dim\_head = 128 output\_dim = 42 context\_size = 200 max\_pos\_emb = 512 dropout = 0.1 conv\_kernel\_size = 15 conv\_expansion\_factor = 2 \*\*kwargs  )

Parameters

* **input\_dim** (`int`, *optional*, defaults to 160) —
  Dimension of the first hidden layer of the encoder.
* **num\_layers** (`int`, *optional*, defaults to 10) —
  Number of encoder blocks.
* **hidden\_dim** (`int`, *optional*, defaults to 1024) —
  The size of the intermediate layers in the conformer encoder.
* **feedforward\_mult** (`int`, *optional*, defaults to 4) —
  Multiplier for the up/down projections in the encoder’s feedforward layers;
  The projections will have intermediate dim of size `hidden_dim * feedforward_mult`.
* **num\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **dim\_head** (`int`, *optional*, defaults to 128) —
  Dimension of attention heads for each attention layer in the Transformer encoder.
* **output\_dim** (`int`, *optional*, defaults to 42) —
  Intermediate dimension of the feedforward projections in the conformer
  to be added to every other encoder block’s output.
* **context\_size** (`int`, *optional*, defaults to 200) —
  Context size to be used in conformer attention.
* **max\_pos\_emb** (`int`, *optional*, defaults to 512) —
  Max pos embeds to be used in attention (shaw’s relative positional encoding).
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for fully connected layers in the encoder.
* **conv\_kernel\_size** (`int`, *optional*, defaults to 15) —
  Kernel size to be used for 1D convolution in each conformer block.
* **conv\_expansion\_factor** (`int`, *optional*, defaults to 2) —
  Intermediate dimension to be used in conformer convolutions.

This is the configuration class to store the configuration of a `GraniteSpeechCTCEncoder`. It is used to instantiate
a Granite Speech audio encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the dfefaults will yield a similar configuration to that of the audio encoder of the Granite Speech
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import GraniteSpeechEncoderConfig, GraniteSpeechCTCEncoder

>>> # Initializing a GraniteSpeechEncoderConfig
>>> configuration = GraniteSpeechEncoderConfig()

>>> # Initializing a GraniteSpeechCTCEncoder (with random weights)
>>> model = GraniteSpeechCTCEncoder(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## GraniteSpeechProcessor

### class transformers.GraniteSpeechProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granite_speech/processing_granite_speech.py#L32)

( audio\_processor tokenizer audio\_token = '<|audio|>' chat\_template = None  )

## GraniteSpeechFeatureExtractor

### class transformers.GraniteSpeechFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granite_speech/feature_extraction_granite_speech.py#L38)

( sampling\_rate: int = 16000 n\_fft: int = 512 win\_length: int = 400 hop\_length: int = 160 n\_mels: int = 80 projector\_window\_size: int = 15 projector\_downsample\_rate: int = 5 \*\*kwargs  )

## GraniteSpeechForConditionalGeneration

### class transformers.GraniteSpeechForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granite_speech/modeling_granite_speech.py#L313)

( config: GraniteSpeechConfig  )

Parameters

* **config** ([GraniteSpeechConfig](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Granite Speech model, which consists of an audio encoder, projector, and language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granite_speech/modeling_granite_speech.py#L356)

( input\_ids: LongTensor = None input\_features: FloatTensor = None input\_features\_mask: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*lm\_kwargs  ) → `transformers.models.granite_speech.modeling_granite_speech.GraniteSpeechCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [GraniteSpeechFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechFeatureExtractor). See `GraniteSpeechFeatureExtractor.__call__()` for details ([GraniteSpeechProcessor](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechProcessor) uses
  [GraniteSpeechFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechFeatureExtractor) for processing audios).
* **input\_features\_mask** (`torch.Tensor`, *optional*) —
  Mask to be applied to audio features prior to scattering into the language embeddings.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
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
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
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
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.granite_speech.modeling_granite_speech.GraniteSpeechCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.granite_speech.modeling_granite_speech.GraniteSpeechCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GraniteSpeechConfig](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [GraniteSpeechForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/granite_speech#transformers.GraniteSpeechForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("None")
>>> model = GraniteSpeechForConditionalGeneration.from_pretrained("None")

>>> # audio file is decoded on the fly
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> # transcribe speech
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
...

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> # compute loss
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
...
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/granite_speech.md)
