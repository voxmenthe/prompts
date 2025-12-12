*This model was released on 2025-06-17 and added to Hugging Face Transformers on 2025-06-25.*

# Kyutai Speech-To-Text

## Overview

[Kyutai STT](https://kyutai.org/next/stt) is a speech-to-text model architecture based on the [Mimi codec](https://huggingface.co/docs/transformers/en/model_doc/mimi), which encodes audio into discrete tokens in a streaming fashion, and a [Moshi-like](https://huggingface.co/docs/transformers/en/model_doc/moshi) autoregressive decoder. Kyutai’s lab has released two model checkpoints:

* [kyutai/stt-1b-en\_fr](https://huggingface.co/kyutai/stt-1b-en_fr): a 1B-parameter model capable of transcribing both English and French
* [kyutai/stt-2.6b-en](https://huggingface.co/kyutai/stt-2.6b-en): a 2.6B-parameter model focused solely on English, optimized for maximum transcription accuracy

![](https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/kyutai_stt.png)

## Usage Tips

### Inference


```
import torch
from datasets import load_dataset, Audio
from transformers import infer_device, KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

# 1. load the model and the processor
torch_device = infer_device()
model_id = "kyutai/stt-2.6b-en-trfs"

processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=torch_device, dtype="auto")

# 2. load audio samples
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
ds = ds.cast_column("audio", Audio(sampling_rate=24000))

# 3. prepare the model inputs
inputs = processor(
    ds[0]["audio"]["array"],
)
inputs.to(model.device)

# 4. infer the model
output_tokens = model.generate(**inputs)

# 5. decode the generated tokens
print(processor.batch_decode(output_tokens, skip_special_tokens=True))
```

### Batched Inference


```
import torch
from datasets import load_dataset, Audio
from transformers import infer_device, KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

# 1. load the model and the processor
torch_device = infer_device()
model_id = "kyutai/stt-2.6b-en-trfs"

processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=torch_device, dtype="auto")

# 2. load audio samples
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
ds = ds.cast_column("audio", Audio(sampling_rate=24000))

# 3. prepare the model inputs
audio_arrays = [ds[i]["audio"]["array"] for i in range(4)]
inputs = processor(audio_arrays, return_tensors="pt", padding=True)
inputs = inputs.to(model.device)

# 4. infer the model
output_tokens = model.generate(**inputs)

# 5. decode the generated tokens
decoded_outputs = processor.batch_decode(output_tokens, skip_special_tokens=True)
for output in decoded_outputs:
    print(output)
```

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb).
The original code can be found [here](https://github.com/kyutai-labs/moshi).

## KyutaiSpeechToTextConfig

### class transformers.KyutaiSpeechToTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/configuration_kyutai_speech_to_text.py#L24)

( codebook\_vocab\_size = 2049 vocab\_size = 4001 hidden\_size = 2048 num\_hidden\_layers = 48 num\_attention\_heads = 32 num\_key\_value\_heads = None max\_position\_embeddings = 750 rope\_theta = 100000.0 hidden\_act = 'silu' head\_dim = None initializer\_range = 0.02 use\_cache = True sliding\_window = 375 attention\_dropout = 0.0 ffn\_dim = 11264 rms\_norm\_eps = 1e-08 num\_codebooks = 32 audio\_bos\_token\_id = 2048 audio\_pad\_token\_id = 69569 tie\_word\_embeddings = False pad\_token\_id = 3 bos\_token\_id = 48000 codec\_config = None \*\*kwargs  )

Parameters

* **codebook\_vocab\_size** (`int`, *optional*, defaults to 2049) —
  Vocabulary size of the codebook. Defines the number of different audio tokens that can be represented by each codebook.
* **vocab\_size** (`int`, *optional*, defaults to 4001) —
  Vocabulary size of the model. Defines the number of different tokens that can be represented by the
  `input_ids` passed when calling the model.
* **hidden\_size** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the layers and the pooler layer of the main decoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 48) —
  Number of decoder layers.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the main decoder block.
* **num\_key\_value\_heads** (`int`, *optional*) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details checkout [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
  `num_attention_heads`.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 750) —
  The maximum sequence length that this model might ever be used with. Typically, set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **rope\_theta** (`float`, *optional*, defaults to 100000.0) —
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
* **sliding\_window** (`int`, *optional*, defaults to 375) —
  Sliding window attention window size. If not specified, will default to `3000`.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **ffn\_dim** (`int`, *optional*, defaults to 11264) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the main decoder block. Must be even.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-08) —
  The epsilon used by the rms normalization layers.
* **num\_codebooks** (`int`, *optional*, defaults to 32) —
  The number of audio codebooks for each audio channels.
* **audio\_bos\_token\_id** (`int`, *optional*, defaults to 2048) —
  Beginning of stream token id for codebook tokens.
* **audio\_pad\_token\_id** (`int`, *optional*, defaults to 69569) —
  Padding token id for codebook tokens.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie weight embeddings.
* **pad\_token\_id** (`int`, *optional*, defaults to 3) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 48000) —
  Beginning of stream token id for text tokens.
* **codec\_config** (`PretrainedConfig`, *optional*) —
  Configuration for the codec.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments. Notably:
  + **audio\_encoder\_config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) — An instance of a configuration object that
    defines the audio encoder config.
  + **depth\_\_config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig), *optional*) — An instance of a configuration object that
    defines the depth decoder config.

This is the configuration class to store the configuration of a [KyutaiSpeechToTextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextForConditionalGeneration).
It is used to instantiate a Kyutai Speech-to-Text model according to the specified arguments, defining the model
architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
2.6b-en model.

e.g. [kyutai/stt-2.6b-en-trfs](https://huggingface.co/kyutai/stt-2.6b-en-trfs)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import KyutaiSpeechToTextConfig, KyutaiSpeechToTextForConditionalGeneration

>>> # Initializing a KyutaiSpeechToTextConfig
>>> configuration = KyutaiSpeechToTextConfig()

>>> # Initializing a model
>>> model = KyutaiSpeechToTextForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## KyutaiSpeechToTextProcessor

### class transformers.KyutaiSpeechToTextProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/processing_kyutai_speech_to_text.py#L31)

( \*args \*\*kwargs  )

Constructs a Moshi ASR processor which wraps [EncodecFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor) and
[PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) into a single processor that inherits both the audio feature extraction and
tokenizer functionalities. See the [**call**()](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextProcessor.__call__) for more
information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/processing_kyutai_speech_to_text.py#L42)

( audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), typing.Sequence[numpy.ndarray], typing.Sequence[ForwardRef('torch.Tensor')], NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.kyutai\_speech\_to\_text.processing\_kyutai\_speech\_to\_text.KyutaiSpeechToTextProcessorKwargs]  ) → [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

Parameters

* **audio** (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`) —
  The audio or batch of audio to be prepared. Each audio can be a NumPy array or PyTorch
  tensor.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors of a particular framework. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return NumPy `np.ndarray` objects.
  + `'jax'`: Return JAX `jnp.ndarray` objects.

Returns

[BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

A [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

* **input\_values** — List of audio values to be fed to a model. Returned when `audio` is not `None`.
* **padding\_mask** — List of indices specifying which input values should be ignored by the model.

Main method to prepare audio to be fed as input to the model. This method forwards the `audio`
arguments to KyutaiSpeechToTextFeatureExtractor’s `__call__()`. Please refer
to the docstring of the above method for more information.

## KyutaiSpeechToTextFeatureExtractor

### class transformers.KyutaiSpeechToTextFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/feature_extraction_kyutai_speech_to_text.py#L34)

( feature\_size: int = 1 sampling\_rate: int = 24000 padding\_value: float = 0.0 chunk\_length\_s: typing.Optional[float] = None overlap: typing.Optional[float] = None audio\_delay\_seconds: typing.Optional[float] = 0.0 audio\_silence\_prefix\_seconds: typing.Optional[float] = 0.0 \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 1) —
  The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
* **sampling\_rate** (`int`, *optional*, defaults to 24000) —
  The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value that is used to fill the padding values.
* **chunk\_length\_s** (`float`, *optional*) —
  If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.
* **overlap** (`float`, *optional*) —
  Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following
  formulae : `int((1.0 - self.overlap) * self.chunk_length)`.
* **audio\_delay\_seconds** (`float`, *optional*, defaults to 0.0) —
  The delay in seconds to add after the audio (right padding).
* **audio\_silence\_prefix\_seconds** (`float`, *optional*, defaults to 0.0) —
  The silence prefix in seconds to add before the audio (left padding).

Constructs an KyutaiSpeechToText feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

## KyutaiSpeechToTextForConditionalGeneration

### class transformers.KyutaiSpeechToTextForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py#L1071)

( config  )

Parameters

* **config** ([KyutaiSpeechToTextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextForConditionalGeneration)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Kyutai Speech To Text Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py#L1092)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

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

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([KyutaiSpeechToTextConfig](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [KyutaiSpeechToTextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from datasets import load_dataset, Audio
>>> from transformers import KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

>>> torch_device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model_id = "kyutai/stt-2.6b-en-trfs"

>>> processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
>>> model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)

>>> ds = load_dataset(
...     "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
... )

>>> ds = ds.cast_column("audio", Audio(sampling_rate=24000))
>>> inputs = processor(
...     ds[0]["audio"]["array"],
... )
>>> inputs.to(torch_device)

>>> output_tokens = model.generate(**inputs)
>>> print(processor.batch_decode(output_tokens, skip_special_tokens=True))
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py#L1347)

( \*args \*\*kwargs  )

This method forwards all its arguments to GenerationMixin’s [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate). Please refer to the docstring of this method for more information.

## KyutaiSpeechToTextModel

### class transformers.KyutaiSpeechToTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py#L805)

( config  )

Parameters

* **config** ([KyutaiSpeechToTextModel](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Kyutai Speech To Text Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py#L823)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[list[torch.FloatTensor], transformers.cache\_utils.Cache, NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **past\_key\_values** (`Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]`) —
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
elements depending on the configuration ([KyutaiSpeechToTextConfig](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextConfig)) and inputs.

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

The [KyutaiSpeechToTextModel](/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/kyutai_speech_to_text.md)
