*This model was released on 2025-04-21 and added to Hugging Face Transformers on 2025-06-26.*

# Dia

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

[Dia](https://github.com/nari-labs/dia) is an open-source text-to-speech (TTS) model (1.6B parameters) developed by [Nari Labs](https://huggingface.co/nari-labs).
It can generate highly realistic dialogue from transcript including non-verbal communications such as laughter and coughing.
Furthermore, emotion and tone control is also possible via audio conditioning (voice cloning).

**Model Architecture:**
Dia is an encoder-decoder transformer based on the original transformer architecture. However, some more modern features such as
rotational positional embeddings (RoPE) are also included. For its text portion (encoder), a byte tokenizer is utilized while
for the audio portion (decoder), a pretrained codec model [DAC](./dac) is used - DAC encodes speech into discrete codebook
tokens and decodes them back into audio.

## Usage Tips

### Generation with Text


```
from transformers import AutoProcessor, DiaForConditionalGeneration, infer_device

torch_device = infer_device()
model_checkpoint = "nari-labs/Dia-1.6B-0626"

text = ["[S1] Dia is an open weights text to dialogue model."]
processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, padding=True, return_tensors="pt").to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(**inputs, max_new_tokens=256)  # corresponds to around ~2s

# save audio to a file
outputs = processor.batch_decode(outputs)
processor.save_audio(outputs, "example.wav")
```

### Generation with Text and Audio (Voice Cloning)


```
from datasets import load_dataset, Audio
from transformers import AutoProcessor, DiaForConditionalGeneration, infer_device

torch_device = infer_device()
model_checkpoint = "nari-labs/Dia-1.6B-0626"

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=44100))
audio = ds[-1]["audio"]["array"]
# text is a transcript of the audio + additional text you want as new audio
text = ["[S1] I know. It's going to save me a lot of money, I hope. [S2] I sure hope so for you."]

processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, audio=audio, padding=True, return_tensors="pt").to(torch_device)
prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(**inputs, max_new_tokens=256)  # corresponds to around ~2s

# retrieve actually generated audio and save to a file
outputs = processor.batch_decode(outputs, audio_prompt_len=prompt_len)
processor.save_audio(outputs, "example_with_audio.wav")
```

### Training


```
from datasets import load_dataset, Audio
from transformers import AutoProcessor, DiaForConditionalGeneration, infer_device

torch_device = infer_device()
model_checkpoint = "nari-labs/Dia-1.6B-0626"

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=44100))
audio = ds[-1]["audio"]["array"]
# text is a transcript of the audio
text = ["[S1] I know. It's going to save me a lot of money, I hope."]

processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(
    text=text,
    audio=audio,
    generation=False,
    output_labels=True,
    padding=True,
    return_tensors="pt"
).to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
out = model(**inputs)
out.loss.backward()
```

This model was contributed by [Jaeyong Sung](https://huggingface.co/buttercrab), [Arthur Zucker](https://huggingface.co/ArthurZ),
and [Anton Vlasjuk](https://huggingface.co/AntonV). The original code can be found [here](https://github.com/nari-labs/dia/).

## DiaConfig

### class transformers.DiaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/configuration_dia.py#L282)

( encoder\_config: typing.Optional[transformers.models.dia.configuration\_dia.DiaEncoderConfig] = None decoder\_config: typing.Optional[transformers.models.dia.configuration\_dia.DiaDecoderConfig] = None norm\_eps: float = 1e-05 is\_encoder\_decoder: bool = True pad\_token\_id: int = 1025 eos\_token\_id: int = 1024 bos\_token\_id: int = 1026 delay\_pattern: typing.Optional[list[int]] = None initializer\_range: float = 0.02 use\_cache: bool = True \*\*kwargs  )

Parameters

* **encoder\_config** (`DiaEncoderConfig`, *optional*) —
  Configuration for the encoder part of the model. If not provided, a default `DiaEncoderConfig` will be used.
* **decoder\_config** (`DiaDecoderConfig`, *optional*) —
  Configuration for the decoder part of the model. If not provided, a default `DiaDecoderConfig` will be used.
* **norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the normalization layers.
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Indicating that this model uses an encoder-decoder architecture.
* **pad\_token\_id** (`int`, *optional*, defaults to 1025) —
  Padding token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 1024) —
  End of stream token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 1026) —
  Beginning of stream token id.
* **delay\_pattern** (`list[int]`, *optional*, defaults to `[0, 8, 9, 10, 11, 12, 13, 14, 15]`) —
  The delay pattern for the decoder. The length of this list must match `decoder_config.num_channels`.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).

This is the configuration class to store the configuration of a [DiaModel](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaModel). It is used to instantiate a
Dia model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the
[nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import DiaConfig, DiaModel

>>> # Initializing a DiaConfig with default values
>>> configuration = DiaConfig()

>>> # Initializing a DiaModel (with random weights) from the configuration
>>> model = DiaModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

#### get\_text\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/configuration_dia.py#L371)

( \*args \*\*kwargs  )

Defaulting to audio config as it’s the decoder in this case which is usually the text backbone

## DiaDecoderConfig

### class transformers.DiaDecoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/configuration_dia.py#L141)

( max\_position\_embeddings: int = 3072 num\_hidden\_layers: int = 18 hidden\_size: int = 2048 intermediate\_size: int = 8192 num\_attention\_heads: int = 16 num\_key\_value\_heads: int = 4 head\_dim: int = 128 cross\_num\_attention\_heads: int = 16 cross\_head\_dim: int = 128 cross\_num\_key\_value\_heads: int = 16 cross\_hidden\_size: int = 1024 norm\_eps: float = 1e-05 vocab\_size: int = 1028 hidden\_act: str = 'silu' num\_channels: int = 9 rope\_theta: float = 10000.0 rope\_scaling: typing.Optional[dict] = None initializer\_range: float = 0.02 use\_cache: bool = True is\_encoder\_decoder: bool = True \*\*kwargs  )

Parameters

* **max\_position\_embeddings** (`int`, *optional*, defaults to 3072) —
  The maximum sequence length that this model might ever be used with.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 18) —
  Number of hidden layers in the Transformer decoder.
* **hidden\_size** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the decoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 8192) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer decoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 4) —
  Number of key and value heads for each attention layer in the Transformer decoder.
* **head\_dim** (`int`, *optional*, defaults to 128) —
  Dimensionality of the attention head.
* **cross\_num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each cross-attention layer in the Transformer decoder.
* **cross\_head\_dim** (`int`, *optional*, defaults to 128) —
  Dimensionality of the cross-attention head.
* **cross\_num\_key\_value\_heads** (`int`, *optional*, defaults to 16) —
  Number of key and value heads for each cross-attention layer in the Transformer decoder.
* **cross\_hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the cross-attention layers.
* **norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the normalization layers.
* **vocab\_size** (`int`, *optional*, defaults to 1028) —
  Vocabulary size of the Dia model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [DiaModel](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaModel).
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder. If string, `"gelu"`, `"relu"`,
  `"swish"` and `"gelu_new"` are supported.
* **num\_channels** (`int`, *optional*, defaults to 9) —
  Number of channels for the Dia decoder.
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (`dict`, *optional*) —
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
  `short_factor` (`List[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to short contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `long_factor` (`List[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to long contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `low_freq_factor` (`float`,* optional*):
  Only used with ‘llama3’. Scaling factor applied to low frequency components of the RoPE
  `high_freq_factor` (`float`,* optional\*):
  Only used with ‘llama3’. Scaling factor applied to high frequency components of the RoPE
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Indicating that this model is part of an encoder-decoder architecture.

This is the configuration class to store the configuration of a `DiaDecoder`. It is used to instantiate a Dia
decoder according to the specified arguments, defining the decoder architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## DiaEncoderConfig

### class transformers.DiaEncoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/configuration_dia.py#L27)

( max\_position\_embeddings: int = 1024 num\_hidden\_layers: int = 12 hidden\_size: int = 1024 num\_attention\_heads: int = 16 num\_key\_value\_heads: int = 16 head\_dim: int = 128 intermediate\_size: int = 4096 norm\_eps: float = 1e-05 vocab\_size: int = 256 hidden\_act: str = 'silu' rope\_theta: float = 10000.0 rope\_scaling: typing.Optional[dict] = None initializer\_range: float = 0.02 \*\*kwargs  )

Parameters

* **max\_position\_embeddings** (`int`, *optional*, defaults to 1024) —
  The maximum sequence length that this model might ever be used with.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 16) —
  Number of key and value heads for each attention layer in the Transformer encoder.
* **head\_dim** (`int`, *optional*, defaults to 128) —
  Dimensionality of the attention head.
* **intermediate\_size** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the normalization layers.
* **vocab\_size** (`int`, *optional*, defaults to 256) —
  Vocabulary size of the Dia model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [DiaModel](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaModel).
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"swish"` and `"gelu_new"` are supported.
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (`dict`, *optional*) —
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
  `short_factor` (`List[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to short contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `long_factor` (`List[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to long contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `low_freq_factor` (`float`,* optional*):
  Only used with ‘llama3’. Scaling factor applied to low frequency components of the RoPE
  `high_freq_factor` (`float`,* optional\*):
  Only used with ‘llama3’. Scaling factor applied to high frequency components of the RoPE
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a `DiaEncoder`. It is used to instantiate a Dia
encoder according to the specified arguments, defining the encoder architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## DiaTokenizer

### class transformers.DiaTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/tokenization_dia.py#L26)

( pad\_token: typing.Optional[str] = '<pad>' unk\_token: typing.Optional[str] = '<pad>' max\_length: typing.Optional[int] = 1024 offset: int = 0 \*\*kwargs  )

Parameters

* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **unk\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **max\_length** (`int`, *optional*, defaults to 1024) —
  The maximum length of the sequences when encoding. Sequences longer than this will be truncated.
* **offset** (`int`, *optional*, defaults to 0) —
  The offset of the tokenizer.

Construct a Dia tokenizer. Dia simply uses raw bytes utf-8 encoding except for special tokens `[S1]` and `[S2]`.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828)

( text: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_pair: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_pair\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy, NoneType] = None max\_length: typing.Optional[int] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add special tokens when encoding the sequences. This will use the underlying
  `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
  automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
  automatically.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence is provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **stride** (`int`, *optional*, defaults to 0) —
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
* **is\_split\_into\_words** (`bool`, *optional*, defaults to `False`) —
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **return\_token\_type\_ids** (`bool`, *optional*) —
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
* **return\_overflowing\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
* **return\_special\_tokens\_mask** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return special tokens mask information.
* **return\_offsets\_mapping** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using
  Python’s tokenizer, this method will raise `NotImplementedError`.
* **return\_length** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the lengths of the encoded inputs.
* **verbose** (`bool`, *optional*, defaults to `True`) —
  Whether or not to print more information and warnings.
* \***\*kwargs** — passed to the `self.tokenize()` method

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

* **input\_ids** — List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)
* **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
* **length** — The length of the inputs (when `return_length=True`)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

## DiaFeatureExtractor

### class transformers.DiaFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/feature_extraction_dia.py#L29)

( feature\_size: int = 1 sampling\_rate: int = 16000 padding\_value: float = 0.0 hop\_length: int = 512 \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 1) —
  The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value that is used for padding.
* **hop\_length** (`int`, *optional*, defaults to 512) —
  Overlap length between successive windows.

Constructs an Dia feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/feature_extraction_dia.py#L60)

( raw\_audio: typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]] padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy, NoneType] = None truncation: typing.Optional[bool] = False max\_length: typing.Optional[int] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None sampling\_rate: typing.Optional[int] = None  )

Parameters

* **raw\_audio** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) —
  The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. The numpy array must be of shape
  `(num_samples,)` for mono audio (`feature_size = 1`), or `(2, num_samples)` for stereo audio
  (`feature_size = 2`).
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) —
  Select a strategy to pad the returned sequences (according to the model’s padding side and padding
  index) among:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, *optional*, defaults to `False`) —
  Activates truncation to cut input sequences longer than `max_length` to `max_length`.
* **max\_length** (`int`, *optional*) —
  Maximum length of the returned list and optionally padding length (see above).
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*, default to ‘pt’) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **sampling\_rate** (`int`, *optional*) —
  The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors.

Main method to featurize and prepare for the model one or several sequence(s).

## DiaProcessor

### class transformers.DiaProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/processing_dia.py#L62)

( feature\_extractor tokenizer audio\_tokenizer  )

Parameters

* **feature\_extractor** (`DiaFeatureExtractor`) —
  An instance of [DiaFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaFeatureExtractor). The feature extractor is a required input.
* **tokenizer** (`DiaTokenizer`) —
  An instance of [DiaTokenizer](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaTokenizer). The tokenizer is a required input.
* **audio\_tokenizer** (`DacModel`) —
  An instance of [DacModel](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel) used to encode/decode audio into/from codebooks. It is is a required input.

Constructs a Dia processor which wraps a [DiaFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaFeatureExtractor), [DiaTokenizer](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaTokenizer), and a [DacModel](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel) into
a single processor. It inherits, the audio feature extraction, tokenizer, and audio encode/decode functio-
nalities. See [**call**()](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor.__call__), `~DiaProcessor.encode`, and [decode()](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor.decode) for more
information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/processing_dia.py#L85)

( text: typing.Union[str, list[str]] audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), typing.Sequence[numpy.ndarray], typing.Sequence[ForwardRef('torch.Tensor')], NoneType] = None output\_labels: typing.Optional[bool] = False \*\*kwargs: typing\_extensions.Unpack[transformers.models.dia.processing\_dia.DiaProcessorKwargs]  )

Main method to prepare text(s) and audio to be fed as input to the model. The `audio` argument is
forwarded to the DiaFeatureExtractor’s [**call**()](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaFeatureExtractor.__call__) and subsequently to the
DacModel’s [encode()](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel.encode). The `text` argument to [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Please refer
to the docstring of the above methods for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/processing_dia.py#L258)

( decoder\_input\_ids: torch.Tensor audio\_prompt\_len: typing.Optional[int] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.dia.processing\_dia.DiaProcessorKwargs]  )

Parameters

* **decoder\_input\_ids** (`torch.Tensor`) — The complete output sequence of the decoder.
* **audio\_prompt\_len** (`int`) — The audio prefix length (e.g. when using voice cloning).

Decodes a batch of audio codebook sequences into their respective audio waveforms via the
`audio_tokenizer`. See [decode()](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel.decode) for more information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/processing_dia.py#L329)

( decoder\_input\_ids: torch.Tensor audio\_prompt\_len: typing.Optional[int] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.dia.processing\_dia.DiaProcessorKwargs]  )

Decodes a single sequence of audio codebooks into the respective audio waveform via the
`audio_tokenizer`. See [decode()](/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel.decode) and [batch\_decode()](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor.batch_decode) for more information.

## DiaModel

### class transformers.DiaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/modeling_dia.py#L719)

( config: DiaConfig  )

Parameters

* **config** ([DiaConfig](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Dia model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/modeling_dia.py#L730)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_position\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Union[transformers.modeling\_outputs.BaseModelOutput, tuple, NoneType] = None past\_key\_values: typing.Optional[transformers.cache\_utils.EncoderDecoderCache] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs  ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch\_size \* num\_codebooks, target\_sequence\_length) —
* **or** (batch\_size, target\_sequence\_length, num\_codebooks)`, *optional*) —
  1. (batch\_size \* num\_codebooks, target\_sequence\_length): corresponds to the general use case where
     the audio input codebooks are flattened into the batch dimension. This also aligns with the flat-
     tened audio logits which are used to calculate the loss.
  2. (batch\_size, sequence\_length, num\_codebooks): corresponds to the internally used shape of
     Dia to calculate embeddings and subsequent steps more efficiently.

  If no `decoder_input_ids` are provided, it will create a tensor of `bos_token_id` with shape
  `(batch_size, 1, num_codebooks)`. Indices can be obtained using the [DiaProcessor](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor). See
  [DiaProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor.__call__) for more details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_position\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`) —
  Indices of positions of each input sequence tokens in the position embeddings.
  Used to calculate the position embeddings up to `config.decoder_config.max_position_embeddings`.

  [What are position IDs?](../glossary#position-ids)
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
* **encoder\_outputs** (`Union[~modeling_outputs.BaseModelOutput, tuple, NoneType]`) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`~cache_utils.EncoderDecoderCache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

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

The [DiaModel](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## DiaForConditionalGeneration

### class transformers.DiaForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/modeling_dia.py#L847)

( config: DiaConfig  )

Parameters

* **config** ([DiaConfig](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Dia model consisting of a (byte) text encoder and audio decoder with a prediction head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/modeling_dia.py#L871)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_position\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Union[transformers.modeling\_outputs.BaseModelOutput, tuple, NoneType] = None past\_key\_values: typing.Optional[transformers.cache\_utils.EncoderDecoderCache] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch\_size \* num\_codebooks, target\_sequence\_length) —
* **or** (batch\_size, target\_sequence\_length, num\_codebooks)`, *optional*) —
  1. (batch\_size \* num\_codebooks, target\_sequence\_length): corresponds to the general use case where
     the audio input codebooks are flattened into the batch dimension. This also aligns with the flat-
     tened audio logits which are used to calculate the loss.
  2. (batch\_size, sequence\_length, num\_codebooks): corresponds to the internally used shape of
     Dia to calculate embeddings and subsequent steps more efficiently.

  If no `decoder_input_ids` are provided, it will create a tensor of `bos_token_id` with shape
  `(batch_size, 1, num_codebooks)`. Indices can be obtained using the [DiaProcessor](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor). See
  [DiaProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor.__call__) for more details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_position\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`) —
  Indices of positions of each input sequence tokens in the position embeddings.
  Used to calculate the position embeddings up to `config.decoder_config.max_position_embeddings`.

  [What are position IDs?](../glossary#position-ids)
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
* **encoder\_outputs** (`Union[~modeling_outputs.BaseModelOutput, tuple, NoneType]`) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`~cache_utils.EncoderDecoderCache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size * num_codebooks,)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in
  `[0, ..., config.decoder_config.vocab_size - 1]` or -100. Tokens with indices set to `-100`
  are ignored (masked).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

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

The [DiaForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/generation_dia.py#L406)

( inputs: typing.Optional[torch.Tensor] = None generation\_config: typing.Optional[transformers.generation.configuration\_utils.GenerationConfig] = None logits\_processor: typing.Optional[transformers.generation.logits\_process.LogitsProcessorList] = None stopping\_criteria: typing.Optional[transformers.generation.stopping\_criteria.StoppingCriteriaList] = None prefix\_allowed\_tokens\_fn: typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None synced\_gpus: typing.Optional[bool] = None assistant\_model: typing.Optional[ForwardRef('PreTrainedModel')] = None streamer: typing.Optional[ForwardRef('BaseStreamer')] = None negative\_prompt\_ids: typing.Optional[torch.Tensor] = None negative\_prompt\_attention\_mask: typing.Optional[torch.Tensor] = None use\_model\_defaults: typing.Optional[bool] = None custom\_generate: typing.Optional[str] = None \*\*kwargs  )

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dia.md)
