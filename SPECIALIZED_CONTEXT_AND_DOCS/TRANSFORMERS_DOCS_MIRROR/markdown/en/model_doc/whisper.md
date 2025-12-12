*This model was released on 2022-12-06 and added to Hugging Face Transformers on 2022-10-05.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# Whisper

[Whisper](https://huggingface.co/papers/2212.04356) is a encoder-decoder (sequence-to-sequence) transformer pretrained on 680,000 hours of labeled audio data. This amount of pretraining data enables zero-shot performance on audio tasks in English and many other languages. The decoder allows Whisper to map the encoders learned speech representations to useful outputs, such as text, without additional fine-tuning. Whisper just works out of the box.

You can find all the original Whisper checkpoints under the [Whisper](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013) collection.

> [!NOTE]
> The `head_mask` argument is ignored when using all attention implementation other than “eager”. If you have a `head_mask` and want it to have effect, load the model with `XXXModel.from_pretrained(model_id, attn_implementation="eager")`

Click on the Whisper models in the right sidebar for more examples of how to apply Whisper to different audio tasks.

The example below demonstrates how to automatically transcribe speech into text with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
```

## Notes

* Whisper relies a custom `generate` for inference, make sure to check the docs below.
* The [WhisperProcessor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperProcessor) can be used for preparing audio and decoding predicted ids back into text.

## WhisperConfig

### class transformers.WhisperConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/configuration_whisper.py#L60)

( vocab\_size = 51865 num\_mel\_bins = 80 encoder\_layers = 4 encoder\_attention\_heads = 6 decoder\_layers = 4 decoder\_attention\_heads = 6 decoder\_ffn\_dim = 1536 encoder\_ffn\_dim = 1536 encoder\_layerdrop = 0.0 decoder\_layerdrop = 0.0 decoder\_start\_token\_id = 50257 use\_cache = True is\_encoder\_decoder = True activation\_function = 'gelu' d\_model = 384 dropout = 0.0 attention\_dropout = 0.0 activation\_dropout = 0.0 init\_std = 0.02 scale\_embedding = False max\_source\_positions = 1500 max\_target\_positions = 448 pad\_token\_id = 50256 bos\_token\_id = 50256 eos\_token\_id = 50256 suppress\_tokens = None begin\_suppress\_tokens = [220, 50256] use\_weighted\_layer\_sum = False classifier\_proj\_size = 256 apply\_spec\_augment = False mask\_time\_prob = 0.05 mask\_time\_length = 10 mask\_time\_min\_masks = 2 mask\_feature\_prob = 0.0 mask\_feature\_length = 10 mask\_feature\_min\_masks = 0 median\_filter\_width = 7 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 51865) —
  Vocabulary size of the Whisper model. Defines the number of different tokens that can be represented by the
  `decoder_input_ids` passed when calling [WhisperModel](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperModel)
* **num\_mel\_bins** (`int`, *optional*, defaults to 80) —
  Number of mel features used per input features. Should correspond to the value used in the
  `WhisperProcessor` class.
* **encoder\_layers** (`int`, *optional*, defaults to 4) —
  Number of encoder layers.
* **decoder\_layers** (`int`, *optional*, defaults to 4) —
  Number of decoder layers.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 6) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **decoder\_attention\_heads** (`int`, *optional*, defaults to 6) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 1536) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in encoder.
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 1536) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in decoder.
* **encoder\_layerdrop** (`float`, *optional*, defaults to 0.0) —
  The LayerDrop probability for the encoder. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **decoder\_layerdrop** (`float`, *optional*, defaults to 0.0) —
  The LayerDrop probability for the decoder. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **decoder\_start\_token\_id** (`int`, *optional*, defaults to 50257) —
  Corresponds to the ”<|startoftranscript|>” token, which is automatically used when no `decoder_input_ids`
  are provided to the `generate` function. It is used to guide the model`s generation process depending on
  the task.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Whether the model is used as an encoder/decoder or not.
* **activation\_function** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **d\_model** (`int`, *optional*, defaults to 384) —
  Dimensionality of the layers.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **activation\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for activations inside the fully connected layer.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **scale\_embedding** (`bool`, *optional*, defaults to False) —
  Scale embeddings by diving by sqrt(d\_model).
* **max\_source\_positions** (`int`, *optional*, defaults to 1500) —
  The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
* **max\_target\_positions** (`int`, *optional*, defaults to 448) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **pad\_token\_id** (`int`, *optional*, defaults to 50256) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 50256) —
  Begin of stream token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 50256) —
  End of stream token id.
* **suppress\_tokens** (`list[int]`, *optional*) —
  A list containing the non-speech tokens that will be used by the logit processor in the `generate`
  function. NON\_SPEECH\_TOKENS and NON\_SPEECH\_TOKENS\_MULTI each correspond to the `english-only` and the
  `multilingual` model.
* **begin\_suppress\_tokens** (`list[int]`, *optional*, defaults to `[220,50256]`) —
  A list containing tokens that will be suppressed at the beginning of the sampling process. Initialized as
  the token for `" "` (`blank_token_id`) and the `eos_token_id`
* **use\_weighted\_layer\_sum** (`bool`, *optional*, defaults to `False`) —
  Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
  instance of [WhisperForAudioClassification](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForAudioClassification).
* **classifier\_proj\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the projection before token mean-pooling for classification. Only relevant when using an
  instance of [WhisperForAudioClassification](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForAudioClassification).
* **apply\_spec\_augment** (`bool`, *optional*, defaults to `False`) —
  Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
  [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
  Recognition](https://huggingface.co/papers/1904.08779).
* **mask\_time\_prob** (`float`, *optional*, defaults to 0.05) —
  Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
  procedure generates `mask_time_prob*len(time_axis)/mask_time_length` independent masks over the axis. If
  reasoning from the probability of each feature vector to be chosen as the start of the vector span to be
  masked, *mask\_time\_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
  actual percentage of masked vectors. This is only relevant if `apply_spec_augment == True`.
* **mask\_time\_length** (`int`, *optional*, defaults to 10) —
  Length of vector span along the time axis.
* **mask\_time\_min\_masks** (`int`, *optional*, defaults to 2), —
  The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
  irrespectively of `mask_feature_prob`. Only relevant if ”mask\_time\_prob\*len(time\_axis)/mask\_time\_length <
  mask\_time\_min\_masks”
* **mask\_feature\_prob** (`float`, *optional*, defaults to 0.0) —
  Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
  masking procedure generates `mask_feature_prob*len(feature_axis)/mask_time_length` independent masks over
  the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector
  span to be masked, *mask\_feature\_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
  may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
* **mask\_feature\_length** (`int`, *optional*, defaults to 10) —
  Length of vector span along the feature axis.
* **mask\_feature\_min\_masks** (`int`, *optional*, defaults to 0), —
  The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
  step, irrespectively of `mask_feature_prob`. Only relevant if
  `mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks`.
* **median\_filter\_width** (`int`, *optional*, defaults to 7) —
  Width of the median filter used to smoothen to cross-attention outputs when computing token timestamps.
  Should be an odd number.

This is the configuration class to store the configuration of a [WhisperModel](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperModel). It is used to instantiate a
Whisper model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Whisper
[openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import WhisperConfig, WhisperModel

>>> # Initializing a Whisper tiny style configuration
>>> configuration = WhisperConfig()

>>> # Initializing a model (with random weights) from the tiny style configuration
>>> model = WhisperModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## WhisperTokenizer

### class transformers.WhisperTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper.py#L210)

( vocab\_file merges\_file normalizer\_file = None errors = 'replace' unk\_token = '<|endoftext|>' bos\_token = '<|endoftext|>' eos\_token = '<|endoftext|>' pad\_token = None add\_prefix\_space = False language = None task = None predict\_timestamps = False \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **normalizer\_file** (`str`, *optional*) —
  Path to the normalizer\_file file.
* **errors** (`str`, *optional*, defaults to `"replace"`) —
  Paradigm to follow when decoding bytes to UTF-8. See
  [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
* **unk\_token** (`str`, *optional*, defaults to `"<|endoftext|>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **bos\_token** (`str`, *optional*, defaults to `"<|endoftext|>"`) —
  The beginning of sequence token. The `decoder_start_token_id` is used to set the first token as
  `"<|startoftranscript|>"` when generating.
* **eos\_token** (`str`, *optional*, defaults to `"<|endoftext|>"`) —
  The end of sequence token.
* **pad\_token** (`str`, *optional*) —
  The token used for padding, for example when batching sequences of different lengths.
* **add\_prefix\_space** (`bool`, *optional*, defaults to `False`) —
  Whether or not to add an initial space to the input. This allows to treat the leading word just as any
  other word.
* **language** (`str`, *optional*) —
  The language of the transcription text. The corresponding language id token is appended to the start of the
  sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
  `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
* **task** (`str`, *optional*) —
  Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
  fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
* **predict\_timestamps** (`bool`, *optional*, defaults to `False`) —
  Whether to omit the `<|notimestamps|>` token at the start of the sequence.

Construct a Whisper tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains some of the main methods. Users should refer to
the superclass for more information regarding such methods.

#### set\_prefix\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper.py#L380)

( language: typing.Optional[str] = None task: typing.Optional[str] = None predict\_timestamps: typing.Optional[bool] = None  )

Parameters

* **language** (`str`, *optional*, defaults to `None`) —
  The language of the transcription text.
* **task** (`str`, *optional*, defaults to `None`) —
  Task identifier to append at the start of sequence (if any).
* **predict\_timestamps** (`bool`, *optional*, defaults to `None`) —
  Whether to omit the `<|notimestamps|>` token at the start of the sequence.

Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to

update the prefix tokens as required when fine-tuning. Example:


```
>>> # instantiate the tokenizer and set the prefix token to Spanish
>>> tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="spanish")
>>> # now switch the prefix token from Spanish to French
>>> tokenizer.set_prefix_tokens(language="french")
```

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper.py#L441)

( token\_ids\_0 token\_ids\_1 = None  )

Build model inputs from a sequence by appending eos\_token\_id.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper.py#L449)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

`list[int]`

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) — The first tokenized sequence.
* **token\_ids\_1** (`list[int]`, *optional*) — The second tokenized sequence.

Returns

`list[int]`

The token type ids.

Create the token type IDs corresponding to the sequences passed. [What are token type
IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper.py#L801)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3833)

( sequences: typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None \*\*kwargs  ) → `list[str]`

Parameters

* **sequences** (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces`.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`list[str]`

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper.py#L675)

( token\_ids skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None output\_offsets: bool = False time\_precision: float = 0.02 decode\_with\_timestamps: bool = False normalize: bool = False basic\_normalize: bool = False remove\_diacritics: bool = False \*\*kwargs  ) → `str`

Parameters

* **token\_ids** (`Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding. Will remove the previous tokens (pre-prompt)
  if present.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
* **output\_offsets** (`bool`, *optional*, defaults to `False`) —
  Whether or not to output the offsets of the tokens. This should only be set if the model predicted
  timestamps. If there are previous tokens (pre-prompt) to decode, they will only appear in the decoded
  text if they contain timestamp tokens.
* **time\_precision** (`float`, *optional*, defaults to 0.02) —
  The time ratio to convert from token to time.
* **decode\_with\_timestamps** (`bool`, *optional*, defaults to `False`) —
  Whether or not to decode with timestamps included in the raw text.
* **normalize** (`bool`, *optional*, defaults to `False`) —
  Whether or not to apply the English text normalizer to the decoded text. Only applicable when the
  target text is in English. Otherwise, the basic text normalizer should be applied.
* **basic\_normalize** (`bool`, *optional*, defaults to `False`) —
  Whether or not to apply the Basic text normalizer to the decoded text. Applicable to multilingual
  target text.
* **remove\_diacritics** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove diacritics when applying the Basic text normalizer. Removing diacritics may
  destroy information in the decoded text, hence it should be used with caution.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`str`

The decoded sentence.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

#### basic\_normalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper.py#L524)

( text remove\_diacritics = False  )

Normalize a given string using the `BasicTextNormalizer` class, which performs commons transformation on
multilingual text.

#### normalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper.py#L516)

( text  )

Normalize a given string using the `EnglishTextNormalizer` class, which performs commons transformation on
english text.

## WhisperTokenizerFast

### class transformers.WhisperTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper_fast.py#L44)

( vocab\_file = None merges\_file = None normalizer\_file = None tokenizer\_file = None unk\_token = '<|endoftext|>' bos\_token = '<|endoftext|>' eos\_token = '<|endoftext|>' add\_prefix\_space = False language = None task = None predict\_timestamps = False \*\*kwargs  )

Parameters

* **vocab\_file** (`str`, *optional*) —
  Path to the vocabulary file.
* **merges\_file** (`str`, *optional*) —
  Path to the merges file.
* **normalizer\_file** (`str`, *optional*) —
  Path to the normalizer\_file file.
* **tokenizer\_file** (`str`, *optional*) —
  Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
  contains everything needed to load the tokenizer.
* **unk\_token** (`str`, *optional*, defaults to `"<|endoftext|>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **bos\_token** (`str`, *optional*, defaults to `"<|endoftext|>"`) —
  The beginning of sequence token. The `decoder_start_token_id` is used to set the first token as
  `"<|startoftranscript|>"` when generating.
* **eos\_token** (`str`, *optional*, defaults to `"<|endoftext|>"`) —
  The end of sequence token.
* **add\_prefix\_space** (`bool`, *optional*, defaults to `False`) —
  Whether or not to add an initial space to the input. This allows to treat the leading word just as any
  other word. (Whisper tokenizer detect beginning of words by the preceding space).
* **language** (`str`, *optional*) —
  The language of the transcription text. The corresponding language id token is appended to the start of the
  sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
  `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
* **task** (`str`, *optional*) —
  Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
  fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
* **predict\_timestamps** (`bool`, *optional*, defaults to `False`) —
  Whether to omit the `<|notimestamps|>` token at the start of the sequence.

Construct a “fast” Whisper tokenizer (backed by HuggingFace’s *tokenizers* library).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### set\_prefix\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper_fast.py#L454)

( language: typing.Optional[str] = None task: typing.Optional[str] = None predict\_timestamps: typing.Optional[bool] = None  )

Parameters

* **language** (`str`, *optional*, defaults to `None`) —
  The language of the transcription text.
* **task** (`str`, *optional*, defaults to `None`) —
  Task identifier to append at the start of sequence (if any).
* **predict\_timestamps** (`bool`, *optional*, defaults to `None`) —
  Whether to omit the `<|notimestamps|>` token at the start of the sequence.

Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to

update the prefix tokens as required when fine-tuning. Example:


```
>>> # instantiate the tokenizer and set the prefix token to Spanish
>>> tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny", language="spanish")
>>> # now switch the prefix token from Spanish to French
>>> tokenizer.set_prefix_tokens(language="french")
```

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper_fast.py#L530)

( token\_ids\_0 token\_ids\_1 = None  )

Build model inputs from a sequence by appending eos\_token\_id.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper_fast.py#L538)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

`list[int]`

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) — The first tokenized sequence.
* **token\_ids\_1** (`list[int]`, *optional*) — The second tokenized sequence.

Returns

`list[int]`

The token type ids.

Create the token type IDs corresponding to the sequences passed. [What are token type
IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper_fast.py#L439)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3833)

( sequences: typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None \*\*kwargs  ) → `list[str]`

Parameters

* **sequences** (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces`.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`list[str]`

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper_fast.py#L312)

( token\_ids skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None output\_offsets: bool = False time\_precision: float = 0.02 decode\_with\_timestamps: bool = False normalize: bool = False basic\_normalize: bool = False remove\_diacritics: bool = False \*\*kwargs  ) → `str`

Parameters

* **token\_ids** (`Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding. Will remove the previous tokens (pre-prompt)
  if present.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
* **output\_offsets** (`bool`, *optional*, defaults to `False`) —
  Whether or not to output the offsets of the tokens. This should only be set if the model predicted
  timestamps. If there are previous tokens (pre-prompt) to decode, they will only appear in the decoded
  text if they contain timestamp tokens.
* **time\_precision** (`float`, *optional*, defaults to 0.02) —
  The time ratio to convert from token to time.
* **decode\_with\_timestamps** (`bool`, *optional*, defaults to `False`) —
  Whether or not to decode with timestamps included in the raw text.
* **normalize** (`bool`, *optional*, defaults to `False`) —
  Whether or not to apply the English text normalizer to the decoded text. Only applicable when the
  target text is in English. Otherwise, the basic text normalizer should be applied.
* **basic\_normalize** (`bool`, *optional*, defaults to `False`) —
  Whether or not to apply the Basic text normalizer to the decoded text. Applicable to multilingual
  target text.
* **remove\_diacritics** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove diacritics when applying the Basic text normalizer. Removing diacritics may
  destroy information in the decoded text, hence it should be used with caution.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`str`

The decoded sentence.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

#### basic\_normalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper_fast.py#L429)

( text remove\_diacritics = False  )

Normalize a given string using the `BasicTextNormalizer` class, which performs commons transformation on
multilingual text.

#### normalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/tokenization_whisper_fast.py#L421)

( text  )

Normalize a given string using the `EnglishTextNormalizer` class, which performs commons transformation on
english text.

## WhisperFeatureExtractor

### class transformers.WhisperFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/feature_extraction_whisper.py#L36)

( feature\_size = 80 sampling\_rate = 16000 hop\_length = 160 chunk\_length = 30 n\_fft = 400 padding\_value = 0.0 dither = 0.0 return\_attention\_mask = False \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 80) —
  The feature dimension of the extracted features.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
* **hop\_length** (`int`, *optional*, defaults to 160) —
  Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
* **chunk\_length** (`int`, *optional*, defaults to 30) —
  The maximum number of chunks of `sampling_rate` samples used to trim and pad longer or shorter audio
  sequences.
* **n\_fft** (`int`, *optional*, defaults to 400) —
  Size of the Fourier transform.
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  Padding value used to pad the audio. Should correspond to silences.
* **dither** (`float`, *optional*, defaults to 0.0) —
  Adds dithering. In other words, adds a small Gaussian noise to each frame.
  E.g. use 0.0001 to add dithering with a normal distribution centered
  around 0.0 with standard deviation 0.0001 (assuming [-1,+1] range of raw\_speech).
  The value 0.0 means no dithering.
  Dithering has similar effect as `spectrogram(mel_floor=...)`. It reduces
  the high log\_mel\_fbank values for signals with hard-zero sections,
  when VAD cutoff is present in the signal.

Constructs a Whisper feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time Fourier Transform` which should match pytorch’s `torch.stft` equivalent.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/feature_extraction_whisper.py#L192)

( raw\_speech: typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]] truncation: bool = True pad\_to\_multiple\_of: typing.Optional[int] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_attention\_mask: typing.Optional[bool] = None padding: typing.Optional[str] = 'max\_length' max\_length: typing.Optional[int] = None sampling\_rate: typing.Optional[int] = None do\_normalize: typing.Optional[bool] = None device: typing.Optional[str] = 'cpu' return\_token\_timestamps: typing.Optional[bool] = None \*\*kwargs  )

Parameters

* **raw\_speech** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) —
  The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
  stereo, i.e. single float per timestep.
* **truncation** (`bool`, *optional*, default to `True`) —
  Activates truncation to cut input sequences longer than *max\_length* to *max\_length*.
* **pad\_to\_multiple\_of** (`int`, *optional*, defaults to None) —
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific feature\_extractor’s default.

  [What are attention masks?](../glossary#attention-mask)

  For Whisper models, `attention_mask` should always be passed for batched inference, to avoid subtle
  bugs.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **sampling\_rate** (`int`, *optional*) —
  The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
  pipeline.
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value that is used to fill the padding values / vectors.
* **do\_normalize** (`bool`, *optional*, defaults to `False`) —
  Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
  improve the performance of the model.
* **device** (`str`, *optional*, defaults to `'cpu'`) —
  Specifies the device for computation of the log-mel spectrogram of audio signals in the
  `_torch_extract_fbank_features` method. (e.g., “cpu”, “cuda”)
* **return\_token\_timestamps** (`bool`, *optional*, defaults to `None`) —
  Deprecated. Use `return_attention_mask` instead from which the number of frames can be inferred.

  Whether or not to return the number of frames of the input raw\_speech.
  These num\_frames can be used by the model to compute word level timestamps.

Main method to featurize and prepare for the model one or several sequence(s). Implementation uses PyTorch for
the STFT computation if available, otherwise a slower NumPy based one.

## WhisperProcessor

### class transformers.WhisperProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/processing_whisper.py#L22)

( feature\_extractor tokenizer  )

Parameters

* **feature\_extractor** (`WhisperFeatureExtractor`) —
  An instance of [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor). The feature extractor is a required input.
* **tokenizer** (`WhisperTokenizer`) —
  An instance of [WhisperTokenizer](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperTokenizer). The tokenizer is a required input.

Constructs a Whisper processor which wraps a Whisper feature extractor and a Whisper tokenizer into a single
processor.

[WhisperProcessor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperProcessor) offers all the functionalities of [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor) and [WhisperTokenizer](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperTokenizer). See
the [**call**()](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/processing_whisper.py#L48)

( \*args \*\*kwargs  )

Forwards the `audio` argument to WhisperFeatureExtractor’s [**call**()](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor.__call__) and the `text`
argument to [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Please refer to the docstring of the above two methods for more
information.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1272)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained feature\_extractor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a feature extractor file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`.
  + a path or url to a saved feature extractor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
* \***\*kwargs** —
  Additional keyword arguments passed along to both
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and
  `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L653)

( save\_directory push\_to\_hub: bool = False legacy\_serialization: bool = True \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
  be created if it does not exist).
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **legacy\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether or not to save processor attributes in separate config files (legacy) or in processor’s config
  file as a nested dict. Saving all attributes in a single dict will become the default in future versions.
  Set to `legacy_serialization=True` until then.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it
can be reloaded using the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1419)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1428)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

## WhisperModel

### class transformers.WhisperModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/modeling_whisper.py#L982)

( config: WhisperConfig  )

Parameters

* **config** ([WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Whisper Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/modeling_whisper.py#L1050)

( input\_features: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None decoder\_inputs\_embeds: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_position\_ids: typing.Optional[tuple[torch.LongTensor]] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor). See [WhisperFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor.__call__) for details ([WhisperProcessor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperProcessor) uses
  [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor) for processing audios).
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [WhisperTokenizer](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  Whisper uses the `decoder_start_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read
  `modeling_whisper._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the BART
  paper](https://huggingface.co/papers/1910.13461) for more information on the default strategy.
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
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
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
* **decoder\_inputs\_embeds** (`tuple[torch.FloatTensor]` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **decoder\_position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
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

[transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig)) and inputs.

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

The [WhisperModel](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoFeatureExtractor, WhisperModel
>>> from datasets import load_dataset

>>> model = WhisperModel.from_pretrained("openai/whisper-base")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
>>> input_features = inputs.input_features
>>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
>>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
>>> list(last_hidden_state.shape)
[1, 2, 512]
```

#### \_mask\_input\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/modeling_whisper.py#L1007)

( input\_features: FloatTensor attention\_mask: typing.Optional[torch.LongTensor] = None  )

Masks extracted features along time axis and/or along feature axis according to
[SpecAugment](https://huggingface.co/papers/1904.08779).

## WhisperForConditionalGeneration

### class transformers.WhisperForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/modeling_whisper.py#L1178)

( config: WhisperConfig  )

Parameters

* **config** ([WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Whisper Model with a language modeling head. Can be used for automatic speech recognition.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/modeling_whisper.py#L1213)

( input\_features: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None decoder\_inputs\_embeds: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_position\_ids: typing.Optional[tuple[torch.LongTensor]] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor). See [WhisperFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor.__call__) for details ([WhisperProcessor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperProcessor) uses
  [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor) for processing audios).
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [WhisperTokenizer](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  Whisper uses the `decoder_start_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read
  `modeling_whisper._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the BART
  paper](https://huggingface.co/papers/1910.13461) for more information on the default strategy.
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
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
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
* **decoder\_inputs\_embeds** (`tuple[torch.FloatTensor]` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **decoder\_position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
  or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
  only computed for the tokens with labels in `[0, ..., config.vocab_size]`. `sequence_length` should be smaller than or equal to `config.max_target_positions`.
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

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig)) and inputs.

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

The [WhisperForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoProcessor, WhisperForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
>>> input_features = inputs.input_features

>>> generated_ids = model.generate(inputs=input_features)

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> transcription
' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/generation_whisper.py#L386)

( input\_features: typing.Optional[torch.Tensor] = None generation\_config: typing.Optional[transformers.generation.configuration\_utils.GenerationConfig] = None logits\_processor: typing.Optional[transformers.generation.logits\_process.LogitsProcessorList] = None stopping\_criteria: typing.Optional[transformers.generation.stopping\_criteria.StoppingCriteriaList] = None prefix\_allowed\_tokens\_fn: typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None synced\_gpus: bool = False return\_timestamps: typing.Optional[bool] = None task: typing.Optional[str] = None language: typing.Union[str, list[str], NoneType] = None is\_multilingual: typing.Optional[bool] = None prompt\_ids: typing.Optional[torch.Tensor] = None prompt\_condition\_type: typing.Optional[str] = None condition\_on\_prev\_tokens: typing.Optional[bool] = None temperature: typing.Union[float, tuple[float, ...], NoneType] = None compression\_ratio\_threshold: typing.Optional[float] = None logprob\_threshold: typing.Optional[float] = None no\_speech\_threshold: typing.Optional[float] = None num\_segment\_frames: typing.Optional[int] = None attention\_mask: typing.Optional[torch.Tensor] = None time\_precision: float = 0.02 time\_precision\_features: float = 0.01 return\_token\_timestamps: typing.Optional[bool] = None return\_segments: bool = False return\_dict\_in\_generate: typing.Optional[bool] = None force\_unique\_generate\_call: typing.Optional[bool] = None monitor\_progress: typing.Optional[typing.Callable[[torch.Tensor], NoneType]] = None \*\*kwargs  ) → [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) or `dict[str, Any]` or `torch.LongTensor`

Parameters

* **input\_features** (`torch.Tensor` of shape `(batch_size, feature_size, sequence_length)`, *optional*) —
  Float values of log-mel features extracted from the raw speech waveform. The raw speech waveform can be obtained by
  loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`,
  *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_features`, the [AutoFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoFeatureExtractor) should be used for extracting the mel
  features, padding and conversion into a tensor of type `torch.FloatTensor`.
  See [**call**()](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor.__call__) for details.
* **generation\_config** ([GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig), *optional*) —
  The generation configuration to be used as base parametrization for the generation call. `**kwargs`
  passed to generate matching the attributes of `generation_config` will override them. If
  `generation_config` is not provided, the default will be used, which had the following loading
  priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
  configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)’s
  default values, whose documentation should be checked to parameterize generation.
* **logits\_processor** (`LogitsProcessorList`, *optional*) —
  Custom logits processors that complement the default logits processors built from arguments and
  generation config. If a logit processor is passed that is already created with the arguments or a
  generation config an error is thrown. This feature is intended for advanced users.
* **stopping\_criteria** (`StoppingCriteriaList`, *optional*) —
  Custom stopping criteria that complement the default stopping criteria built from arguments and a
  generation config. If a stopping criteria is passed that is already created with the arguments or a
  generation config an error is thrown. This feature is intended for advanced users.
* **prefix\_allowed\_tokens\_fn** (`Callable[[int, torch.Tensor], list[int]]`, *optional*) —
  If provided, this function constraints the beam search to allowed tokens only at each step. If not
  provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
  `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
  on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
  for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
  Retrieval](https://huggingface.co/papers/2010.00904).
* **synced\_gpus** (`bool`, *optional*, defaults to `False`) —
  Whether to continue running the while loop until max\_length (needed to avoid deadlocking with
  `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
* **return\_timestamps** (`bool`, *optional*) —
  Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
  For audios longer than 30 seconds, it is necessary to set `return_timestamps=True`.
* **task** (`str`, *optional*) —
  Task to use for generation, either “translate” or “transcribe”.
* **language** (`str` or list of `str`, *optional*) —
  Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. For
  batched generation, a list of language tokens can be passed. You can find all the possible language
  tokens in the `model.generation_config.lang_to_id` dictionary.
* **is\_multilingual** (`bool`, *optional*) —
  Whether or not the model is multilingual.
* **prompt\_ids** (`torch.Tensor`, *optional*) —
  Rank-1 tensor of token IDs created by passing text to `get_prompt_ids()` that is
  provided as a prompt to each chunk. This can be used to provide or “prompt-engineer” a context for
  transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
  correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
* **prompt\_condition\_type** (`str`, *optional*) —
  Only relevant for long-form transcription. Condition type of `prompt_ids`. ‘first-segment’ means only the first segment is conditioned on `prompt_ids`. ‘all-segments’ means each segment is conditioned on `prompt_ids`. Make sure to enable `condition_on_prev_tokens` for ‘all-segments’.
  Defaults to ‘first-segment’. For short-term transcription only ‘first-segment’ is possible.
* **condition\_on\_prev\_tokens** (`bool`, *optional*) —
  Only relevant for long-form transcription. Whether to condition each segment on the previous segment.
  As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
  performance.
* **temperature** (`float` or list of `float`, *optional*) —
  The temperature to be used for generation. Passing a single `float` value and `do_sample=True` activates
  generation using sampling. For long-form transcription, temperature fallback can be activated by passing
  a list of float values such as (0.0, 0.2, 0.4, 0.6, 0.8, 1.0). As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
  performance.
* **compression\_ratio\_threshold** (`float`, *optional*) —
  Only relevant for long-form transcription. If defined, the zlib compression rate of each segment will be computed. If the compression rate of
  a segment is higher than `compression_ratio_threshold`, temperature fallback is activated: the generated segment is discarded and the generation is
  repeated using a higher temperature. The intuition behind this feature is that segments with very high compression rates
  suffer from a lot of repetition. The unwanted repetition can be reduced by injecting more randomness by increasing the temperature. If `compression_ratio_threshold` is defined
  make sure that `temperature` is a list of values. A common value for `compression_ratio_threshold` is 1.35.
  As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
  performance.
* **logprob\_threshold** (`float`, *optional*) —
  Only relevant for long-form transcription. If defined, the average log-probability of each segment will be computed. If the log-probability of
  a given segment is lower than `logprob_threshold`, temperature fallback is activated: the generated segment is discarded and the generation is
  repeated using a higher temperature. The intuition behind this feature is that segments of low log-probability
  can be improved by injecting more randomness by increasing the temperature. If `logprob_threshold` is defined
  make sure that `temperature` is a list of values. A common value for `logprob_threshold` is -1.0.
  As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
  performance.
* **no\_speech\_threshold** (`float`, *optional*) —
  Only relevant for long-form transcription. If defined, the “no-speech” token combined with the `logprob_threshold`
  is used to determine whether a segment contains only silence. In this case, the transcription for this segment
  is skipped.
  As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
  performance.
* **num\_segment\_frames** (`int`, *optional*) —
  The number of frames a single segment is made of. If not defined, `num_segment_frames` defaults to the model’s stride
  times the maximum input length.
* **attention\_mask** (`torch.Tensor`, *optional*) —
  `attention_mask` needs to be passed when doing long-form transcription using a batch size > 1.
* **time\_precision** (`int`, *optional*, defaults to 0.02) —
  The duration of output token in seconds. *E.g.* 0.02 means that a generated token on average accounts
  for 20 ms.
* **time\_precision\_features** (`int`, *optional*, defaults to 0.01) —
  The duration represented by a feature frame in seconds.
* **return\_token\_timestamps** (`bool`, *optional*) —
  Whether to return token-level timestamps with the text. This can be used with or without the
  `return_timestamps` option. To get word-level timestamps, use the tokenizer to group the tokens into
  words.
* **return\_segments** (`bool`, *optional*, defaults to `False`) —
  Whether to additionally return a list of all segments. Note that this option can only be enabled
  when doing long-form transcription.
* **return\_dict\_in\_generate** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of just returning the generated tokens.
  Note that when doing long-form transcription, `return_dict_in_generate` can only be enabled when
  `return_segments` is set True. In this case the generation outputs of each segment is added to each
  segment.
* **force\_unique\_generate\_call** (`bool`, *optional*) —
  Whether to force a unique call to the underlying GenerationMixin’s [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) method. This is useful for assisted decoding and testing purposes to ensure
  that only one call to [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) is made and therefore decoder input token ids and eos token ids are returned.
* **monitor\_progress** (`Callable[[torch.Tensor], None]`, *optional*) —
  If provided, this function can be called to report the progress of the audio transcription. The function
  takes a tensor argument `p` of shape `(n, 2)`, where `n` is the batch size. `p[i, 0]` contains the
  index of the audio frame that is currently being transcribed for batch item `i`. `p[i, 1]` contains
  the total number of frames for batch item `i`. No return value is expected.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
  forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
  specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder\_*.

Returns

[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) or `dict[str, Any]` or `torch.LongTensor`

One of the following:

* [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) when `return_dict_in_generate=True` and (`return_timestamps=False` or `force_unique_generate_call=True`), including the decoder input ids and end of sequence id.
* `dict[str, Any]` when (`return_dict_in_generate=True` and `return_timestamps=True`) or `return_segments=True` or `return_token_timestamps=True`.
* `torch.LongTensor` in all other cases, excluding the decoder input ids and end of sequence id.

The possible [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) types are:

* [GenerateEncoderDecoderOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput)
* [GenerateBeamEncoderDecoderOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput)

`segments` is a list of lists (one list per batch element) of `segment`.
A `segment` is a dictionary with keys `start`, `end`, `tokens`, `idxs`, and `result`.

* `start`: the start timestamp of the segment.
* `end`: the end timestamp of the segment.
* `tokens`: the tokens of the segment, excluding the decoder input ids and end of sequence id.
* `idxs`: the start (included) and end (excluded) indices of the `tokens` of the segment in the underlying call to GenerationMixin’s [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) (present in `result`).
* `result`: the result of the underlying call to GenerationMixin’s [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate).

When `return_timestamps=True`, `return_dict_in_generate=True` applies to each call of the underlying GenerationMixin’s [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate), with outputs stored in `result` of each `segment`.

Transcribes or translates log-mel input features to a sequence of auto-regressively generated token ids.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
model’s default generation configuration. You can override any `generation_config` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

For an overview of generation strategies and code examples, check out the [following
guide](./generation_strategies).

Example:

* *Longform transcription*: To transcribe or translate audios longer than 30 seconds, process the audio files without truncation and pass all mel features at once to generate. It is necessary to set `return_timestamps=True`.
  Indeed, long-form transcription uses a sequential algorithm based on timestamps predictions, with heuristics like compression ratio threshold, log probability threshold and temperature fallback. This algorithm is described in the [the Whisper original paper](https://cdn.openai.com/papers/whisper.pdf), section *3.8. Long-form Transcription*.


```
>>> import torch
>>> from transformers import AutoProcessor, WhisperForConditionalGeneration
>>> from datasets import load_dataset, Audio

>>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
>>> model.cuda()
>>> # load audios > 30 seconds
>>> ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
>>> # resample to 16kHz
>>> ds = ds.cast_column("audio", Audio(sampling_rate=16000))
>>> # take first 8 audios and retrieve array
>>> audio = ds[:8]["audio"]
>>> audio = [x["array"] for x in audio]

>>> # make sure to NOT truncate the input audio, to return the `attention_mask` and to pad to the longest audio
>>> inputs = processor(audio, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000)
>>> inputs = inputs.to("cuda", torch.float32)

>>> # transcribe audio to ids
>>> generated_ids = model.generate(**inputs, return_timestamps=True)

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> transcription[0]
" Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories developing the central headline pawns, definitely maneuvering an oso topical night to F6, fainting a classic Sicilian, nade door variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a fisher's shows in Lip Nitsky attack that culminates in the elegant lethal slow-played, all-passant checkmate that is my nightly monologue. But sometimes, sometimes, folks, I. CHEERING AND APPLAUSE Sometimes I startle away, cubside down in the monkey bars of a condemned playground on a super fun site. Get all hept up on goofballs. Rummage that were discarded tag bag of defective toys. Yank out a fist bowl of disembodied doll limbs, toss them on a stained kid's place mat from a defunct dennies. set up a table inside a rusty cargo container down by the Wharf and challenged toothless drifters to the godless bughouse blitz of tournament that is my segment. Meanwhile."
```

The `monitor_progress` callback can be used to monitor the progress of the transcription:


```
>>> from tqdm import tqdm

>>> # prepare inputs like above

>>> # define a callback to monitor the progress of the transcription.
>>> with tqdm(desc="Progress") as pbar:
>>>     def monitor_progress(p_batch):
>>>         i = torch.argmax(p_batch[:, 1])
>>>         p = p_batch[i].detach().cpu()
>>>         pbar.total = int(p[1])
>>>         pbar.n = int(p[0])
>>>         pbar.update()

>>>     # transcribe audio to ids
>>>     generated_ids = model.generate(**inputs, return_timestamps=True, monitor_progress=monitor_progress)

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> transcription[0]
Progress:  95%|█████████████████████████████████████████████████████████████████████████████████████████████████▎    | 8497/8901 [00:04<00:00, 2052.79it/s]
" Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories developing the central headline pawns, definitely maneuvering an oso topical night to F6, fainting a classic Sicilian, nade door variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a fisher's shows in Lip Nitsky attack that culminates in the elegant lethal slow-played, all-passant checkmate that is my nightly monologue. But sometimes, sometimes, folks, I. CHEERING AND APPLAUSE Sometimes I startle away, cubside down in the monkey bars of a condemned playground on a super fun site. Get all hept up on goofballs. Rummage that were discarded tag bag of defective toys. Yank out a fist bowl of disembodied doll limbs, toss them on a stained kid's place mat from a defunct dennies. set up a table inside a rusty cargo container down by the Wharf and challenged toothless drifters to the godless bughouse blitz of tournament that is my segment. Meanwhile."
```

* *Shortform transcription*: If passed mel input features are <= 30 seconds, there are two possibilities:
  + `return_timestamps=False`: the whole audio will be transcribed with a single call to GenerationMixin’s [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate).
  + `return_timestamps=True`: the audio will be transcribed using the same logic as long-form transcription.


```
>>> import torch
>>> from transformers import AutoProcessor, WhisperForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
>>> input_features = inputs.input_features

>>> generated_ids = model.generate(inputs=input_features)

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> transcription
' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

## WhisperForCausalLM

### class transformers.WhisperForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/modeling_whisper.py#L1371)

( config  )

Parameters

* **config** ([WhisperForCausalLM](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Whisper decoder with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/modeling_whisper.py#L1403)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[torch.FloatTensor]] = None head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

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
* **encoder\_outputs** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
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

Returns

[transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Cross attentions weights after the attention softmax, used to compute the weighted average in the
  cross-attention heads.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.

The [WhisperForCausalLM](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import WhisperForCausalLM, WhisperForConditionalGeneration, WhisperProcessor
>>> import torch
>>> from datasets import load_dataset

>>> processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

>>> assistant_model = WhisperForCausalLM.from_pretrained("distil-whisper/distil-large-v2")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> sample = ds[0]["audio"]
>>> input_features = processor(
...     sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
... ).input_features

>>> predicted_ids = model.generate(input_features, assistant_model=assistant_model)

>>> # decode token ids to text
>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
>>> transcription
' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.'
```

## WhisperForAudioClassification

### class transformers.WhisperForAudioClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/modeling_whisper.py#L1512)

( config  )

Parameters

* **config** ([WhisperForAudioClassification](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForAudioClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Whisper Encoder Model with a sequence classification head on top (a linear layer over the pooled output) for tasks
like SUPERB Keyword Spotting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/whisper/modeling_whisper.py#L1539)

( input\_features: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.LongTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor). See [WhisperFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor.__call__) for details ([WhisperProcessor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperProcessor) uses
  [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor) for processing audios).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([WhisperConfig](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [WhisperForAudioClassification](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperForAudioClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoFeatureExtractor, WhisperForAudioClassification
>>> from datasets import load_dataset

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
>>> model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

>>> ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
>>> sample = next(iter(ds))

>>> inputs = feature_extractor(
...     sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
... )
>>> input_features = inputs.input_features

>>> with torch.no_grad():
...     logits = model(input_features).logits

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'Afrikaans'
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/whisper.md)
