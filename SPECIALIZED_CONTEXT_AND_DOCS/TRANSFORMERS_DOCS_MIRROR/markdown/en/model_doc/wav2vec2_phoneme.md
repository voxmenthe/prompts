*This model was released on 2021-09-23 and added to Hugging Face Transformers on 2021-12-17.*

# Wav2Vec2Phoneme

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Wav2Vec2Phoneme model was proposed in [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition (Xu et al.,
2021)](https://huggingface.co/papers/2109.11680) by Qiantong Xu, Alexei Baevski, Michael Auli.

The abstract from the paper is the following:

*Recent progress in self-training, self-supervised pretraining and unsupervised learning enabled well performing speech
recognition systems without any labeled data. However, in many cases there is labeled data available for related
languages which is not utilized by these methods. This paper extends previous work on zero-shot cross-lingual transfer
learning by fine-tuning a multilingually pretrained wav2vec 2.0 model to transcribe unseen languages. This is done by
mapping phonemes of the training languages to the target language using articulatory features. Experiments show that
this simple method significantly outperforms prior work which introduced task-specific architectures and used only part
of a monolingually pretrained model.*

Relevant checkpoints can be found under <https://huggingface.co/models?other=phoneme-recognition>.

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten)

The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec).

## Usage tips

* Wav2Vec2Phoneme uses the exact same architecture as Wav2Vec2
* Wav2Vec2Phoneme is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
* Wav2Vec2Phoneme model was trained using connectionist temporal classification (CTC) so the model output has to be
  decoded using [Wav2Vec2PhonemeCTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer).
* Wav2Vec2Phoneme can be fine-tuned on multiple language at once and decode unseen languages in a single forward pass
  to a sequence of phonemes
* By default, the model outputs a sequence of phonemes. In order to transform the phonemes to a sequence of words one
  should make use of a dictionary and language model.

Wav2Vec2Phoneme’s architecture is based on the Wav2Vec2 model, for API reference, check out [`Wav2Vec2`](wav2vec2)’s documentation page
except for the tokenizer.

## Wav2Vec2PhonemeCTCTokenizer

### class transformers.Wav2Vec2PhonemeCTCTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L80)

( vocab\_file bos\_token = '<s>' eos\_token = '</s>' unk\_token = '<unk>' pad\_token = '<pad>' phone\_delimiter\_token = ' ' word\_delimiter\_token = None do\_phonemize = True phonemizer\_lang = 'en-us' phonemizer\_backend = 'espeak' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  File containing the vocabulary.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sentence token.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sentence token.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **do\_phonemize** (`bool`, *optional*, defaults to `True`) —
  Whether the tokenizer should phonetize the input or not. Only if a sequence of phonemes is passed to the
  tokenizer, `do_phonemize` should be set to `False`.
* **phonemizer\_lang** (`str`, *optional*, defaults to `"en-us"`) —
  The language of the phoneme set to which the tokenizer should phonetize the input text to.
* **phonemizer\_backend** (`str`, *optional*. defaults to `"espeak"`) —
  The backend phonetization library that shall be used by the phonemizer library. Defaults to `espeak-ng`.
  See the [phonemizer package](https://github.com/bootphon/phonemizer#readme). for more information.
* \***\*kwargs** —
  Additional keyword arguments passed along to [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)

Constructs a Wav2Vec2PhonemeCTC tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains some of the main methods. Users should refer to
the superclass for more information regarding such methods.

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

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L510)

( sequences: typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None output\_char\_offsets: bool = False \*\*kwargs  ) → `list[str]` or `~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`

Parameters

* **sequences** (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces.
* **output\_char\_offsets** (`bool`, *optional*, defaults to `False`) —
  Whether or not to output character offsets. Character offsets can be used in combination with the
  sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

  Please take a look at the Example of `~models.wav2vec2.tokenization_wav2vec2.decode` to better
  understand how to make use of `output_word_offsets`.
  `~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode` works analogous with phonemes
  and batched output.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`list[str]` or `~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`

The
decoded sentence. Will be a
`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput` when
`output_char_offsets == True`.

Convert a list of lists of token ids into a list of strings by calling decode.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L454)

( token\_ids: typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None output\_char\_offsets: bool = False \*\*kwargs  ) → `str` or `~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`

Parameters

* **token\_ids** (`Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces.
* **output\_char\_offsets** (`bool`, *optional*, defaults to `False`) —
  Whether or not to output character offsets. Character offsets can be used in combination with the
  sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

  Please take a look at the Example of `~models.wav2vec2.tokenization_wav2vec2.decode` to better
  understand how to make use of `output_word_offsets`.
  `~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode` works the same way with
  phonemes.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`str` or `~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`

The decoded
sentence. Will be a `~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`
when `output_char_offsets == True`.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

#### phonemize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L252)

( text: str phonemizer\_lang: typing.Optional[str] = None  )

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/wav2vec2_phoneme.md)
