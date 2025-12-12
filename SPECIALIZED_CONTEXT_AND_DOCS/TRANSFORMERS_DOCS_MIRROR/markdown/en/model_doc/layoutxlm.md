*This model was released on 2021-04-18 and added to Hugging Face Transformers on 2021-11-03.*

# LayoutXLM

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

LayoutXLM was proposed in [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://huggingface.co/papers/2104.08836) by Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha
Zhang, Furu Wei. It’s a multilingual extension of the [LayoutLMv2 model](https://huggingface.co/papers/2012.14740) trained
on 53 languages.

The abstract from the paper is the following:

*Multimodal pre-training with text, layout, and image has achieved SOTA performance for visually-rich document
understanding tasks recently, which demonstrates the great potential for joint learning across different modalities. In
this paper, we present LayoutXLM, a multimodal pre-trained model for multilingual document understanding, which aims to
bridge the language barriers for visually-rich document understanding. To accurately evaluate LayoutXLM, we also
introduce a multilingual form understanding benchmark dataset named XFUN, which includes form understanding samples in
7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese), and key-value pairs are manually labeled
for each language. Experiment results show that the LayoutXLM model has significantly outperformed the existing SOTA
cross-lingual pre-trained models on the XFUN dataset.*

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/microsoft/unilm).

## Usage tips and examples

One can directly plug in the weights of LayoutXLM into a LayoutLMv2 model, like so:


```
from transformers import LayoutLMv2Model

model = LayoutLMv2Model.from_pretrained("microsoft/layoutxlm-base")
```

Note that LayoutXLM has its own tokenizer, based on
[LayoutXLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer)/[LayoutXLMTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizerFast). You can initialize it as
follows:


```
from transformers import LayoutXLMTokenizer

tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
```

Similar to LayoutLMv2, you can use [LayoutXLMProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMProcessor) (which internally applies
[LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor) and
[LayoutXLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer)/[LayoutXLMTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizerFast) in sequence) to prepare all
data for the model.

As LayoutXLM’s architecture is equivalent to that of LayoutLMv2, one can refer to [LayoutLMv2’s documentation page](layoutlmv2) for all tips, code examples and notebooks.

## LayoutXLMTokenizer

### class transformers.LayoutXLMTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L148)

( vocab\_file bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' cls\_token\_box = [0, 0, 0, 0] sep\_token\_box = [1000, 1000, 1000, 1000] pad\_token\_box = [0, 0, 0, 0] pad\_token\_label = -100 only\_label\_first\_subword = True sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **cls\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **mask\_token** (`str`, *optional*, defaults to `"<mask>"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **cls\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [CLS] token.
* **sep\_token\_box** (`list[int]`, *optional*, defaults to `[1000, 1000, 1000, 1000]`) —
  The bounding box to use for the special [SEP] token.
* **pad\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [PAD] token.
* **pad\_token\_label** (`int`, *optional*, defaults to -100) —
  The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch’s
  CrossEntropyLoss.
* **only\_label\_first\_subword** (`bool`, *optional*, defaults to `True`) —
  Whether or not to only label the first subword, in case word labels are provided.
* **sp\_model\_kwargs** (`dict`, *optional*) —
  Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
  SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
  to set:
  + `enable_sampling`: Enable subword regularization.
  + `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

    - `nbest_size = {0,1}`: No sampling is performed.
    - `nbest_size > 1`: samples from the nbest\_size results.
    - `nbest_size < 0`: assuming that nbest\_size is infinite and samples from the all hypothesis (lattice)
      using forward-filtering-and-backward-sampling algorithm.
  + `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
    BPE-dropout.
* **sp\_model** (`SentencePieceProcessor`) —
  The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).

Adapted from [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) and [XLNetTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer). Based on
[SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L439)

( text: typing.Union[str, list[str], list[list[str]]] text\_pair: typing.Union[list[str], list[list[str]], NoneType] = None boxes: typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None word\_labels: typing.Union[list[int], list[list[int]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
  (words of a single example or questions of a batch of examples) or a list of list of strings (batch of
  words).
* **text\_pair** (`list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
  (pretokenized string).
* **boxes** (`list[list[int]]`, `list[list[list[int]]]`) —
  Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.
* **word\_labels** (`list[int]`, `list[list[int]]`, *optional*) —
  Word-level integer labels (for token classification tasks such as FUNSD, CORD).
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to encode the sequences with the special tokens relative to their model.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
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
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
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
* **bbox** — List of bounding boxes to be fed to a model.
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)
* **labels** — List of labels to be fed to a model. (when `word_labels` is specified).
* **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
* **length** — The length of the inputs (when `return_length=True`).

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L311)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLM-RoBERTa sequence has the following format:

* single sequence: `<s> X </s>`
* pair of sequences: `<s> A </s></s> B </s>`

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L337)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L365)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
not make use of token type ids, therefore a list of zeros is returned.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L422)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## LayoutXLMTokenizerFast

### class transformers.LayoutXLMTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm_fast.py#L149)

( vocab\_file = None tokenizer\_file = None bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' cls\_token\_box = [0, 0, 0, 0] sep\_token\_box = [1000, 1000, 1000, 1000] pad\_token\_box = [0, 0, 0, 0] pad\_token\_label = -100 only\_label\_first\_subword = True \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **cls\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **mask\_token** (`str`, *optional*, defaults to `"<mask>"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **cls\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [CLS] token.
* **sep\_token\_box** (`list[int]`, *optional*, defaults to `[1000, 1000, 1000, 1000]`) —
  The bounding box to use for the special [SEP] token.
* **pad\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [PAD] token.
* **pad\_token\_label** (`int`, *optional*, defaults to -100) —
  The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch’s
  CrossEntropyLoss.
* **only\_label\_first\_subword** (`bool`, *optional*, defaults to `True`) —
  Whether or not to only label the first subword, in case word labels are provided.
* **additional\_special\_tokens** (`list[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`) —
  Additional special tokens used by the tokenizer.

Construct a “fast” LayoutXLM tokenizer (backed by HuggingFace’s *tokenizers* library). Adapted from
[RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) and [XLNetTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer). Based on
[BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm_fast.py#L263)

( text: typing.Union[str, list[str], list[list[str]]] text\_pair: typing.Union[list[str], list[list[str]], NoneType] = None boxes: typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None word\_labels: typing.Union[list[int], list[list[int]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
  (words of a single example or questions of a batch of examples) or a list of list of strings (batch of
  words).
* **text\_pair** (`list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
  (pretokenized string).
* **boxes** (`list[list[int]]`, `list[list[list[int]]]`) —
  Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.
* **word\_labels** (`list[int]`, `list[list[int]]`, *optional*) —
  Word-level integer labels (for token classification tasks such as FUNSD, CORD).
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to encode the sequences with the special tokens relative to their model.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
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
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
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
* **bbox** — List of bounding boxes to be fed to a model.
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)
* **labels** — List of labels to be fed to a model. (when `word_labels` is specified).
* **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
* **length** — The length of the inputs (when `return_length=True`).

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.

## LayoutXLMProcessor

### class transformers.LayoutXLMProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/processing_layoutxlm.py#L27)

( image\_processor = None tokenizer = None \*\*kwargs  )

Parameters

* **image\_processor** (`LayoutLMv2ImageProcessor`, *optional*) —
  An instance of [LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor). The image processor is a required input.
* **tokenizer** (`LayoutXLMTokenizer` or `LayoutXLMTokenizerFast`, *optional*) —
  An instance of [LayoutXLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer) or [LayoutXLMTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizerFast). The tokenizer is a required input.

Constructs a LayoutXLM processor which combines a LayoutXLM image processor and a LayoutXLM tokenizer into a single
processor.

[LayoutXLMProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMProcessor) offers all the functionalities you need to prepare data for the model.

It first uses [LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor) to resize document images to a fixed size, and optionally applies OCR to
get words and normalized bounding boxes. These are then provided to [LayoutXLMTokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer) or
[LayoutXLMTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizerFast), which turns the words and bounding boxes into token-level `input_ids`,
`attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
into token-level `labels` for token classification tasks (such as FUNSD, CORD).

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/processing_layoutxlm.py#L68)

( images text: typing.Union[str, list[str], list[list[str]]] = None text\_pair: typing.Union[list[str], list[list[str]], NoneType] = None boxes: typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None word\_labels: typing.Union[list[int], list[list[int]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None \*\*kwargs  )

This method first forwards the `images` argument to `~LayoutLMv2ImagePrpcessor.__call__`. In case
`LayoutLMv2ImagePrpcessor` was initialized with `apply_ocr` set to `True`, it passes the obtained words and
bounding boxes along with the additional arguments to [**call**()](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer.__call__) and returns the output,
together with resized `images`. In case `LayoutLMv2ImagePrpcessor` was initialized with `apply_ocr` set to
`False`, it passes the words (`text`/`` text_pair`) and `boxes` specified by the user along with the additional arguments to [__call__()](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer.__call__) and returns the output, together with resized `images ``.

Please refer to the docstring of the above two methods for more information.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/layoutxlm.md)
