*This model was released on 2021-10-15 and added to Hugging Face Transformers on 2021-12-07.*

# mLUKE

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The mLUKE model was proposed in [mLUKE: The Power of Entity Representations in Multilingual Pretrained Language Models](https://huggingface.co/papers/2110.08151) by Ryokan Ri, Ikuya Yamada, and Yoshimasa Tsuruoka. It’s a multilingual extension
of the [LUKE model](https://huggingface.co/papers/2010.01057) trained on the basis of XLM-RoBERTa.

It is based on XLM-RoBERTa and adds entity embeddings, which helps improve performance on various downstream tasks
involving reasoning about entities such as named entity recognition, extractive question answering, relation
classification, cloze-style knowledge completion.

The abstract from the paper is the following:

*Recent studies have shown that multilingual pretrained language models can be effectively improved with cross-lingual
alignment information from Wikipedia entities. However, existing methods only exploit entity information in pretraining
and do not explicitly use entities in downstream tasks. In this study, we explore the effectiveness of leveraging
entity representations for downstream cross-lingual tasks. We train a multilingual language model with 24 languages
with entity representations and show the model consistently outperforms word-based pretrained models in various
cross-lingual transfer tasks. We also analyze the model and the key insight is that incorporating entity
representations into the input allows us to extract more language-agnostic features. We also evaluate the model with a
multilingual cloze prompt task with the mLAMA dataset. We show that entity-based prompt elicits correct factual
knowledge more likely than using only word representations.*

This model was contributed by [ryo0634](https://huggingface.co/ryo0634). The original code can be found [here](https://github.com/studio-ousia/luke).

## Usage tips

One can directly plug in the weights of mLUKE into a LUKE model, like so:


```
from transformers import LukeModel

model = LukeModel.from_pretrained("studio-ousia/mluke-base")
```

Note that mLUKE has its own tokenizer, [MLukeTokenizer](/docs/transformers/v4.56.2/en/model_doc/mluke#transformers.MLukeTokenizer). You can initialize it as follows:


```
from transformers import MLukeTokenizer

tokenizer = MLukeTokenizer.from_pretrained("studio-ousia/mluke-base")
```

As mLUKE’s architecture is equivalent to that of LUKE, one can refer to [LUKE’s documentation page](luke) for all
tips, code examples and notebooks.

## MLukeTokenizer

### class transformers.MLukeTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mluke/tokenization_mluke.py#L133)

( vocab\_file entity\_vocab\_file bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' task = None max\_entity\_length = 32 max\_mention\_length = 30 entity\_token\_1 = '<ent>' entity\_token\_2 = '<ent2>' entity\_unk\_token = '[UNK]' entity\_pad\_token = '[PAD]' entity\_mask\_token = '[MASK]' entity\_mask2\_token = '[MASK2]' sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **entity\_vocab\_file** (`str`) —
  Path to the entity vocabulary file.
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
* **task** (`str`, *optional*) —
  Task for which you want to prepare sequences. One of `"entity_classification"`,
  `"entity_pair_classification"`, or `"entity_span_classification"`. If you specify this argument, the entity
  sequence is automatically created based on the given entity span(s).
* **max\_entity\_length** (`int`, *optional*, defaults to 32) —
  The maximum length of `entity_ids`.
* **max\_mention\_length** (`int`, *optional*, defaults to 30) —
  The maximum number of tokens inside an entity span.
* **entity\_token\_1** (`str`, *optional*, defaults to `<ent>`) —
  The special token used to represent an entity span in a word token sequence. This token is only used when
  `task` is set to `"entity_classification"` or `"entity_pair_classification"`.
* **entity\_token\_2** (`str`, *optional*, defaults to `<ent2>`) —
  The special token used to represent an entity span in a word token sequence. This token is only used when
  `task` is set to `"entity_pair_classification"`.
* **additional\_special\_tokens** (`list[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`) —
  Additional special tokens used by the tokenizer.
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

Adapted from [XLMRobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) and [LukeTokenizer](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeTokenizer). Based on
[SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mluke/tokenization_mluke.py#L386)

( text: typing.Union[str, list[str]] text\_pair: typing.Union[str, list[str], NoneType] = None entity\_spans: typing.Union[list[tuple[int, int]], list[list[tuple[int, int]]], NoneType] = None entity\_spans\_pair: typing.Union[list[tuple[int, int]], list[list[tuple[int, int]]], NoneType] = None entities: typing.Union[list[str], list[list[str]], NoneType] = None entities\_pair: typing.Union[list[str], list[list[str]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None max\_entity\_length: typing.Optional[int] = None stride: int = 0 is\_split\_into\_words: typing.Optional[bool] = False pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
  tokenizer does not support tokenization based on pretokenized strings.
* **text\_pair** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
  tokenizer does not support tokenization based on pretokenized strings.
* **entity\_spans** (`list[tuple[int, int]]`, `list[list[tuple[int, int]]]`, *optional*) —
  The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
  with two integers denoting character-based start and end positions of entities. If you specify
  `"entity_classification"` or `"entity_pair_classification"` as the `task` argument in the constructor,
  the length of each sequence must be 1 or 2, respectively. If you specify `entities`, the length of each
  sequence must be equal to the length of each sequence of `entities`.
* **entity\_spans\_pair** (`list[tuple[int, int]]`, `list[list[tuple[int, int]]]`, *optional*) —
  The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
  with two integers denoting character-based start and end positions of entities. If you specify the
  `task` argument in the constructor, this argument is ignored. If you specify `entities_pair`, the
  length of each sequence must be equal to the length of each sequence of `entities_pair`.
* **entities** (`list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
  representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
  Angeles). This argument is ignored if you specify the `task` argument in the constructor. The length of
  each sequence must be equal to the length of each sequence of `entity_spans`. If you specify
  `entity_spans` without specifying this argument, the entity sequence or the batch of entity sequences
  is automatically constructed by filling it with the [MASK] entity.
* **entities\_pair** (`list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
  representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
  Angeles). This argument is ignored if you specify the `task` argument in the constructor. The length of
  each sequence must be equal to the length of each sequence of `entity_spans_pair`. If you specify
  `entity_spans_pair` without specifying this argument, the entity sequence or the batch of entity
  sequences is automatically constructed by filling it with the [MASK] entity.
* **max\_entity\_length** (`int`, *optional*) —
  The maximum length of `entity_ids`.
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
* **entity\_ids** — List of entity ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)
* **entity\_position\_ids** — List of entity positions in the input sequence to be fed to a model.
* **entity\_token\_type\_ids** — List of entity token type ids to be fed to a model (when
  `return_token_type_ids=True` or if *“entity\_token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
* **entity\_attention\_mask** — List of indices specifying which entities should be attended to by the model
  (when `return_attention_mask=True` or if *“entity\_attention\_mask”* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)
* **entity\_start\_positions** — List of the start positions of entities in the word token sequence (when
  `task="entity_span_classification"`).
* **entity\_end\_positions** — List of the end positions of entities in the word token sequence (when
  `task="entity_span_classification"`).
* **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
* **length** — The length of the inputs (when `return_length=True`)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences, depending on the task you want to prepare them for.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mluke/tokenization_mluke.py#L1533)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mluke.md)
