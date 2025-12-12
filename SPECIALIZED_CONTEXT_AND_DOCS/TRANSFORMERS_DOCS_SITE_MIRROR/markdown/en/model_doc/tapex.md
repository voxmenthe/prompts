# TAPEX

This model is in maintenance mode only, we don't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

## Overview

The TAPEX model was proposed in [TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://huggingface.co/papers/2107.07653) by Qian Liu,
Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, Jian-Guang Lou. TAPEX pre-trains a BART model to solve synthetic SQL queries, after
which it can be fine-tuned to answer natural language questions related to tabular data, as well as performing table fact checking.

TAPEX has been fine-tuned on several datasets:

- [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253) (Sequential Question Answering by Microsoft)
- [WTQ](https://github.com/ppasupat/WikiTableQuestions) (Wiki Table Questions by Stanford University)
- [WikiSQL](https://github.com/salesforce/WikiSQL) (by Salesforce)
- [TabFact](https://tabfact.github.io/) (by USCB NLP Lab).

The abstract from the paper is the following:

*Recent progress in language model pre-training has achieved a great success via leveraging large-scale unstructured textual data. However, it is
still a challenge to apply pre-training on structured tabular data due to the absence of large-scale high-quality tabular data. In this paper, we
propose TAPEX to show that table pre-training can be achieved by learning a neural SQL executor over a synthetic corpus, which is obtained by automatically
synthesizing executable SQL queries and their execution outputs. TAPEX addresses the data scarcity challenge via guiding the language model to mimic a SQL
executor on the diverse, large-scale and high-quality synthetic corpus. We evaluate TAPEX on four benchmark datasets. Experimental results demonstrate that
TAPEX outperforms previous table pre-training approaches by a large margin and achieves new state-of-the-art results on all of them. This includes improvements
on the weakly-supervised WikiSQL denotation accuracy to 89.5% (+2.3%), the WikiTableQuestions denotation accuracy to 57.5% (+4.8%), the SQA denotation accuracy
to 74.5% (+3.5%), and the TabFact accuracy to 84.2% (+3.2%). To our knowledge, this is the first work to exploit table pre-training via synthetic executable programs
and to achieve new state-of-the-art results on various downstream tasks.*

## Usage tips

- TAPEX is a generative (seq2seq) model. One can directly plug in the weights of TAPEX into a BART model.
- TAPEX has checkpoints on the hub that are either pre-trained only, or fine-tuned on WTQ, SQA, WikiSQL and TabFact.
- Sentences + tables are presented to the model as `sentence + " " + linearized table`. The linearized table has the following format:
  `col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...`.
- TAPEX has its own tokenizer, that allows to prepare all data for the model easily. One can pass Pandas DataFrames and strings to the tokenizer,
  and it will automatically create the `input_ids` and `attention_mask` (as shown in the usage examples below).

### Usage: inference

Below, we illustrate how to use TAPEX for table question answering. As one can see, one can directly plug in the weights of TAPEX into a BART model.
We use the [Auto API](auto), which will automatically instantiate the appropriate tokenizer ([TapexTokenizer](/docs/transformers/main/en/model_doc/tapex#transformers.TapexTokenizer)) and model ([BartForConditionalGeneration](/docs/transformers/main/en/model_doc/bart#transformers.BartForConditionalGeneration)) for us,
based on the configuration file of the checkpoint on the hub.

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large-finetuned-wtq")

>>> # prepare table + question
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> question = "how many movies does Leonardo Di Caprio have?"

>>> encoding = tokenizer(table, question, return_tensors="pt")

>>> # let the model generate an answer autoregressively
>>> outputs = model.generate(**encoding)

>>> # decode back to text
>>> predicted_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
>>> print(predicted_answer)
53
```

Note that [TapexTokenizer](/docs/transformers/main/en/model_doc/tapex#transformers.TapexTokenizer) also supports batched inference. Hence, one can provide a batch of different tables/questions, or a batch of a single table
and multiple questions, or a batch of a single query and multiple tables. Let's illustrate this:

```python
>>> # prepare table + question
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> questions = [
...     "how many movies does Leonardo Di Caprio have?",
...     "which actor has 69 movies?",
...     "what's the first name of the actor who has 87 movies?",
... ]
>>> encoding = tokenizer(table, questions, padding=True, return_tensors="pt")

>>> # let the model generate an answer autoregressively
>>> outputs = model.generate(**encoding)

>>> # decode back to text
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
[' 53', ' george clooney', ' brad pitt']
```

In case one wants to do table verification (i.e. the task of determining whether a given sentence is supported or refuted by the contents
of a table), one can instantiate a [BartForSequenceClassification](/docs/transformers/main/en/model_doc/bart#transformers.BartForSequenceClassification) model. TAPEX has checkpoints on the hub fine-tuned on TabFact, an important
benchmark for table fact checking (it achieves 84% accuracy). The code example below again leverages the [Auto API](auto).

```python
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-tabfact")
>>> model = AutoModelForSequenceClassification.from_pretrained("microsoft/tapex-large-finetuned-tabfact")

>>> # prepare table + sentence
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> sentence = "George Clooney has 30 movies"

>>> encoding = tokenizer(table, sentence, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**encoding)

>>> # print prediction
>>> predicted_class_idx = outputs.logits[0].argmax(dim=0).item()
>>> print(model.config.id2label[predicted_class_idx])
Refused
```

TAPEX architecture is the same as BART, except for tokenization. Refer to [BART documentation](bart) for information on
configuration classes and their parameters. TAPEX-specific tokenizer is documented below.

## TapexTokenizer[[transformers.TapexTokenizer]]

#### transformers.TapexTokenizer[[transformers.TapexTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tapex/tokenization_tapex.py#L176)

Construct a TAPEX tokenizer. Based on byte-level Byte-Pair-Encoding (BPE).

This tokenizer can be used to flatten one or more table(s) and concatenate them with one or more related sentences
to be used by TAPEX models. The format that the TAPEX tokenizer creates is the following:

sentence col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...

The tokenizer supports a single table + single query, a single table and multiple queries (in which case the table
will be duplicated for every query), a single query and multiple tables (in which case the query will be duplicated
for every table), and multiple tables and queries. In other words, you can provide a batch of tables + questions to
the tokenizer for instance to prepare them for the model.

Tokenization itself is based on the BPE algorithm. It is identical to the one used by BART, RoBERTa and GPT-2.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

__call__transformers.TapexTokenizer.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tapex/tokenization_tapex.py#L494[{"name": "table", "val": ": typing.Union[ForwardRef('pd.DataFrame'), list['pd.DataFrame']] = None"}, {"name": "query", "val": ": typing.Union[str, list[str], NoneType] = None"}, {"name": "answer", "val": ": typing.Union[str, list[str], NoneType] = None"}, {"name": "add_special_tokens", "val": ": bool = True"}, {"name": "padding", "val": ": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"}, {"name": "truncation", "val": ": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"}, {"name": "max_length", "val": ": typing.Optional[int] = None"}, {"name": "stride", "val": ": int = 0"}, {"name": "pad_to_multiple_of", "val": ": typing.Optional[int] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "return_token_type_ids", "val": ": typing.Optional[bool] = None"}, {"name": "return_attention_mask", "val": ": typing.Optional[bool] = None"}, {"name": "return_overflowing_tokens", "val": ": bool = False"}, {"name": "return_special_tokens_mask", "val": ": bool = False"}, {"name": "return_offsets_mapping", "val": ": bool = False"}, {"name": "return_length", "val": ": bool = False"}, {"name": "verbose", "val": ": bool = True"}, {"name": "**kwargs", "val": ""}]- **table** (`pd.DataFrame`, `list[pd.DataFrame]`) --
  Table(s) containing tabular data.
- **query** (`str` or `list[str]`, *optional*) --
  Sentence or batch of sentences related to one or more table(s) to be encoded. Note that the number of
  sentences must match the number of tables.
- **answer** (`str` or `list[str]`, *optional*) --
  Optionally, the corresponding answer to the questions as supervision.

- **add_special_tokens** (`bool`, *optional*, defaults to `True`) --
  Whether or not to add special tokens when encoding the sequences. This will use the underlying
  `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
  automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
  automatically.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Activates and controls padding. Accepts the following values:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence is provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) --
  Activates and controls truncation. Accepts the following values:

  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
- **max_length** (`int`, *optional*) --
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
- **stride** (`int`, *optional*, defaults to 0) --
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
- **is_split_into_words** (`bool`, *optional*, defaults to `False`) --
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
- **pad_to_multiple_of** (`int`, *optional*) --
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
- **padding_side** (`str`, *optional*) --
  The side on which the model should have padding applied. Should be selected between ['right', 'left'].
  Default value is picked from the class attribute of the same name.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors instead of list of python integers. Acceptable values are:

  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.

- **add_special_tokens** (`bool`, *optional*, defaults to `True`) --
  Whether or not to encode the sequences with the special tokens relative to their model.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Activates and controls padding. Accepts the following values:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **truncation** (`bool`, `str`, `TapexTruncationStrategy` or [TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), --
  *optional*, defaults to `False`):

  Activates and controls truncation. Accepts the following values:

  - `'drop_rows_to_fit'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will truncate
    row by row, removing rows from the table.
  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
- **max_length** (`int`, *optional*) --
  Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to
  `None`, this will use the predefined model maximum length if a maximum length is required by one of the
  truncation/padding parameters. If the model has no specific maximum input length (like XLNet)
  truncation/padding to a maximum length will be deactivated.
- **stride** (`int`, *optional*, defaults to 0) --
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
- **pad_to_multiple_of** (`int`, *optional*) --
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors instead of list of python integers. Acceptable values are:

  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.0

Main method to tokenize and prepare for the model one or several table-sequence pair(s).

**Parameters:**

vocab_file (`str`) : Path to the vocabulary file.

merges_file (`str`) : Path to the merges file.

do_lower_case (`bool`, *optional*, defaults to `True`) : Whether or not to lowercase the input when tokenizing.

errors (`str`, *optional*, defaults to `"replace"`) : Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

add_prefix_space (`bool`, *optional*, defaults to `False`) : Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (BART tokenizer detect beginning of words by the preceding space).

max_cell_length (`int`, *optional*, defaults to 15) : Maximum number of characters per cell when linearizing a table. If this number is exceeded, truncation takes place.
#### save_vocabulary[[transformers.TapexTokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/tapex/tokenization_tapex.py#L465)
