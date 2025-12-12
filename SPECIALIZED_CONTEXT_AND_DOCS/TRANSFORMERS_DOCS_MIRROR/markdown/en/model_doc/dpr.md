*This model was released on 2020-04-10 and added to Hugging Face Transformers on 2020-11-16.*

# DPR

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

Dense Passage Retrieval (DPR) is a set of tools and models for state-of-the-art open-domain Q&A research. It was
introduced in [Dense Passage Retrieval for Open-Domain Question Answering](https://huggingface.co/papers/2004.04906) by
Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih.

The abstract from the paper is the following:

*Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional
sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can
be practically implemented using dense representations alone, where embeddings are learned from a small number of
questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets,
our dense retriever outperforms a strong Lucene-BM25 system largely by 9%-19% absolute in terms of top-20 passage
retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA
benchmarks.*

This model was contributed by [lhoestq](https://huggingface.co/lhoestq). The original code can be found [here](https://github.com/facebookresearch/DPR).

## Usage tips

* DPR consists in three models:

  + Question encoder: encode questions as vectors
  + Context encoder: encode contexts as vectors
  + Reader: extract the answer of the questions inside retrieved contexts, along with a relevance score (high if the inferred span actually answers the question).

## DPRConfig

### class transformers.DPRConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/configuration_dpr.py#L24)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 position\_embedding\_type = 'absolute' projection\_dim: int = 0 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the DPR model. Defines the different tokens that can be represented by the *inputs\_ids*
  passed to the forward method of [BertModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the *token\_type\_ids* passed into [BertModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  Padding token id.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"absolute"`) —
  Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
  positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
  [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
  For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
  with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
* **projection\_dim** (`int`, *optional*, defaults to 0) —
  Dimension of the projection for the context and question encoders. If it is set to zero (default), then no
  projection is done.

[DPRConfig](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRConfig) is the configuration class to store the configuration of a *DPRModel*.

This is the configuration class to store the configuration of a [DPRContextEncoder](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRContextEncoder), [DPRQuestionEncoder](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoder), or a
[DPRReader](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRReader). It is used to instantiate the components of the DPR model according to the specified arguments,
defining the model component architectures. Instantiating a configuration with the defaults will yield a similar
configuration to that of the DPRContextEncoder
[facebook/dpr-ctx\_encoder-single-nq-base](https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base)
architecture.

This class is a subclass of [BertConfig](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig). Please check the superclass for the documentation of all kwargs.

Example:


```
>>> from transformers import DPRConfig, DPRContextEncoder

>>> # Initializing a DPR facebook/dpr-ctx_encoder-single-nq-base style configuration
>>> configuration = DPRConfig()

>>> # Initializing a model (with random weights) from the facebook/dpr-ctx_encoder-single-nq-base style configuration
>>> model = DPRContextEncoder(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## DPRContextEncoderTokenizer

### class transformers.DPRContextEncoderTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/tokenization_dpr.py#L30)

( vocab\_file do\_lower\_case = True do\_basic\_tokenize = True never\_split = None unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' tokenize\_chinese\_chars = True strip\_accents = None clean\_up\_tokenization\_spaces = True \*\*kwargs  )

Construct a DPRContextEncoder tokenizer.

[DPRContextEncoderTokenizer](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRContextEncoderTokenizer) is identical to [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) and runs end-to-end tokenization: punctuation
splitting and wordpiece.

Refer to superclass [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) for usage examples and documentation concerning parameters.

## DPRContextEncoderTokenizerFast

### class transformers.DPRContextEncoderTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/tokenization_dpr_fast.py#L31)

( vocab\_file = None tokenizer\_file = None do\_lower\_case = True unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' tokenize\_chinese\_chars = True strip\_accents = None \*\*kwargs  )

Construct a “fast” DPRContextEncoder tokenizer (backed by HuggingFace’s *tokenizers* library).

[DPRContextEncoderTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRContextEncoderTokenizerFast) is identical to [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) and runs end-to-end tokenization:
punctuation splitting and wordpiece.

Refer to superclass [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) for usage examples and documentation concerning parameters.

## DPRQuestionEncoderTokenizer

### class transformers.DPRQuestionEncoderTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/tokenization_dpr.py#L43)

( vocab\_file do\_lower\_case = True do\_basic\_tokenize = True never\_split = None unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' tokenize\_chinese\_chars = True strip\_accents = None clean\_up\_tokenization\_spaces = True \*\*kwargs  )

Constructs a DPRQuestionEncoder tokenizer.

[DPRQuestionEncoderTokenizer](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoderTokenizer) is identical to [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) and runs end-to-end tokenization: punctuation
splitting and wordpiece.

Refer to superclass [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) for usage examples and documentation concerning parameters.

## DPRQuestionEncoderTokenizerFast

### class transformers.DPRQuestionEncoderTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/tokenization_dpr_fast.py#L45)

( vocab\_file = None tokenizer\_file = None do\_lower\_case = True unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' tokenize\_chinese\_chars = True strip\_accents = None \*\*kwargs  )

Constructs a “fast” DPRQuestionEncoder tokenizer (backed by HuggingFace’s *tokenizers* library).

[DPRQuestionEncoderTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoderTokenizerFast) is identical to [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) and runs end-to-end tokenization:
punctuation splitting and wordpiece.

Refer to superclass [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) for usage examples and documentation concerning parameters.

## DPRReaderTokenizer

### class transformers.DPRReaderTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/tokenization_dpr.py#L306)

( vocab\_file do\_lower\_case = True do\_basic\_tokenize = True never\_split = None unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' tokenize\_chinese\_chars = True strip\_accents = None clean\_up\_tokenization\_spaces = True \*\*kwargs  ) → `dict[str, list[list[int]]]`

Parameters

* **questions** (`str` or `list[str]`) —
  The questions to be encoded. You can specify one question for many passages. In this case, the question
  will be duplicated like `[questions] * n_passages`. Otherwise you have to specify as many questions as in
  `titles` or `texts`.
* **titles** (`str` or `list[str]`) —
  The passages titles to be encoded. This can be a string or a list of strings if there are several passages.
* **texts** (`str` or `list[str]`) —
  The passages texts to be encoded. This can be a string or a list of strings if there are several passages.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
    if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to
    the maximum acceptable input length for the model if that argument is not provided. This will truncate
    token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch
    of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided. This will only truncate the first
    sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided. This will only truncate the
    second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether or not to return the attention mask. If not set, will return the attention mask according to the
  specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)

Returns

`dict[str, list[list[int]]]`

A dictionary with the following keys:

* `input_ids`: List of token ids to be fed to a model.
* `attention_mask`: List of indices specifying which tokens should be attended to by the model.

Construct a DPRReader tokenizer.

[DPRReaderTokenizer](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRReaderTokenizer) is almost identical to [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) and runs end-to-end tokenization: punctuation
splitting and wordpiece. The difference is that is has three inputs strings: question, titles and texts that are
combined to be fed to the [DPRReader](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRReader) model.

Refer to superclass [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) for usage examples and documentation concerning parameters.

Return a dictionary with the token ids of the input strings and other information to give to `.decode_best_spans`.
It converts the strings of a question and different passages (title and text) in a sequence of IDs (integers),
using the tokenizer and vocabulary. The resulting `input_ids` is a matrix of size `(n_passages, sequence_length)`

with the format:


```
[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>
```

## DPRReaderTokenizerFast

### class transformers.DPRReaderTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/tokenization_dpr_fast.py#L304)

( vocab\_file = None tokenizer\_file = None do\_lower\_case = True unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' tokenize\_chinese\_chars = True strip\_accents = None \*\*kwargs  ) → `dict[str, list[list[int]]]`

Parameters

* **questions** (`str` or `list[str]`) —
  The questions to be encoded. You can specify one question for many passages. In this case, the question
  will be duplicated like `[questions] * n_passages`. Otherwise you have to specify as many questions as in
  `titles` or `texts`.
* **titles** (`str` or `list[str]`) —
  The passages titles to be encoded. This can be a string or a list of strings if there are several passages.
* **texts** (`str` or `list[str]`) —
  The passages texts to be encoded. This can be a string or a list of strings if there are several passages.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
    if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to
    the maximum acceptable input length for the model if that argument is not provided. This will truncate
    token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch
    of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided. This will only truncate the first
    sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided. This will only truncate the
    second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether or not to return the attention mask. If not set, will return the attention mask according to the
  specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)

Returns

`dict[str, list[list[int]]]`

A dictionary with the following keys:

* `input_ids`: List of token ids to be fed to a model.
* `attention_mask`: List of indices specifying which tokens should be attended to by the model.

Constructs a “fast” DPRReader tokenizer (backed by HuggingFace’s *tokenizers* library).

[DPRReaderTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRReaderTokenizerFast) is almost identical to [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) and runs end-to-end tokenization:
punctuation splitting and wordpiece. The difference is that is has three inputs strings: question, titles and texts
that are combined to be fed to the [DPRReader](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRReader) model.

Refer to superclass [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) for usage examples and documentation concerning parameters.

Return a dictionary with the token ids of the input strings and other information to give to `.decode_best_spans`.
It converts the strings of a question and different passages (title and text) in a sequence of IDs (integers),
using the tokenizer and vocabulary. The resulting `input_ids` is a matrix of size `(n_passages, sequence_length)`
with the format:

[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>

## DPR specific outputs

### class transformers.models.dpr.modeling\_dpr.DPRContextEncoderOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/modeling_dpr.py#L48)

( pooler\_output: FloatTensor hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`) —
  The DPR encoder outputs the *pooler\_output* that corresponds to the context representation. Last layer
  hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
  This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Class for outputs of [DPRQuestionEncoder](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoder).

### class transformers.models.dpr.modeling\_dpr.DPRQuestionEncoderOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/modeling_dpr.py#L67)

( pooler\_output: FloatTensor hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`) —
  The DPR encoder outputs the *pooler\_output* that corresponds to the question representation. Last layer
  hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
  This output is to be used to embed questions for nearest neighbors queries with context embeddings.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Class for outputs of [DPRQuestionEncoder](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoder).

### class transformers.DPRReaderOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/modeling_dpr.py#L86)

( start\_logits: FloatTensor end\_logits: typing.Optional[torch.FloatTensor] = None relevance\_logits: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **start\_logits** (`torch.FloatTensor` of shape `(n_passages, sequence_length)`) —
  Logits of the start index of the span for each passage.
* **end\_logits** (`torch.FloatTensor` of shape `(n_passages, sequence_length)`) —
  Logits of the end index of the span for each passage.
* **relevance\_logits** (`torch.FloatTensor` of shape `(n_passages, )`) —
  Outputs of the QA classifier of the DPRReader that corresponds to the scores of each passage to answer the
  question, compared to all the other passages.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Class for outputs of [DPRQuestionEncoder](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoder).

## DPRContextEncoder

### class transformers.DPRContextEncoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/modeling_dpr.py#L286)

( config: DPRConfig  )

Parameters

* **config** ([DPRConfig](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DPRContextEncoder transformer outputting pooler outputs as context representations.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/modeling_dpr.py#L294)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.dpr.modeling\_dpr.DPRContextEncoderOutput](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.models.dpr.modeling_dpr.DPRContextEncoderOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. To match pretraining, DPR input sequence should be
  formatted with [CLS] and [SEP] tokens as follows:

  (a) For sequence pairs (for a pair title+text for example):

Returns

[transformers.models.dpr.modeling\_dpr.DPRContextEncoderOutput](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.models.dpr.modeling_dpr.DPRContextEncoderOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.dpr.modeling\_dpr.DPRContextEncoderOutput](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.models.dpr.modeling_dpr.DPRContextEncoderOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPRConfig](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRConfig)) and inputs.

* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`) — The DPR encoder outputs the *pooler\_output* that corresponds to the context representation. Last layer
  hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
  This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DPRContextEncoder](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRContextEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

>>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
>>> model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
>>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
>>> embeddings = model(input_ids).pooler_output
```

## DPRQuestionEncoder

### class transformers.DPRQuestionEncoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/modeling_dpr.py#L391)

( config: DPRConfig  )

Parameters

* **config** ([DPRConfig](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DPRQuestionEncoder transformer outputting pooler outputs as question representations.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/modeling_dpr.py#L399)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.dpr.modeling\_dpr.DPRQuestionEncoderOutput](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.models.dpr.modeling_dpr.DPRQuestionEncoderOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. To match pretraining, DPR input sequence should be
  formatted with [CLS] and [SEP] tokens as follows:

  (a) For sequence pairs (for a pair title+text for example):

Returns

[transformers.models.dpr.modeling\_dpr.DPRQuestionEncoderOutput](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.models.dpr.modeling_dpr.DPRQuestionEncoderOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.dpr.modeling\_dpr.DPRQuestionEncoderOutput](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.models.dpr.modeling_dpr.DPRQuestionEncoderOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPRConfig](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRConfig)) and inputs.

* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, embeddings_size)`) — The DPR encoder outputs the *pooler\_output* that corresponds to the question representation. Last layer
  hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
  This output is to be used to embed questions for nearest neighbors queries with context embeddings.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DPRQuestionEncoder](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRQuestionEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

>>> tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
>>> model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
>>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
>>> embeddings = model(input_ids).pooler_output
```

## DPRReader

### class transformers.DPRReader

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/modeling_dpr.py#L497)

( config: DPRConfig  )

Parameters

* **config** ([DPRConfig](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DPRReader transformer outputting span predictions.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpr/modeling_dpr.py#L505)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.dpr.modeling\_dpr.DPRReaderOutput](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRReaderOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`tuple[torch.LongTensor]` of shapes `(n_passages, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. It has to be a sequence triplet with 1) the question
  and 2) the passages titles and 3) the passages texts To match pretraining, DPR `input_ids` sequence should
  be formatted with [CLS] and [SEP] with the format:

  `[CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>`

  DPR is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right
  rather than the left.

  Indices can be obtained using [DPRReaderTokenizer](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRReaderTokenizer). See this class documentation for more details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(n_passages, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.dpr.modeling\_dpr.DPRReaderOutput](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRReaderOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.dpr.modeling\_dpr.DPRReaderOutput](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRReaderOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPRConfig](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRConfig)) and inputs.

* **start\_logits** (`torch.FloatTensor` of shape `(n_passages, sequence_length)`) — Logits of the start index of the span for each passage.
* **end\_logits** (`torch.FloatTensor` of shape `(n_passages, sequence_length)`) — Logits of the end index of the span for each passage.
* **relevance\_logits** (`torch.FloatTensor` of shape `(n_passages, )`) — Outputs of the QA classifier of the DPRReader that corresponds to the scores of each passage to answer the
  question, compared to all the other passages.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DPRReader](/docs/transformers/v4.56.2/en/model_doc/dpr#transformers.DPRReader) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import DPRReader, DPRReaderTokenizer

>>> tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
>>> model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
>>> encoded_inputs = tokenizer(
...     questions=["What is love ?"],
...     titles=["Haddaway"],
...     texts=["'What Is Love' is a song recorded by the artist Haddaway"],
...     return_tensors="pt",
... )
>>> outputs = model(**encoded_inputs)
>>> start_logits = outputs.start_logits
>>> end_logits = outputs.end_logits
>>> relevance_logits = outputs.relevance_logits
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dpr.md)
