*This model was released on 2020-06-12 and added to Hugging Face Transformers on 2023-06-20.*

# RetriBERT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

This model is in maintenance mode only, so we won’t accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

## Overview

The [RetriBERT](https://huggingface.co/yjernite/retribert-base-uncased/tree/main) model was proposed in the blog post [Explain Anything Like I’m Five: A Model for Open Domain Long Form
Question Answering](https://yjernite.github.io/lfqa.html). RetriBERT is a small model that uses either a single or
pair of BERT encoders with lower-dimension projection for dense semantic indexing of text.

This model was contributed by [yjernite](https://huggingface.co/yjernite). Code to train and use the model can be
found [here](https://github.com/huggingface/transformers/tree/main/examples/research-projects/distillation).

## RetriBertConfig

### class transformers.RetriBertConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/retribert/configuration_retribert.py#L24)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 8 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 share\_encoders = True projection\_dim = 128 pad\_token\_id = 0 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the RetriBERT model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [RetriBertModel](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertModel)
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
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
* **share\_encoders** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use the same Bert-type encoder for the queries and document
* **projection\_dim** (`int`, *optional*, defaults to 128) —
  Final dimension of the query and document representation after projection

This is the configuration class to store the configuration of a [RetriBertModel](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertModel). It is used to instantiate a
RetriBertModel model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the RetriBERT
[yjernite/retribert-base-uncased](https://huggingface.co/yjernite/retribert-base-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## RetriBertTokenizer

### class transformers.RetriBertTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/retribert/tokenization_retribert.py#L51)

( vocab\_file do\_lower\_case = True do\_basic\_tokenize = True never\_split = None unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' tokenize\_chinese\_chars = True strip\_accents = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  File containing the vocabulary.
* **do\_lower\_case** (`bool`, *optional*, defaults to `True`) —
  Whether or not to lowercase the input when tokenizing.
* **do\_basic\_tokenize** (`bool`, *optional*, defaults to `True`) —
  Whether or not to do basic tokenization before WordPiece.
* **never\_split** (`Iterable`, *optional*) —
  Collection of tokens which will never be split during tokenization. Only has an effect when
  `do_basic_tokenize=True`
* **unk\_token** (`str`, *optional*, defaults to `"[UNK]"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **sep\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"[PAD]"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"[CLS]"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"[MASK]"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **tokenize\_chinese\_chars** (`bool`, *optional*, defaults to `True`) —
  Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see this
  [issue](https://github.com/huggingface/transformers/issues/328)).
* **strip\_accents** (`bool`, *optional*) —
  Whether or not to strip all accents. If this option is not specified, then it will be determined by the
  value for `lowercase` (as in the original BERT).

Constructs a RetriBERT tokenizer.

[RetriBertTokenizer](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertTokenizer) is identical to [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) and runs end-to-end tokenization: punctuation splitting
and wordpiece.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer
to: this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/retribert/tokenization_retribert.py#L183)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`List[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A BERT sequence has the following format:

* single sequence: `[CLS] X [SEP]`
* pair of sequences: `[CLS] A [SEP] B [SEP]`

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/retribert/tokenization_retribert.py#L178)

( tokens  )

Converts a sequence of tokens (string) in a single string.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/retribert/tokenization_retribert.py#L208)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

`List[int]`

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

## RetriBertTokenizerFast

### class transformers.RetriBertTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/retribert/tokenization_retribert_fast.py#L32)

( vocab\_file = None tokenizer\_file = None do\_lower\_case = True unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' tokenize\_chinese\_chars = True strip\_accents = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  File containing the vocabulary.
* **do\_lower\_case** (`bool`, *optional*, defaults to `True`) —
  Whether or not to lowercase the input when tokenizing.
* **unk\_token** (`str`, *optional*, defaults to `"[UNK]"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **sep\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"[PAD]"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"[CLS]"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"[MASK]"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **clean\_text** (`bool`, *optional*, defaults to `True`) —
  Whether or not to clean the text before tokenization by removing any control characters and replacing all
  whitespaces by the classic one.
* **tokenize\_chinese\_chars** (`bool`, *optional*, defaults to `True`) —
  Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
  issue](https://github.com/huggingface/transformers/issues/328)).
* **strip\_accents** (`bool`, *optional*) —
  Whether or not to strip all accents. If this option is not specified, then it will be determined by the
  value for `lowercase` (as in the original BERT).
* **wordpieces\_prefix** (`str`, *optional*, defaults to `"##"`) —
  The prefix for subwords.

Construct a “fast” RetriBERT tokenizer (backed by HuggingFace’s *tokenizers* library).

[RetriBertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertTokenizerFast) is identical to [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) and runs end-to-end tokenization: punctuation
splitting and wordpiece.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/retribert/tokenization_retribert_fast.py#L121)

( token\_ids\_0 token\_ids\_1 = None  ) → `List[int]`

Parameters

* **token\_ids\_0** (`List[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`List[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`List[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A BERT sequence has the following format:

* single sequence: `[CLS] X [SEP]`
* pair of sequences: `[CLS] A [SEP] B [SEP]`

## RetriBertModel

### class transformers.RetriBertModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/retribert/modeling_retribert.py#L82)

( config: RetriBertConfig  )

Parameters

* **config** ([RetriBertConfig](/docs/transformers/v4.56.2/en/model_doc/retribert#transformers.RetriBertConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bert Based model to embed queries or document for document retrieval.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/retribert/modeling_retribert.py#L170)

( input\_ids\_query: LongTensor attention\_mask\_query: typing.Optional[torch.FloatTensor] input\_ids\_doc: LongTensor attention\_mask\_doc: typing.Optional[torch.FloatTensor] checkpoint\_batch\_size: int = -1  ) → `torch.FloatTensor“

Parameters

* **input\_ids\_query** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary for the queries in a batch.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask\_query** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **input\_ids\_doc** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary for the documents in a batch.
* **attention\_mask\_doc** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on documents padding token indices.
* **checkpoint\_batch\_size** (`int`, *optional*, defaults to `-1`) —
  If greater than 0, uses gradient checkpointing to only compute sequence representation on
  `checkpoint_batch_size` examples at a time on the GPU. All query representations are still compared to
  all document representations in the batch.

Returns

`torch.FloatTensor“

The bidirectional cross-entropy loss obtained while trying to match each query to its
corresponding document and each document to its corresponding query in the batch

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/retribert.md)
