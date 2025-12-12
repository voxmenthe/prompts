# REALM

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

## Overview

The REALM model was proposed in [REALM: Retrieval-Augmented Language Model Pre-Training](https://huggingface.co/papers/2002.08909) by Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat and Ming-Wei Chang. It's a
retrieval-augmented language model that firstly retrieves documents from a textual knowledge corpus and then
utilizes retrieved documents to process question answering tasks.

The abstract from the paper is the following:

*Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks
such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network,
requiring ever-larger networks to cover more facts. To capture knowledge in a more modular and interpretable way, we
augment language model pre-training with a latent knowledge retriever, which allows the model to retrieve and attend
over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference. For the
first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language
modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents. We
demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the
challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both
explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous
methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as
interpretability and modularity.*

This model was contributed by [qqaatw](https://huggingface.co/qqaatw). The original code can be found
[here](https://github.com/google-research/language/tree/master/language/realm).

## RealmConfig[[transformers.RealmConfig]]

#### transformers.RealmConfig[[transformers.RealmConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/configuration_realm.py#L24)

This is the configuration class to store the configuration of

1. [RealmEmbedder](/docs/transformers/main/en/model_doc/realm#transformers.RealmEmbedder)
2. [RealmScorer](/docs/transformers/main/en/model_doc/realm#transformers.RealmScorer)
3. [RealmKnowledgeAugEncoder](/docs/transformers/main/en/model_doc/realm#transformers.RealmKnowledgeAugEncoder)
4. [RealmRetriever](/docs/transformers/main/en/model_doc/realm#transformers.RealmRetriever)
5. [RealmReader](/docs/transformers/main/en/model_doc/realm#transformers.RealmReader)
6. [RealmForOpenQA](/docs/transformers/main/en/model_doc/realm#transformers.RealmForOpenQA)

It is used to instantiate an REALM model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the REALM
[google/realm-cc-news-pretrained-embedder](https://huggingface.co/google/realm-cc-news-pretrained-embedder)
architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import RealmConfig, RealmEmbedder

>>> # Initializing a REALM realm-cc-news-pretrained-* style configuration
>>> configuration = RealmConfig()

>>> # Initializing a model (with random weights) from the google/realm-cc-news-pretrained-embedder style configuration
>>> model = RealmEmbedder(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 30522) : Vocabulary size of the REALM model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [RealmEmbedder](/docs/transformers/main/en/model_doc/realm#transformers.RealmEmbedder), [RealmScorer](/docs/transformers/main/en/model_doc/realm#transformers.RealmScorer), [RealmKnowledgeAugEncoder](/docs/transformers/main/en/model_doc/realm#transformers.RealmKnowledgeAugEncoder), or [RealmReader](/docs/transformers/main/en/model_doc/realm#transformers.RealmReader).

hidden_size (`int`, *optional*, defaults to 768) : Dimension of the encoder layers and the pooler layer.

retriever_proj_size (`int`, *optional*, defaults to 128) : Dimension of the retriever(embedder) projection.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

num_candidates (`int`, *optional*, defaults to 8) : Number of candidates inputted to the RealmScorer or RealmKnowledgeAugEncoder.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

type_vocab_size (`int`, *optional*, defaults to 2) : The vocabulary size of the `token_type_ids` passed when calling [RealmEmbedder](/docs/transformers/main/en/model_doc/realm#transformers.RealmEmbedder), [RealmScorer](/docs/transformers/main/en/model_doc/realm#transformers.RealmScorer), [RealmKnowledgeAugEncoder](/docs/transformers/main/en/model_doc/realm#transformers.RealmKnowledgeAugEncoder), or [RealmReader](/docs/transformers/main/en/model_doc/realm#transformers.RealmReader).

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

span_hidden_size (`int`, *optional*, defaults to 256) : Dimension of the reader's spans.

max_span_width (`int`, *optional*, defaults to 10) : Max span width of the reader.

reader_layer_norm_eps (`float`, *optional*, defaults to 1e-3) : The epsilon used by the reader's layer normalization layers.

reader_beam_size (`int`, *optional*, defaults to 5) : Beam size of the reader.

reader_seq_len (`int`, *optional*, defaults to 288+32) : Maximum sequence length of the reader.

num_block_records (`int`, *optional*, defaults to 13353718) : Number of block records.

searcher_beam_size (`int`, *optional*, defaults to 5000) : Beam size of the searcher. Note that when eval mode is enabled, *searcher_beam_size* will be the same as *reader_beam_size*.

## RealmTokenizer[[transformers.RealmTokenizer]]

#### transformers.RealmTokenizer[[transformers.RealmTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/tokenization_realm.py#L52)

Construct a REALM tokenizer.

[RealmTokenizer](/docs/transformers/main/en/model_doc/realm#transformers.RealmTokenizer) is identical to [BertTokenizer](/docs/transformers/main/en/model_doc/bert#transformers.BertTokenizer) and runs end-to-end tokenization: punctuation splitting and
wordpiece.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

build_inputs_with_special_tokenstransformers.RealmTokenizer.build_inputs_with_special_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/tokenization_realm.py#L254[{"name": "token_ids_0", "val": ": list"}, {"name": "token_ids_1", "val": ": typing.Optional[list[int]] = None"}]- **token_ids_0** (`List[int]`) --
  List of IDs to which the special tokens will be added.
- **token_ids_1** (`List[int]`, *optional*) --
  Optional second list of IDs for sequence pairs.0`List[int]`List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A REALM sequence has the following format:

- single sequence: `[CLS] X [SEP]`
- pair of sequences: `[CLS] A [SEP] B [SEP]`

**Parameters:**

vocab_file (`str`) : File containing the vocabulary.

do_lower_case (`bool`, *optional*, defaults to `True`) : Whether or not to lowercase the input when tokenizing.

do_basic_tokenize (`bool`, *optional*, defaults to `True`) : Whether or not to do basic tokenization before WordPiece.

never_split (`Iterable`, *optional*) : Collection of tokens which will never be split during tokenization. Only has an effect when `do_basic_tokenize=True`

unk_token (`str`, *optional*, defaults to `"[UNK]"`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

sep_token (`str`, *optional*, defaults to `"[SEP]"`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

pad_token (`str`, *optional*, defaults to `"[PAD]"`) : The token used for padding, for example when batching sequences of different lengths.

cls_token (`str`, *optional*, defaults to `"[CLS]"`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

mask_token (`str`, *optional*, defaults to `"[MASK]"`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

tokenize_chinese_chars (`bool`, *optional*, defaults to `True`) : Whether or not to tokenize Chinese characters.  This should likely be deactivated for Japanese (see this [issue](https://github.com/huggingface/transformers/issues/328)).

strip_accents (`bool`, *optional*) : Whether or not to strip all accents. If this option is not specified, then it will be determined by the value for `lowercase` (as in the original BERT).

**Returns:**

``List[int]``

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
#### get_special_tokens_mask[[transformers.RealmTokenizer.get_special_tokens_mask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/tokenization_realm.py#L279)

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

**Parameters:**

token_ids_0 (`List[int]`) : List of IDs.

token_ids_1 (`List[int]`, *optional*) : Optional second list of IDs for sequence pairs.

already_has_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the token list is already formatted with special tokens for the model.

**Returns:**

``List[int]``

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
#### create_token_type_ids_from_sequences[[transformers.RealmTokenizer.create_token_type_ids_from_sequences]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3483)

Create the token type IDs corresponding to the sequences passed. [What are token type
IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

**Parameters:**

token_ids_0 (`list[int]`) : The first tokenized sequence.

token_ids_1 (`list[int]`, *optional*) : The second tokenized sequence.

**Returns:**

``list[int]``

The token type ids.
#### save_vocabulary[[transformers.RealmTokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/tokenization_realm.py#L307)
#### batch_encode_candidates[[transformers.RealmTokenizer.batch_encode_candidates]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/tokenization_realm.py#L181)

Encode a batch of text or text pair. This method is similar to regular __call__ method but has the following
differences:

1. Handle additional num_candidate axis. (batch_size, num_candidates, text)
2. Always pad the sequences to *max_length*.
3. Must specify *max_length* in order to stack packs of candidates into a batch.

- single sequence: `[CLS] X [SEP]`
- pair of sequences: `[CLS] A [SEP] B [SEP]`

Example:

```python
>>> from transformers import RealmTokenizer

>>> # batch_size = 2, num_candidates = 2
>>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]

>>> tokenizer = RealmTokenizer.from_pretrained("google/realm-cc-news-pretrained-encoder")
>>> tokenized_text = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")
```

**Parameters:**

text (`List[List[str]]`) : The batch of sequences to be encoded. Each sequence must be in this format: (batch_size, num_candidates, text).

text_pair (`List[List[str]]`, *optional*) : The batch of sequences to be encoded. Each sequence must be in this format: (batch_size, num_candidates, text).

- ****kwargs** : Keyword arguments of the __call__ method.

**Returns:**

`[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)`

Encoded text or text pair.

## RealmTokenizerFast[[transformers.RealmTokenizerFast]]

#### transformers.RealmTokenizerFast[[transformers.RealmTokenizerFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/tokenization_realm_fast.py#L33)

Construct a "fast" REALM tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

[RealmTokenizerFast](/docs/transformers/main/en/model_doc/realm#transformers.RealmTokenizerFast) is identical to [BertTokenizerFast](/docs/transformers/main/en/model_doc/bert#transformers.BertTokenizerFast) and runs end-to-end tokenization: punctuation
splitting and wordpiece.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

batch_encode_candidatestransformers.RealmTokenizerFast.batch_encode_candidateshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/tokenization_realm_fast.py#L121[{"name": "text", "val": ""}, {"name": "**kwargs", "val": ""}]- **text** (`List[List[str]]`) --
  The batch of sequences to be encoded. Each sequence must be in this format: (batch_size,
  num_candidates, text).
- **text_pair** (`List[List[str]]`, *optional*) --
  The batch of sequences to be encoded. Each sequence must be in this format: (batch_size,
  num_candidates, text).
- ****kwargs** --
  Keyword arguments of the __call__ method.0[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)Encoded text or text pair.

Encode a batch of text or text pair. This method is similar to regular __call__ method but has the following
differences:

1. Handle additional num_candidate axis. (batch_size, num_candidates, text)
2. Always pad the sequences to *max_length*.
3. Must specify *max_length* in order to stack packs of candidates into a batch.

- single sequence: `[CLS] X [SEP]`
- pair of sequences: `[CLS] A [SEP] B [SEP]`

Example:

```python
>>> from transformers import RealmTokenizerFast

>>> # batch_size = 2, num_candidates = 2
>>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]

>>> tokenizer = RealmTokenizerFast.from_pretrained("google/realm-cc-news-pretrained-encoder")
>>> tokenized_text = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")
```

**Parameters:**

vocab_file (`str`) : File containing the vocabulary.

do_lower_case (`bool`, *optional*, defaults to `True`) : Whether or not to lowercase the input when tokenizing.

unk_token (`str`, *optional*, defaults to `"[UNK]"`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

sep_token (`str`, *optional*, defaults to `"[SEP]"`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

pad_token (`str`, *optional*, defaults to `"[PAD]"`) : The token used for padding, for example when batching sequences of different lengths.

cls_token (`str`, *optional*, defaults to `"[CLS]"`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

mask_token (`str`, *optional*, defaults to `"[MASK]"`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

clean_text (`bool`, *optional*, defaults to `True`) : Whether or not to clean the text before tokenization by removing any control characters and replacing all whitespaces by the classic one.

tokenize_chinese_chars (`bool`, *optional*, defaults to `True`) : Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this issue](https://github.com/huggingface/transformers/issues/328)).

strip_accents (`bool`, *optional*) : Whether or not to strip all accents. If this option is not specified, then it will be determined by the value for `lowercase` (as in the original BERT).

wordpieces_prefix (`str`, *optional*, defaults to `"##"`) : The prefix for subwords.

**Returns:**

`[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)`

Encoded text or text pair.

## RealmRetriever[[transformers.RealmRetriever]]

#### transformers.RealmRetriever[[transformers.RealmRetriever]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/retrieval_realm.py#L63)

The retriever of REALM outputting the retrieved evidence block and whether the block has answers as well as answer
positions."

block_has_answertransformers.RealmRetriever.block_has_answerhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/retrieval_realm.py#L128[{"name": "concat_inputs", "val": ""}, {"name": "answer_ids", "val": ""}]
check if retrieved_blocks has answers.

**Parameters:**

block_records (`np.ndarray`) : A numpy array which contains evidence texts.

tokenizer ([RealmTokenizer](/docs/transformers/main/en/model_doc/realm#transformers.RealmTokenizer)) : The tokenizer to encode retrieved texts.

## RealmEmbedder[[transformers.RealmEmbedder]]

#### transformers.RealmEmbedder[[transformers.RealmEmbedder]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L942)

The embedder of REALM outputting projected score that will be used to calculate relevance score.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.RealmEmbedder.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L961[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
  config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
  model's internal embedding lookup matrix.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.deprecated.realm.modeling_realm.RealmEmbedderOutput` or `tuple(torch.FloatTensor)`A `transformers.models.deprecated.realm.modeling_realm.RealmEmbedderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) and inputs.

- **projected_score** (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`) -- Projected score.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [RealmEmbedder](/docs/transformers/main/en/model_doc/realm#transformers.RealmEmbedder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, RealmEmbedder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-embedder")
>>> model = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> projected_score = outputs.projected_score
```

**Parameters:**

config ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.deprecated.realm.modeling_realm.RealmEmbedderOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.realm.modeling_realm.RealmEmbedderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) and inputs.

- **projected_score** (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`) -- Projected score.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## RealmScorer[[transformers.RealmScorer]]

#### transformers.RealmScorer[[transformers.RealmScorer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L1025)

The scorer of REALM outputting relevance scores representing the score of document candidates (before softmax).
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.RealmScorer.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L1041[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "candidate_input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "candidate_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "candidate_token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "candidate_inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
  config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
  model's internal embedding lookup matrix.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

- **candidate_input_ids** (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`) --
  Indices of candidate input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **candidate_attention_mask** (`torch.FloatTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **candidate_token_type_ids** (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **candidate_inputs_embeds** (`torch.FloatTensor` of shape `(batch_size * num_candidates, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `candidate_input_ids` you can choose to directly pass an embedded
  representation. This is useful if you want more control over how to convert *candidate_input_ids* indices
  into associated vectors than the model's internal embedding lookup matrix.0`transformers.models.deprecated.realm.modeling_realm.RealmScorerOutput` or `tuple(torch.FloatTensor)`A `transformers.models.deprecated.realm.modeling_realm.RealmScorerOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) and inputs.

- **relevance_score** (`torch.FloatTensor` of shape `(batch_size, config.num_candidates)`) -- The relevance score of document candidates (before softmax).
- **query_score** (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`) -- Query score derived from the query embedder.
- **candidate_score** (`torch.FloatTensor` of shape `(batch_size, config.num_candidates, config.retriever_proj_size)`) -- Candidate score derived from the embedder.
The [RealmScorer](/docs/transformers/main/en/model_doc/realm#transformers.RealmScorer) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> import torch
>>> from transformers import AutoTokenizer, RealmScorer

>>> tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-scorer")
>>> model = RealmScorer.from_pretrained("google/realm-cc-news-pretrained-scorer", num_candidates=2)

>>> # batch_size = 2, num_candidates = 2
>>> input_texts = ["How are you?", "What is the item in the picture?"]
>>> candidates_texts = [["Hello world!", "Nice to meet you!"], ["A cute cat.", "An adorable dog."]]

>>> inputs = tokenizer(input_texts, return_tensors="pt")
>>> candidates_inputs = tokenizer.batch_encode_candidates(candidates_texts, max_length=10, return_tensors="pt")

>>> outputs = model(
...     **inputs,
...     candidate_input_ids=candidates_inputs.input_ids,
...     candidate_attention_mask=candidates_inputs.attention_mask,
...     candidate_token_type_ids=candidates_inputs.token_type_ids,
... )
>>> relevance_score = outputs.relevance_score
```

**Parameters:**

config ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

query_embedder ([RealmEmbedder](/docs/transformers/main/en/model_doc/realm#transformers.RealmEmbedder)) : Embedder for input sequences. If not specified, it will use the same embedder as candidate sequences.

**Returns:**

``transformers.models.deprecated.realm.modeling_realm.RealmScorerOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.realm.modeling_realm.RealmScorerOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) and inputs.

- **relevance_score** (`torch.FloatTensor` of shape `(batch_size, config.num_candidates)`) -- The relevance score of document candidates (before softmax).
- **query_score** (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`) -- Query score derived from the query embedder.
- **candidate_score** (`torch.FloatTensor` of shape `(batch_size, config.num_candidates, config.retriever_proj_size)`) -- Candidate score derived from the embedder.

## RealmKnowledgeAugEncoder[[transformers.RealmKnowledgeAugEncoder]]

#### transformers.RealmKnowledgeAugEncoder[[transformers.RealmKnowledgeAugEncoder]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L1170)

The knowledge-augmented encoder of REALM outputting masked language model logits and marginal log-likelihood loss.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.RealmKnowledgeAugEncoder.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L1195[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "relevance_score", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "mlm_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
  config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, num_candidates, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
  model's internal embedding lookup matrix.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

- **relevance_score** (`torch.FloatTensor` of shape `(batch_size, num_candidates)`, *optional*) --
  Relevance score derived from RealmScorer, must be specified if you want to compute the masked language
  modeling loss.

- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
  config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

- **mlm_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid calculating joint loss on certain positions. If not specified, the loss will not be masked.
  Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.0[transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Masked language modeling (MLM) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [RealmKnowledgeAugEncoder](/docs/transformers/main/en/model_doc/realm#transformers.RealmKnowledgeAugEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> import torch
>>> from transformers import AutoTokenizer, RealmKnowledgeAugEncoder

>>> tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-encoder")
>>> model = RealmKnowledgeAugEncoder.from_pretrained(
...     "google/realm-cc-news-pretrained-encoder", num_candidates=2
... )

>>> # batch_size = 2, num_candidates = 2
>>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]

>>> inputs = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

**Parameters:**

config ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Masked language modeling (MLM) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## RealmReader[[transformers.RealmReader]]

#### transformers.RealmReader[[transformers.RealmReader]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L1321)

The reader of REALM.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.RealmReader.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L1332[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "relevance_score", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "block_mask", "val": ": typing.Optional[torch.BoolTensor] = None"}, {"name": "start_positions", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "end_positions", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "has_answers", "val": ": typing.Optional[torch.BoolTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **input_ids** (`torch.LongTensor` of shape `(reader_beam_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(reader_beam_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(reader_beam_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(reader_beam_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
  config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(reader_beam_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
  model's internal embedding lookup matrix.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

- **relevance_score** (`torch.FloatTensor` of shape `(searcher_beam_size,)`, *optional*) --
  Relevance score, which must be specified if you want to compute the logits and marginal log loss.
- **block_mask** (`torch.BoolTensor` of shape `(searcher_beam_size, sequence_length)`, *optional*) --
  The mask of the evidence block, which must be specified if you want to compute the logits and marginal log
  loss.
- **start_positions** (`torch.LongTensor` of shape `(searcher_beam_size,)`, *optional*) --
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **end_positions** (`torch.LongTensor` of shape `(searcher_beam_size,)`, *optional*) --
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **has_answers** (`torch.BoolTensor` of shape `(searcher_beam_size,)`, *optional*) --
  Whether or not the evidence block has answer(s).0`transformers.models.deprecated.realm.modeling_realm.RealmReaderOutput` or `tuple(torch.FloatTensor)`A `transformers.models.deprecated.realm.modeling_realm.RealmReaderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided) -- Total loss.
- **retriever_loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided) -- Retriever loss.
- **reader_loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided) -- Reader loss.
- **retriever_correct** (`torch.BoolTensor` of shape `(config.searcher_beam_size,)`, *optional*) -- Whether or not an evidence block contains answer.
- **reader_correct** (`torch.BoolTensor` of shape `(config.reader_beam_size, num_candidates)`, *optional*) -- Whether or not a span candidate contains answer.
- **block_idx** (`torch.LongTensor` of shape `()`) -- The index of the retrieved evidence block in which the predicted answer is most likely.
- **candidate** (`torch.LongTensor` of shape `()`) -- The index of the retrieved span candidates in which the predicted answer is most likely.
- **start_pos** (`torch.IntTensor` of shape `()`) -- Predicted answer starting position in *RealmReader*'s inputs.
- **end_pos** (`torch.IntTensor` of shape `()`) -- Predicted answer ending position in *RealmReader*'s inputs.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [RealmReader](/docs/transformers/main/en/model_doc/realm#transformers.RealmReader) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.deprecated.realm.modeling_realm.RealmReaderOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.realm.modeling_realm.RealmReaderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided) -- Total loss.
- **retriever_loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided) -- Retriever loss.
- **reader_loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided) -- Reader loss.
- **retriever_correct** (`torch.BoolTensor` of shape `(config.searcher_beam_size,)`, *optional*) -- Whether or not an evidence block contains answer.
- **reader_correct** (`torch.BoolTensor` of shape `(config.reader_beam_size, num_candidates)`, *optional*) -- Whether or not a span candidate contains answer.
- **block_idx** (`torch.LongTensor` of shape `()`) -- The index of the retrieved evidence block in which the predicted answer is most likely.
- **candidate** (`torch.LongTensor` of shape `()`) -- The index of the retrieved span candidates in which the predicted answer is most likely.
- **start_pos** (`torch.IntTensor` of shape `()`) -- Predicted answer starting position in *RealmReader*'s inputs.
- **end_pos** (`torch.IntTensor` of shape `()`) -- Predicted answer ending position in *RealmReader*'s inputs.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## RealmForOpenQA[[transformers.RealmForOpenQA]]

#### transformers.RealmForOpenQA[[transformers.RealmForOpenQA]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L1523)

`RealmForOpenQA` for end-to-end open domain question answering.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

block_embedding_totransformers.RealmForOpenQA.block_embedding_tohttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L1546[{"name": "device", "val": ""}]- **device** (`str` or `torch.device`) --
  The device to which `self.block_emb` will be sent.0
Send `self.block_emb` to a specific device.

**Parameters:**

config ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
#### forward[[transformers.RealmForOpenQA.forward]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/realm/modeling_realm.py#L1556)

The [RealmForOpenQA](/docs/transformers/main/en/model_doc/realm#transformers.RealmForOpenQA) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> import torch
>>> from transformers import RealmForOpenQA, RealmRetriever, AutoTokenizer

>>> retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa")
>>> tokenizer = AutoTokenizer.from_pretrained("google/realm-orqa-nq-openqa")
>>> model = RealmForOpenQA.from_pretrained("google/realm-orqa-nq-openqa", retriever=retriever)

>>> question = "Who is the pioneer in modern computer science?"
>>> question_ids = tokenizer([question], return_tensors="pt")
>>> answer_ids = tokenizer(
...     ["alan mathison turing"],
...     add_special_tokens=False,
...     return_token_type_ids=False,
...     return_attention_mask=False,
... ).input_ids

>>> reader_output, predicted_answer_ids = model(**question_ids, answer_ids=answer_ids, return_dict=False)
>>> predicted_answer = tokenizer.decode(predicted_answer_ids)
>>> loss = reader_output.loss
```

**Parameters:**

input_ids (`torch.LongTensor` of shape `(1, sequence_length)`) : Indices of input sequence tokens in the vocabulary.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.FloatTensor` of shape `(1, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.LongTensor` of shape `(1, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token (should not be used in this model by design).  [What are token type IDs?](../glossary#token-type-ids)

answer_ids (`list` of shape `(num_answers, answer_length)`, *optional*) : Answer ids for computing the marginal log-likelihood loss. Indices should be in `[-1, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-1` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

``transformers.models.deprecated.realm.modeling_realm.RealmForOpenQAOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.realm.modeling_realm.RealmForOpenQAOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/main/en/model_doc/realm#transformers.RealmConfig)) and inputs.

- **reader_output** (`dict`) -- Reader output.
- **predicted_answer_ids** (`torch.LongTensor` of shape `(answer_sequence_length)`) -- Predicted answer ids.
