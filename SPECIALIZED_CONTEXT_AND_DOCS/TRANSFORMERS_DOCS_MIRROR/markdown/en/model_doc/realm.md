*This model was released on 2020-02-10 and added to Hugging Face Transformers on 2023-06-20.*

# REALM

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

This model is in maintenance mode only, we don’t accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

## Overview

The REALM model was proposed in [REALM: Retrieval-Augmented Language Model Pre-Training](https://huggingface.co/papers/2002.08909) by Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat and Ming-Wei Chang. It’s a
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

## RealmConfig

### class transformers.RealmConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/configuration_realm.py#L24)

( vocab\_size = 30522 hidden\_size = 768 retriever\_proj\_size = 128 num\_hidden\_layers = 12 num\_attention\_heads = 12 num\_candidates = 8 intermediate\_size = 3072 hidden\_act = 'gelu\_new' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 span\_hidden\_size = 256 max\_span\_width = 10 reader\_layer\_norm\_eps = 0.001 reader\_beam\_size = 5 reader\_seq\_len = 320 num\_block\_records = 13353718 searcher\_beam\_size = 5000 pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the REALM model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [RealmEmbedder](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmEmbedder), [RealmScorer](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmScorer), [RealmKnowledgeAugEncoder](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmKnowledgeAugEncoder), or
  [RealmReader](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmReader).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimension of the encoder layers and the pooler layer.
* **retriever\_proj\_size** (`int`, *optional*, defaults to 128) —
  Dimension of the retriever(embedder) projection.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_candidates** (`int`, *optional*, defaults to 8) —
  Number of candidates inputted to the RealmScorer or RealmKnowledgeAugEncoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu_new"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the `token_type_ids` passed when calling [RealmEmbedder](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmEmbedder), [RealmScorer](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmScorer),
  [RealmKnowledgeAugEncoder](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmKnowledgeAugEncoder), or [RealmReader](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmReader).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **span\_hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimension of the reader’s spans.
* **max\_span\_width** (`int`, *optional*, defaults to 10) —
  Max span width of the reader.
* **reader\_layer\_norm\_eps** (`float`, *optional*, defaults to 1e-3) —
  The epsilon used by the reader’s layer normalization layers.
* **reader\_beam\_size** (`int`, *optional*, defaults to 5) —
  Beam size of the reader.
* **reader\_seq\_len** (`int`, *optional*, defaults to 288+32) —
  Maximum sequence length of the reader.
* **num\_block\_records** (`int`, *optional*, defaults to 13353718) —
  Number of block records.
* **searcher\_beam\_size** (`int`, *optional*, defaults to 5000) —
  Beam size of the searcher. Note that when eval mode is enabled, *searcher\_beam\_size* will be the same as
  *reader\_beam\_size*.

This is the configuration class to store the configuration of

1. [RealmEmbedder](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmEmbedder)
2. [RealmScorer](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmScorer)
3. [RealmKnowledgeAugEncoder](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmKnowledgeAugEncoder)
4. [RealmRetriever](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmRetriever)
5. [RealmReader](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmReader)
6. [RealmForOpenQA](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmForOpenQA)

It is used to instantiate an REALM model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the REALM
[google/realm-cc-news-pretrained-embedder](https://huggingface.co/google/realm-cc-news-pretrained-embedder)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import RealmConfig, RealmEmbedder

>>> # Initializing a REALM realm-cc-news-pretrained-* style configuration
>>> configuration = RealmConfig()

>>> # Initializing a model (with random weights) from the google/realm-cc-news-pretrained-embedder style configuration
>>> model = RealmEmbedder(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## RealmTokenizer

### class transformers.RealmTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/tokenization_realm.py#L52)

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
  Whether or not to tokenize Chinese characters.

  This should likely be deactivated for Japanese (see this
  [issue](https://github.com/huggingface/transformers/issues/328)).
* **strip\_accents** (`bool`, *optional*) —
  Whether or not to strip all accents. If this option is not specified, then it will be determined by the
  value for `lowercase` (as in the original BERT).

Construct a REALM tokenizer.

[RealmTokenizer](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmTokenizer) is identical to [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) and runs end-to-end tokenization: punctuation splitting and
wordpiece.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/tokenization_realm.py#L254)

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
adding special tokens. A REALM sequence has the following format:

* single sequence: `[CLS] X [SEP]`
* pair of sequences: `[CLS] A [SEP] B [SEP]`

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/tokenization_realm.py#L279)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/tokenization_realm.py#L307)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

#### batch\_encode\_candidates

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/tokenization_realm.py#L181)

( text \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`List[List[str]]`) —
  The batch of sequences to be encoded. Each sequence must be in this format: (batch\_size,
  num\_candidates, text).
* **text\_pair** (`List[List[str]]`, *optional*) —
  The batch of sequences to be encoded. Each sequence must be in this format: (batch\_size,
  num\_candidates, text).
* \***\*kwargs** —
  Keyword arguments of the **call** method.

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Encoded text or text pair.

Encode a batch of text or text pair. This method is similar to regular **call** method but has the following
differences:

1. Handle additional num\_candidate axis. (batch\_size, num\_candidates, text)
2. Always pad the sequences to *max\_length*.
3. Must specify *max\_length* in order to stack packs of candidates into a batch.

* single sequence: `[CLS] X [SEP]`
* pair of sequences: `[CLS] A [SEP] B [SEP]`

Example:


```
>>> from transformers import RealmTokenizer

>>> # batch_size = 2, num_candidates = 2
>>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]

>>> tokenizer = RealmTokenizer.from_pretrained("google/realm-cc-news-pretrained-encoder")
>>> tokenized_text = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")
```

## RealmTokenizerFast

### class transformers.RealmTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/tokenization_realm_fast.py#L33)

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

Construct a “fast” REALM tokenizer (backed by HuggingFace’s *tokenizers* library). Based on WordPiece.

[RealmTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmTokenizerFast) is identical to [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) and runs end-to-end tokenization: punctuation
splitting and wordpiece.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### batch\_encode\_candidates

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/tokenization_realm_fast.py#L121)

( text \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`List[List[str]]`) —
  The batch of sequences to be encoded. Each sequence must be in this format: (batch\_size,
  num\_candidates, text).
* **text\_pair** (`List[List[str]]`, *optional*) —
  The batch of sequences to be encoded. Each sequence must be in this format: (batch\_size,
  num\_candidates, text).
* \***\*kwargs** —
  Keyword arguments of the **call** method.

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Encoded text or text pair.

Encode a batch of text or text pair. This method is similar to regular **call** method but has the following
differences:

1. Handle additional num\_candidate axis. (batch\_size, num\_candidates, text)
2. Always pad the sequences to *max\_length*.
3. Must specify *max\_length* in order to stack packs of candidates into a batch.

* single sequence: `[CLS] X [SEP]`
* pair of sequences: `[CLS] A [SEP] B [SEP]`

Example:


```
>>> from transformers import RealmTokenizerFast

>>> # batch_size = 2, num_candidates = 2
>>> text = [["Hello world!", "Nice to meet you!"], ["The cute cat.", "The adorable dog."]]

>>> tokenizer = RealmTokenizerFast.from_pretrained("google/realm-cc-news-pretrained-encoder")
>>> tokenized_text = tokenizer.batch_encode_candidates(text, max_length=10, return_tensors="pt")
```

## RealmRetriever

### class transformers.RealmRetriever

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/retrieval_realm.py#L73)

( block\_records tokenizer  )

Parameters

* **block\_records** (`np.ndarray`) —
  A numpy array which contains evidence texts.
* **tokenizer** ([RealmTokenizer](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmTokenizer)) —
  The tokenizer to encode retrieved texts.

The retriever of REALM outputting the retrieved evidence block and whether the block has answers as well as answer
positions.”

#### block\_has\_answer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/retrieval_realm.py#L138)

( concat\_inputs answer\_ids  )

check if retrieved\_blocks has answers.

## RealmEmbedder

### class transformers.RealmEmbedder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1127)

( config  )

Parameters

* **config** ([RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The embedder of REALM outputting projected score that will be used to calculate relevance score.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1143)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deprecated.realm.modeling_realm.RealmEmbedderOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert *input\_ids* indices into associated vectors than the
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

`transformers.models.deprecated.realm.modeling_realm.RealmEmbedderOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deprecated.realm.modeling_realm.RealmEmbedderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig)) and inputs.

* **projected\_score** (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`) — Projected score.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [RealmEmbedder](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmEmbedder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, RealmEmbedder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-embedder")
>>> model = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> projected_score = outputs.projected_score
```

## RealmScorer

### class transformers.RealmScorer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1209)

( config query\_embedder = None  )

Parameters

* **config** ([RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **query\_embedder** ([RealmEmbedder](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmEmbedder)) —
  Embedder for input sequences. If not specified, it will use the same embedder as candidate sequences.

The scorer of REALM outputting relevance scores representing the score of document candidates (before softmax).
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1225)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None candidate\_input\_ids: typing.Optional[torch.LongTensor] = None candidate\_attention\_mask: typing.Optional[torch.FloatTensor] = None candidate\_token\_type\_ids: typing.Optional[torch.LongTensor] = None candidate\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deprecated.realm.modeling_realm.RealmScorerOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert *input\_ids* indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **candidate\_input\_ids** (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`) —
  Indices of candidate input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **candidate\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **candidate\_token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **candidate\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size * num_candidates, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `candidate_input_ids` you can choose to directly pass an embedded
  representation. This is useful if you want more control over how to convert *candidate\_input\_ids* indices
  into associated vectors than the model’s internal embedding lookup matrix.

Returns

`transformers.models.deprecated.realm.modeling_realm.RealmScorerOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deprecated.realm.modeling_realm.RealmScorerOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig)) and inputs.

* **relevance\_score** (`torch.FloatTensor` of shape `(batch_size, config.num_candidates)`) — The relevance score of document candidates (before softmax).
* **query\_score** (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`) — Query score derived from the query embedder.
* **candidate\_score** (`torch.FloatTensor` of shape `(batch_size, config.num_candidates, config.retriever_proj_size)`) — Candidate score derived from the embedder.

The [RealmScorer](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmScorer) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

## RealmKnowledgeAugEncoder

### class transformers.RealmKnowledgeAugEncoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1357)

( config  )

Parameters

* **config** ([RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The knowledge-augmented encoder of REALM outputting masked language model logits and marginal log-likelihood loss.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1379)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None relevance\_score: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None mlm\_mask: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, num_candidates, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_candidates, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert *input\_ids* indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **relevance\_score** (`torch.FloatTensor` of shape `(batch_size, num_candidates)`, *optional*) —
  Relevance score derived from RealmScorer, must be specified if you want to compute the masked language
  modeling loss.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **mlm\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid calculating joint loss on certain positions. If not specified, the loss will not be masked.
  Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

Returns

[transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [RealmKnowledgeAugEncoder](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmKnowledgeAugEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

## RealmReader

### class transformers.RealmReader

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1507)

( config  )

Parameters

* **config** ([RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The reader of REALM.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1518)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None relevance\_score: typing.Optional[torch.FloatTensor] = None block\_mask: typing.Optional[torch.BoolTensor] = None start\_positions: typing.Optional[torch.LongTensor] = None end\_positions: typing.Optional[torch.LongTensor] = None has\_answers: typing.Optional[torch.BoolTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deprecated.realm.modeling_realm.RealmReaderOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(reader_beam_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(reader_beam_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(reader_beam_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(reader_beam_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(reader_beam_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert *input\_ids* indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **relevance\_score** (`torch.FloatTensor` of shape `(searcher_beam_size,)`, *optional*) —
  Relevance score, which must be specified if you want to compute the logits and marginal log loss.
* **block\_mask** (`torch.BoolTensor` of shape `(searcher_beam_size, sequence_length)`, *optional*) —
  The mask of the evidence block, which must be specified if you want to compute the logits and marginal log
  loss.
* **start\_positions** (`torch.LongTensor` of shape `(searcher_beam_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.LongTensor` of shape `(searcher_beam_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **has\_answers** (`torch.BoolTensor` of shape `(searcher_beam_size,)`, *optional*) —
  Whether or not the evidence block has answer(s).

Returns

`transformers.models.deprecated.realm.modeling_realm.RealmReaderOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deprecated.realm.modeling_realm.RealmReaderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided) — Total loss.
* **retriever\_loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided) — Retriever loss.
* **reader\_loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided) — Reader loss.
* **retriever\_correct** (`torch.BoolTensor` of shape `(config.searcher_beam_size,)`, *optional*) — Whether or not an evidence block contains answer.
* **reader\_correct** (`torch.BoolTensor` of shape `(config.reader_beam_size, num_candidates)`, *optional*) — Whether or not a span candidate contains answer.
* **block\_idx** (`torch.LongTensor` of shape `()`) — The index of the retrieved evidence block in which the predicted answer is most likely.
* **candidate** (`torch.LongTensor` of shape `()`) — The index of the retrieved span candidates in which the predicted answer is most likely.
* **start\_pos** (`torch.IntTensor` of shape `()`) — Predicted answer starting position in *RealmReader*’s inputs.
* **end\_pos** (`torch.IntTensor` of shape `()`) — Predicted answer ending position in *RealmReader*’s inputs.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [RealmReader](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmReader) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## RealmForOpenQA

### class transformers.RealmForOpenQA

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1711)

( config retriever = None  )

Parameters

* **config** ([RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

`RealmForOpenQA` for end-to-end open domain question answering.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

#### block\_embedding\_to

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1734)

( device  )

Parameters

* **device** (`str` or `torch.device`) —
  The device to which `self.block_emb` will be sent.

Send `self.block_emb` to a specific device.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/realm/modeling_realm.py#L1744)

( input\_ids: typing.Optional[torch.LongTensor] attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None answer\_ids: typing.Optional[torch.LongTensor] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deprecated.realm.modeling_realm.RealmForOpenQAOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(1, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(1, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(1, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token (should not be used in this model by design).

  [What are token type IDs?](../glossary#token-type-ids)
* **answer\_ids** (`list` of shape `(num_answers, answer_length)`, *optional*) —
  Answer ids for computing the marginal log-likelihood loss. Indices should be in `[-1, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-1` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.deprecated.realm.modeling_realm.RealmForOpenQAOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deprecated.realm.modeling_realm.RealmForOpenQAOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RealmConfig](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmConfig)) and inputs.

* **reader\_output** (`dict`) — Reader output.
* **predicted\_answer\_ids** (`torch.LongTensor` of shape `(answer_sequence_length)`) — Predicted answer ids.

The [RealmForOpenQA](/docs/transformers/v4.56.2/en/model_doc/realm#transformers.RealmForOpenQA) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/realm.md)
