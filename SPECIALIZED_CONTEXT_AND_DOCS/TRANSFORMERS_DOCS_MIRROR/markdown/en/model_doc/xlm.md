*This model was released on 2019-01-22 and added to Hugging Face Transformers on 2020-11-16.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# XLM

[XLM](https://huggingface.co/papers/1901.07291) demonstrates cross-lingual pretraining with two approaches, unsupervised training on a single language and supervised training on more than one language with a cross-lingual language model objective. The XLM model supports the causal language modeling objective, masked language modeling, and translation language modeling (an extension of the [BERT](./bert)) masked language modeling objective to multiple language inputs).

You can find all the original XLM checkpoints under the [Facebook AI community](https://huggingface.co/FacebookAI?search_models=xlm-mlm) organization.

Click on the XLM models in the right sidebar for more examples of how to apply XLM to different cross-lingual tasks like classification, translation, and question answering.

The example below demonstrates how to predict the `<mask>` token with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="facebook/xlm-roberta-xl",
    dtype=torch.float16,
    device=0
)
pipeline("Bonjour, je suis un modèle <mask>.")
```

## XLMConfig

### class transformers.XLMConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/configuration_xlm.py#L28)

( vocab\_size = 30145 emb\_dim = 2048 n\_layers = 12 n\_heads = 16 dropout = 0.1 attention\_dropout = 0.1 gelu\_activation = True sinusoidal\_embeddings = False causal = False asm = False n\_langs = 1 use\_lang\_emb = True max\_position\_embeddings = 512 embed\_init\_std = 0.02209708691207961 layer\_norm\_eps = 1e-12 init\_std = 0.02 bos\_index = 0 eos\_index = 1 pad\_index = 2 unk\_index = 3 mask\_index = 5 is\_encoder = True summary\_type = 'first' summary\_use\_proj = True summary\_activation = None summary\_proj\_to\_labels = True summary\_first\_dropout = 0.1 start\_n\_top = 5 end\_n\_top = 5 mask\_token\_id = 0 lang\_id = 0 pad\_token\_id = 2 bos\_token\_id = 0 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30145) —
  Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [XLMModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMModel) or `TFXLMModel`.
* **emb\_dim** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the encoder layers and the pooler layer.
* **n\_layer** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **n\_head** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for the attention mechanism
* **gelu\_activation** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use *gelu* for the activations instead of *relu*.
* **sinusoidal\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.
* **causal** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should behave in a causal manner. Causal models use a triangular attention mask in
  order to only attend to the left-side context instead if a bidirectional context.
* **asm** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the prediction
  layer.
* **n\_langs** (`int`, *optional*, defaults to 1) —
  The number of languages the model handles. Set to 1 for monolingual models.
* **use\_lang\_emb** (`bool`, *optional*, defaults to `True`) —
  Whether to use language embeddings. Some models use additional language embeddings, see [the multilingual
  models page](http://huggingface.co/transformers/multilingual.html#xlm-language-embeddings) for information
  on how to use them.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **embed\_init\_std** (`float`, *optional*, defaults to 2048^-0.5) —
  The standard deviation of the truncated\_normal\_initializer for initializing the embedding matrices.
* **init\_std** (`int`, *optional*, defaults to 50257) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices except the
  embedding matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **bos\_index** (`int`, *optional*, defaults to 0) —
  The index of the beginning of sentence token in the vocabulary.
* **eos\_index** (`int`, *optional*, defaults to 1) —
  The index of the end of sentence token in the vocabulary.
* **pad\_index** (`int`, *optional*, defaults to 2) —
  The index of the padding token in the vocabulary.
* **unk\_index** (`int`, *optional*, defaults to 3) —
  The index of the unknown token in the vocabulary.
* **mask\_index** (`int`, *optional*, defaults to 5) —
  The index of the masking token in the vocabulary.
* **is\_encoder(`bool`,** *optional*, defaults to `True`) —
  Whether or not the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.
* **summary\_type** (`string`, *optional*, defaults to “first”) —
  Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

  Has to be one of the following options:

  + `"last"`: Take the last token hidden state (like XLNet).
  + `"first"`: Take the first token hidden state (like BERT).
  + `"mean"`: Take the mean of all tokens hidden states.
  + `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
  + `"attn"`: Not implemented now, use multi-head attention.
* **summary\_use\_proj** (`bool`, *optional*, defaults to `True`) —
  Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

  Whether or not to add a projection after the vector extraction.
* **summary\_activation** (`str`, *optional*) —
  Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

  Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
* **summary\_proj\_to\_labels** (`bool`, *optional*, defaults to `True`) —
  Used in the sequence classification and multiple choice models.

  Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
* **summary\_first\_dropout** (`float`, *optional*, defaults to 0.1) —
  Used in the sequence classification and multiple choice models.

  The dropout ratio to be used after the projection and activation.
* **start\_n\_top** (`int`, *optional*, defaults to 5) —
  Used in the SQuAD evaluation script.
* **end\_n\_top** (`int`, *optional*, defaults to 5) —
  Used in the SQuAD evaluation script.
* **mask\_token\_id** (`int`, *optional*, defaults to 0) —
  Model agnostic parameter to identify masked tokens when generating text in an MLM context.
* **lang\_id** (`int`, *optional*, defaults to 1) —
  The ID of the language used by the model. This parameter is used when generating text in a given language.

This is the configuration class to store the configuration of a [XLMModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMModel) or a `TFXLMModel`. It is used to
instantiate a XLM model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the
[FacebookAI/xlm-mlm-en-2048](https://huggingface.co/FacebookAI/xlm-mlm-en-2048) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import XLMConfig, XLMModel

>>> # Initializing a XLM configuration
>>> configuration = XLMConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = XLMModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## XLMTokenizer

### class transformers.XLMTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/tokenization_xlm.py#L136)

( vocab\_file merges\_file unk\_token = '<unk>' bos\_token = '<s>' sep\_token = '</s>' pad\_token = '<pad>' cls\_token = '</s>' mask\_token = '<special1>' additional\_special\_tokens = ['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>'] lang2id = None id2lang = None do\_lowercase\_and\_remove\_accent = True \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Vocabulary file.
* **merges\_file** (`str`) —
  Merges file.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"<special1>"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **additional\_special\_tokens** (`List[str]`, *optional*, defaults to `['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>']`) —
  List of additional special tokens.
* **lang2id** (`Dict[str, int]`, *optional*) —
  Dictionary mapping languages string identifiers to their IDs.
* **id2lang** (`Dict[int, str]`, *optional*) —
  Dictionary mapping language IDs to their string identifiers.
* **do\_lowercase\_and\_remove\_accent** (`bool`, *optional*, defaults to `True`) —
  Whether to lowercase and remove accents when tokenizing.

Construct an XLM tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

* Moses preprocessing and tokenization for most supported languages.
* Language specific tokenization for Chinese (Jieba), Japanese (KyTea) and Thai (PyThaiNLP).
* Optionally lowercases and normalizes all inputs text.
* The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like
  ”**classify**”) to a vocabulary.
* The `lang2id` attribute maps the languages supported by the model with their IDs if provided (automatically set
  for pretrained vocabularies).
* The `id2lang` attributes does reverse mapping if provided (automatically set for pretrained vocabularies).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/tokenization_xlm.py#L475)

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
adding special tokens. An XLM sequence has the following format:

* single sequence: `<s> X </s>`
* pair of sequences: `<s> A </s> B </s>`

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/tokenization_xlm.py#L502)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/tokenization_xlm.py#L530)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## XLM specific outputs

### class transformers.models.xlm.modeling\_xlm.XLMForQuestionAnsweringOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L665)

( loss: typing.Optional[torch.FloatTensor] = None start\_top\_log\_probs: typing.Optional[torch.FloatTensor] = None start\_top\_index: typing.Optional[torch.LongTensor] = None end\_top\_log\_probs: typing.Optional[torch.FloatTensor] = None end\_top\_index: typing.Optional[torch.LongTensor] = None cls\_logits: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided) —
  Classification loss as the sum of start token, end token (and is\_impossible if provided) classification
  losses.
* **start\_top\_log\_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) —
  Log probabilities for the top config.start\_n\_top start token possibilities (beam-search).
* **start\_top\_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) —
  Indices for the top config.start\_n\_top start token possibilities (beam-search).
* **end\_top\_log\_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) —
  Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
  (beam-search).
* **end\_top\_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) —
  Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
* **cls\_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided) —
  Log probabilities for the `is_impossible` label of the answers.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Base class for outputs of question answering models using a `XLMSQuADHead`.

## XLMModel

### class transformers.XLMModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L694)

( config  )

Parameters

* **config** ([XLMModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Xlm Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L779)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None langs: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None lengths: typing.Optional[torch.Tensor] = None cache: typing.Optional[dict[str, torch.Tensor]] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None \*\*kwargs  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention\_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
* **cache** (`dict[str, torch.FloatTensor]`, *optional*) —
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
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
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLMModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## XLMWithLMHeadModel

### class transformers.XLMWithLMHeadModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L967)

( config  )

Parameters

* **config** ([XLMWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMWithLMHeadModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L999)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None langs: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None lengths: typing.Optional[torch.Tensor] = None cache: typing.Optional[dict[str, torch.Tensor]] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None \*\*kwargs  ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention\_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
* **cache** (`dict[str, torch.FloatTensor]`, *optional*) —
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
  `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
  are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLMWithLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMWithLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoTokenizer, XLMWithLMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
>>> model = XLMWithLMHeadModel.from_pretrained("FacebookAI/xlm-mlm-en-2048")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## XLMForSequenceClassification

### class transformers.XLMForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L1078)

( config  )

Parameters

* **config** ([XLMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLM Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L1090)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None langs: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None lengths: typing.Optional[torch.Tensor] = None cache: typing.Optional[dict[str, torch.Tensor]] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention\_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
* **cache** (`dict[str, torch.FloatTensor]`, *optional*) —
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
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
elements depending on the configuration ([XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLMForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, XLMForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
>>> model = XLMForSequenceClassification.from_pretrained("FacebookAI/xlm-mlm-en-2048")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = XLMForSequenceClassification.from_pretrained("FacebookAI/xlm-mlm-en-2048", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, XLMForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
>>> model = XLMForSequenceClassification.from_pretrained("FacebookAI/xlm-mlm-en-2048", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = XLMForSequenceClassification.from_pretrained(
...     "FacebookAI/xlm-mlm-en-2048", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## XLMForMultipleChoice

### class transformers.XLMForMultipleChoice

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L1496)

( config \*inputs \*\*kwargs  )

Parameters

* **config** ([XLMForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForMultipleChoice)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Xlm Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L1507)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None langs: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None lengths: typing.Optional[torch.Tensor] = None cache: typing.Optional[dict[str, torch.Tensor]] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **langs** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention\_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
* **cache** (`dict[str, torch.FloatTensor]`, *optional*) —
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
  `input_ids` above)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — *num\_choices* is the second dimension of the input tensors. (see *input\_ids* above).

  Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLMForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLMForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
>>> model = XLMForMultipleChoice.from_pretrained("FacebookAI/xlm-mlm-en-2048")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."
>>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

>>> # the linear classifier still needs to be trained
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## XLMForTokenClassification

### class transformers.XLMForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L1408)

( config  )

Parameters

* **config** ([XLMForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForTokenClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Xlm transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L1420)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None langs: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None lengths: typing.Optional[torch.Tensor] = None cache: typing.Optional[dict[str, torch.Tensor]] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention\_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
* **cache** (`dict[str, torch.FloatTensor]`, *optional*) —
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLMForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLMForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
>>> model = XLMForTokenClassification.from_pretrained("FacebookAI/xlm-mlm-en-2048")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
... )

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_token_class_ids = logits.argmax(-1)

>>> # Note that tokens are classified rather then input words which means that
>>> # there might be more predicted token classes than words.
>>> # Multiple token classes might account for the same word
>>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
>>> predicted_tokens_classes
...

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

## XLMForQuestionAnsweringSimple

### class transformers.XLMForQuestionAnsweringSimple

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L1189)

( config  )

Parameters

* **config** ([XLMForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForQuestionAnsweringSimple)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L1199)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None langs: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None lengths: typing.Optional[torch.Tensor] = None cache: typing.Optional[dict[str, torch.Tensor]] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None start\_positions: typing.Optional[torch.Tensor] = None end\_positions: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention\_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
* **cache** (`dict[str, torch.FloatTensor]`, *optional*) —
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **start\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLMForQuestionAnsweringSimple](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForQuestionAnsweringSimple) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLMForQuestionAnsweringSimple
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
>>> model = XLMForQuestionAnsweringSimple.from_pretrained("FacebookAI/xlm-mlm-en-2048")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
...

>>> # target is "nice puppet"
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
...
```

## XLMForQuestionAnswering

### class transformers.XLMForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L1289)

( config  )

Parameters

* **config** ([XLMForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForQuestionAnswering)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Xlm transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm/modeling_xlm.py#L1299)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None langs: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None lengths: typing.Optional[torch.Tensor] = None cache: typing.Optional[dict[str, torch.Tensor]] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None start\_positions: typing.Optional[torch.Tensor] = None end\_positions: typing.Optional[torch.Tensor] = None is\_impossible: typing.Optional[torch.Tensor] = None cls\_index: typing.Optional[torch.Tensor] = None p\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.xlm.modeling\_xlm.XLMForQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.models.xlm.modeling_xlm.XLMForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention\_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
* **cache** (`dict[str, torch.FloatTensor]`, *optional*) —
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **start\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **is\_impossible** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels whether a question has an answer or no answer (SQuAD 2.0)
* **cls\_index** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the classification token to use as input for computing plausibility of the
  answer.
* **p\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Optional mask of tokens which can’t be in answers (e.g. [CLS], [PAD], …). 1.0 means token should be
  masked. 0.0 mean token is not masked.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.xlm.modeling\_xlm.XLMForQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.models.xlm.modeling_xlm.XLMForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.xlm.modeling\_xlm.XLMForQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.models.xlm.modeling_xlm.XLMForQuestionAnsweringOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMConfig](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided) — Classification loss as the sum of start token, end token (and is\_impossible if provided) classification
  losses.
* **start\_top\_log\_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) — Log probabilities for the top config.start\_n\_top start token possibilities (beam-search).
* **start\_top\_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) — Indices for the top config.start\_n\_top start token possibilities (beam-search).
* **end\_top\_log\_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) — Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
  (beam-search).
* **end\_top\_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) — Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
* **cls\_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided) — Log probabilities for the `is_impossible` label of the answers.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLMForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLMForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
>>> model = XLMForQuestionAnswering.from_pretrained("FacebookAI/xlm-mlm-en-2048")

>>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
...     0
... )  # Batch size 1
>>> start_positions = torch.tensor([1])
>>> end_positions = torch.tensor([3])

>>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
>>> loss = outputs.loss
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/xlm.md)
