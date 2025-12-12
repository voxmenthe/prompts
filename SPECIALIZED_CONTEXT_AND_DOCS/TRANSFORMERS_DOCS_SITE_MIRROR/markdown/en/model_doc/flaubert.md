# FlauBERT

## Overview

The FlauBERT model was proposed in the paper [FlauBERT: Unsupervised Language Model Pre-training for French](https://huggingface.co/papers/1912.05372) by Hang Le et al. It's a transformer model pretrained using a masked language
modeling (MLM) objective (like BERT).

The abstract from the paper is the following:

*Language models have become a key step to achieve state-of-the art results in many different Natural Language
Processing (NLP) tasks. Leveraging the huge amount of unlabeled texts nowadays available, they provide an efficient way
to pre-train continuous word representations that can be fine-tuned for a downstream task, along with their
contextualization at the sentence level. This has been widely demonstrated for English using contextualized
representations (Dai and Le, 2015; Peters et al., 2018; Howard and Ruder, 2018; Radford et al., 2018; Devlin et al.,
2019; Yang et al., 2019b). In this paper, we introduce and share FlauBERT, a model learned on a very large and
heterogeneous French corpus. Models of different sizes are trained using the new CNRS (French National Centre for
Scientific Research) Jean Zay supercomputer. We apply our French language models to diverse NLP tasks (text
classification, paraphrasing, natural language inference, parsing, word sense disambiguation) and show that most of the
time they outperform other pretraining approaches. Different versions of FlauBERT as well as a unified evaluation
protocol for the downstream tasks, called FLUE (French Language Understanding Evaluation), are shared to the research
community for further reproducible experiments in French NLP.*

This model was contributed by [formiel](https://huggingface.co/formiel). The original code can be found [here](https://github.com/getalp/Flaubert).

Tips:

- Like RoBERTa, without the sentence ordering prediction (so just trained on the MLM objective).

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## FlaubertConfig[[transformers.FlaubertConfig]]

#### transformers.FlaubertConfig[[transformers.FlaubertConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/configuration_flaubert.py#L24)

This is the configuration class to store the configuration of a [FlaubertModel](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertModel). It is
used to instantiate a FlauBERT model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the FlauBERT
[flaubert/flaubert_base_uncased](https://huggingface.co/flaubert/flaubert_base_uncased) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

**Parameters:**

pre_norm (`bool`, *optional*, defaults to `False`) : Whether to apply the layer normalization before or after the feed forward layer following the attention in each layer (Vaswani et al., Tensor2Tensor for Neural Machine Translation. 2018)

layerdrop (`float`, *optional*, defaults to 0.0) : Probability to drop layers during training (Fan et al., Reducing Transformer Depth on Demand with Structured Dropout. ICLR 2020)

vocab_size (`int`, *optional*, defaults to 30145) : Vocabulary size of the FlauBERT model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [FlaubertModel](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertModel).

emb_dim (`int`, *optional*, defaults to 2048) : Dimensionality of the encoder layers and the pooler layer.

n_layer (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

n_head (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer encoder.

dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for the attention mechanism

gelu_activation (`bool`, *optional*, defaults to `True`) : Whether or not to use a *gelu* activation instead of *relu*.

sinusoidal_embeddings (`bool`, *optional*, defaults to `False`) : Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.

causal (`bool`, *optional*, defaults to `False`) : Whether or not the model should behave in a causal manner. Causal models use a triangular attention mask in order to only attend to the left-side context instead if a bidirectional context.

asm (`bool`, *optional*, defaults to `False`) : Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the prediction layer.

n_langs (`int`, *optional*, defaults to 1) : The number of languages the model handles. Set to 1 for monolingual models.

use_lang_emb (`bool`, *optional*, defaults to `True`) : Whether to use language embeddings. Some models use additional language embeddings, see [the multilingual models page](http://huggingface.co/transformers/multilingual.html#xlm-language-embeddings) for information on how to use them.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

embed_init_std (`float`, *optional*, defaults to 2048^-0.5) : The standard deviation of the truncated_normal_initializer for initializing the embedding matrices.

init_std (`int`, *optional*, defaults to 50257) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices except the embedding matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

bos_index (`int`, *optional*, defaults to 0) : The index of the beginning of sentence token in the vocabulary.

eos_index (`int`, *optional*, defaults to 1) : The index of the end of sentence token in the vocabulary.

pad_index (`int`, *optional*, defaults to 2) : The index of the padding token in the vocabulary.

unk_index (`int`, *optional*, defaults to 3) : The index of the unknown token in the vocabulary.

mask_index (`int`, *optional*, defaults to 5) : The index of the masking token in the vocabulary.

is_encoder(`bool`, *optional*, defaults to `True`) : Whether or not the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.

summary_type (`string`, *optional*, defaults to "first") : Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.  Has to be one of the following options:  - `"last"`: Take the last token hidden state (like XLNet). - `"first"`: Take the first token hidden state (like BERT). - `"mean"`: Take the mean of all tokens hidden states. - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2). - `"attn"`: Not implemented now, use multi-head attention.

summary_use_proj (`bool`, *optional*, defaults to `True`) : Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.  Whether or not to add a projection after the vector extraction.

summary_activation (`str`, *optional*) : Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.  Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.

summary_proj_to_labels (`bool`, *optional*, defaults to `True`) : Used in the sequence classification and multiple choice models.  Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.

summary_first_dropout (`float`, *optional*, defaults to 0.1) : Used in the sequence classification and multiple choice models.  The dropout ratio to be used after the projection and activation.

start_n_top (`int`, *optional*, defaults to 5) : Used in the SQuAD evaluation script.

end_n_top (`int`, *optional*, defaults to 5) : Used in the SQuAD evaluation script.

mask_token_id (`int`, *optional*, defaults to 0) : Model agnostic parameter to identify masked tokens when generating text in an MLM context.

lang_id (`int`, *optional*, defaults to 1) : The ID of the language used by the model. This parameter is used when generating text in a given language.

## FlaubertTokenizer[[transformers.FlaubertTokenizer]]

#### transformers.FlaubertTokenizer[[transformers.FlaubertTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/tokenization_flaubert.py#L123)

Construct a Flaubert tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

- Moses preprocessing and tokenization.
- Normalizing all inputs text.
- The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like
  "__classify__") to a vocabulary.
- The argument `do_lowercase` controls lower casing (automatically set for pretrained vocabularies).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

build_inputs_with_special_tokenstransformers.FlaubertTokenizer.build_inputs_with_special_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/tokenization_flaubert.py#L432[{"name": "token_ids_0", "val": ": list"}, {"name": "token_ids_1", "val": ": typing.Optional[list[int]] = None"}]- **token_ids_0** (`List[int]`) --
  List of IDs to which the special tokens will be added.
- **token_ids_1** (`List[int]`, *optional*) --
  Optional second list of IDs for sequence pairs.0`List[int]`List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLM sequence has the following format:

- single sequence: ` X `
- pair of sequences: ` A  B `

**Parameters:**

vocab_file (`str`) : Vocabulary file.

merges_file (`str`) : Merges file.

do_lowercase (`bool`, *optional*, defaults to `False`) : Controls lower casing.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

additional_special_tokens (`List[str]`, *optional*, defaults to `['', '', '', '', '', '', '', '', '', '']`) : List of additional special tokens.

lang2id (`Dict[str, int]`, *optional*) : Dictionary mapping languages string identifiers to their IDs.

id2lang (`Dict[int, str]`, *optional*) : Dictionary mapping language IDs to their string identifiers.

**Returns:**

``List[int]``

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
#### convert_tokens_to_string[[transformers.FlaubertTokenizer.convert_tokens_to_string]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/tokenization_flaubert.py#L426)

Converts a sequence of tokens (string) in a single string.
#### get_special_tokens_mask[[transformers.FlaubertTokenizer.get_special_tokens_mask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/tokenization_flaubert.py#L460)

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

**Parameters:**

token_ids_0 (`List[int]`) : List of IDs.

token_ids_1 (`List[int]`, *optional*) : Optional second list of IDs for sequence pairs.

already_has_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the token list is already formatted with special tokens for the model.

**Returns:**

``List[int]``

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

## FlaubertModel[[transformers.FlaubertModel]]

#### transformers.FlaubertModel[[transformers.FlaubertModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L705)

The bare Flaubert Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlaubertModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L780[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "langs", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "lengths", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "cache", "val": ": typing.Optional[dict[str, torch.FloatTensor]] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use `attention_mask` for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`:
- **cache** (`dict[str, torch.FloatTensor]`, *optional*) --
  Dictionary strings to `torch.FloatTensor` that contains precomputed hidden-states (key and values in the
  attention blocks) as computed by the model (see `cache` output below). Can be used to speed up sequential
  decoding. The dictionary object will be modified in-place during the forward pass to add newly computed
  hidden-states.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **cache_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.0[transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [FlaubertModel](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([FlaubertModel](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## FlaubertWithLMHeadModel[[transformers.FlaubertWithLMHeadModel]]

#### transformers.FlaubertWithLMHeadModel[[transformers.FlaubertWithLMHeadModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L959)

The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlaubertWithLMHeadModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L991[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "langs", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "lengths", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "cache", "val": ": typing.Optional[dict[str, torch.Tensor]] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use `attention_mask` for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`:
- **cache** (`dict[str, torch.FloatTensor]`, *optional*) --
  Dictionary strings to `torch.FloatTensor` that contains precomputed hidden-states (key and values in the
  attention blocks) as computed by the model (see `cache` output below). Can be used to speed up sequential
  decoding. The dictionary object will be modified in-place during the forward pass to add newly computed
  hidden-states.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
  `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
  are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Masked language modeling (MLM) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [FlaubertWithLMHeadModel](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertWithLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> import torch
>>> from transformers import AutoTokenizer, FlaubertWithLMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
>>> model = FlaubertWithLMHeadModel.from_pretrained("flaubert/flaubert_base_uncased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

**Parameters:**

config ([FlaubertWithLMHeadModel](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertWithLMHeadModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Masked language modeling (MLM) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## FlaubertForSequenceClassification[[transformers.FlaubertForSequenceClassification]]

#### transformers.FlaubertForSequenceClassification[[transformers.FlaubertForSequenceClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L1068)

Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlaubertForSequenceClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L1080[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "langs", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "lengths", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "cache", "val": ": typing.Optional[dict[str, torch.Tensor]] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
- **cache** (`dict[str, torch.FloatTensor]`, *optional*) --
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [FlaubertForSequenceClassification](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, FlaubertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
>>> model = FlaubertForSequenceClassification.from_pretrained("flaubert/flaubert_base_uncased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = FlaubertForSequenceClassification.from_pretrained("flaubert/flaubert_base_uncased", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, FlaubertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
>>> model = FlaubertForSequenceClassification.from_pretrained("flaubert/flaubert_base_uncased", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = FlaubertForSequenceClassification.from_pretrained(
...     "flaubert/flaubert_base_uncased", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

**Parameters:**

config ([FlaubertForSequenceClassification](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertForSequenceClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## FlaubertForMultipleChoice[[transformers.FlaubertForMultipleChoice]]

#### transformers.FlaubertForMultipleChoice[[transformers.FlaubertForMultipleChoice]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L1521)

The Flaubert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlaubertForMultipleChoice.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L1532[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "langs", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "lengths", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "cache", "val": ": typing.Optional[dict[str, torch.Tensor]] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **langs** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) --
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
  config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
- **cache** (`dict[str, torch.FloatTensor]`, *optional*) --
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
  num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
  `input_ids` above)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.MultipleChoiceModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.MultipleChoiceModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) -- *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

  Classification scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [FlaubertForMultipleChoice](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, FlaubertForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
>>> model = FlaubertForMultipleChoice.from_pretrained("flaubert/flaubert_base_uncased")

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

**Parameters:**

config ([FlaubertForMultipleChoice](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertForMultipleChoice)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.MultipleChoiceModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.MultipleChoiceModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) -- *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

  Classification scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## FlaubertForTokenClassification[[transformers.FlaubertForTokenClassification]]

#### transformers.FlaubertForTokenClassification[[transformers.FlaubertForTokenClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L1174)

The Flaubert transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlaubertForTokenClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L1186[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "langs", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "lengths", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "cache", "val": ": typing.Optional[dict[str, torch.Tensor]] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
- **cache** (`dict[str, torch.FloatTensor]`, *optional*) --
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided)  -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Classification scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [FlaubertForTokenClassification](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, FlaubertForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
>>> model = FlaubertForTokenClassification.from_pretrained("flaubert/flaubert_base_uncased")

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

**Parameters:**

config ([FlaubertForTokenClassification](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertForTokenClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided)  -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Classification scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## FlaubertForQuestionAnsweringSimple[[transformers.FlaubertForQuestionAnsweringSimple]]

#### transformers.FlaubertForQuestionAnsweringSimple[[transformers.FlaubertForQuestionAnsweringSimple]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L1267)

Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlaubertForQuestionAnsweringSimple.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L1277[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "langs", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "lengths", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "cache", "val": ": typing.Optional[dict[str, torch.Tensor]] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "start_positions", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "end_positions", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
- **cache** (`dict[str, torch.FloatTensor]`, *optional*) --
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **start_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **end_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-end scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [FlaubertForQuestionAnsweringSimple](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertForQuestionAnsweringSimple) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, FlaubertForQuestionAnsweringSimple
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
>>> model = FlaubertForQuestionAnsweringSimple.from_pretrained("flaubert/flaubert_base_uncased")

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

**Parameters:**

config ([FlaubertForQuestionAnsweringSimple](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertForQuestionAnsweringSimple)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-end scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## FlaubertForQuestionAnswering[[transformers.FlaubertForQuestionAnswering]]

#### transformers.FlaubertForQuestionAnswering[[transformers.FlaubertForQuestionAnswering]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L1402)

The Flaubert transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlaubertForQuestionAnswering.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flaubert/modeling_flaubert.py#L1412[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "langs", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "lengths", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "cache", "val": ": typing.Optional[dict[str, torch.Tensor]] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "start_positions", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "end_positions", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "is_impossible", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "cls_index", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "p_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **langs** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
  languages ids which can be obtained from the language names by using two conversion mappings provided in
  the configuration of the model (only provided for multilingual models). More precisely, the *language name
  to language id* mapping is in `model.config.lang2id` (which is a dictionary string to int) and the
  *language id to language name* mapping is in `model.config.id2lang` (dictionary int to string).

  See usage examples detailed in the [multilingual documentation](../multilingual).
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **lengths** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Length of each sentence that can be used to avoid performing attention on padding token indices. You can
  also use *attention_mask* for the same result (see above), kept here for compatibility. Indices selected in
  `[0, ..., input_ids.size(-1)]`.
- **cache** (`dict[str, torch.FloatTensor]`, *optional*) --
  Instance of `EncoderDecoderCache` that contains precomputed KV states. Can be used to speed up sequential
  decoding.
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **start_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **end_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **is_impossible** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels whether a question has an answer or no answer (SQuAD 2.0)
- **cls_index** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the classification token to use as input for computing plausibility of the
  answer.
- **p_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...). 1.0 means token should be
  masked. 0.0 mean token is not masked.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.flaubert.modeling_flaubert.FlaubertForQuestionAnsweringOutput` or `tuple(torch.FloatTensor)`A `transformers.models.flaubert.modeling_flaubert.FlaubertForQuestionAnsweringOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided) -- Classification loss as the sum of start token, end token (and is_impossible if provided) classification
  losses.
- **start_top_log_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the top config.start_n_top start token possibilities (beam-search).
- **start_top_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Indices for the top config.start_n_top start token possibilities (beam-search).
- **end_top_log_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
  (beam-search).
- **end_top_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
- **cls_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the `is_impossible` label of the answers.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [FlaubertForQuestionAnswering](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, FlaubertForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
>>> model = FlaubertForQuestionAnswering.from_pretrained("FacebookAI/xlm-mlm-en-2048")

>>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
...     0
... )  # Batch size 1
>>> start_positions = torch.tensor([1])
>>> end_positions = torch.tensor([3])

>>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
>>> loss = outputs.loss
```

**Parameters:**

config ([FlaubertForQuestionAnswering](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertForQuestionAnswering)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.flaubert.modeling_flaubert.FlaubertForQuestionAnsweringOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.flaubert.modeling_flaubert.FlaubertForQuestionAnsweringOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlaubertConfig](/docs/transformers/main/en/model_doc/flaubert#transformers.FlaubertConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided) -- Classification loss as the sum of start token, end token (and is_impossible if provided) classification
  losses.
- **start_top_log_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the top config.start_n_top start token possibilities (beam-search).
- **start_top_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Indices for the top config.start_n_top start token possibilities (beam-search).
- **end_top_log_probs** (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
  (beam-search).
- **end_top_index** (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
- **cls_logits** (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided) -- Log probabilities for the `is_impossible` label of the answers.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
