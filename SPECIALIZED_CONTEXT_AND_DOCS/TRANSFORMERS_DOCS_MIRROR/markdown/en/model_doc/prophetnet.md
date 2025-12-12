*This model was released on 2020-01-13 and added to Hugging Face Transformers on 2020-11-16.*

# ProphetNet

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The ProphetNet model was proposed in [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training,](https://huggingface.co/papers/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei
Zhang, Ming Zhou on 13 Jan, 2020.

ProphetNet is an encoder-decoder model and can predict n-future tokens for “ngram” language modeling instead of just
the next token.

The abstract from the paper is the following:

*In this paper, we present a new sequence-to-sequence pretraining model called ProphetNet, which introduces a novel
self-supervised objective named future n-gram prediction and the proposed n-stream self-attention mechanism. Instead of
the optimization of one-step ahead prediction in traditional sequence-to-sequence model, the ProphetNet is optimized by
n-step ahead prediction which predicts the next n tokens simultaneously based on previous context tokens at each time
step. The future n-gram prediction explicitly encourages the model to plan for the future tokens and prevent
overfitting on strong local correlations. We pre-train ProphetNet using a base scale dataset (16GB) and a large scale
dataset (160GB) respectively. Then we conduct experiments on CNN/DailyMail, Gigaword, and SQuAD 1.1 benchmarks for
abstractive summarization and question generation tasks. Experimental results show that ProphetNet achieves new
state-of-the-art results on all these datasets compared to the models using the same scale pretraining corpus.*

The Authors’ code can be found [here](https://github.com/microsoft/ProphetNet).

## Usage tips

* ProphetNet is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than
  the left.
* The model architecture is based on the original Transformer, but replaces the “standard” self-attention mechanism in the decoder by a main self-attention mechanism and a self and n-stream (predict) self-attention mechanism.

## Resources

* [Causal language modeling task guide](../tasks/language_modeling)
* [Translation task guide](../tasks/translation)
* [Summarization task guide](../tasks/summarization)

## ProphetNetConfig

### class transformers.ProphetNetConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/configuration_prophetnet.py#L26)

( activation\_dropout: typing.Optional[float] = 0.1 activation\_function: typing.Union[str, typing.Callable, NoneType] = 'gelu' vocab\_size: typing.Optional[int] = 30522 hidden\_size: typing.Optional[int] = 1024 encoder\_ffn\_dim: typing.Optional[int] = 4096 num\_encoder\_layers: typing.Optional[int] = 12 num\_encoder\_attention\_heads: typing.Optional[int] = 16 decoder\_ffn\_dim: typing.Optional[int] = 4096 num\_decoder\_layers: typing.Optional[int] = 12 num\_decoder\_attention\_heads: typing.Optional[int] = 16 attention\_dropout: typing.Optional[float] = 0.1 dropout: typing.Optional[float] = 0.1 max\_position\_embeddings: typing.Optional[int] = 512 init\_std: typing.Optional[float] = 0.02 is\_encoder\_decoder: typing.Optional[bool] = True add\_cross\_attention: typing.Optional[bool] = True decoder\_start\_token\_id: typing.Optional[int] = 0 ngram: typing.Optional[int] = 2 num\_buckets: typing.Optional[int] = 32 relative\_max\_distance: typing.Optional[int] = 128 disable\_ngram\_loss: typing.Optional[bool] = False eps: typing.Optional[float] = 0.0 use\_cache: typing.Optional[bool] = True pad\_token\_id: typing.Optional[int] = 0 bos\_token\_id: typing.Optional[int] = 1 eos\_token\_id: typing.Optional[int] = 2 \*\*kwargs  )

Parameters

* **activation\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for activations inside the fully connected layer.
* **activation\_function** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the ProphetNET model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [ProphetNetModel](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetModel).
* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the layers and the pooler layer.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in decoder.
* **num\_encoder\_layers** (`int`, *optional*, defaults to 12) —
  Number of encoder layers.
* **num\_encoder\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the `intermediate` (often named feed-forward) layer in decoder.
* **num\_decoder\_layers** (`int`, *optional*, defaults to 12) —
  Number of decoder layers.
* **num\_decoder\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **add\_cross\_attention** (`bool`, *optional*, defaults to `True`) —
  Whether cross-attention layers should be added to the model.
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Whether this is an encoder/decoder model.
* **pad\_token\_id** (`int`, *optional*, defaults to 1) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 0) —
  Beginning of stream token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  End of stream token id.
* **ngram** (`int`, *optional*, defaults to 2) —
  Number of future tokens to predict. Set to 1 to be same as traditional Language model to predict next first
  token.
* **num\_buckets** (`int`, *optional*, defaults to 32) —
  The number of buckets to use for each attention layer. This is for relative position calculation. See the
  [T5 paper](see <https://huggingface.co/papers/1910.10683>) for more details.
* **relative\_max\_distance** (`int`, *optional*, defaults to 128) —
  Relative distances greater than this number will be put into the last same bucket. This is for relative
  position calculation. See the [T5 paper](see <https://huggingface.co/papers/1910.10683>) for more details.
* **disable\_ngram\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether be trained predicting only the next first token.
* **eps** (`float`, *optional*, defaults to 0.0) —
  Controls the `epsilon` parameter value for label smoothing in the loss calculation. If set to 0, no label
  smoothing is performed.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).

This is the configuration class to store the configuration of a [ProphetNetModel](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetModel). It is used to instantiate a
ProphetNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the ProphetNet
[microsoft/prophetnet-large-uncased](https://huggingface.co/microsoft/prophetnet-large-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## ProphetNetTokenizer

### class transformers.ProphetNetTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/tokenization_prophetnet.py#L272)

( vocab\_file: str do\_lower\_case: typing.Optional[bool] = True do\_basic\_tokenize: typing.Optional[bool] = True never\_split: typing.Optional[collections.abc.Iterable] = None unk\_token: typing.Optional[str] = '[UNK]' sep\_token: typing.Optional[str] = '[SEP]' x\_sep\_token: typing.Optional[str] = '[X\_SEP]' pad\_token: typing.Optional[str] = '[PAD]' mask\_token: typing.Optional[str] = '[MASK]' tokenize\_chinese\_chars: typing.Optional[bool] = True strip\_accents: typing.Optional[bool] = None clean\_up\_tokenization\_spaces: bool = True \*\*kwargs  )

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
* **x\_sep\_token** (`str`, *optional*, defaults to `"[X_SEP]"`) —
  Special second separator token, which can be generated by [ProphetNetForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetForConditionalGeneration). It is
  used to separate bullet-point like sentences in summarization, *e.g.*.
* **pad\_token** (`str`, *optional*, defaults to `"[PAD]"`) —
  The token used for padding, for example when batching sequences of different lengths.
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
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*, defaults to `True`) —
  Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
  extra spaces.

Construct a ProphetNetTokenizer. Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/tokenization_prophetnet.py#L455)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/tokenization_prophetnet.py#L400)

( tokens: str  )

Converts a sequence of tokens (string) in a single string.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/tokenization_prophetnet.py#L405)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: typing.Optional[bool] = False  ) → `List[int]`

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

## ProphetNet specific outputs

### class transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqLMOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L122)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None logits\_ngram: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_ngram\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_ngram\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, config.vocab_size)`) —
  Prediction scores of the main stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **logits\_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) —
  Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) —
  List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_ngram\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **decoder\_ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

Base class for sequence-to-sequence language models outputs.

### class transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L184)

( last\_hidden\_state: FloatTensor last\_hidden\_state\_ngram: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_ngram\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_ngram\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, hidden_size)`) —
  Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **last\_hidden\_state\_ngram** (`torch.FloatTensor` of shape `(batch_size,ngram * decoder_sequence_length, config.vocab_size)`, *optional*) —
  Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) —
  List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_ngram\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **decoder\_ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

Base class for model encoder’s outputs that also contains : pre-computed hidden states that can speed up sequential
decoding.

### class transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L243)

( last\_hidden\_state: FloatTensor last\_hidden\_state\_ngram: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[tuple[torch.FloatTensor]] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None hidden\_states\_ngram: typing.Optional[tuple[torch.FloatTensor]] = None attentions: typing.Optional[tuple[torch.FloatTensor]] = None ngram\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, hidden_size)`) —
  Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **last\_hidden\_state\_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) —
  Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) —
  List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **hidden\_states\_ngram** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

### class transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderLMOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L288)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None logits\_ngram: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[tuple[torch.FloatTensor]] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None hidden\_states\_ngram: typing.Optional[tuple[torch.FloatTensor]] = None attentions: typing.Optional[tuple[torch.FloatTensor]] = None ngram\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) —
  Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, config.vocab_size)`) —
  Prediction scores of the main stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **logits\_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) —
  Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) —
  List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **hidden\_states\_ngram** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

## ProphetNetModel

### class transformers.ProphetNetModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1477)

( config: ProphetNetConfig  )

Parameters

* **config** ([ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Prophetnet Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1513)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None inputs\_embeds: typing.Optional[torch.Tensor] = None decoder\_inputs\_embeds: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqModelOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput) or `tuple(torch.FloatTensor)`

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
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  ProphetNet uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`tuple`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **decoder\_inputs\_embeds** (`torch.Tensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
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

[transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqModelOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqModelOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, hidden_size)`) — Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **last\_hidden\_state\_ngram** (`torch.FloatTensor` of shape `(batch_size,ngram * decoder_sequence_length, config.vocab_size)`, *optional*) — Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_ngram\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **decoder\_ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [ProphetNetModel](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, ProphetNetModel

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = ProphetNetModel.from_pretrained("microsoft/prophetnet-large-uncased")

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  # Batch size 1
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

>>> last_hidden_states = outputs.last_hidden_state  # main stream hidden states
>>> last_hidden_states_ngram = outputs.last_hidden_state_ngram  # predict hidden states
```

## ProphetNetEncoder

### class transformers.ProphetNetEncoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1021)

( config: ProphetNetConfig word\_embeddings: Embedding = None  )

Parameters

* **config** ([ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **word\_embeddings** (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*) —
  The word embedding parameters. This can be used to initialize [ProphetNetEncoder](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetEncoder) with pre-defined word
  embeddings instead of randomly initialized word embeddings.

The standalone encoder part of the ProphetNetModel.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1050)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

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

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ProphetNetEncoder](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, ProphetNetEncoder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = ProphetNetEncoder.from_pretrained("patrickvonplaten/prophetnet-large-uncased-standalone")
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## ProphetNetDecoder

### class transformers.ProphetNetDecoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1143)

( config: ProphetNetConfig word\_embeddings: typing.Optional[torch.nn.modules.sparse.Embedding] = None  )

Parameters

* **config** ([ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **word\_embeddings** (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*) —
  The word embedding parameters. This can be used to initialize [ProphetNetEncoder](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetEncoder) with pre-defined word
  embeddings instead of randomly initialized word embeddings.

The standalone decoder part of the ProphetNetModel.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1181)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None encoder\_hidden\_states: typing.Optional[torch.Tensor] = None encoder\_attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None inputs\_embeds: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderModelOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput) or `tuple(torch.FloatTensor)`

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
* **encoder\_hidden\_states** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **past\_key\_values** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
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

[transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderModelOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderModelOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, hidden_size)`) — Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **last\_hidden\_state\_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) — Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **hidden\_states\_ngram** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [ProphetNetDecoder](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetDecoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, ProphetNetDecoder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = ProphetNetDecoder.from_pretrained("microsoft/prophetnet-large-uncased", add_cross_attention=False)
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## ProphetNetForConditionalGeneration

### class transformers.ProphetNetForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1628)

( config: ProphetNetConfig  )

Parameters

* **config** ([ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ProphetNet Model with a language modeling head. Can be used for sequence generation tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1649)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None inputs\_embeds: typing.Optional[torch.Tensor] = None decoder\_inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqLMOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

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
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  ProphetNet uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`torch.Tensor`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **decoder\_inputs\_embeds** (`torch.Tensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
  labels in `[0, ..., config.vocab_size]`
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
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

[transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqLMOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetSeq2SeqLMOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, config.vocab_size)`) — Prediction scores of the main stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **logits\_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) — Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_ngram\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **decoder\_ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [ProphetNetForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, ProphetNetForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  # Batch size 1
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

>>> logits_next_token = outputs.logits  # logits to predict next token as usual
>>> logits_ngram_next_tokens = outputs.logits_ngram  # logits to predict 2nd, 3rd, ... next tokens
```

## ProphetNetForCausalLM

### class transformers.ProphetNetForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1816)

( config: ProphetNetConfig  )

Parameters

* **config** ([ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone decoder part of the ProphetNetModel with a lm head on top. The model can be used for causal

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1855)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None encoder\_hidden\_states: typing.Optional[torch.Tensor] = None encoder\_attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderLMOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput) or `tuple(torch.FloatTensor)`

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
* **encoder\_hidden\_states** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **past\_key\_values** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
  `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
  ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderLMOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.prophetnet.modeling\_prophetnet.ProphetNetDecoderLMOutput](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, config.vocab_size)`) — Prediction scores of the main stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **logits\_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) — Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **hidden\_states\_ngram** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [ProphetNetForCausalLM](/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, ProphetNetForCausalLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = ProphetNetForCausalLM.from_pretrained("microsoft/prophetnet-large-uncased")
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits

>>> # Model can also be used with EncoderDecoder framework
>>> from transformers import BertTokenizer, EncoderDecoderModel, AutoTokenizer
>>> import torch

>>> tokenizer_enc = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")
>>> tokenizer_dec = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google-bert/bert-large-uncased", "microsoft/prophetnet-large-uncased"
... )

>>> ARTICLE = (
...     "the us state department said wednesday it had received no "
...     "formal word from bolivia that it was expelling the us ambassador there "
...     "but said the charges made against him are `` baseless ."
... )
>>> input_ids = tokenizer_enc(ARTICLE, return_tensors="pt").input_ids
>>> labels = tokenizer_dec(
...     "us rejects charges against its ambassador in bolivia", return_tensors="pt"
... ).input_ids
>>> outputs = model(input_ids=input_ids, decoder_input_ids=labels[:, :-1], labels=labels[:, 1:])

>>> loss = outputs.loss
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/prophetnet.md)
