*This model was released on 2020-01-13 and added to Hugging Face Transformers on 2023-06-20.*

# XLM-ProphetNet

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

This model is in maintenance mode only, we don’t accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

[![Models](https://img.shields.io/badge/All_model_pages-xprophetnet-blueviolet)](https://huggingface.co/models?filter=xprophetnet) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/xprophetnet-large-wiki100-cased-xglue-ntg)

**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) and assign
@patrickvonplaten

## Overview

The XLM-ProphetNet model was proposed in [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training,](https://huggingface.co/papers/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei
Zhang, Ming Zhou on 13 Jan, 2020.

XLM-ProphetNet is an encoder-decoder model and can predict n-future tokens for “ngram” language modeling instead of
just the next token. Its architecture is identical to ProhpetNet, but the model was trained on the multi-lingual
“wiki100” Wikipedia dump. XLM-ProphetNet’s model architecture and pretraining objective is same as ProphetNet, but XLM-ProphetNet was pre-trained on the cross-lingual dataset XGLUE.

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

## Resources

* [Causal language modeling task guide](../tasks/language_modeling)
* [Translation task guide](../tasks/translation)
* [Summarization task guide](../tasks/summarization)

## XLMProphetNetConfig

### class transformers.XLMProphetNetConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/configuration_xlm_prophetnet.py#L26)

( activation\_dropout: typing.Optional[float] = 0.1 activation\_function: typing.Union[str, typing.Callable, NoneType] = 'gelu' vocab\_size: typing.Optional[int] = 30522 hidden\_size: typing.Optional[int] = 1024 encoder\_ffn\_dim: typing.Optional[int] = 4096 num\_encoder\_layers: typing.Optional[int] = 12 num\_encoder\_attention\_heads: typing.Optional[int] = 16 decoder\_ffn\_dim: typing.Optional[int] = 4096 num\_decoder\_layers: typing.Optional[int] = 12 num\_decoder\_attention\_heads: typing.Optional[int] = 16 attention\_dropout: typing.Optional[float] = 0.1 dropout: typing.Optional[float] = 0.1 max\_position\_embeddings: typing.Optional[int] = 512 init\_std: typing.Optional[float] = 0.02 is\_encoder\_decoder: typing.Optional[bool] = True add\_cross\_attention: typing.Optional[bool] = True decoder\_start\_token\_id: typing.Optional[int] = 0 ngram: typing.Optional[int] = 2 num\_buckets: typing.Optional[int] = 32 relative\_max\_distance: typing.Optional[int] = 128 disable\_ngram\_loss: typing.Optional[bool] = False eps: typing.Optional[float] = 0.0 use\_cache: typing.Optional[bool] = True pad\_token\_id: typing.Optional[int] = 0 bos\_token\_id: typing.Optional[int] = 1 eos\_token\_id: typing.Optional[int] = 2 \*\*kwargs  )

Parameters

* **activation\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for activations inside the fully connected layer.
* **activation\_function** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the ProphetNET model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [XLMProphetNetModel](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel).
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

This is the configuration class to store the configuration of a [XLMProphetNetModel](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel). It is used to instantiate a
XLMProphetNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the XLMProphetNet
[microsoft/xprophetnet-large-wiki100-cased](https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## XLMProphetNetTokenizer

### class transformers.XLMProphetNetTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L43)

( vocab\_file bos\_token = '[SEP]' eos\_token = '[SEP]' sep\_token = '[SEP]' unk\_token = '[UNK]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **bos\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **sep\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **unk\_token** (`str`, *optional*, defaults to `"[UNK]"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"[PAD]"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"[CLS]"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"[MASK]"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
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

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L296)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs to which the special tokens will be added
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A XLMProphetNet sequence has the following format:

* single sequence: `X [SEP]`
* pair of sequences: `A [SEP] B [SEP]`

#### convert\_tokens\_to\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L274)

( tokens  )

Converts a sequence of tokens (strings for sub-words) in a single string.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L223)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLMProphetNet
does not make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L195)

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

## XLMProphetNetModel

### class transformers.XLMProphetNetModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1700)

( config: XLMProphetNetConfig  )

Parameters

* **config** ([XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare XLMProphetNet Model outputting raw hidden-states without any specific head on top.
This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1736)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None inputs\_embeds: typing.Optional[torch.Tensor] = None decoder\_inputs\_embeds: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

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

  XLMProphetNet uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **head\_mask** (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
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
* **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) —
  Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
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

`transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, hidden_size)`) — Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **last\_hidden\_state\_ngram** (`torch.FloatTensor` of shape `(batch_size,ngram * decoder_sequence_length, config.vocab_size)`, *optional*) — Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_ngram\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **decoder\_ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, encoder_sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, encoder_sequence_length, encoder_sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [XLMProphetNetModel](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLMProphetNetModel

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetModel.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  # Batch size 1
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

>>> last_hidden_states = outputs.last_hidden_state  # main stream hidden states
>>> last_hidden_states_ngram = outputs.last_hidden_state_ngram  # predict hidden states
```

## XLMProphetNetEncoder

### class transformers.XLMProphetNetEncoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1229)

( config: XLMProphetNetConfig word\_embeddings: Embedding = None  )

Parameters

* **config** ([XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone encoder part of the XLMProphetNetModel.
This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.

word\_embeddings (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
The word embedding parameters. This can be used to initialize [XLMProphetNetEncoder](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder) with pre-defined word
embeddings instead of randomly initialized word embeddings.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1259)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
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
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [XLMProphetNetEncoder](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLMProphetNetEncoder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetEncoder.from_pretrained("patrickvonplaten/prophetnet-large-uncased-standalone")
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## XLMProphetNetDecoder

### class transformers.XLMProphetNetDecoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1354)

( config: XLMProphetNetConfig word\_embeddings: typing.Optional[torch.nn.modules.sparse.Embedding] = None  )

Parameters

* **config** ([XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone decoder part of the XLMProphetNetModel.
This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.

word\_embeddings (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
The word embedding parameters. This can be used to initialize [XLMProphetNetEncoder](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder) with pre-defined word
embeddings instead of randomly initialized word embeddings.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1391)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None encoder\_hidden\_states: typing.Optional[torch.Tensor] = None encoder\_attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None inputs\_embeds: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
  the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) —
  Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

Returns

`transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, hidden_size)`) — Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **last\_hidden\_state\_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) — Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
* **ngram\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the

The [XLMProphetNetDecoder](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetDecoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLMProphetNetDecoder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetDecoder.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone", add_cross_attention=False)
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## XLMProphetNetForConditionalGeneration

### class transformers.XLMProphetNetForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1831)

( config: XLMProphetNetConfig  )

Parameters

* **config** ([XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The XLMProphetNet Model with a language modeling head. Can be used for sequence generation tasks.
This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1852)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None inputs\_embeds: typing.Optional[torch.Tensor] = None decoder\_inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqLMOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

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

  XLMProphetNet uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
* **head\_mask** (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
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
* **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) —
  Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
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
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
  labels in `[0, ..., config.vocab_size]`

Returns

`transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqLMOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, config.vocab_size)`) — Prediction scores of the main stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **logits\_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) — Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_ngram\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **decoder\_ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, encoder_sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, encoder_sequence_length, encoder_sequence_length)`. Attentions weights of the encoder, after the attention
  softmax, used to compute the weighted average in the self-attention heads.

The [XLMProphetNetForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLMProphetNetForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetForConditionalGeneration.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")

>>> input_ids = tokenizer(
...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
... ).input_ids  # Batch size 1
>>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
>>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

>>> logits_next_token = outputs.logits  # logits to predict next token as usual
>>> logits_ngram_next_tokens = outputs.logits_ngram  # logits to predict 2nd, 3rd, ... next tokens
```

## XLMProphetNetForCausalLM

### class transformers.XLMProphetNetForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L2041)

( config: XLMProphetNetConfig  )

Parameters

* **config** ([XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) — Model configuration class with all the parameters of the model.
  Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The standalone decoder part of the XLMProphetNetModel with a lm head on top. The model can be used for causal language modeling.
This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L2080)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None encoder\_hidden\_states: typing.Optional[torch.Tensor] = None encoder\_attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderLMOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
  the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) —
  Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
  `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
  ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`

Returns

`transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderLMOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, config.vocab_size)`) — Prediction scores of the main stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **logits\_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) — Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
  SoftMax).
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
* **ngram\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **ngram\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the

The [XLMProphetNetForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, XLMProphetNetForCausalLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetForCausalLM.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits

>>> # Model can also be used with EncoderDecoder framework
>>> from transformers import BertTokenizer, EncoderDecoderModel, AutoTokenizer
>>> import torch

>>> tokenizer_enc = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")
>>> tokenizer_dec = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google-bert/bert-large-uncased", "patrickvonplaten/xprophetnet-large-uncased-standalone"
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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/xlm-prophetnet.md)
