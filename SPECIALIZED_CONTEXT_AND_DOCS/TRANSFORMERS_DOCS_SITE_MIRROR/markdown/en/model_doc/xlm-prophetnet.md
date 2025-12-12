# XLM-ProphetNet

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) and assign
@patrickvonplaten

## Overview

The XLM-ProphetNet model was proposed in [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training,](https://huggingface.co/papers/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei
Zhang, Ming Zhou on 13 Jan, 2020.

XLM-ProphetNet is an encoder-decoder model and can predict n-future tokens for "ngram" language modeling instead of
just the next token. Its architecture is identical to ProhpetNet, but the model was trained on the multi-lingual
"wiki100" Wikipedia dump. XLM-ProphetNet's model architecture and pretraining objective is same as ProphetNet, but XLM-ProphetNet was pre-trained on the cross-lingual dataset XGLUE.

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

The Authors' code can be found [here](https://github.com/microsoft/ProphetNet).

## Resources

- [Causal language modeling task guide](../tasks/language_modeling)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## XLMProphetNetConfig[[transformers.XLMProphetNetConfig]]

#### transformers.XLMProphetNetConfig[[transformers.XLMProphetNetConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/configuration_xlm_prophetnet.py#L27)

This is the configuration class to store the configuration of a [XLMProphetNetModel](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel). It is used to instantiate a
XLMProphetNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the XLMProphetNet
[microsoft/xprophetnet-large-wiki100-cased](https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased)
architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

**Parameters:**

activation_dropout (`float`, *optional*, defaults to 0.1) : The dropout ratio for activations inside the fully connected layer.

activation_function (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

vocab_size (`int`, *optional*, defaults to 30522) : Vocabulary size of the ProphetNET model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [XLMProphetNetModel](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel).

hidden_size (`int`, *optional*, defaults to 1024) : Dimensionality of the layers and the pooler layer.

encoder_ffn_dim (`int`, *optional*, defaults to 4096) : Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.

num_encoder_layers (`int`, *optional*, defaults to 12) : Number of encoder layers.

num_encoder_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer encoder.

decoder_ffn_dim (`int`, *optional*, defaults to 4096) : Dimensionality of the `intermediate` (often named feed-forward) layer in decoder.

num_decoder_layers (`int`, *optional*, defaults to 12) : Number of decoder layers.

num_decoder_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer decoder.

attention_dropout (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

init_std (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

add_cross_attention (`bool`, *optional*, defaults to `True`) : Whether cross-attention layers should be added to the model.

is_encoder_decoder (`bool`, *optional*, defaults to `True`) : Whether this is an encoder/decoder model.

pad_token_id (`int`, *optional*, defaults to 1) : Padding token id.

bos_token_id (`int`, *optional*, defaults to 0) : Beginning of stream token id.

eos_token_id (`int`, *optional*, defaults to 2) : End of stream token id.

ngram (`int`, *optional*, defaults to 2) : Number of future tokens to predict. Set to 1 to be same as traditional Language model to predict next first token.

num_buckets (`int`, *optional*, defaults to 32) : The number of buckets to use for each attention layer. This is for relative position calculation. See the [T5 paper](see https://huggingface.co/papers/1910.10683) for more details.

relative_max_distance (`int`, *optional*, defaults to 128) : Relative distances greater than this number will be put into the last same bucket. This is for relative position calculation. See the [T5 paper](see https://huggingface.co/papers/1910.10683) for more details.

disable_ngram_loss (`bool`, *optional*, defaults to `False`) : Whether be trained predicting only the next first token.

eps (`float`, *optional*, defaults to 0.0) : Controls the `epsilon` parameter value for label smoothing in the loss calculation. If set to 0, no label smoothing is performed.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models).

## XLMProphetNetTokenizer[[transformers.XLMProphetNetTokenizer]]

#### transformers.XLMProphetNetTokenizer[[transformers.XLMProphetNetTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L43)

Adapted from [RobertaTokenizer](/docs/transformers/main/en/model_doc/roberta#transformers.RobertaTokenizer) and [XLNetTokenizer](/docs/transformers/main/en/model_doc/xlnet#transformers.XLNetTokenizer). Based on
[SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

build_inputs_with_special_tokenstransformers.XLMProphetNetTokenizer.build_inputs_with_special_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L296[{"name": "token_ids_0", "val": ": list"}, {"name": "token_ids_1", "val": ": typing.Optional[list[int]] = None"}]- **token_ids_0** (`list[int]`) --
  List of IDs to which the special tokens will be added
- **token_ids_1** (`list[int]`, *optional*) --
  Optional second list of IDs for sequence pairs.0`list[int]`list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A XLMProphetNet sequence has the following format:

- single sequence: `X [SEP]`
- pair of sequences: `A [SEP] B [SEP]`

**Parameters:**

vocab_file (`str`) : Path to the vocabulary file.

bos_token (`str`, *optional*, defaults to `"[SEP]"`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `"[SEP]"`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

sep_token (`str`, *optional*, defaults to `"[SEP]"`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

unk_token (`str`, *optional*, defaults to `"[UNK]"`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `"[PAD]"`) : The token used for padding, for example when batching sequences of different lengths.

cls_token (`str`, *optional*, defaults to `"[CLS]"`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

mask_token (`str`, *optional*, defaults to `"[MASK]"`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

sp_model_kwargs (`dict`, *optional*) : Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things, to set:  - `enable_sampling`: Enable subword regularization. - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.  - `nbest_size = {0,1}`: No sampling is performed. - `nbest_size > 1`: samples from the nbest_size results. - `nbest_size >> from transformers import AutoTokenizer, XLMProphetNetModel

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

**Parameters:**

config ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, hidden_size)`) -- Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **last_hidden_state_ngram** (`torch.FloatTensor` of shape `(batch_size,ngram * decoder_sequence_length, config.vocab_size)`, *optional*) -- Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
- **past_key_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
  num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_ngram_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **decoder_ngram_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, encoder_sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  encoder_sequence_length, encoder_sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

## XLMProphetNetEncoder[[transformers.XLMProphetNetEncoder]]

#### transformers.XLMProphetNetEncoder[[transformers.XLMProphetNetEncoder]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1155)

The standalone encoder part of the XLMProphetNetModel.
This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.

word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
The word embedding parameters. This can be used to initialize [XLMProphetNetEncoder](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder) with pre-defined word
embeddings instead of randomly initialized word embeddings.

forwardtransformers.XLMProphetNetEncoder.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1181[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [XLMProphetNetEncoder](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, XLMProphetNetEncoder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetEncoder.from_pretrained("patrickvonplaten/prophetnet-large-uncased-standalone")
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

**Parameters:**

config ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## XLMProphetNetDecoder[[transformers.XLMProphetNetDecoder]]

#### transformers.XLMProphetNetDecoder[[transformers.XLMProphetNetDecoder]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1269)

The standalone decoder part of the XLMProphetNetModel.
This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.

word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
The word embedding parameters. This can be used to initialize [XLMProphetNetEncoder](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder) with pre-defined word
embeddings instead of randomly initialized word embeddings.

forwardtransformers.XLMProphetNetDecoder.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1302[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_hidden_states", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

- **encoder_hidden_states**  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
  the model is configured as a decoder.
- **encoder_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

- **past_key_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) --
  Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.0`transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, hidden_size)`) -- Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **last_hidden_state_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) -- Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
- **past_key_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
  num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
- **ngram_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **ngram_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the
The [XLMProphetNetDecoder](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetDecoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, XLMProphetNetDecoder
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
>>> model = XLMProphetNetDecoder.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone", add_cross_attention=False)
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

**Parameters:**

config ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, hidden_size)`) -- Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **last_hidden_state_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) -- Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
- **past_key_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
  num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
- **ngram_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **ngram_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the

## XLMProphetNetForConditionalGeneration[[transformers.XLMProphetNetForConditionalGeneration]]

#### transformers.XLMProphetNetForConditionalGeneration[[transformers.XLMProphetNetForConditionalGeneration]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1718)

The XLMProphetNet Model with a language modeling head. Can be used for sequence generation tasks.
This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.

forwardtransformers.XLMProphetNetForConditionalGeneration.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1735[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "decoder_input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.BoolTensor] = None"}, {"name": "encoder_outputs", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "decoder_inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  XLMProphetNet uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).

- **decoder_attention_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.
- **encoder_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) --
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
- **past_key_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) --
  Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
  config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
  labels in `[0, ..., config.vocab_size]`0`transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqLMOutput` or `tuple(torch.FloatTensor)`A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, config.vocab_size)`) -- Prediction scores of the main stream language modeling head (scores for each vocabulary token before
  SoftMax).
- **logits_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) -- Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
  SoftMax).
- **past_key_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
  num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_ngram_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **decoder_ngram_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, encoder_sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  encoder_sequence_length, encoder_sequence_length)`. Attentions weights of the encoder, after the attention
  softmax, used to compute the weighted average in the self-attention heads.
The [XLMProphetNetForConditionalGeneration](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
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

**Parameters:**

config ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqLMOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, config.vocab_size)`) -- Prediction scores of the main stream language modeling head (scores for each vocabulary token before
  SoftMax).
- **logits_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) -- Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
  SoftMax).
- **past_key_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
  num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_ngram_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **decoder_ngram_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, encoder_sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  encoder_sequence_length, encoder_sequence_length)`. Attentions weights of the encoder, after the attention
  softmax, used to compute the weighted average in the self-attention heads.

## XLMProphetNetForCausalLM[[transformers.XLMProphetNetForCausalLM]]

#### transformers.XLMProphetNetForCausalLM[[transformers.XLMProphetNetForCausalLM]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1912)

The standalone decoder part of the XLMProphetNetModel with a lm head on top. The model can be used for causal language modeling.
This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

Original ProphetNet code can be found [here](https://github.com/microsoft/ProphetNet). Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file `convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.

forwardtransformers.XLMProphetNetForCausalLM.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1945[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_hidden_states", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

- **encoder_hidden_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
  the model is configured as a decoder.
- **encoder_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

- **past_key_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) --
  Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
  `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
  ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`0`transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderLMOutput` or `tuple(torch.FloatTensor)`A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, config.vocab_size)`) -- Prediction scores of the main stream language modeling head (scores for each vocabulary token before
  SoftMax).
- **logits_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) -- Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
  SoftMax).
- **past_key_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
  num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
- **ngram_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **ngram_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the
The [XLMProphetNetForCausalLM](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
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

**Parameters:**

config ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderLMOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XLMProphetNetConfig](/docs/transformers/main/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, decoder_sequence_length, config.vocab_size)`) -- Prediction scores of the main stream language modeling head (scores for each vocabulary token before
  SoftMax).
- **logits_ngram** (`torch.FloatTensor` of shape `(batch_size, ngram * decoder_sequence_length, config.vocab_size)`) -- Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
  SoftMax).
- **past_key_values** (`list[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
  num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, decoder_sequence_length, hidden_size)`.

  Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
- **ngram_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, ngram * decoder_sequence_length, hidden_size)`.

  Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
  outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **ngram_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  decoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
  weighted average in the
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_attn_heads,
  encoder_sequence_length, decoder_sequence_length)`.

  Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
  compute the weighted average in the
