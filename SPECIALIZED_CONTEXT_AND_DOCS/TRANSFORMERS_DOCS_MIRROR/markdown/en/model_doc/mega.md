*This model was released on 2022-09-21 and added to Hugging Face Transformers on 2023-06-20.*

# MEGA

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

This model is in maintenance mode only, we don’t accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

## Overview

The MEGA model was proposed in [Mega: Moving Average Equipped Gated Attention](https://huggingface.co/papers/2209.10655) by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, and Luke Zettlemoyer.
MEGA proposes a new approach to self-attention with each encoder layer having a multi-headed exponential moving average in addition to a single head of standard dot-product attention, giving the attention mechanism
stronger positional biases. This allows MEGA to perform competitively to Transformers on standard benchmarks including LRA
while also having significantly fewer parameters. MEGA’s compute efficiency allows it to scale to very long sequences, making it an
attractive option for long-document NLP tasks.

The abstract from the paper is the following:

*The design choices in the Transformer attention mechanism, including weak inductive bias and quadratic computational complexity, have limited its application for modeling long sequences. In this paper, we introduce Mega, a simple, theoretically grounded, single-head gated attention mechanism equipped with (exponential) moving average to incorporate inductive bias of position-aware local dependencies into the position-agnostic attention mechanism. We further propose a variant of Mega that offers linear time and space complexity yet yields only minimal quality loss, by efficiently splitting the whole sequence into multiple chunks with fixed length. Extensive experiments on a wide range of sequence modeling benchmarks, including the Long Range Arena, neural machine translation, auto-regressive language modeling, and image and speech classification, show that Mega achieves significant improvements over other sequence models, including variants of Transformers and recent state space models.*

This model was contributed by [mnaylor](https://huggingface.co/mnaylor).
The original code can be found [here](https://github.com/facebookresearch/mega).

## Usage tips

* MEGA can perform quite well with relatively few parameters. See Appendix D in the MEGA paper for examples of architectural specs which perform well in various settings. If using MEGA as a decoder, be sure to set `bidirectional=False` to avoid errors with default bidirectional.
* Mega-chunk is a variant of mega that reduces time and spaces complexity from quadratic to linear. Utilize chunking with MegaConfig.use\_chunking and control chunk size with MegaConfig.chunk\_size

## Implementation Notes

* The original implementation of MEGA had an inconsistent expectation of attention masks for padding and causal self-attention between the softmax attention and Laplace/squared ReLU method. This implementation addresses that inconsistency.
* The original implementation did not include token type embeddings; this implementation adds support for these, with the option controlled by MegaConfig.add\_token\_type\_embeddings

## MegaConfig

### class transformers.MegaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/configuration_mega.py#L28)

( vocab\_size = 30522 hidden\_size = 128 num\_hidden\_layers = 4 intermediate\_size = 256 ema\_projection\_size = 16 bidirectional = True shared\_representation\_size = 64 use\_chunking = False chunk\_size = -1 truncation = None normalize\_before\_mega = True normalization\_type = 'scalenorm' norm\_affine = True activation = 'silu' attention\_activation = 'softmax' dropout\_prob = 0.1 hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 use\_feature\_dropout = False use\_normalized\_ffn = True nffn\_hidden\_size = 256 normalize\_before\_ffn = True nffn\_activation\_dropout\_prob = 0.1 max\_positions = 2048 add\_token\_type\_embeddings = False type\_vocab\_size = 2 initializer\_range = 0.02 ema\_delta\_alpha\_range = 0.2 ema\_beta\_range = 0.02 ema\_gamma\_omega\_range = 1.0 pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 relative\_positional\_bias = 'rotary' classifier\_dropout = None use\_cache = True add\_lm\_hidden\_dense\_layer = True \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the Mega model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [MegaModel](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaModel).
* **hidden\_size** (`int`, *optional*, defaults to 128) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 4) —
  Number of hidden layers in the Mega encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the hidden size (self-attention value projection) within the Mega encoder
* **ema\_projection\_size** (`int`, *optional*, defaults to 16) —
  Dimensionality of the MegaMultiDimensionDampedEma
* **bidirectional** (`bool`, *optional*, defaults to `True`) —
  Whether the MegaMultiDimensionDampedEma used in Mega’s self-attention should work bidirectionally (`True`)
  or unidirectionally (`False`). Bidirectional EMA is incompatible with causal decoding, so this should be
  False if you intend to use the model as a decoder.
* **shared\_representation\_size** (`int`, *optional*, defaults to 64) —
  Dimensionality of the linear projection for shared representation of self-attention queries and keys
* **use\_chunking** (`bool`, *optional*, defaults to `False`) —
  Whether to chunk inputs for linear self-attention complexity (described as Mega-chunk in the paper)
* **chunk\_size** (`int`, *optional*, defaults to -1) —
  If `use_chunking` is set to `True`, determines the size of the chunks to apply to the input sequence. If
  chunking is used, input sequences must be padded to a multiple of `chunk_size`
* **truncation** (`int`, *optional*) —
  If specified, the sequence length for which to truncate MegaMultiDimensionDampedEma
* **normalize\_before\_mega** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize before (`True`) or after (`False`) passing through Mega encoder blocks
* **normalization\_type** (`str`, *optional*, defaults to `"scalenorm"`) —
  Type of normalization to use in Mega encoder blocks. Choose one of `"scalenorm"`, `"layernorm"`,
  `"rmsnorm"`, `"batchnorm"`, or `"syncbatchnorm"` (GPU required for syncbatchnorm)
* **norm\_affine** (`bool`, *optional*, defaults to `True`) —
  If `True`, applies a parameterized affine transformation to inputs during normalization
* **activation** (`str`, *optional*, defaults to `"silu"`) —
  Activation function to apply within Mega encoder blocks. Choose one of `"silu"`, `"relu"`, `"linear"`,
  `"gelu"`, or `"gelu_accurate"`
* **attention\_activation** (`str`, *optional*, defaults to `"softmax"`) —
  Activation function to apply for single-headed self-attention (a la Transformer). Choose one of
  `"softmax"`, `"laplace"`, or `"relu2"`
* **dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for EMA self-attention
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **use\_feature\_dropout** (`bool`, *optional*, defaults to `False`) —
  Whether to use feature-based (`True`) or standard dropout (`False`)
* **use\_normalized\_ffn** (`bool`, *optional*, defaults to `True`) —
  Whether to use the normalized feed-forward sub-layer in Mega blocks (`True`) or pass Mega encoder output
  as-is (`False`)
* **nffn\_hidden\_size** (`int`, *optional*, defaults to 256) —
  If using the normalized feed-forward network (NFFN) layer within Mega (`use_normalized_ffn = True`), this
  is the hidden size of the NFFN
* **normalize\_before\_ffn** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize before (`True`) or after (`False`) the feed-forward portion of NFFN
* **nffn\_activation\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the NFFN component.
* **max\_positions** (`int`, *optional*, defaults to 2048) —
  The maximum sequence length to use for positional representations. For `"simple"` relative positional bias,
  this is a hard limit on input length; `"rotary"` relative positional bias will extrapolate to longer
  sequences
* **add\_token\_type\_embeddings** (`bool`, *optional*, defaults to `True`) —
  Whether to account for token types in embeddings. Left as optional to maintain compatibility with original
  implementation while adding support for token types.
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the `token_type_ids` passed when calling [MegaModel](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaModel). Only used if
  `add_token_type_embeddings = True`
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **ema\_delta\_alpha\_range** (`float`, *optional*, defaults to 0.2) —
  The standard deviation for initializing the delta (damping factor) and alpha (decay factor) parameters in
  MegaMultiDimensionDampedEma.
* **ema\_beta\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation for initializing the beta parameter (expansion matrix) in
  MegaMultiDimensionDampedEma.
* **ema\_gamma\_omega\_range** (`float`, *optional*, defaults to 1.0) —
  The standard deviation for initializing the gamma (projection matrix) and omega (residual weight)
  parameters in MultiDimensionEMA.
* **relative\_positional\_bias** (`str`, *optional*, defaults to `"rotary"`) —
  Type of relative positional encoding. Choose one of `"rotary"` or `"simple"`. If `"simple"` is selected,
  `max_positions` is used as a limit on input size, while `"rotary"` extrapolates beyond `max_positions`.
* **is\_decoder** (`bool`, *optional*, defaults to `False`) —
  Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **classifier\_dropout** (`float`, *optional*) —
  The dropout ratio for the classification head.
* **add\_lm\_hidden\_dense\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to include a hidden layer for projection between encoder outputs and LM heads (`True`) or pass
  hidden states directly to LM head (`False`). Remains optional for compatibility with original
  implementation

This is the configuration class to store the configuration of a [MegaModel](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaModel). It is used to instantiate a Mega
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Mega
[mnaylor/mega-base-wikitext](https://huggingface.co/mnaylor/mega-base-wikitext) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import MegaConfig, MegaModel

>>> # Initializing a Mega configuration
>>> configuration = MegaConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = MegaModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MegaModel

### class transformers.MegaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1444)

( config: MegaConfig add\_pooling\_layer = True  )

Parameters

* **config** ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) — Model configuration class with all the parameters of the
  model. Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare MEGA Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added after self-attention, following the architecture described in *Mega: Moving Average
Equipped Gated Attention*\_ by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig,
Jonathan May, and Luke Zettlemoyer

To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to
`True` and `bidirectional` set to `False`. To be used in a Seq2Seq model, the model needs to initialized with both
`is_decoder=True` and `bidirectional=False` argument as well as `add_cross_attention` set to `True`; an
`encoder_hidden_states` is then expected as an input to the forward pass.

.. \_*Mega: Moving Average Equipped Gated Attention*: <https://huggingface.co/papers/2209.10655>

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1479)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None encoder\_hidden\_states: typing.Optional[torch.Tensor] = None encoder\_attention\_mask: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

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
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    This parameter can only be used when the model is initialized with `add_token_type_embeddings` parameter
    set to `True`. All the value in this tensor should be always < config.type\_vocab\_size.

  [What are token type IDs?](../glossary#token-type-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
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
* **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
  the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) —
  Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.

The [MegaModel](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, MegaModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
>>> model = MegaModel.from_pretrained("mnaylor/mega-base-wikitext")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## MegaForCausalLM

### class transformers.MegaForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1644)

( config: MegaConfig  )

Parameters

* **config** ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) — Model configuration class with all the parameters of the
  model. Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MEGA Model with a `language modeling` head on top for CLM fine-tuning.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1667)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None encoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

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
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    This parameter can only be used when the model is initialized with `add_token_type_embeddings` parameter
    set to `True`. All the value in this tensor should be always < config.type\_vocab\_size.

  [What are token type IDs?](../glossary#token-type-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
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
* **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
  the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
  `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
  ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) —
  Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
  don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
  `decoder_input_ids` of shape `(batch_size, sequence_length)`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

[transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Cross attentions weights after the attention softmax, used to compute the weighted average in the
  cross-attention heads.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.

The [MegaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, MegaForCausalLM, AutoConfig
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
>>> config = AutoConfig.from_pretrained("mnaylor/mega-base-wikitext")
>>> config.is_decoder = True
>>> config.bidirectional = False
>>> model = MegaForCausalLM.from_pretrained(
...     "mnaylor/mega-base-wikitext", config=config, ignore_mismatched_sizes=True
... )

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> prediction_logits = outputs.logits
```

## MegaForMaskedLM

### class transformers.MegaForMaskedLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1799)

( config: MegaConfig  )

Parameters

* **config** ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) — Model configuration class with all the parameters of the
  model. Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MEGA Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1830)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None encoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

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
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    This parameter can only be used when the model is initialized with `add_token_type_embeddings` parameter
    set to `True`. All the value in this tensor should be always < config.type\_vocab\_size.

  [What are token type IDs?](../glossary#token-type-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **kwargs** (`dict[str, any]`, optional, defaults to *{}*) —
  Used to hide legacy arguments that have been deprecated.

Returns

[transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MegaForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, MegaForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
>>> model = MegaForMaskedLM.from_pretrained("mnaylor/mega-base-wikitext")

>>> inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # retrieve index of <mask>
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
>>> tokenizer.decode(predicted_token_id)
' Paris'

>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> # mask labels of non-<mask> tokens
>>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

>>> outputs = model(**inputs, labels=labels)
>>> round(outputs.loss.item(), 2)
0.1
```

## MegaForSequenceClassification

### class transformers.MegaForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1903)

( config  )

Parameters

* **config** ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) — Model configuration class with all the parameters of the
  model. Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MEGA Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1915)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

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
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    This parameter can only be used when the model is initialized with `add_token_type_embeddings` parameter
    set to `True`. All the value in this tensor should be always < config.type\_vocab\_size.

  [What are token type IDs?](../glossary#token-type-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
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
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MegaForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, MegaForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
>>> model = MegaForSequenceClassification.from_pretrained("mnaylor/mega-base-wikitext")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = MegaForSequenceClassification.from_pretrained("mnaylor/mega-base-wikitext", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, MegaForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
>>> model = MegaForSequenceClassification.from_pretrained("mnaylor/mega-base-wikitext", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = MegaForSequenceClassification.from_pretrained(
...     "mnaylor/mega-base-wikitext", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## MegaForMultipleChoice

### class transformers.MegaForMultipleChoice

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1994)

( config  )

Parameters

* **config** ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) — Model configuration class with all the parameters of the
  model. Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MEGA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L2005)

( input\_ids: typing.Optional[torch.LongTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    This parameter can only be used when the model is initialized with `add_token_type_embeddings` parameter
    set to `True`. All the value in this tensor should be always < config.type\_vocab\_size.

  [What are token type IDs?](../glossary#token-type-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*) —
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
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
  `input_ids` above)

Returns

[transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — *num\_choices* is the second dimension of the input tensors. (see *input\_ids* above).

  Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MegaForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, MegaForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
>>> model = MegaForMultipleChoice.from_pretrained("mnaylor/mega-base-wikitext")

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

## MegaForTokenClassification

### class transformers.MegaForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L2079)

( config  )

Parameters

* **config** ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) — Model configuration class with all the parameters of the
  model. Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MEGA Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L2094)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

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
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    This parameter can only be used when the model is initialized with `add_token_type_embeddings` parameter
    set to `True`. All the value in this tensor should be always < config.type\_vocab\_size.

  [What are token type IDs?](../glossary#token-type-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MegaForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, MegaForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
>>> model = MegaForTokenClassification.from_pretrained("mnaylor/mega-base-wikitext")

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

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
```

## MegaForQuestionAnswering

### class transformers.MegaForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L2179)

( config  )

Parameters

* **config** ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) — Model configuration class with all the parameters of the
  model. Initializing with a config file does not load the weights associated with the model, only the
  configuration. Check out the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MEGA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L2190)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None start\_positions: typing.Optional[torch.LongTensor] = None end\_positions: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

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
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    This parameter can only be used when the model is initialized with `add_token_type_embeddings` parameter
    set to `True`. All the value in this tensor should be always < config.type\_vocab\_size.

  [What are token type IDs?](../glossary#token-type-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
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
* **start\_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.

Returns

[transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MegaConfig](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MegaForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, MegaForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("mnaylor/mega-base-wikitext")
>>> model = MegaForQuestionAnswering.from_pretrained("mnaylor/mega-base-wikitext")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

>>> # target is "nice puppet"
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mega.md)
