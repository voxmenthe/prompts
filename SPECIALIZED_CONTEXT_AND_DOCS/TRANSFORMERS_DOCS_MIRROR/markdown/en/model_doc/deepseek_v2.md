*This model was released on 2024-05-07 and added to Hugging Face Transformers on 2025-07-09.*

# DeepSeek-V2

## Overview

The DeepSeek-V2 model was proposed in [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://huggingface.co/papers/2405.04434) by DeepSeek-AI Team.

The abstract from the paper is the following:
We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector, while DeepSeekMoE enables training strong models at an economical cost through sparse computation. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. We pretrain DeepSeek-V2 on a high-quality and multi-source corpus consisting of 8.1T tokens, and further perform Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unlock its potential. Evaluation results show that, even with only 21B activated parameters, DeepSeek-V2 and its chat versions still achieve top-tier performance among open-source models.

This model was contributed by [VladOS95-cyber](https://github.com/VladOS95-cyber).
The original code can be found [here](https://huggingface.co/deepseek-ai/DeepSeek-V2).

### Usage tips

The model uses Multi-head Latent Attention (MLA) and DeepSeekMoE architectures for efficient inference and cost-effective training. It employs an auxiliary-loss-free strategy for load balancing and multi-token prediction training objective. The model can be used for various language tasks after being pre-trained on 14.8 trillion tokens and going through Supervised Fine-Tuning and Reinforcement Learning stages.

## DeepseekV2Config

### class transformers.DeepseekV2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/configuration_deepseek_v2.py#L27)

( vocab\_size = 32000 hidden\_size = 4096 intermediate\_size = 11008 num\_hidden\_layers = 32 num\_attention\_heads = 32 num\_key\_value\_heads = None hidden\_act = 'silu' max\_position\_embeddings = 2048 initializer\_range = 0.02 rms\_norm\_eps = 1e-06 use\_cache = True pad\_token\_id = None bos\_token\_id = 1 eos\_token\_id = 2 tie\_word\_embeddings = False rope\_theta = 10000.0 rope\_scaling = None attention\_bias = False attention\_dropout = 0.0 mlp\_bias = False aux\_loss\_alpha = 0.001 first\_k\_dense\_replace = 0 kv\_lora\_rank = 512 q\_lora\_rank = 1536 n\_group = None n\_routed\_experts = 64 n\_shared\_experts = 2 qk\_nope\_head\_dim = 128 qk\_rope\_head\_dim = 64 routed\_scaling\_factor = 1.0 seq\_aux = True topk\_group = None topk\_method = 'greedy' v\_head\_dim = 128 num\_experts\_per\_tok = None norm\_topk\_prob = False moe\_intermediate\_size = 1407 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32000) —
  Vocabulary size of the DeepSeek model. Defines the number of different tokens that can be represented by the
  `input_ids` passed when calling [DeepseekV2Model](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Model).
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 11008) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 32) —
  Number of hidden layers in the Transformer decoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **num\_key\_value\_heads** (`int`, *optional*) —
  The number of key-value heads used to implement Grouped Query Attention (GQA). If
  `num_key_value_heads=num_attention_heads`, the model will use Multi-Head Attention (MHA). If
  `num_key_value_heads=1`, the model will use Multi-Query Attention (MQA). Otherwise, GQA is used.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 2048) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated normal initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon value used by the RMS normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/value attentions (useful for inference optimization).
* **pad\_token\_id** (`int`, *optional*) —
  Padding token ID.
* **bos\_token\_id** (`int`, *optional*, defaults to 1) —
  Beginning-of-sequence token ID.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  End-of-sequence token ID.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie input and output embeddings.
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the Rotary Position Embeddings (RoPE).
* **rope\_scaling** (`Dict`, *optional*) —
  Configuration for scaling RoPE embeddings. Supports `linear` and `dynamic` scaling strategies.
* **attention\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value, and output projection layers during self-attention.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability applied to attention weights.
* **mlp\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias term in the MLP layers.
* **aux\_loss\_alpha** (`float`, *optional*, defaults to 0.001) —
  Weight coefficient for auxiliary loss in Mixture of Experts (MoE) models.
* **first\_k\_dense\_replace** (`int`, *optional*, defaults to 0) —
  Number of dense layers in the shallow layers before switching to MoE layers.
* **kv\_lora\_rank** (`int`, *optional*, defaults to 512) —
  Rank of the LoRA decomposition for key-value projections.
* **q\_lora\_rank** (`int`, *optional*, defaults to 1536) —
  Rank of the LoRA decomposition for query projections.
  Specifically, it determines the dimensionality to which the query (q) vectors are compressed before being expanded back to their original size.
  It reduces computational overhead while maintaining model performance.
* **n\_group** (`int`, *optional*) —
  Number of groups for routed experts.
* **n\_routed\_experts** (`int`, *optional*, defaults to 64) —
  Number of routed experts (None indicates a dense model).
* **n\_shared\_experts** (`int`, *optional*, defaults to 2) —
  Number of shared experts (None indicates a dense model).
* **qk\_nope\_head\_dim** (`int`, *optional*, defaults to 128) —
  The head dimension for the QK (query-key) projections when using NOPE (Neural Operator Position Encoding).
* **qk\_rope\_head\_dim** (`int`, *optional*, defaults to 64) —
  The head dimension for QK projections when using RoPE.
* **routed\_scaling\_factor** (`float`, *optional*, defaults to 1.0) —
  Scaling factor for routed experts in MoE models.
* **seq\_aux** (`bool`, *optional*, defaults to `True`) —
  Whether to compute the auxiliary loss for each individual sequence.
* **topk\_group** (`int`, *optional*) —
  Number of selected groups per token for expert selection.
* **topk\_method** (`str`, *optional*, defaults to `"greedy"`) —
  The method used for selecting top-k experts in the routed gate mechanism.
* **v\_head\_dim** (`int`, *optional*, defaults to 128) —
  The dimension of value projections in the attention layers.
* **num\_experts\_per\_tok** (`int`, *optional*) —
  The number of experts selected per token. If `None`, the model behaves as a dense Transformer.
* **norm\_topk\_prob** (`bool`, *optional*, defaults to `False`) —
  Whether to normalize the probability distribution over top-k selected experts.
* **moe\_intermediate\_size** (`int`, *optional*, defaults to 1407) —
  Dimension of the MoE (Mixture of Experts) representations.

This is the configuration class to store the configuration of a [DeepseekV2Model](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Model). It is used to instantiate a DeepSeek
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of DeepSeek-V2-Lite” [deepseek-ai/DeepSeek-V2-Lite”](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite%22).
Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import DeepseekV2Model, DeepseekV2Config
>>> # Initializing a DeepSeek-V2 style configuration
>>> configuration = DeepseekV2Config()
>>> # Accessing the model configuration
>>> model = DeepseekV2Model(configuration)
>>> print(model.config)
```

## DeepseekV2Model

### class transformers.DeepseekV2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L477)

( config: DeepseekV2Config  )

Parameters

* **config** ([DeepseekV2Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Deepseek V2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L494)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DeepseekV2Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DeepseekV2Model](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## DeepseekV2ForCausalLM

### class transformers.DeepseekV2ForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L556)

( config  )

Parameters

* **config** ([DeepseekV2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2ForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Deepseek V2 Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L570)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DeepseekV2Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DeepseekV2ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2ForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, DeepseekV2ForCausalLM

>>> model = DeepseekV2ForCausalLM.from_pretrained("meta-deepseek_v2/DeepseekV2-2-7b-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-deepseek_v2/DeepseekV2-2-7b-hf")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

## DeepseekV2ForSequenceClassification

### class transformers.DeepseekV2ForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L631)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L111)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

`transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The `GenericForSequenceClassification` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v2.md)
