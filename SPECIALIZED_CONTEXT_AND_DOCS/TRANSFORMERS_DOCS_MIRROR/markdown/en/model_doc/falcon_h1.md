*This model was released on 2025-05-21 and added to Hugging Face Transformers on 2025-05-21.*

# FalconH1

## Overview

The [FalconH1](https://huggingface.co/blog/tiiuae/falcon-h1) model was developed by the TII Pretraining team. A comprehensive research paper covering the architecture, pretraining dynamics, experimental results, and conclusions is forthcoming. You can read more about this series in [this website](https://github.com/tiiuae/Falcon-H1).

## Contributors

This model was contributed by [DhiyaEddine](https://huggingface.co/DhiyaEddine), [ybelkada](https://huggingface.co/ybelkada), [JingweiZuo](https://huggingface.co/JingweiZuo), [IlyasChahed](https://huggingface.co/IChahed), and [MaksimVelikanov](https://huggingface.co/yellowvm).
The original code can be found [here](https://github.com/tiiuae/Falcon-H1).

## FalconH1Config

| Model | Depth | Dim | Attn Heads | KV | Mamba Heads | d\_head | d\_state | Ctx Len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 0.5B | 36 | 1024 | 8 | 2 | 24 | 64 / 64 | 128 | 4K, 16K-SFT |
| H1 1.5B | 24 | 2048 | 8 | 2 | 48 | 128 / 64 | 256 | 128K |
| H1 1.5B-d | 66 | 1280 | 6 | 2 | 24 | 128 / 64 | 256 | 128K |
| H1 3B | 32 | 2560 | 10 | 2 | 32 | 128 / 128 | 256 | 128K |
| H1 7B | 44 | 3072 | 12 | 2 | 24 | 128 / 128 | 256 | 256K |
| H1 34B | 72 | 5120 | 20 | 4 | 32 | 128 / 128 | 256 | 256K |

### class transformers.FalconH1Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_h1/configuration_falcon_h1.py#L24)

( vocab\_size = 128000 tie\_word\_embeddings = False hidden\_size = 4096 intermediate\_size = 14336 num\_hidden\_layers = 32 num\_attention\_heads = 32 num\_key\_value\_heads = 8 hidden\_act = 'silu' initializer\_range = 0.02 rms\_norm\_eps = 1e-05 use\_cache = True num\_logits\_to\_keep = 1 pad\_token\_id = 0 bos\_token\_id = 1 eos\_token\_id = 2 max\_position\_embeddings = 8192 attention\_dropout = 0.0 mamba\_d\_ssm = 1024 mamba\_n\_heads = 128 mamba\_d\_head = 'auto' mamba\_n\_groups = 1 mamba\_d\_state = 256 mamba\_d\_conv = 4 mamba\_expand = 2 mamba\_chunk\_size = 256 mamba\_conv\_bias = True mamba\_proj\_bias = False mamba\_norm\_before\_gate = True mamba\_rms\_norm = False projectors\_bias = False rope\_theta = 100000.0 rope\_scaling = None lm\_head\_multiplier = 1.0 embedding\_multiplier = 1.0 mlp\_multipliers = None key\_multiplier = None attention\_out\_multiplier = None attention\_in\_multiplier = None ssm\_multipliers = None ssm\_in\_multiplier = None ssm\_out\_multiplier = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 128000) —
  Vocabulary size of the FalconH1 model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [FalconH1Model](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1Model)
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether the model’s input and output word embeddings should be tied. Note that this is only relevant if the
  model has a output word embedding layer.
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 14336) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 32) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 8) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **num\_logits\_to\_keep** (`int` or `None`, *optional*, defaults to 1) —
  Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
  integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
  logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
  sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
  significantly.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the padding token.
* **bos\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the “beginning-of-sequence” token.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  The id of the “end-of-sequence” token.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 8192) —
  Max cached sequence length for the model
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **mamba\_d\_ssm** (`int`, *optional*, defaults to 1024) —
  The dimension of the SSM state space latents.
* **mamba\_n\_heads** (`int`, *optional*, defaults to 128) —
  The number of mamba heads used in the v2 implementation.
* **mamba\_d\_head** (`int`, *optional*, defaults to `"auto"`) —
  Head embeddding dimension size
* **mamba\_n\_groups** (`int`, *optional*, defaults to 1) —
  The number of the mamba groups used in the v2 implementation.
* **mamba\_d\_state** (`int`, *optional*, defaults to 256) —
  The dimension the mamba state space latents
* **mamba\_d\_conv** (`int`, *optional*, defaults to 4) —
  The size of the mamba convolution kernel
* **mamba\_expand** (`int`, *optional*, defaults to 2) —
  Expanding factor (relative to hidden\_size) used to determine the mamba intermediate size
* **mamba\_chunk\_size** (`int`, *optional*, defaults to 256) —
  The chunks in which to break the sequence when doing prefill/training
* **mamba\_conv\_bias** (`bool`, *optional*, defaults to `True`) —
  Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
* **mamba\_proj\_bias** (`bool`, *optional*, defaults to `False`) —
  Flag indicating whether or not to use bias in the input and output projections ([“in\_proj”, “out\_proj”]) of the mamba mixer block
* **mamba\_norm\_before\_gate** (`bool`, *optional*, defaults to `True`) —
  Whether to use RMSNorm before the gate in the Mamba block
* **mamba\_rms\_norm** (`bool`, *optional*, defaults to `False`) —
  Whether to use RMSNorm instead of LayerNorm in the Mamba block
* **projectors\_bias** (`bool`, *optional*, defaults to `False`) —
  Flag indicating whether or not to use bias in the input and output projections ([“in\_proj”, “out\_proj”]) of the attention block
* **rope\_theta** (`float`, *optional*, defaults to 100000.0) —
  The theta value used for the RoPE embeddings.
* **rope\_scaling** (`float`, *optional*) —
  The scaling value used for the RoPE embeddings. If `None`, no scaling is applied.
* **lm\_head\_multiplier** (`float`, *optional*, defaults to 1.0) —
  The multiplier for the LM head. This is used to scale the output of the LM head.
* **embedding\_multiplier** (`float`, *optional*, defaults to 1.0) —
  The multiplier for the embedding layer. This is used to scale the output of the embedding layer.
* **mlp\_multipliers** (`list[float]`, *optional*) —
  The multipliers for the MLP layers. This is used to scale the output of the MLP layers. The first value is
  the multiplier of gate layer, the second value is the multiplier of the down\_proj layer.
* **key\_multiplier** (`float`, *optional*) —
  The multiplier for the key layer. This is used to scale the output of the key layer.
* **attention\_out\_multiplier** (`float`, *optional*) —
  The multiplier for the attention output layer. This is used to scale the output of the attention output
* **attention\_in\_multiplier** (`float`, *optional*) —
  The multiplier for the attention input layer. This is used to scale the output of the attention input layer.
* **ssm\_multipliers** (`list[float]`, *optional*) —
  The multipliers for the SSM layers. This is used to scale the output of the SSM layers.
* **ssm\_in\_multiplier** (`float`, *optional*) —
  The multiplier for the SSM input layer. This is used to scale the output of the SSM input layer.
* **ssm\_out\_multiplier** (`float`, *optional*) —
  The multiplier for the SSM output layer. This is used to scale the output of the SSM output layer.

This is the configuration class to store the configuration of a [FalconH1Model](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1Model). It is used to instantiate a
FalconH1Model model according to the specified arguments, defining the model architecture. Instantiating a configuration
with defaults taken from [ibm-fms/FalconH1-9.8b-2.2T-hf](https://huggingface.co/ibm-fms/FalconH1-9.8b-2.2T-hf).
The FalconH1Model is a hybrid [mamba2](https://github.com/state-spaces/mamba) architecture with SwiGLU.
The checkpoints are jointly trained by IBM, Princeton, and UIUC.
Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## FalconH1ForCausalLM


```
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon-H1-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon-H1-7B-Instruct")

message = ["Mamba is a snake with following properties  "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

### class transformers.FalconH1ForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_h1/modeling_falcon_h1.py#L1467)

( config  )

Parameters

* **config** ([FalconH1ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1ForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Falcon H1 Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon_h1/modeling_falcon_h1.py#L1481)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.models.falcon\_h1.modeling\_falcon\_h1.FalconHybridMambaAttentionDynamicCache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
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
* **past\_key\_values** (`~models.falcon_h1.modeling_falcon_h1.FalconHybridMambaAttentionDynamicCache`, *optional*) —
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
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
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
elements depending on the configuration ([FalconH1Config](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1Config)) and inputs.

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

The [FalconH1ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/falcon_h1#transformers.FalconH1ForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, FalconH1ForCausalLM

>>> model = FalconH1ForCausalLM.from_pretrained("...")
>>> tokenizer = AutoTokenizer.from_pretrained("...")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

This HF implementation is contributed by [younesbelkada](https://github.com/younesbelkada) and [DhiaEddineRhaiem](https://github.com/dhiaEddineRhaiem).

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/falcon_h1.md)
