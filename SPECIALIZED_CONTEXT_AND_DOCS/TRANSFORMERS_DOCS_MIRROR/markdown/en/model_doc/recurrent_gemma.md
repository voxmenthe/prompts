*This model was released on 2024-04-11 and added to Hugging Face Transformers on 2024-04-10.*

# RecurrentGemma

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Recurrent Gemma model was proposed in [RecurrentGemma: Moving Past Transformers for Efficient Open Language Models](https://huggingface.co/papers/2404.07839) by the Griffin, RLHF and Gemma Teams of Google.

The abstract from the paper is the following:

*We introduce RecurrentGemma, an open language model which uses Google’s novel Griffin architecture. Griffin combines linear recurrences with local attention to achieve excellent performance on language. It has a fixed-sized state, which reduces memory use and enables efficient inference on long sequences. We provide a pre-trained model with 2B non-embedding parameters, and an instruction tuned variant. Both models achieve comparable performance to Gemma-2B despite being trained on fewer tokens.*

Tips:

* The original checkpoints can be converted using the conversion script [`src/transformers/models/recurrent_gemma/convert_recurrent_gemma_weights_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/recurrent_gemma/convert_recurrent_gemma_to_hf.py).

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ). The original code can be found [here](https://github.com/google-deepmind/recurrentgemma).

## RecurrentGemmaConfig

### class transformers.RecurrentGemmaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/recurrent_gemma/configuration_recurrent_gemma.py#L24)

( num\_hidden\_layers = 26 vocab\_size = 256000 hidden\_size = 2560 intermediate\_size = 7680 num\_attention\_heads = 10 lru\_width = None attention\_window\_size = 2048 conv1d\_width = 4 logits\_soft\_cap = 30.0 rms\_norm\_eps = 1e-06 use\_cache = True pad\_token\_id = 0 eos\_token\_id = 1 bos\_token\_id = 2 hidden\_activation = 'gelu\_pytorch\_tanh' partial\_rotary\_factor = 0.5 rope\_theta = 10000.0 block\_types = ('recurrent', 'recurrent', 'attention') attention\_dropout = 0.0 num\_key\_value\_heads = None attention\_bias = False w\_init\_variance\_scale = 0.01 \*\*kwargs  )

Parameters

* **num\_hidden\_layers** (`int`, *optional*, defaults to 26) —
  The number of hidden layers in the model.
* **vocab\_size** (`int`, *optional*, defaults to 256000) —
  Vocabulary size of the RecurrentGemma model. Defines the number of
  different tokens that can be represented by the
  `inputs_ids` passed when calling [RecurrentGemmaModel](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaModel)
* **hidden\_size** (`int`, *optional*, defaults to 2560) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 7680) —
  Dimension of the MLP representations.
* **num\_attention\_heads** (`int`, *optional*, defaults to 10) —
  The number of heads for the attention block and the number of
  heads/blocks for the block-diagonal layers used in the RG-LRU gates.
  This number must divide `hidden_size` and `lru_width`.
* **lru\_width** (`int` or `None`, *optional*) —
  Dimension of the hidden representations of the RG-LRU. If `None`
  this will be set to `hidden_size`.
  Whether to scale the output of the embeddings by `sqrt(hidden_size)`.
* **attention\_window\_size** (`int`, *optional*, defaults to 2048) —
  The size of the attention window used in the attention block.
* **conv1d\_width** (`int`, *optional*, defaults to 4) —
  The kernel size of conv1d layers used in the recurrent blocks.
* **logits\_soft\_cap** (`float`, *optional*, defaults to 30.0) —
  The value at which the logits should be soft-capped to after the transformer and LM-head computation in the Causal LM architecture.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether the model should return the last key/values
  attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  Padding token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 1) —
  End of stream token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 2) —
  Beginning of stream token id.
* **hidden\_activation** (`` str` or `function ``, *optional*, defaults to `"gelu_pytorch_tanh"`) —
  The hidden activation used in the recurrent block as well as the MLP layer of the decoder layers.
* **partial\_rotary\_factor** (`float`, *optional*, defaults to 0.5) —
  The partial rotary factor used in the initialization of the rotary embeddings.
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **block\_types** (`list[str]`, *optional*, defaults to `('recurrent', 'recurrent', 'attention')`) —
  List of aleternating blocks that will be repeated to initialize the `temporal_block` layer.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) — dropout value to use after the attention softmax.
* **num\_key\_value\_heads** (`16`, *optional*, defaults to 16) — Number of key value heads to use GQA.
* **attention\_bias** (`bool`, *optional*, defaults to `False`) — whether or not the linear q,k,v of the Attention layer should have bias
* **w\_init\_variance\_scale** (`float`, *optional*, defaults to 0.01) — weight initialization variance.

This is the configuration class to store the configuration of a [RecurrentGemmaModel](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaModel). It is used to instantiate a RecurrentGemma
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the RecurrentGemma-7B.

e.g. [google/recurrentgemma-2b](https://huggingface.co/google/recurrentgemma-2b)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import RecurrentGemmaModel, RecurrentGemmaConfig

>>> # Initializing a RecurrentGemma recurrentgemma-2b style configuration
>>> configuration = RecurrentGemmaConfig()

>>> # Initializing a model from the recurrentgemma-2b style configuration
>>> model = RecurrentGemmaModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## RecurrentGemmaModel

### class transformers.RecurrentGemmaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py#L573)

( config: RecurrentGemmaConfig  )

Parameters

* **config** ([RecurrentGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Recurrent Gemma Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py#L592)

( input\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None cache\_position: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.BaseModelOutputWithNoAttention` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.BaseModelOutputWithNoAttention` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RecurrentGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [RecurrentGemmaModel](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## RecurrentGemmaForCausalLM

### class transformers.RecurrentGemmaForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py#L690)

( config  )

Parameters

* **config** ([RecurrentGemmaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Recurrent Gemma Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py#L702)

( input\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None use\_cache: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

[transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RecurrentGemmaConfig](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [RecurrentGemmaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/recurrent_gemma#transformers.RecurrentGemmaForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, RecurrentGemmaForCausalLM

>>> model = RecurrentGemmaForCausalLM.from_pretrained("google/recurrentgemma-2b")
>>> tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b")

>>> prompt = "What is your favorite condiment?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"What is your favorite condiment?"
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/recurrent_gemma.md)
