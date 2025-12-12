# RWKV

## Overview

The RWKV model (version 4) was proposed in [this repo](https://github.com/BlinkDL/RWKV-LM)

It suggests a tweak in the traditional Transformer attention to make it linear. This way, the model can be used as recurrent network: passing inputs for timestamp 0 and timestamp 1 together is the same as passing inputs at timestamp 0, then inputs at timestamp 1 along with the state of timestamp 0 (see example below).

This can be more efficient than a regular Transformer and can deal with sentence of any length (even if the model uses a fixed context length for training).

This model was contributed by [sgugger](https://huggingface.co/sgugger).
The original code can be found [here](https://github.com/BlinkDL/RWKV-LM).

## Usage example

```py
import torch
from transformers import AutoTokenizer, RwkvConfig, RwkvModel

model = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile")
tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

inputs = tokenizer("This is an example.", return_tensors="pt")
# Feed everything to the model
outputs = model(inputs["input_ids"])
output_whole = outputs.last_hidden_state

outputs = model(inputs["input_ids"][:, :2])
output_one = outputs.last_hidden_state

# Using the state computed on the first inputs, we will get the same output
outputs = model(inputs["input_ids"][:, 2:], state=outputs.state)
output_two = outputs.last_hidden_state

torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5)
```

If you want to make sure the model stops generating when `'\n\n'` is detected, we recommend using the following stopping criteria:

```python
from transformers import StoppingCriteria

class RwkvStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [187,187], eos_token_id = 537):
        self.eos_sequence = eos_sequence
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_2_ids = input_ids[:,-2:].tolist()
        return self.eos_sequence in last_2_ids

output = model.generate(inputs["input_ids"], max_new_tokens=64, stopping_criteria = [RwkvStoppingCriteria()])
```

## RwkvConfig[[transformers.RwkvConfig]]

#### transformers.RwkvConfig[[transformers.RwkvConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/configuration_rwkv.py#L25)

This is the configuration class to store the configuration of a [RwkvModel](/docs/transformers/main/en/model_doc/rwkv#transformers.RwkvModel). It is used to instantiate a RWKV
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the RWVK-4
[RWKV/rwkv-4-169m-pile](https://huggingface.co/RWKV/rwkv-4-169m-pile) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import RwkvConfig, RwkvModel

>>> # Initializing a Rwkv configuration
>>> configuration = RwkvConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = RwkvModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 50277) : Vocabulary size of the RWKV model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [RwkvModel](/docs/transformers/main/en/model_doc/rwkv#transformers.RwkvModel).

context_length (`int`, *optional*, defaults to 1024) : The maximum sequence length that this model can be used with in a single forward (using it in RNN mode lets use any sequence length).

hidden_size (`int`, *optional*, defaults to 4096) : Dimensionality of the embeddings and hidden states.

num_hidden_layers (`int`, *optional*, defaults to 32) : Number of hidden layers in the model.

attention_hidden_size (`int`, *optional*) : Dimensionality of the attention hidden states. Will default to `hidden_size` if unset.

intermediate_size (`int`, *optional*) : Dimensionality of the inner feed-forward layers. Will default to 4 times `hidden_size` if unset.

layer_norm_epsilon (`float`, *optional*, defaults to 1e-05) : The epsilon to use in the layer normalization layers.

bos_token_id (`int`, *optional*, defaults to 0) : The id of the beginning of sentence token in the vocabulary. Defaults to 0 as RWKV uses the same tokenizer as GPTNeoX.

eos_token_id (`int`, *optional*, defaults to 0) : The id of the end of sentence token in the vocabulary. Defaults to 0 as RWKV uses the same tokenizer as GPTNeoX.

rescale_every (`int`, *optional*, defaults to 6) : At inference, the hidden states (and weights of the corresponding output layers) are divided by 2 every `rescale_every` layer. If set to 0 or a negative number, no rescale is done.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether or not to tie the word embeddings with the input token embeddings.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last state.

## RwkvModel[[transformers.RwkvModel]]

#### transformers.RwkvModel[[transformers.RwkvModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py#L491)

The bare Rwkv Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.RwkvModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py#L512[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "state", "val": ": typing.Optional[list[torch.FloatTensor]] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) --
  `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
  `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
  sequence tokens in the vocabulary.

  If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
  `input_ids`.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **state** (`tuple` of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*) --
  If passed along, the model uses the previous state in all the blocks (which will give the output for the
  `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
- **use_cache** (`bool`, *optional*) --
  If set to `True`, the last state is returned and can be used to quickly generate the next logits.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.rwkv.modeling_rwkv.RwkvOutput` or `tuple(torch.FloatTensor)`A `transformers.models.rwkv.modeling_rwkv.RwkvOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RwkvConfig](/docs/transformers/main/en/model_doc/rwkv#transformers.RwkvConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **state** (`list` of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`) -- The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
  avoid providing the old `input_ids`.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [RwkvModel](/docs/transformers/main/en/model_doc/rwkv#transformers.RwkvModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([RwkvModel](/docs/transformers/main/en/model_doc/rwkv#transformers.RwkvModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.rwkv.modeling_rwkv.RwkvOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.rwkv.modeling_rwkv.RwkvOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RwkvConfig](/docs/transformers/main/en/model_doc/rwkv#transformers.RwkvConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **state** (`list` of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`) -- The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
  avoid providing the old `input_ids`.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## RwkvLMHeadModel[[transformers.RwkvForCausalLM]]

#### transformers.RwkvForCausalLM[[transformers.RwkvForCausalLM]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py#L671)

The RWKV Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.RwkvForCausalLM.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py#L711[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "state", "val": ": typing.Optional[list[torch.FloatTensor]] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "logits_to_keep", "val": ": typing.Union[int, torch.Tensor] = 0"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) --
  `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
  `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
  sequence tokens in the vocabulary.

  If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
  `input_ids`.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **state** (`tuple` of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*) --
  If passed along, the model uses the previous state in all the blocks (which will give the output for the
  `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
  `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
  are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
- **use_cache** (`bool`, *optional*) --
  If set to `True`, the last state is returned and can be used to quickly generate the next logits.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **logits_to_keep** (`Union[int, torch.Tensor]`, defaults to `0`) --
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).0`transformers.models.rwkv.modeling_rwkv.RwkvCausalLMOutput` or `tuple(torch.FloatTensor)`A `transformers.models.rwkv.modeling_rwkv.RwkvCausalLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RwkvConfig](/docs/transformers/main/en/model_doc/rwkv#transformers.RwkvConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **state** (`list` of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`) -- The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
  avoid providing the old `input_ids`.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [RwkvForCausalLM](/docs/transformers/main/en/model_doc/rwkv#transformers.RwkvForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
```

**Parameters:**

config ([RwkvForCausalLM](/docs/transformers/main/en/model_doc/rwkv#transformers.RwkvForCausalLM)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.rwkv.modeling_rwkv.RwkvCausalLMOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.rwkv.modeling_rwkv.RwkvCausalLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RwkvConfig](/docs/transformers/main/en/model_doc/rwkv#transformers.RwkvConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **state** (`list` of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`) -- The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
  avoid providing the old `input_ids`.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Rwkv attention and the recurrent formulas

In a traditional auto-regressive Transformer, attention is written as

$$O = \hbox{softmax}(QK^{T} / \sqrt{d}) V$$

with $Q$, $K$ and $V$ are matrices of shape `seq_len x hidden_size` named query, key and value (they are actually bigger matrices with a batch dimension and an attention head dimension but we're only interested in the last two, which is where the matrix product is taken, so for the sake of simplicity we only consider those two). The product $QK^{T}$ then has shape `seq_len x seq_len` and we can take the matrix product with $V$ to get the output $O$ of the same shape as the others.  

Replacing the softmax by its value gives:

$$O_{i} = \frac{\sum_{j=1}^{i} e^{Q_{i} K_{j}^{T} / \sqrt{d}} V_{j}}{\sum_{j=1}^{i} e^{Q_{i} K_{j}^{T} / \sqrt{d}}}$$

Note that the entries in $QK^{T}$ corresponding to $j > i$ are masked (the sum stops at j) because the attention is not allowed to look at future tokens (only past ones).

In comparison, the RWKV attention is given by

$$O_{i} = \sigma(R_{i}) \frac{\sum_{j=1}^{i} e^{W_{i-j} + K_{j}} V_{j}}{\sum_{j=1}^{i} e^{W_{i-j} + K_{j}}}$$

where $R$ is a new matrix called receptance by the author, $K$ and $V$ are still the key and value ($\sigma$ here is the sigmoid function). $W$ is a new vector that represents the position of the token and is given by

$$W_{0} = u \hbox{  and  } W_{k} = (k-1)w \hbox{ for } k \geq 1$$

with $u$ and $w$ learnable parameters called in the code `time_first` and `time_decay` respectively. The numerator and denominator can both be expressed recursively. Naming them $N_{i}$ and $D_{i}$ we have:

$$N_{i} = e^{u + K_{i}} V_{i} + \hat{N}_{i} \hbox{  where  } \hat{N}_{i} = e^{K_{i-1}} V_{i-1} + e^{w + K_{i-2}} V_{i-2} \cdots + e^{(i-2)w + K_{1}} V_{1}$$

so $\hat{N}_{i}$ (called `numerator_state` in the code) satisfies

$$\hat{N}_{0} = 0 \hbox{  and  } \hat{N}_{j+1} = e^{K_{j}} V_{j} + e^{w} \hat{N}_{j}$$

and

$$D_{i} = e^{u + K_{i}} + \hat{D}_{i} \hbox{  where  } \hat{D}_{i} = e^{K_{i-1}} + e^{w + K_{i-2}} \cdots + e^{(i-2)w + K_{1}}$$

so $\hat{D}_{i}$ (called `denominator_state` in the code) satisfies

$$\hat{D}_{0} = 0 \hbox{  and  } \hat{D}_{j+1} = e^{K_{j}} + e^{w} \hat{D}_{j}$$

The actual recurrent formula used are a tiny bit more complex, as for numerical stability we don't want to compute exponentials of big numbers. Usually the softmax is not computed as is, but the exponential of the maximum term is divided of the numerator and denominator:

$$\frac{e^{x_{i}}}{\sum_{j=1}^{n} e^{x_{j}}} = \frac{e^{x_{i} - M}}{\sum_{j=1}^{n} e^{x_{j} - M}}$$

with $M$ the maximum of all $x_{j}$. So here on top of saving the numerator state ($\hat{N}$) and the denominator state ($\hat{D}$) we also keep track of the maximum of all terms encountered in the exponentials. So we actually use

$$\tilde{N}_{i} = e^{-M_{i}} \hat{N}_{i} \hbox{  and  } \tilde{D}_{i} = e^{-M_{i}} \hat{D}_{i}$$

defined by the following recurrent formulas:

$$\tilde{N}_{0} = 0 \hbox{  and  } \tilde{N}_{j+1} = e^{K_{j} - q} V_{j} + e^{w + M_{j} - q} \tilde{N}_{j} \hbox{  where  } q = \max(K_{j}, w + M_{j})$$

and

$$\tilde{D}_{0} = 0 \hbox{  and  } \tilde{D}_{j+1} = e^{K_{j} - q} + e^{w + M_{j} - q} \tilde{D}_{j} \hbox{  where  } q = \max(K_{j}, w + M_{j})$$

and $M_{j+1} = q$. With those, we can then compute

$$N_{i} = e^{u + K_{i} - q} V_{i} + e^{M_{i}} \tilde{N}_{i} \hbox{  where  } q = \max(u + K_{i}, M_{i})$$

and

$$D_{i} = e^{u + K_{i} - q} + e^{M_{i}} \tilde{D}_{i} \hbox{  where  } q = \max(u + K_{i}, M_{i})$$

which finally gives us

$$O_{i} = \sigma(R_{i}) \frac{N_{i}}{D_{i}}$$
