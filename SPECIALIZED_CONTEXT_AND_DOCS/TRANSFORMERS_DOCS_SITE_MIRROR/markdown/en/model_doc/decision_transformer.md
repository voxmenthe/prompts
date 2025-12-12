# Decision Transformer

## Overview

The Decision Transformer model was proposed in [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://huggingface.co/papers/2106.01345)  
by Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.

The abstract from the paper is the following:

*We introduce a framework that abstracts Reinforcement Learning (RL) as a sequence modeling problem.
This allows us to draw upon the simplicity and scalability of the Transformer architecture, and associated advances
 in language modeling such as GPT-x and BERT. In particular, we present Decision Transformer, an architecture that
 casts the problem of RL as conditional sequence modeling. Unlike prior approaches to RL that fit value functions or
 compute policy gradients, Decision Transformer simply outputs the optimal actions by leveraging a causally masked
 Transformer. By conditioning an autoregressive model on the desired return (reward), past states, and actions, our
 Decision Transformer model can generate future actions that achieve the desired return. Despite its simplicity,
 Decision Transformer matches or exceeds the performance of state-of-the-art model-free offline RL baselines on
 Atari, OpenAI Gym, and Key-to-Door tasks.*

This version of the model is for tasks where the state is a vector.

This model was contributed by [edbeeching](https://huggingface.co/edbeeching). The original code can be found [here](https://github.com/kzl/decision-transformer).

## DecisionTransformerConfig[[transformers.DecisionTransformerConfig]]

#### transformers.DecisionTransformerConfig[[transformers.DecisionTransformerConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/decision_transformer/configuration_decision_transformer.py#L24)

This is the configuration class to store the configuration of a [DecisionTransformerModel](/docs/transformers/main/en/model_doc/decision_transformer#transformers.DecisionTransformerModel). It is used to
instantiate a Decision Transformer model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the standard
DecisionTransformer architecture. Many of the config options are used to instantiate the GPT2 model that is used as
part of the architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import DecisionTransformerConfig, DecisionTransformerModel

>>> # Initializing a DecisionTransformer configuration
>>> configuration = DecisionTransformerConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = DecisionTransformerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

state_dim (`int`, *optional*, defaults to 17) : The state size for the RL environment

act_dim (`int`, *optional*, defaults to 4) : The size of the output action space

hidden_size (`int`, *optional*, defaults to 128) : The size of the hidden layers

max_ep_len (`int`, *optional*, defaults to 4096) : The maximum length of an episode in the environment

action_tanh (`bool`, *optional*, defaults to True) : Whether to use a tanh activation on action prediction

vocab_size (`int`, *optional*, defaults to 50257) : Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [DecisionTransformerModel](/docs/transformers/main/en/model_doc/decision_transformer#transformers.DecisionTransformerModel).

n_positions (`int`, *optional*, defaults to 1024) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

n_layer (`int`, *optional*, defaults to 3) : Number of hidden layers in the Transformer encoder.

n_head (`int`, *optional*, defaults to 1) : Number of attention heads for each attention layer in the Transformer encoder.

n_inner (`int`, *optional*) : Dimensionality of the inner feed-forward layers. If unset, will default to 4 times `n_embd`.

activation_function (`str`, *optional*, defaults to `"gelu"`) : Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.

resid_pdrop (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

embd_pdrop (`int`, *optional*, defaults to 0.1) : The dropout ratio for the embeddings.

attn_pdrop (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention.

layer_norm_epsilon (`float`, *optional*, defaults to 1e-5) : The epsilon to use in the layer normalization layers.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

scale_attn_weights (`bool`, *optional*, defaults to `True`) : Scale attention weights by dividing by sqrt(hidden_size)..

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models).

scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`) : Whether to additionally scale attention weights by `1 / layer_idx + 1`.

reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`) : Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention dot-product/softmax to float() when training with mixed precision.

## DecisionTransformerGPT2Model[[transformers.DecisionTransformerGPT2Model]]

#### transformers.DecisionTransformerGPT2Model[[transformers.DecisionTransformerGPT2Model]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L394)

forwardtransformers.DecisionTransformerGPT2Model.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L420[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "encoder_hidden_states", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]

## DecisionTransformerModel[[transformers.DecisionTransformerModel]]

#### transformers.DecisionTransformerModel[[transformers.DecisionTransformerModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L616)

The Decision Transformer Model

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.DecisionTransformerModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L649[{"name": "states", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "actions", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "rewards", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "returns_to_go", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "timesteps", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **states** (`torch.FloatTensor` of shape `(batch_size, episode_length, state_dim)`) --
  The states for each step in the trajectory
- **actions** (`torch.FloatTensor` of shape `(batch_size, episode_length, act_dim)`) --
  The actions taken by the "expert" policy for the current state, these are masked for auto regressive
  prediction
- **rewards** (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`) --
  The rewards for each state, action
- **returns_to_go** (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`) --
  The returns for each state in the trajectory
- **timesteps** (`torch.LongTensor` of shape `(batch_size, episode_length)`) --
  The timestep for each step in the trajectory
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerOutput` or `tuple(torch.FloatTensor)`A `transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DecisionTransformerConfig](/docs/transformers/main/en/model_doc/decision_transformer#transformers.DecisionTransformerConfig)) and inputs.

- **state_preds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, state_dim)`) -- Environment state predictions
- **action_preds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_dim)`) -- Model action predictions
- **return_preds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`) -- Predicted returns for each state
- **hidden_states** (`torch.FloatTensor`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`torch.FloatTensor`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
The [DecisionTransformerModel](/docs/transformers/main/en/model_doc/decision_transformer#transformers.DecisionTransformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import DecisionTransformerModel
>>> import torch

>>> model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
>>> # evaluation
>>> model = model.to(device)
>>> model.eval()

>>> env = gym.make("Hopper-v3")
>>> state_dim = env.observation_space.shape[0]
>>> act_dim = env.action_space.shape[0]

>>> state = env.reset()
>>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
>>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
>>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
>>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
>>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
>>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

>>> # forward pass
>>> with torch.no_grad():
...     state_preds, action_preds, return_preds = model(
...         states=states,
...         actions=actions,
...         rewards=rewards,
...         returns_to_go=target_return,
...         timesteps=timesteps,
...         attention_mask=attention_mask,
...         return_dict=False,
...     )
```

**Parameters:**

config ([DecisionTransformerModel](/docs/transformers/main/en/model_doc/decision_transformer#transformers.DecisionTransformerModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DecisionTransformerConfig](/docs/transformers/main/en/model_doc/decision_transformer#transformers.DecisionTransformerConfig)) and inputs.

- **state_preds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, state_dim)`) -- Environment state predictions
- **action_preds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_dim)`) -- Model action predictions
- **return_preds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`) -- Predicted returns for each state
- **hidden_states** (`torch.FloatTensor`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`torch.FloatTensor`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
