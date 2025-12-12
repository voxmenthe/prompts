*This model was released on 2021-06-02 and added to Hugging Face Transformers on 2022-03-23.*

# Decision Transformer

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

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

## DecisionTransformerConfig

### class transformers.DecisionTransformerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/decision_transformer/configuration_decision_transformer.py#L24)

( state\_dim = 17 act\_dim = 4 hidden\_size = 128 max\_ep\_len = 4096 action\_tanh = True vocab\_size = 1 n\_positions = 1024 n\_layer = 3 n\_head = 1 n\_inner = None activation\_function = 'relu' resid\_pdrop = 0.1 embd\_pdrop = 0.1 attn\_pdrop = 0.1 layer\_norm\_epsilon = 1e-05 initializer\_range = 0.02 scale\_attn\_weights = True use\_cache = True bos\_token\_id = 50256 eos\_token\_id = 50256 scale\_attn\_by\_inverse\_layer\_idx = False reorder\_and\_upcast\_attn = False \*\*kwargs  )

Parameters

* **state\_dim** (`int`, *optional*, defaults to 17) —
  The state size for the RL environment
* **act\_dim** (`int`, *optional*, defaults to 4) —
  The size of the output action space
* **hidden\_size** (`int`, *optional*, defaults to 128) —
  The size of the hidden layers
* **max\_ep\_len** (`int`, *optional*, defaults to 4096) —
  The maximum length of an episode in the environment
* **action\_tanh** (`bool`, *optional*, defaults to True) —
  Whether to use a tanh activation on action prediction
* **vocab\_size** (`int`, *optional*, defaults to 50257) —
  Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [DecisionTransformerModel](/docs/transformers/v4.56.2/en/model_doc/decision_transformer#transformers.DecisionTransformerModel).
* **n\_positions** (`int`, *optional*, defaults to 1024) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **n\_layer** (`int`, *optional*, defaults to 3) —
  Number of hidden layers in the Transformer encoder.
* **n\_head** (`int`, *optional*, defaults to 1) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **n\_inner** (`int`, *optional*) —
  Dimensionality of the inner feed-forward layers. If unset, will default to 4 times `n_embd`.
* **activation\_function** (`str`, *optional*, defaults to `"gelu"`) —
  Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
* **resid\_pdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **embd\_pdrop** (`int`, *optional*, defaults to 0.1) —
  The dropout ratio for the embeddings.
* **attn\_pdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention.
* **layer\_norm\_epsilon** (`float`, *optional*, defaults to 1e-5) —
  The epsilon to use in the layer normalization layers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **scale\_attn\_weights** (`bool`, *optional*, defaults to `True`) —
  Scale attention weights by dividing by sqrt(hidden\_size)..
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **scale\_attn\_by\_inverse\_layer\_idx** (`bool`, *optional*, defaults to `False`) —
  Whether to additionally scale attention weights by `1 / layer_idx + 1`.
* **reorder\_and\_upcast\_attn** (`bool`, *optional*, defaults to `False`) —
  Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
  dot-product/softmax to float() when training with mixed precision.

This is the configuration class to store the configuration of a [DecisionTransformerModel](/docs/transformers/v4.56.2/en/model_doc/decision_transformer#transformers.DecisionTransformerModel). It is used to
instantiate a Decision Transformer model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the standard
DecisionTransformer architecture. Many of the config options are used to instantiate the GPT2 model that is used as
part of the architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import DecisionTransformerConfig, DecisionTransformerModel

>>> # Initializing a DecisionTransformer configuration
>>> configuration = DecisionTransformerConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = DecisionTransformerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## DecisionTransformerGPT2Model

### class transformers.DecisionTransformerGPT2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L498)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L527)

( input\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None cache\_position: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[torch.Tensor] = None encoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  )

## DecisionTransformerModel

### class transformers.DecisionTransformerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L773)

( config  )

Parameters

* **config** ([DecisionTransformerModel](/docs/transformers/v4.56.2/en/model_doc/decision_transformer#transformers.DecisionTransformerModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Decision Transformer Model

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L806)

( states: typing.Optional[torch.FloatTensor] = None actions: typing.Optional[torch.FloatTensor] = None rewards: typing.Optional[torch.FloatTensor] = None returns\_to\_go: typing.Optional[torch.FloatTensor] = None timesteps: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerOutput` or `tuple(torch.FloatTensor)`

Parameters

* **states** (`torch.FloatTensor` of shape `(batch_size, episode_length, state_dim)`) —
  The states for each step in the trajectory
* **actions** (`torch.FloatTensor` of shape `(batch_size, episode_length, act_dim)`) —
  The actions taken by the “expert” policy for the current state, these are masked for auto regressive
  prediction
* **rewards** (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`) —
  The rewards for each state, action
* **returns\_to\_go** (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`) —
  The returns for each state in the trajectory
* **timesteps** (`torch.LongTensor` of shape `(batch_size, episode_length)`) —
  The timestep for each step in the trajectory
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.decision_transformer.modeling_decision_transformer.DecisionTransformerOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DecisionTransformerConfig](/docs/transformers/v4.56.2/en/model_doc/decision_transformer#transformers.DecisionTransformerConfig)) and inputs.

* **state\_preds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, state_dim)`) — Environment state predictions
* **action\_preds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_dim)`) — Model action predictions
* **return\_preds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`) — Predicted returns for each state
* **hidden\_states** (`torch.FloatTensor`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`torch.FloatTensor`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.

The [DecisionTransformerModel](/docs/transformers/v4.56.2/en/model_doc/decision_transformer#transformers.DecisionTransformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/decision_transformer.md)
