# Trajectory Transformer

This model is in maintenance mode only, so we won't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

## Overview

The Trajectory Transformer model was proposed in [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://huggingface.co/papers/2106.02039)  by Michael Janner, Qiyang Li, Sergey Levine.

The abstract from the paper is the following:

*Reinforcement learning (RL) is typically concerned with estimating stationary policies or single-step models,
leveraging the Markov property to factorize problems in time. However, we can also view RL as a generic sequence
modeling problem, with the goal being to produce a sequence of actions that leads to a sequence of high rewards.
Viewed in this way, it is tempting to consider whether high-capacity sequence prediction models that work well
in other domains, such as natural-language processing, can also provide effective solutions to the RL problem.
To this end, we explore how RL can be tackled with the tools of sequence modeling, using a Transformer architecture
to model distributions over trajectories and repurposing beam search as a planning algorithm. Framing RL as sequence
modeling problem simplifies a range of design decisions, allowing us to dispense with many of the components common
in offline RL algorithms. We demonstrate the flexibility of this approach across long-horizon dynamics prediction,
imitation learning, goal-conditioned RL, and offline RL. Further, we show that this approach can be combined with
existing model-free algorithms to yield a state-of-the-art planner in sparse-reward, long-horizon tasks.*

This model was contributed by [CarlCochet](https://huggingface.co/CarlCochet). The original code can be found [here](https://github.com/jannerm/trajectory-transformer).

## Usage tips

This Transformer is used for deep reinforcement learning. To use it, you need to create sequences from
actions, states and rewards from all previous timesteps. This model will treat all these elements together
as one big sequence (a trajectory).

## TrajectoryTransformerConfig[[transformers.TrajectoryTransformerConfig]]

#### transformers.TrajectoryTransformerConfig[[transformers.TrajectoryTransformerConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/trajectory_transformer/configuration_trajectory_transformer.py#L24)

This is the configuration class to store the configuration of a [TrajectoryTransformerModel](/docs/transformers/main/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerModel). It is used to
instantiate an TrajectoryTransformer model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the
TrajectoryTransformer
[CarlCochet/trajectory-transformer-halfcheetah-medium-v2](https://huggingface.co/CarlCochet/trajectory-transformer-halfcheetah-medium-v2)
architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import TrajectoryTransformerConfig, TrajectoryTransformerModel

>>> # Initializing a TrajectoryTransformer CarlCochet/trajectory-transformer-halfcheetah-medium-v2 style configuration
>>> configuration = TrajectoryTransformerConfig()

>>> # Initializing a model (with random weights) from the CarlCochet/trajectory-transformer-halfcheetah-medium-v2 style configuration
>>> model = TrajectoryTransformerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 100) : Vocabulary size of the TrajectoryTransformer model. Defines the number of different tokens that can be represented by the `trajectories` passed when calling [TrajectoryTransformerModel](/docs/transformers/main/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerModel)

action_weight (`int`, *optional*, defaults to 5) : Weight of the action in the loss function

reward_weight (`int`, *optional*, defaults to 1) : Weight of the reward in the loss function

value_weight (`int`, *optional*, defaults to 1) : Weight of the value in the loss function

block_size (`int`, *optional*, defaults to 249) : Size of the blocks in the trajectory transformer.

action_dim (`int`, *optional*, defaults to 6) : Dimension of the action space.

observation_dim (`int`, *optional*, defaults to 17) : Dimension of the observation space.

transition_dim (`int`, *optional*, defaults to 25) : Dimension of the transition space.

n_layer (`int`, *optional*, defaults to 4) : Number of hidden layers in the Transformer encoder.

n_head (`int`, *optional*, defaults to 4) : Number of attention heads for each attention layer in the Transformer encoder.

n_embd (`int`, *optional*, defaults to 128) : Dimensionality of the embeddings and hidden states.

resid_pdrop (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

embd_pdrop (`int`, *optional*, defaults to 0.1) : The dropout ratio for the embeddings.

attn_pdrop (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

kaiming_initializer_range (`float, *optional*, defaults to 1) : A coefficient scaling the negative slope of the kaiming initializer rectifier for EinLinear layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.

Example --

## TrajectoryTransformerModel[[transformers.TrajectoryTransformerModel]]

#### transformers.TrajectoryTransformerModel[[transformers.TrajectoryTransformerModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/trajectory_transformer/modeling_trajectory_transformer.py#L326)

The bare TrajectoryTransformer Model transformer outputting raw hidden-states without any specific head on top.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

the full GPT language model, with a context size of block_size

forwardtransformers.TrajectoryTransformerModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/trajectory_transformer/modeling_trajectory_transformer.py#L387[{"name": "trajectories", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "targets", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **trajectories** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Batch of trajectories, where a trajectory is a sequence of states, actions and rewards.
- **past_key_values** (`tuple[tuple[torch.Tensor]]` of length `config.n_layers`, *optional*) --
  Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
  `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
  their past given to this model should not be passed as `input_ids` as they have already been computed.
- **targets** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Desired targets used to compute the loss.
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
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
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.deprecated.trajectory_transformer.modeling_trajectory_transformer.TrajectoryTransformerOutput` or `tuple(torch.FloatTensor)`A `transformers.models.deprecated.trajectory_transformer.modeling_trajectory_transformer.TrajectoryTransformerOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TrajectoryTransformerConfig](/docs/transformers/main/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`tuple[tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
  sequence_length, embed_size_per_head)`). Contains pre-computed hidden-states (key and values in the
  attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
  plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. GPT2Attentions weights after the attention softmax, used to compute the weighted average
  in the self-attention heads.
The [TrajectoryTransformerModel](/docs/transformers/main/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import TrajectoryTransformerModel
>>> import torch

>>> model = TrajectoryTransformerModel.from_pretrained(
...     "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"
... )
>>> model.to(device)
>>> model.eval()

>>> observations_dim, action_dim, batch_size = 17, 6, 256
>>> seq_length = observations_dim + action_dim + 1

>>> trajectories = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(
...     device
... )
>>> targets = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(batch_size)]).to(device)

>>> outputs = model(
...     trajectories,
...     targets=targets,
...     use_cache=True,
...     output_attentions=True,
...     output_hidden_states=True,
...     return_dict=True,
... )
```

**Parameters:**

config ([TrajectoryTransformerConfig](/docs/transformers/main/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.deprecated.trajectory_transformer.modeling_trajectory_transformer.TrajectoryTransformerOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.trajectory_transformer.modeling_trajectory_transformer.TrajectoryTransformerOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TrajectoryTransformerConfig](/docs/transformers/main/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`tuple[tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
  sequence_length, embed_size_per_head)`). Contains pre-computed hidden-states (key and values in the
  attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
  plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. GPT2Attentions weights after the attention softmax, used to compute the weighted average
  in the self-attention heads.
