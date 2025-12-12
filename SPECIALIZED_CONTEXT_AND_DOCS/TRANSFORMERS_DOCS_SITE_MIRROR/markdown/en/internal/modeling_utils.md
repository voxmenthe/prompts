# Custom Layers and Utilities

This page lists all the custom layers used by the library, as well as the utility functions and classes it provides for modeling.

Most of those are only useful if you are studying the code of the models in the library.

## Layers[[transformers.GradientCheckpointingLayer]]

#### transformers.GradientCheckpointingLayer[[transformers.GradientCheckpointingLayer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_layers.py#L35)

Base class for layers with gradient checkpointing.

This class enables gradient checkpointing functionality for a layer. By default, gradient checkpointing is disabled
(`gradient_checkpointing = False`). When `model.set_gradient_checkpointing()` is called, gradient checkpointing is
enabled by setting `gradient_checkpointing = True` and assigning a checkpointing function to `_gradient_checkpointing_func`.

Important:

When using gradient checkpointing with `use_reentrant=True`, inputs that require gradients (e.g. hidden states)
must be passed as positional arguments (`*args`) rather than keyword arguments to properly propagate gradients.

Example:

```python
>>> # Correct - hidden_states passed as positional arg
>>> out = self.layer(hidden_states, attention_mask=attention_mask)

>>> # Incorrect - hidden_states passed as keyword arg
>>> out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask)
```

## Attention Functions[[transformers.AttentionInterface]]

#### transformers.AttentionInterface[[transformers.AttentionInterface]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L4658)

Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
it needs to declare a new instance of this class inside the `modeling_.py`, and declare it on that instance.

registertransformers.AttentionInterface.registerhttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#L1029[{"name": "key", "val": ": str"}, {"name": "value", "val": ": Callable"}]

## Attention Mask Functions[[transformers.AttentionMaskInterface]]

#### transformers.AttentionMaskInterface[[transformers.AttentionMaskInterface]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py#L631)

registertransformers.AttentionMaskInterface.registerhttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#L1029[{"name": "key", "val": ": str"}, {"name": "value", "val": ": Callable"}]

## Rotary Position Embedding Functions[[transformers.dynamic_rope_update]]

#### transformers.dynamic_rope_update[[transformers.dynamic_rope_update]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L33)

Decorator function to update the RoPE parameters in the forward pass, if the model is using a dynamic RoPE
(i.e. a RoPE implementation that may recompute its frequencies in the forward pass).

**Parameters:**

rope_forward (Callable) : The forward pass of the RoPE implementation.

**Returns:**

The decorated forward pass.

## Pytorch custom modules[[transformers.Conv1D]]

#### transformers.Conv1D[[transformers.Conv1D]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L97)

1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

Basically works like a linear layer but the weights are transposed.

**Parameters:**

nf (`int`) : The number of output features.

nx (`int`) : The number of input features.

## PyTorch Helper Functions[[transformers.apply_chunking_to_forward]]

#### transformers.apply_chunking_to_forward[[transformers.apply_chunking_to_forward]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L126)

This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
`chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
applying `forward_fn` to `input_tensors`.

Examples:

```python
# rename the usual forward() fn to forward_chunk()
def forward_chunk(self, hidden_states):
    hidden_states = self.decoder(hidden_states)
    return hidden_states

# implement a chunked forward function
def forward(self, hidden_states):
    return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
```

**Parameters:**

forward_fn (`Callable[..., torch.Tensor]`) : The forward function of the model.

chunk_size (`int`) : The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.

chunk_dim (`int`) : The dimension over which the `input_tensors` should be chunked.

input_tensors (`tuple[torch.Tensor]`) : The input tensors of `forward_fn` which will be chunked

**Returns:**

``torch.Tensor``

A tensor with the same shape as the `forward_fn` would have given if applied`.

#### transformers.pytorch_utils.prune_linear_layer[[transformers.pytorch_utils.prune_linear_layer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L63)

Prune a linear layer to keep only entries in index.

Used to remove heads.

**Parameters:**

layer (`torch.nn.Linear`) : The layer to prune.

index (`torch.LongTensor`) : The indices to keep in the layer.

dim (`int`, *optional*, defaults to 0) : The dimension on which to keep the indices.

**Returns:**

``torch.nn.Linear``

The pruned layer as a new layer with `requires_grad=True`.
