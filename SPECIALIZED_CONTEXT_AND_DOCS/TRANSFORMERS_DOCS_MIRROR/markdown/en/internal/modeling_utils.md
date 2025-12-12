# Custom Layers and Utilities

This page lists all the custom layers used by the library, as well as the utility functions and classes it provides for modeling.

Most of those are only useful if you are studying the code of the models in the library.

## Layers

### class transformers.GradientCheckpointingLayer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L35)

( \*args \*\*kwargs  )

Base class for layers with gradient checkpointing.

This class enables gradient checkpointing functionality for a layer. By default, gradient checkpointing is disabled
(`gradient_checkpointing = False`). When `model.set_gradient_checkpointing()` is called, gradient checkpointing is
enabled by setting `gradient_checkpointing = True` and assigning a checkpointing function to `_gradient_checkpointing_func`.

Important:

When using gradient checkpointing with `use_reentrant=True`, inputs that require gradients (e.g. hidden states)
must be passed as positional arguments (`*args`) rather than keyword arguments to properly propagate gradients.

Example:


```
>>> # Correct - hidden_states passed as positional arg
>>> out = self.layer(hidden_states, attention_mask=attention_mask)

>>> # Incorrect - hidden_states passed as keyword arg
>>> out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask)
```

## Attention Functions

### class transformers.AttentionInterface

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L6250)

( )

Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.

#### register

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/generic.py#L1127)

( key: str value: typing.Callable  )

## Attention Mask Functions

### class transformers.AttentionMaskInterface

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/masking_utils.py#L621)

( )

#### register

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/generic.py#L1127)

( key: str value: typing.Callable  )

## Rotary Position Embedding Functions

#### transformers.dynamic\_rope\_update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_rope_utils.py#L30)

( rope\_forward  )

Parameters

* **rope\_forward** (Callable) —
  The forward pass of the RoPE implementation.

Decorator function to update the RoPE parameters in the forward pass, if the model is using a dynamic RoPE
(i.e. a RoPE implementation that may recompute its frequencies in the forward pass).

## Pytorch custom modules

### class transformers.Conv1D

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/pytorch_utils.py#L98)

( nf nx  )

Parameters

* **nf** (`int`) — The number of output features.
* **nx** (`int`) — The number of input features.

1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

Basically works like a linear layer but the weights are transposed.

## PyTorch Helper Functions

#### transformers.apply\_chunking\_to\_forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/pytorch_utils.py#L182)

( forward\_fn: Callable[..., torch.Tensor] chunk\_size: int chunk\_dim: int \*input\_tensors  ) → `torch.Tensor`

Parameters

* **forward\_fn** (`Callable[..., torch.Tensor]`) —
  The forward function of the model.
* **chunk\_size** (`int`) —
  The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
* **chunk\_dim** (`int`) —
  The dimension over which the `input_tensors` should be chunked.
* **input\_tensors** (`tuple[torch.Tensor]`) —
  The input tensors of `forward_fn` which will be chunked

Returns

`torch.Tensor`

A tensor with the same shape as the `forward_fn` would have given if applied`.

This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
`chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
applying `forward_fn` to `input_tensors`.

Examples:


```
# rename the usual forward() fn to forward_chunk()
def forward_chunk(self, hidden_states):
    hidden_states = self.decoder(hidden_states)
    return hidden_states


# implement a chunked forward function
def forward(self, hidden_states):
    return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
```

#### transformers.pytorch\_utils.find\_pruneable\_heads\_and\_indices

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/pytorch_utils.py#L260)

( heads: list[int] n\_heads: int head\_size: int already\_pruned\_heads: set[int]  ) → `tuple[Set[int], torch.LongTensor]`

Parameters

* **heads** (`list[int]`) — List of the indices of heads to prune.
* **n\_heads** (`int`) — The number of heads in the model.
* **head\_size** (`int`) — The size of each head.
* **already\_pruned\_heads** (`Set[int]`) — A set of already pruned heads.

Returns

`tuple[Set[int], torch.LongTensor]`

A tuple with the indices of heads to prune taking `already_pruned_heads`
into account and the indices of rows/columns to keep in the layer weight.

Finds the heads and their indices taking `already_pruned_heads` into account.

#### transformers.prune\_layer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/pytorch_utils.py#L160)

( layer: nn.Linear | Conv1D index: torch.LongTensor dim: int | None = None  ) → `torch.nn.Linear` or [Conv1D](/docs/transformers/v4.56.2/en/internal/modeling_utils#transformers.Conv1D)

Parameters

* **layer** (`Union[torch.nn.Linear, Conv1D]`) — The layer to prune.
* **index** (`torch.LongTensor`) — The indices to keep in the layer.
* **dim** (`int`, *optional*) — The dimension on which to keep the indices.

Returns

`torch.nn.Linear` or [Conv1D](/docs/transformers/v4.56.2/en/internal/modeling_utils#transformers.Conv1D)

The pruned layer as a new layer with `requires_grad=True`.

Prune a Conv1D or linear layer to keep only entries in index.

Used to remove heads.

#### transformers.pytorch\_utils.prune\_conv1d\_layer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/pytorch_utils.py#L127)

( layer: Conv1D index: torch.LongTensor dim: int = 1  ) → [Conv1D](/docs/transformers/v4.56.2/en/internal/modeling_utils#transformers.Conv1D)

Parameters

* **layer** ([Conv1D](/docs/transformers/v4.56.2/en/internal/modeling_utils#transformers.Conv1D)) — The layer to prune.
* **index** (`torch.LongTensor`) — The indices to keep in the layer.
* **dim** (`int`, *optional*, defaults to 1) — The dimension on which to keep the indices.

Returns

[Conv1D](/docs/transformers/v4.56.2/en/internal/modeling_utils#transformers.Conv1D)

The pruned layer as a new layer with `requires_grad=True`.

Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
are transposed.

Used to remove heads.

#### transformers.pytorch\_utils.prune\_linear\_layer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/pytorch_utils.py#L64)

( layer: nn.Linear index: torch.LongTensor dim: int = 0  ) → `torch.nn.Linear`

Parameters

* **layer** (`torch.nn.Linear`) — The layer to prune.
* **index** (`torch.LongTensor`) — The indices to keep in the layer.
* **dim** (`int`, *optional*, defaults to 0) — The dimension on which to keep the indices.

Returns

`torch.nn.Linear`

The pruned layer as a new layer with `requires_grad=True`.

Prune a linear layer to keep only entries in index.

Used to remove heads.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/modeling_utils.md)
