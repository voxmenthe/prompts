# Reformer

## Overview

The Reformer model was proposed in the paper [Reformer: The Efficient Transformer](https://huggingface.co/papers/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

The abstract from the paper is the following:

*Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can
be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of
Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its
complexity from O(L^2) to O(Llog(L)), where L is the length of the sequence. Furthermore, we use reversible residual
layers instead of the standard residuals, which allows storing activations only once in the training process instead of
N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models
while being much more memory-efficient and much faster on long sequences.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The Authors' code can be
found [here](https://github.com/google/trax/tree/master/trax/models/reformer).

## Usage tips

- Reformer does **not** work with *torch.nn.DataParallel* due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035).
- Use Axial position encoding (see below for more details). It's a mechanism to avoid having a huge positional encoding matrix (when the sequence length is very big) by factorizing it into smaller matrices.
- Replace traditional attention by LSH (local-sensitive hashing) attention (see below for more details). It's a technique to avoid computing the full product query-key in the attention layers.
- Avoid storing the intermediate results of each layer by using reversible transformer layers to obtain them during the backward pass (subtracting the residuals from the input of the next layer gives them back) or recomputing them for results inside a given layer (less efficient than storing them but saves memory).
- Compute the feedforward operations by chunks and not on the whole batch.

### Axial Positional Encodings

Axial Positional Encodings were first implemented in Google's [trax library](https://github.com/google/trax/blob/4d99ad4965bab1deba227539758d59f0df0fef48/trax/layers/research/position_encodings.py#L29)
and developed by the authors of this model's paper. In models that are treating very long input sequences, the
conventional position id encodings store an embeddings vector of size $d$ being the `config.hidden_size` for
every position $i, \ldots, n_s$, with $n_s$ being `config.max_embedding_size`. This means that having
a sequence length of $n_s = 2^{19} \approx 0.5M$ and a `config.hidden_size` of $d = 2^{10} \approx 1000$
would result in a position encoding matrix:

$$X_{i,j}, \text{ with } i \in \left[1,\ldots, d\right] \text{ and } j \in \left[1,\ldots, n_s\right]$$

which alone has over 500M parameters to store. Axial positional encodings factorize $X_{i,j}$ into two matrices:

$$X^{1}_{i,j}, \text{ with } i \in \left[1,\ldots, d^1\right] \text{ and } j \in \left[1,\ldots, n_s^1\right]$$

and

$$X^{2}_{i,j}, \text{ with } i \in \left[1,\ldots, d^2\right] \text{ and } j \in \left[1,\ldots, n_s^2\right]$$

with:

$$d = d^1 + d^2 \text{ and } n_s = n_s^1 \times n_s^2 .$$

Therefore the following holds:

$$X_{i,j} = \begin{cases}
X^{1}_{i, k}, & \text{if }\ i >> from transformers import ReformerConfig, ReformerModel

>>> # Initializing a Reformer configuration
>>> configuration = ReformerConfig()

>>> # Initializing a Reformer model (with random weights)
>>> model = ReformerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

attention_head_size (`int`, *optional*, defaults to 64) : Dimensionality of the projected key, query and value vectors

attn_layers (`list[str]`, *optional*, defaults to `["local", "lsh", "local", "lsh", "local", "lsh"]`) : List of attention layer types in ascending order. It can be chosen between a LSHSelfAttention layer (`"lsh"`) and a LocalSelfAttention layer (`"local"`).  For more information on LSHSelfAttention layer, see [LSH Self Attention](reformer#lsh-self-attention). For more information on LocalSelfAttention layer, see [Local Self Attention](reformer#local-self-attention).

axial_pos_embds (`bool`, *optional*, defaults to `True`) : Whether or not to use axial position embeddings. For more information on how axial position embeddings work, see [Axial Position Encodings](reformer#axial-positional-encodings).

axial_norm_std (`float`, *optional*, defaults to 1.0) : The standard deviation of the normal_initializer for initializing the weight matrices of the axial positional encodings.

axial_pos_shape (`list[int]`, *optional*, defaults to `[64, 64]`) : The position dims of the axial position encodings. During training, the product of the position dims has to be equal to the sequence length.  For more information on how axial position embeddings work, see [Axial Position Encodings](reformer#axial-positional-encodings).

axial_pos_embds_dim (`list[int]`, *optional*, defaults to `[64, 192]`) : The embedding dims of the axial position encodings. The sum of the embedding dims has to be equal to the hidden size.  For more information on how axial position embeddings work, see [Axial Position Encodings](reformer#axial-positional-encodings).

chunk_size_lm_head (`int`, *optional*, defaults to 0) : The chunk size of the final language model feed forward head layer. A chunk size of 0 means that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes n "`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

additional_special_tokens (`list[str]`, *optional*) : Additional special tokens used by the tokenizer.

vocab (`str` or `dict[str, int]`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from `vocab_file`.

merges (`str` or `list[str]`, *optional*) : Custom merges list. If not provided, merges are loaded from `vocab_file`.

## ReformerTokenizerFast[[transformers.ReformerTokenizer]]

#### transformers.ReformerTokenizer[[transformers.ReformerTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/tokenization_reformer.py#L34)

Construct a Reformer tokenizer (backed by HuggingFace's tokenizers library). Based on
[BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=bpe#models).

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

**Parameters:**

vocab_file (`str`) : [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that contains the vocabulary necessary to instantiate a tokenizer.

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

additional_special_tokens (`list[str]`, *optional*) : Additional special tokens used by the tokenizer.

vocab (`str` or `dict[str, int]`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from `vocab_file`.

merges (`str` or `list[str]`, *optional*) : Custom merges list. If not provided, merges are loaded from `vocab_file`.

## ReformerModel[[transformers.ReformerModel]]

#### transformers.ReformerModel[[transformers.ReformerModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py#L1916)

The bare Reformer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.ReformerModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py#L1936[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "num_hashes", "val": ": typing.Optional[int] = None"}, {"name": "past_buckets_states", "val": ": typing.Optional[transformers.models.reformer.modeling_reformer.ReformerDynamicCache] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. During training the input_ids sequence_length has to be
  a multiple of the relevant model's chunk lengths (lsh's, local's or both). During evaluation, the indices
  are automatically padded to be a multiple of the chunk length.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **num_hashes** (`int`, *optional*) --
  The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
  the default defined in `config.num_hashes`.

  For more information, see `num_hashes` in [ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig).
- **past_buckets_states** (`ReformerDynamicCache`, *optional*) --
  List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
  being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
  second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

  Contains precomputed hidden-states and buckets (only relevant for LSH Self-Attention). Can be used to speed
  up sequential decoding.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.reformer.modeling_reformer.ReformerModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.reformer.modeling_reformer.ReformerModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`) -- Sequence of hidden-states at the last layer of the model.

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
- **past_buckets_states** (`list[tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
  being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
  second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

  Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
  up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [ReformerModel](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([ReformerModel](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.reformer.modeling_reformer.ReformerModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.reformer.modeling_reformer.ReformerModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`) -- Sequence of hidden-states at the last layer of the model.

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
- **past_buckets_states** (`list[tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
  being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
  second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

  Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
  up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## ReformerModelWithLMHead[[transformers.ReformerModelWithLMHead]]

#### transformers.ReformerModelWithLMHead[[transformers.ReformerModelWithLMHead]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py#L2135)

Reformer Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.ReformerModelWithLMHead.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py#L2161[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "num_hashes", "val": ": typing.Optional[int] = None"}, {"name": "past_buckets_states", "val": ": typing.Optional[list[tuple[torch.Tensor]]] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "logits_to_keep", "val": ": typing.Union[int, torch.Tensor] = 0"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. During training the input_ids sequence_length has to be
  a multiple of the relevant model's chunk lengths (lsh's, local's or both). During evaluation, the indices
  are automatically padded to be a multiple of the chunk length.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **num_hashes** (`int`, *optional*) --
  The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
  the default defined in `config.num_hashes`.

  For more information, see `num_hashes` in [ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig).
- **past_buckets_states** (`list[tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*) --
  List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
  being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
  second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

  Contains precomputed hidden-states and buckets (only relevant for LSH Self-Attention). Can be used to speed
  up sequential decoding.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
  config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
  labels in `[0, ..., config.vocab_size]`
- **logits_to_keep** (`Union[int, torch.Tensor]`, defaults to `0`) --
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).0[transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [ReformerModelWithLMHead](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerModelWithLMHead) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> import torch
>>> from transformers import AutoTokenizer, ReformerModelWithLMHead

>>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
>>> model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

**Parameters:**

config ([ReformerModelWithLMHead](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerModelWithLMHead)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## ReformerForMaskedLM[[transformers.ReformerForMaskedLM]]

#### transformers.ReformerForMaskedLM[[transformers.ReformerForMaskedLM]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py#L2269)

The Reformer Model with a `language modeling` head on top."

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.ReformerForMaskedLM.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py#L2289[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "num_hashes", "val": ": typing.Optional[int] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. During training the input_ids sequence_length has to be
  a multiple of the relevant model's chunk lengths (lsh's, local's or both). During evaluation, the indices
  are automatically padded to be a multiple of the chunk length.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **num_hashes** (`int`, *optional*) --
  The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
  the default defined in `config.num_hashes`.

  For more information, see `num_hashes` in [ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig).
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
  config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
  the loss is only computed for the tokens with labels

  

  This example uses a false checkpoint since we don't have any available pretrained model for the masked language
  modeling task with the Reformer architecture.

  
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Masked language modeling (MLM) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [ReformerForMaskedLM](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> import torch
>>> from transformers import AutoTokenizer, ReformerForMaskedLM

>>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-reformer")
>>> model = ReformerForMaskedLM.from_pretrained("hf-internal-testing/tiny-random-reformer")

>>> # add mask_token
>>> tokenizer.add_special_tokens({"mask_token": "[MASK]"})
>>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

>>> # resize model's embedding matrix
>>> model.resize_token_embeddings(new_num_tokens=model.config.vocab_size + 1)
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # retrieve index of [MASK]
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
>>> predicted_token = tokenizer.decode(predicted_token_id)
```

```python
>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> # mask labels of non-[MASK] tokens
>>> labels = torch.where(
...     inputs.input_ids == tokenizer.mask_token_id, labels[:, : inputs["input_ids"].shape[-1]], -100
... )

>>> outputs = model(**inputs, labels=labels)
>>> loss = round(outputs.loss.item(), 2)
```

**Parameters:**

config ([ReformerForMaskedLM](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerForMaskedLM)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.MaskedLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Masked language modeling (MLM) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## ReformerForSequenceClassification[[transformers.ReformerForSequenceClassification]]

#### transformers.ReformerForSequenceClassification[[transformers.ReformerForSequenceClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py#L2407)

Reformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.ReformerForSequenceClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py#L2421[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "num_hashes", "val": ": typing.Optional[int] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. During training the input_ids sequence_length has to be
  a multiple of the relevant model's chunk lengths (lsh's, local's or both). During evaluation, the indices
  are automatically padded to be a multiple of the chunk length.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **num_hashes** (`int`, *optional*) --
  The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
  the default defined in `config.num_hashes`.

  For more information, see `num_hashes` in [ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig).
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [ReformerForSequenceClassification](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, ReformerForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
>>> model = ReformerForSequenceClassification.from_pretrained("google/reformer-crime-and-punishment")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> label = model.config.id2label[predicted_class_id]
```

```python
>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = ReformerForSequenceClassification.from_pretrained(
...     "google/reformer-crime-and-punishment", num_labels=num_labels
... )

>>> labels = torch.tensor(1)
>>> loss = model(**inputs, labels=labels).loss
```

**Parameters:**

config ([ReformerForSequenceClassification](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerForSequenceClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## ReformerForQuestionAnswering[[transformers.ReformerForQuestionAnswering]]

#### transformers.ReformerForQuestionAnswering[[transformers.ReformerForQuestionAnswering]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py#L2558)

The Reformer transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.ReformerForQuestionAnswering.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/reformer/modeling_reformer.py#L2570[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "num_hashes", "val": ": typing.Optional[int] = None"}, {"name": "start_positions", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "end_positions", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. During training the input_ids sequence_length has to be
  a multiple of the relevant model's chunk lengths (lsh's, local's or both). During evaluation, the indices
  are automatically padded to be a multiple of the chunk length.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **num_hashes** (`int`, *optional*) --
  The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
  the default defined in `config.num_hashes`.

  For more information, see `num_hashes` in [ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig).
- **start_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **end_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-end scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [ReformerForQuestionAnswering](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, ReformerForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
>>> model = ReformerForQuestionAnswering.from_pretrained("google/reformer-crime-and-punishment")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
...

>>> # target is "nice puppet"
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
...
```

**Parameters:**

config ([ReformerForQuestionAnswering](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerForQuestionAnswering)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/main/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-end scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
