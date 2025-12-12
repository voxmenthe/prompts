*This model was released on 2024-05-07 and added to Hugging Face Transformers on 2025-07-25.*

# xLSTM

## Overview

The xLSTM model was proposed in [xLSTM: Extended Long Short-Term Memory](https://huggingface.co/papers/2405.04517) by Maximilian Beck*, Korbinian Pöppel*, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter and Sepp Hochreiter.
xLSTM updates the original LSTM architecture to be competitive with Transformer models by introducing exponential gating, matrix memory expansion, and parallelizable training and ingestion.

The [7B model](https://huggingface.co/NX-AI/xLSTM-7b) variant was trained by the xLSTM team Maximilian Beck, Korbinian Pöppel, Phillip Lippe, Richard Kurle, Patrick Blies, Sebastian Böck and Sepp Hochreiter at NXAI.

The abstract from the paper is the following:

*In the 1990s, the constant error carousel and gating were introduced as the central ideas of the Long Short-Term Memory (LSTM). Since then, LSTMs have stood the test of time and contributed to numerous deep learning success stories, in particular they constituted the first Large Language Models (LLMs). However, the advent of the Transformer technology with parallelizable self-attention at its core marked the dawn of a new era, outpacing LSTMs at scale. We now raise a simple question: How far do we get in language modeling when scaling LSTMs to billions of parameters, leveraging the latest techniques from modern LLMs, but mitigating known limitations of LSTMs? Firstly, we introduce exponential gating with appropriate normalization and stabilization techniques. Secondly, we modify the LSTM memory structure, obtaining: (i) sLSTM with a scalar memory, a scalar update, and new memory mixing, (ii) mLSTM that is fully parallelizable with a matrix memory and a covariance update rule. Integrating these LSTM extensions into residual block backbones yields xLSTM blocks that are then residually stacked into xLSTM architectures. Exponential gating and modified memory structures boost xLSTM capabilities to perform favorably when compared to state-of-the-art Transformers and State Space Models, both in performance and scaling.*

This model was contributed by [NX-AI](https://huggingface.co/NX-AI).
The original code can be found [here](https://github.com/NX-AI/xlstm).

## xLSTMConfig

### class transformers.xLSTMConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlstm/configuration_xlstm.py#L60)

( vocab\_size: int = 50304 hidden\_size: int = 4096 embedding\_dim: typing.Optional[int] = None num\_hidden\_layers: typing.Optional[int] = 32 num\_blocks: typing.Optional[int] = None num\_heads: int = 8 use\_bias: bool = False norm\_reduction\_force\_float32: bool = True tie\_word\_embeddings: bool = False add\_out\_norm: bool = True norm\_eps: float = 1e-06 qk\_dim\_factor: float = 0.5 v\_dim\_factor: float = 1.0 chunkwise\_kernel: typing.Literal['chunkwise--native\_autograd', 'parallel--native\_autograd'] = 'chunkwise--native\_autograd' sequence\_kernel: typing.Literal['native\_sequence\_\_native'] = 'native\_sequence\_\_native' step\_kernel: typing.Literal['native'] = 'native' mode: typing.Literal['train', 'train\_with\_padding', 'inference'] = 'inference' chunk\_size: int = 64 return\_last\_states: bool = True autocast\_kernel\_dtype: typing.Literal['float32', 'bfloat16', 'float16'] = 'bfloat16' eps: float = 1e-06 inference\_state\_dtype: typing.Literal['float32', 'bfloat16', 'float16'] = 'float32' ffn\_proj\_factor: float = 2.667 ffn\_round\_up\_to\_multiple\_of: int = 64 gate\_soft\_cap: float = 15.0 output\_logit\_soft\_cap: float = 30.0 weight\_mode: typing.Literal['single', 'fused'] = 'single' use\_cache: bool = True pad\_token\_id: int = 1 bos\_token\_id: int = 0 eos\_token\_id: int = 2 max\_inference\_chunksize: int = 16384 \*\*kwargs  )

Parameters

* **vocab\_size** (int, optional, *optional*, defaults to 50304) —
  Vocabulary size of the xLSTM model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [xLSTMModel](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMModel). Defaults to the GPT2-NeoX tokenizer size.
* **hidden\_size** (int, optional, *optional*, defaults to 4096) —
  Dimensionality of the embeddings or hidden states.
* **embedding\_dim** (int, optional, *optional*, defaults to 4096) —
  Dimensionality of the embeddings or hidden states, use hidde\_size if None.
* **num\_hidden\_layers** (int, optional, *optional*, defaults to 32) —
  Number of blocks of the xLSTM model.
* **num\_blocks** (int, optional, *optional*, defaults to 32) —
  Number of blocks of the xLSTM model, use num\_hidden\_layers if None.
* **num\_heads** (int, optional, *optional*, defaults to 8) —
  Number of heads for the xLSTM Layer/Cell.
* **use\_bias** (bool, optional, *optional*, defaults to `False`) —
  Whether to use biases in the xLSTM model.
* **norm\_reduction\_force\_float32** (bool, optional, *optional*, defaults to `True`) —
  Whether to force the float32 norm reduction op to be done in fp32 precision.
* **tie\_word\_embeddings** (bool, optional, *optional*, defaults to `False`) —
  Whether to tie word embeddings to the lm head weights.
* **add\_out\_norm** (bool, optional, *optional*, defaults to `True`) —
  Whether to add an output norm after the blocks before the LMHead.
* **norm\_eps** (float, optional, *optional*, defaults to 1e-06) —
  Norm eps for RMSNorm and Layer Norm.
* **qk\_dim\_factor** (float, optional, *optional*, defaults to 0.5) —
  Scale factor for the query and key dimension.
* **v\_dim\_factor** (float, optional, *optional*, defaults to 1.0) —
  Scale factor for the value dimension.
* **chunkwise\_kernel** (ChunkwiseKernelType, optional, *optional*, defaults to `"chunkwise--native_autograd"`) —
  Kernel type for chunkwise processing mode.
* **sequence\_kernel** (SequenceKernelType, optional, *optional*, defaults to `"native_sequence__native"`) —
  Kernel type for sequence processing mode.
* **step\_kernel** (StepKernelType, optional, *optional*, defaults to `"native"`) —
  Kernel type for step processing mode.
* **mode** (BackendModeType, optional, *optional*, defaults to `"inference"`) —
  Operation mode (inference is needed for generation).
* **chunk\_size** (int, optional, *optional*, defaults to 64) —
  Internal chunk size.
* **return\_last\_states** (bool, optional, *optional*, defaults to `True`) —
  If to return the last states / cache internally. Needed as True for generation.
* **autocast\_kernel\_dtype** (DtypeType, optional, *optional*, defaults to `"bfloat16"`) —
  Kernel dtype for the states.
* **eps** (float, optional, *optional*, defaults to 1e-06) —
  Epsilon for the mLSTM cell post norm.
* **inference\_state\_dtype** (DtypeType, optional, *optional*, defaults to `"float32"`) —
  Kernel dtype for states in inference.
* **ffn\_proj\_factor** (float, optional, *optional*, defaults to 2.667) —
  Size factor of the post-up projection gated Feed Forward network.
* **ffn\_round\_up\_to\_multiple\_of** (int, optional, *optional*, defaults to 64) —
  Size factor round value of the post-up projection gated Feed Forward network.
* **gate\_soft\_cap** (float, optional, *optional*, defaults to 15.0) —
  Gate soft cap scale.
* **output\_logit\_soft\_cap** (float, optional, *optional*, defaults to 30.0) —
  Output logit soft cap scale.
* **weight\_mode** (`Literal`, *optional*, defaults to `"single"`) —
  Whether parallel linear layers are separated or fused (single).
* **use\_cache** (bool, optional, *optional*, defaults to `True`) —
  Whether to use the cache (xLSTMCache).
* **pad\_token\_id** (int, optional, *optional*, defaults to 1) —
  Pad token id needed for generation.
* **bos\_token\_id** (int, optional, *optional*, defaults to 0) —
  BOS token id needed for generation.
* **eos\_token\_id** (int, optional, *optional*, defaults to 2) —
  EOS token id needed for generation.
* **max\_inference\_chunksize** (int, optional, *optional*, defaults to 16384) —
  Limit the chunk size for inference to save memory.

This is the configuration class to store the configuration of a `xLSTM`. It is used to instantiate a xLSTM
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the xLSTM-7b [NX-AI/xLSTM-7b](https://huggingface.co/NX-AI/xLSTM-7b) model.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import xLSTMConfig, xLSTMModel

>>> # Initializing a xLSTM configuration
>>> configuration = xLSTMConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = xLSTMModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## xLSTMModel

### class transformers.xLSTMModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlstm/modeling_xlstm.py#L1377)

( config  )

Parameters

* **config** ([xLSTMModel](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Xlstm Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlstm/modeling_xlstm.py#L1394)

( input\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.LongTensor] = None cache\_params: typing.Optional[transformers.models.xlstm.modeling\_xlstm.xLSTMCache] = None use\_cache: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None \*\*kwargs  ) → `transformers.models.xlstm.modeling_xlstm.xLSTMOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **inputs\_embeds** (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **cache\_params** (`xLSTMCache`, *optional*) —
  The xLSTMCache that carries the RNN states.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.models.xlstm.modeling_xlstm.xLSTMOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.xlstm.modeling_xlstm.xLSTMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([xLSTMConfig](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the model.
* **cache\_params** (`~models.xlstm.modeling_xlstm.xLSTMCache`, *optional*, defaults to `None`) — The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
  avoid providing the old `input_ids`.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [xLSTMModel](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## xLSTMLMHeadModel

### class transformers.xLSTMForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlstm/modeling_xlstm.py#L1515)

( config  )

Parameters

* **config** ([xLSTMForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Xlstm Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlstm/modeling_xlstm.py#L1561)

( input\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None cache\_params: typing.Optional[transformers.models.xlstm.modeling\_xlstm.xLSTMCache] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None \*\*kwargs  ) → `transformers.models.xlstm.modeling_xlstm.xLSTMCausalLMOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **cache\_params** (`xLSTMCache`, *optional*) —
  The xLSTMCache that carries the RNN states.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.models.xlstm.modeling_xlstm.xLSTMCausalLMOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.xlstm.modeling_xlstm.xLSTMCausalLMOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([xLSTMConfig](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **cache\_params** (`xLSTMCache`, *optional*, carrying the RNN states) — The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
  avoid providing the old `input_ids`.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [xLSTMForCausalLM](/docs/transformers/v4.56.2/en/model_doc/xlstm#transformers.xLSTMForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/xlstm.md)
