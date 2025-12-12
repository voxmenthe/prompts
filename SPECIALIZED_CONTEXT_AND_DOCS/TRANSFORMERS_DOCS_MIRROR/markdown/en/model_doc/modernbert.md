*This model was released on 2024-12-18 and added to Hugging Face Transformers on 2024-12-19.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# ModernBERT

[ModernBERT](https://huggingface.co/papers/2412.13663) is a modernized version of `BERT` trained on 2T tokens. It brings many improvements to the original architecture such as rotary positional embeddings to support sequences of up to 8192 tokens, unpadding to avoid wasting compute on padding tokens, GeGLU layers, and alternating attention.

You can find all the original ModernBERT checkpoints under the [ModernBERT](https://huggingface.co/collections/answerdotai/modernbert-67627ad707a4acbf33c41deb) collection.

Click on the ModernBERT models in the right sidebar for more examples of how to apply ModernBERT to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel), and from the command line.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="answerdotai/ModernBERT-base",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

## ModernBertConfig

### class transformers.ModernBertConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/configuration_modernbert.py#L27)

( vocab\_size = 50368 hidden\_size = 768 intermediate\_size = 1152 num\_hidden\_layers = 22 num\_attention\_heads = 12 hidden\_activation = 'gelu' max\_position\_embeddings = 8192 initializer\_range = 0.02 initializer\_cutoff\_factor = 2.0 norm\_eps = 1e-05 norm\_bias = False pad\_token\_id = 50283 eos\_token\_id = 50282 bos\_token\_id = 50281 cls\_token\_id = 50281 sep\_token\_id = 50282 global\_rope\_theta = 160000.0 attention\_bias = False attention\_dropout = 0.0 global\_attn\_every\_n\_layers = 3 local\_attention = 128 local\_rope\_theta = 10000.0 embedding\_dropout = 0.0 mlp\_bias = False mlp\_dropout = 0.0 decoder\_bias = True classifier\_pooling: typing.Literal['cls', 'mean'] = 'cls' classifier\_dropout = 0.0 classifier\_bias = False classifier\_activation = 'gelu' deterministic\_flash\_attn = False sparse\_prediction = False sparse\_pred\_ignore\_index = -100 reference\_compile = None repad\_logits\_with\_grad = False \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 50368) —
  Vocabulary size of the ModernBert model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [ModernBertModel](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertModel)
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 1152) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 22) —
  Number of hidden layers in the Transformer decoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **hidden\_activation** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the decoder. Will default to `"gelu"`
  if not specified.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 8192) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **initializer\_cutoff\_factor** (`float`, *optional*, defaults to 2.0) —
  The cutoff factor for the truncated\_normal\_initializer for initializing all weight matrices.
* **norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **norm\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use bias in the normalization layers.
* **pad\_token\_id** (`int`, *optional*, defaults to 50283) —
  Padding token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 50282) —
  End of stream token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 50281) —
  Beginning of stream token id.
* **cls\_token\_id** (`int`, *optional*, defaults to 50281) —
  Classification token id.
* **sep\_token\_id** (`int`, *optional*, defaults to 50282) —
  Separation token id.
* **global\_rope\_theta** (`float`, *optional*, defaults to 160000.0) —
  The base period of the global RoPE embeddings.
* **attention\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **global\_attn\_every\_n\_layers** (`int`, *optional*, defaults to 3) —
  The number of layers between global attention layers.
* **local\_attention** (`int`, *optional*, defaults to 128) —
  The window size for local attention.
* **local\_rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the local RoPE embeddings.
* **embedding\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the embeddings.
* **mlp\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use bias in the MLP layers.
* **mlp\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the MLP layers.
* **decoder\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to use bias in the decoder layers.
* **classifier\_pooling** (`str`, *optional*, defaults to `"cls"`) —
  The pooling method for the classifier. Should be either `"cls"` or `"mean"`. In local attention layers, the
  CLS token doesn’t attend to all tokens on long sequences.
* **classifier\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the classifier.
* **classifier\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use bias in the classifier.
* **classifier\_activation** (`str`, *optional*, defaults to `"gelu"`) —
  The activation function for the classifier.
* **deterministic\_flash\_attn** (`bool`, *optional*, defaults to `False`) —
  Whether to use deterministic flash attention. If `False`, inference will be faster but not deterministic.
* **sparse\_prediction** (`bool`, *optional*, defaults to `False`) —
  Whether to use sparse prediction for the masked language model instead of returning the full dense logits.
* **sparse\_pred\_ignore\_index** (`int`, *optional*, defaults to -100) —
  The index to ignore for the sparse prediction.
* **reference\_compile** (`bool`, *optional*) —
  Whether to compile the layers of the model which were compiled during pretraining. If `None`, then parts of
  the model will be compiled if 1) `triton` is installed, 2) the model is not on MPS, 3) the model is not
  shared between devices, and 4) the model is not resized after initialization. If `True`, then the model may
  be faster in some scenarios.
* **repad\_logits\_with\_grad** (`bool`, *optional*, defaults to `False`) —
  When True, ModernBertForMaskedLM keeps track of the logits’ gradient when repadding for output. This only
  applies when using Flash Attention 2 with passed labels. Otherwise output logits always have a gradient.

This is the configuration class to store the configuration of a [ModernBertModel](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertModel). It is used to instantiate an ModernBert
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the ModernBERT-base.
e.g. [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import ModernBertModel, ModernBertConfig

>>> # Initializing a ModernBert style configuration
>>> configuration = ModernBertConfig()

>>> # Initializing a model from the modernbert-base style configuration
>>> model = ModernBertModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

Pytorch

Hide Pytorch content

## ModernBertModel

### class transformers.ModernBertModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L764)

( config: ModernBertConfig  )

Parameters

* **config** ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Modernbert Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L782)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None sliding\_window\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None indices: typing.Optional[torch.Tensor] = None cu\_seqlens: typing.Optional[torch.Tensor] = None max\_seqlen: typing.Optional[int] = None batch\_size: typing.Optional[int] = None seq\_len: typing.Optional[int] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **sliding\_window\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
  perform global attention, while the rest perform local attention. This mask is used to avoid attending to
  far-away tokens in the local attention layers when not using Flash Attention.
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **indices** (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*) —
  Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
* **cu\_seqlens** (`torch.Tensor` of shape `(batch + 1,)`, *optional*) —
  Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
* **max\_seqlen** (`int`, *optional*) —
  Maximum sequence length in the batch excluding padding tokens. Used to unpad input\_ids and pad output tensors.
* **batch\_size** (`int`, *optional*) —
  Batch size of the input sequences. Used to pad the output tensors.
* **seq\_len** (`int`, *optional*) —
  Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ModernBertModel](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## ModernBertForMaskedLM

### class transformers.ModernBertForMaskedLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L954)

( config: ModernBertConfig  )

Parameters

* **config** ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ModernBert Model with a decoder head on top that is used for masked language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L980)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None sliding\_window\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None indices: typing.Optional[torch.Tensor] = None cu\_seqlens: typing.Optional[torch.Tensor] = None max\_seqlen: typing.Optional[int] = None batch\_size: typing.Optional[int] = None seq\_len: typing.Optional[int] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **sliding\_window\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
  perform global attention, while the rest perform local attention. This mask is used to avoid attending to
  far-away tokens in the local attention layers when not using Flash Attention.
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **indices** (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*) —
  Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
* **cu\_seqlens** (`torch.Tensor` of shape `(batch + 1,)`, *optional*) —
  Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
* **max\_seqlen** (`int`, *optional*) —
  Maximum sequence length in the batch excluding padding tokens. Used to unpad input\_ids and pad output tensors.
* **batch\_size** (`int`, *optional*) —
  Batch size of the input sequences. Used to pad the output tensors.
* **seq\_len** (`int`, *optional*) —
  Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ModernBertForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, ModernBertForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
>>> model = ModernBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")

>>> inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # retrieve index of <mask>
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
>>> tokenizer.decode(predicted_token_id)
...

>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> # mask labels of non-<mask> tokens
>>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

>>> outputs = model(**inputs, labels=labels)
>>> round(outputs.loss.item(), 2)
...
```

## ModernBertForSequenceClassification

### class transformers.ModernBertForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L1098)

( config: ModernBertConfig  )

Parameters

* **config** ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ModernBert Model with a sequence classification head on top that performs pooling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L1112)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None sliding\_window\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None indices: typing.Optional[torch.Tensor] = None cu\_seqlens: typing.Optional[torch.Tensor] = None max\_seqlen: typing.Optional[int] = None batch\_size: typing.Optional[int] = None seq\_len: typing.Optional[int] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **sliding\_window\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
  perform global attention, while the rest perform local attention. This mask is used to avoid attending to
  far-away tokens in the local attention layers when not using Flash Attention.
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **indices** (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*) —
  Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
* **cu\_seqlens** (`torch.Tensor` of shape `(batch + 1,)`, *optional*) —
  Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
* **max\_seqlen** (`int`, *optional*) —
  Maximum sequence length in the batch excluding padding tokens. Used to unpad input\_ids and pad output tensors.
* **batch\_size** (`int`, *optional*) —
  Batch size of the input sequences. Used to pad the output tensors.
* **seq\_len** (`int`, *optional*) —
  Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ModernBertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, ModernBertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
>>> model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, ModernBertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
>>> model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = ModernBertForSequenceClassification.from_pretrained(
...     "answerdotai/ModernBERT-base", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## ModernBertForTokenClassification

### class transformers.ModernBertForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L1235)

( config: ModernBertConfig  )

Parameters

* **config** ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ModernBert Model with a token classification head on top, e.g. for Named Entity Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L1248)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None sliding\_window\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None indices: typing.Optional[torch.Tensor] = None cu\_seqlens: typing.Optional[torch.Tensor] = None max\_seqlen: typing.Optional[int] = None batch\_size: typing.Optional[int] = None seq\_len: typing.Optional[int] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **sliding\_window\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
  perform global attention, while the rest perform local attention. This mask is used to avoid attending to
  far-away tokens in the local attention layers when not using Flash Attention.
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
* **indices** (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*) —
  Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
* **cu\_seqlens** (`torch.Tensor` of shape `(batch + 1,)`, *optional*) —
  Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
* **max\_seqlen** (`int`, *optional*) —
  Maximum sequence length in the batch excluding padding tokens. Used to unpad input\_ids and pad output tensors.
* **batch\_size** (`int`, *optional*) —
  Batch size of the input sequences. Used to pad the output tensors.
* **seq\_len** (`int`, *optional*) —
  Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ModernBertForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, ModernBertForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
>>> model = ModernBertForTokenClassification.from_pretrained("answerdotai/ModernBERT-base")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
... )

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_token_class_ids = logits.argmax(-1)

>>> # Note that tokens are classified rather then input words which means that
>>> # there might be more predicted token classes than words.
>>> # Multiple token classes might account for the same word
>>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
>>> predicted_tokens_classes
...

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

## ModernBertForMultipleChoice

### class transformers.ModernBertForMultipleChoice

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L1422)

( config: ModernBertConfig  )

Parameters

* **config** ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ModernBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L1435)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None sliding\_window\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None indices: typing.Optional[torch.Tensor] = None cu\_seqlens: typing.Optional[torch.Tensor] = None max\_seqlen: typing.Optional[int] = None batch\_size: typing.Optional[int] = None seq\_len: typing.Optional[int] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **sliding\_window\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
  perform global attention, while the rest perform local attention. This mask is used to avoid attending to
  far-away tokens in the local attention layers when not using Flash Attention.
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors.
* **indices** (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*) —
  Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
* **cu\_seqlens** (`torch.Tensor` of shape `(batch + 1,)`, *optional*) —
  Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
* **max\_seqlen** (`int`, *optional*) —
  Maximum sequence length in the batch excluding padding tokens. Used to unpad input\_ids and pad output tensors.
* **batch\_size** (`int`, *optional*) —
  Batch size of the input sequences. Used to pad the output tensors.
* **seq\_len** (`int`, *optional*) —
  Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — *num\_choices* is the second dimension of the input tensors. (see *input\_ids* above).

  Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ModernBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, ModernBertForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
>>> model = ModernBertForMultipleChoice.from_pretrained("answerdotai/ModernBERT-base")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."
>>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

>>> # the linear classifier still needs to be trained
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## ModernBertForQuestionAnswering

### class transformers.ModernBertForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L1326)

( config: ModernBertConfig  )

Parameters

* **config** ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Modernbert transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/modernbert/modeling_modernbert.py#L1338)

( input\_ids: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None sliding\_window\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None start\_positions: typing.Optional[torch.Tensor] = None end\_positions: typing.Optional[torch.Tensor] = None indices: typing.Optional[torch.Tensor] = None cu\_seqlens: typing.Optional[torch.Tensor] = None max\_seqlen: typing.Optional[int] = None batch\_size: typing.Optional[int] = None seq\_len: typing.Optional[int] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **sliding\_window\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
  perform global attention, while the rest perform local attention. This mask is used to avoid attending to
  far-away tokens in the local attention layers when not using Flash Attention.
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **start\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **indices** (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*) —
  Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
* **cu\_seqlens** (`torch.Tensor` of shape `(batch + 1,)`, *optional*) —
  Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
* **max\_seqlen** (`int`, *optional*) —
  Maximum sequence length in the batch excluding padding tokens. Used to unpad input\_ids and pad output tensors.
* **batch\_size** (`int`, *optional*) —
  Batch size of the input sequences. Used to pad the output tensors.
* **seq\_len** (`int`, *optional*) —
  Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ModernBertConfig](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ModernBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/modernbert#transformers.ModernBertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, ModernBertForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
>>> model = ModernBertForQuestionAnswering.from_pretrained("answerdotai/ModernBERT-base")

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

### Usage tips

The ModernBert model can be fine-tuned using the HuggingFace Transformers library with its [official script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py) for question-answering tasks.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/modernbert.md)
