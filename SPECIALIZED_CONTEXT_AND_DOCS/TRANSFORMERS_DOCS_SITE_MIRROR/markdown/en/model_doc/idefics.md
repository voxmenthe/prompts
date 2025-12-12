# IDEFICS

## Overview

The IDEFICS model was proposed in [OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents](https://huggingface.co/papers/2306.16527) by Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh

The abstract from the paper is the following:

*Large multimodal models trained on natural documents, which interleave images and text, outperform models trained on image-text pairs on various multimodal benchmarks that require reasoning over one or multiple images to generate a text. However, the datasets used to train these models have not been released, and the collection process has not been fully specified. We introduce the OBELICS dataset, an open web-scale filtered dataset of interleaved image-text documents comprising 141 million web pages extracted from Common Crawl, 353 million associated images, and 115 billion text tokens. We describe the dataset creation process, present comprehensive filtering rules, and provide an analysis of the dataset's content. To show the viability of OBELISC, we train an 80 billion parameters vision and language model on the dataset and obtain competitive performance on various multimodal benchmarks. We release the code to reproduce the dataset along with the dataset itself.*

This model was contributed by [HuggingFaceM4](https://huggingface.co/HuggingFaceM4). The original code can be found [here](). (TODO: don't have a public link yet).

IDEFICS modeling code in Transformers is for finetuning and inferencing the pre-trained IDEFICS models.

To train a new IDEFICS model from scratch use the m4 codebase (a link will be provided once it's made public)

## IdeficsConfig[[transformers.IdeficsConfig]]

#### transformers.IdeficsConfig[[transformers.IdeficsConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/configuration_idefics.py#L154)

This is the configuration class to store the configuration of a [IdeficsModel](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsModel). It is used to instantiate an
Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Idefics-9B.

e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import IdeficsModel, IdeficsConfig

>>> # Initializing a Idefics idefics-9b style configuration
>>> configuration = IdeficsConfig()

>>> # Initializing a model from the idefics-9b style configuration
>>> model = IdeficsModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

additional_vocab_size (`int`, *optional*, defaults to 0) : Additional vocabulary size of the model, typically for the special "" token. Additional vocab tokens are always trainable whereas regular vocab tokens can be frozen or not.

vocab_size (`int`, *optional*, defaults to 32000) : Vocabulary size of the Idefics model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [~IdeficsModel](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsModel)

hidden_size (`int`, *optional*, defaults to 4096) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to 11008) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to 32) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 32) : Number of attention heads for each attention layer in the Transformer encoder.

dropout (`float`, *optional*, defaults to 0.0) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

hidden_act (`str` or `function`, *optional*, defaults to `"silu"`) : The non-linear activation function (function or string) in the decoder.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

alpha_initializer (`str`, *optional*, defaults to `"zeros"`) : Initialization type for the alphas.

alphas_initializer_range (`float`, *optional*, defaults to 0.0) : The standard deviation of the truncated_normal_initializer for initializing the alphas in the Gated Cross Attention.

alpha_type (`str`, *optional*, defaults to `"float"`) : Whether the gating alphas should be vectors or single floats.

rms_norm_eps (`float`, *optional*, defaults to 1e-6) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.

pad_token_id (`int`, *optional*, defaults to 0) : Padding token id.

bos_token_id (`int`, *optional*, defaults to 1) : Beginning of stream token id.

eos_token_id (`int`, *optional*, defaults to 2) : End of stream token id.

tie_word_embeddings(`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings

cross_layer_interval (`int`, *optional*, default to 1) : Interval for cross attention (from text to image) layers.

qk_layer_norms (`bool`, *optional*, defaults to `False`) : Whether to add layer norm after q and k

freeze_text_layers (`bool`, *optional*, defaults to `True`) : Whether to freeze text layers

freeze_text_module_exceptions (`bool`, *optional*, defaults to `[]`) : Exceptions to freezing text layers when `freeze_text_layers` is `True`

freeze_lm_head (`bool`, *optional*, defaults to `False`) : Whether to freeze lm head

freeze_vision_layers (`bool`, *optional*, defaults to `True`) : Whether to freeze vision layers

freeze_vision_module_exceptions (`bool`, *optional*, defaults to `[]`) : Exceptions to freezing vision layers when `freeze_vision_layers` is `True`

use_resampler (`bool`, *optional*, defaults to `False`) : Whether to use the Resampler

vision_config (`IdeficsVisionConfig`,  *optional*) : Custom vision config or dict

perceiver_config (`IdeficsPerceiverConfig`,  *optional*) : Custom perceiver config or dict

## IdeficsModel[[transformers.IdeficsModel]]

#### transformers.IdeficsModel[[transformers.IdeficsModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/modeling_idefics.py#L858)

The bare Idefics Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.IdeficsModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/modeling_idefics.py#L933[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "image_encoder_embeddings", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "perceiver_embeddings", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "image_attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": typing.Optional[bool] = False"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [IdeficsImageProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsImageProcessor). See [IdeficsImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([IdeficsProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsProcessor) uses
  [IdeficsImageProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsImageProcessor) for processing images).
- **image_encoder_embeddings** (`torch.FloatTensor`, *optional*) --
  The output of the image encoder.
- **perceiver_embeddings** (`torch.FloatTensor`, *optional*) --
  The output of the perceiver resampler.
- **image_attention_mask** (`torch.LongTensor`, *optional*) --
  The attention mask for the image encoder.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **interpolate_pos_encoding** (`bool`, *optional*, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.0`transformers.models.idefics.modeling_idefics.IdeficsBaseModelOutputWithPast` or `tuple(torch.FloatTensor)`A `transformers.models.idefics.modeling_idefics.IdeficsBaseModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([IdeficsConfig](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`tuple(torch.FloatTensor)`, *optional*) -- Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
  sequence_length, hidden_size)`.

  image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
The [IdeficsModel](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([IdeficsConfig](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.idefics.modeling_idefics.IdeficsBaseModelOutputWithPast` or `tuple(torch.FloatTensor)``

A `transformers.models.idefics.modeling_idefics.IdeficsBaseModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([IdeficsConfig](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`tuple(torch.FloatTensor)`, *optional*) -- Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
  sequence_length, hidden_size)`.

  image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver

## IdeficsForVisionText2Text[[transformers.IdeficsForVisionText2Text]]

#### transformers.IdeficsForVisionText2Text[[transformers.IdeficsForVisionText2Text]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/modeling_idefics.py#L1096)

forwardtransformers.IdeficsForVisionText2Text.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/modeling_idefics.py#L1119[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "image_encoder_embeddings", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "perceiver_embeddings", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "image_attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": typing.Optional[bool] = False"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "logits_to_keep", "val": ": typing.Union[int, torch.Tensor] = 0"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [IdeficsImageProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsImageProcessor). See [IdeficsImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([IdeficsProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsProcessor) uses
  [IdeficsImageProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsImageProcessor) for processing images).
- **image_encoder_embeddings** (`torch.FloatTensor`, *optional*) --
  The output of the image encoder.
- **perceiver_embeddings** (`torch.FloatTensor`, *optional*) --
  The output of the perceiver resampler.
- **image_attention_mask** (`torch.LongTensor`, *optional*) --
  The attention mask for the image encoder.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **interpolate_pos_encoding** (`bool`, *optional*, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
- **logits_to_keep** (`Union[int, torch.Tensor]`, defaults to `0`) --
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).0`transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`A `transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([IdeficsConfig](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`tuple(torch.FloatTensor)`, *optional*) -- Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
  sequence_length, hidden_size)`.

  image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
The [IdeficsForVisionText2Text](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsForVisionText2Text) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoProcessor, IdeficsForVisionText2Text

>>> model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")
>>> processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b")

>>> dogs_image_url_1 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image1.jpeg"
>>> dogs_image_url_2 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image2.jpeg"

>>> prompts = [
...     [
...         "User:",
...         dogs_image_url_1,
...         "Describe this image.\nAssistant: An image of two dogs.\n",
...         "User:",
...         dogs_image_url_2,
...         "Describe this image.\nAssistant:",
...     ]
... ]
>>> inputs = processor(prompts, return_tensors="pt")
>>> generate_ids = model.generate(**inputs, max_new_tokens=6)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True)
```

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [IdeficsImageProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsImageProcessor). See [IdeficsImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([IdeficsProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsProcessor) uses [IdeficsImageProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsImageProcessor) for processing images).

image_encoder_embeddings (`torch.FloatTensor`, *optional*) : The output of the image encoder.

perceiver_embeddings (`torch.FloatTensor`, *optional*) : The output of the perceiver resampler.

image_attention_mask (`torch.LongTensor`, *optional*) : The attention mask for the image encoder.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

interpolate_pos_encoding (`bool`, *optional*, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*) : Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`, this tensor is not affected by padding. It is used to update the cache in the correct position and to infer the complete sequence length.

logits_to_keep (`Union[int, torch.Tensor]`, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:**

``transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast` or `tuple(torch.FloatTensor)``

A `transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([IdeficsConfig](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`tuple(torch.FloatTensor)`, *optional*) -- Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
  sequence_length, hidden_size)`.

  image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver

## IdeficsImageProcessor[[transformers.IdeficsImageProcessor]]

#### transformers.IdeficsImageProcessor[[transformers.IdeficsImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/image_processing_idefics.py#L67)

Constructs a Idefics image processor.

preprocesstransformers.IdeficsImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/image_processing_idefics.py#L114[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "image_num_channels", "val": ": typing.Optional[int] = 3"}, {"name": "image_size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "transform", "val": ": typing.Optional[collections.abc.Callable] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = "}, {"name": "**kwargs", "val": ""}]- **images** (`ImageInput`) --
  A list of images to preprocess.
- **image_size** (`int`, *optional*, defaults to `self.image_size`) --
  Resize to image size
- **image_num_channels** (`int`, *optional*, defaults to `self.image_num_channels`) --
  Number of image channels.
- **image_mean** (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`) --
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can
  be overridden by the `image_mean` parameter in the `preprocess` method.
- **image_std** (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`) --
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess`
  method. Can be overridden by the `image_std` parameter in the `preprocess` method.
- **transform** (`Callable`, *optional*, defaults to `None`) --
  A custom transform function that accepts a single image can be passed for training. For example,
  `torchvision.Compose` can be used to compose multiple transforms. If `None` - an inference mode is
  assumed - and then a preset of inference-specific transforms will be applied to the images
- **do_rescale** (`bool`, *optional*, defaults to `True`) --
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
- **rescale_factor** (`int` or `float`, *optional*, defaults to `1/255`) --
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.0a PyTorch tensor of the processed images

Preprocess a batch of images.

**Parameters:**

image_size (`int`, *optional*, defaults to 224) : Resize to image size

image_mean (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`) : Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method. Can be overridden by the `image_std` parameter in the `preprocess` method.

image_num_channels (`int`, *optional*, defaults to 3) : Number of image channels.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in the `preprocess` method.

rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess` method.

**Returns:**

a PyTorch tensor of the processed images

## IdeficsProcessor[[transformers.IdeficsProcessor]]

#### transformers.IdeficsProcessor[[transformers.IdeficsProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/processing_idefics.py#L137)

Constructs a IDEFICS processor which wraps a LLama tokenizer and IDEFICS image processor into a single processor.

[IdeficsProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsProcessor) offers all the functionalities of [IdeficsImageProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsImageProcessor) and [LlamaTokenizerFast](/docs/transformers/main/en/model_doc/llama2#transformers.LlamaTokenizer). See
the docstring of [__call__()](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsProcessor.__call__) and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

__call__transformers.IdeficsProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics/processing_idefics.py#L173[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]], str, list[str], list[list[str]]] = None"}, {"name": "text", "val": ": typing.Union[str, list[str], list[list[str]], list[list[list[str]]]] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.idefics.processing_idefics.IdeficsProcessorKwargs]"}]- **images** (`Union[ImageInput, list[ImageInput], str, list[str], list[list[str]]]`) --
  either a single image or a batched list of images - can be passed in when text contains only text prompts,
  in order to use the image-text-to-text behavior.
- **text** (`Union[list[TextInput], [list[list[TextInput]]]]`) --
  either a single prompt or a batched list of prompts - see the detailed description immediately after
  the end of the arguments doc section.
- **return_tensors** (`str` or `TensorType`, *optional*, defaults to `TensorType.PYTORCH`) --
  The type of tensors to return. Can be one of:
  - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.0a dict with entries`input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be
directly passed to `model.generate`
This method takes batched or non-batched prompts made of text and images and converts them into prompts that
the model was trained on and prepares the image pixel values for the model to process.

Detailed explanation:

Each entry in `text` is either a text to be passed as is or an image that will be processed.

An image can be either an image object (`PIL.Image`) or a url from which the image can be retrieved.

When the processor encounters an image it'll inject ``
entry into the prompt.

Example:

```python
checkpoint = "HuggingFaceM4/idefics-9b"
processor = AutoProcessor.from_pretrained(checkpoint)
url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
img = processor.image_processor.fetch_images([url])[0]

prompts = [
    "User:",
    img,
    "Describe this image.
t: An image of two kittens in grass.

    "User:",
    "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg",
    "Describe this image.
t:",
]

inputs = processor(text=prompts, return_tensors="pt")
generated_ids = model.generate(**inputs, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

In this example the `prompts` will be converted into:

```
User:Describe this image.
Assistant: An image of two kittens in grass.
User:Describe this image.
Assistant:'
```

and the two images will be massaged using [IdeficsImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) method and placed inside the
`pixel_values` dict entry of the return value.

This example also exemplifies that images can be passed as objects or as text urls. It can be seen that the
first image is passed as object and the second one as a url.

To do training do:

```python
image_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            (w, h), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=self.image_mean, std=self.image_std),
    ]
)
inputs = processor(text=prompts, transform=image_transform, return_tensors="pt")
```

In order to help debug prompt generation enable `debug=True` which will show you what's happening.

**Parameters:**

image_processor (`IdeficsImageProcessor`) : An instance of [IdeficsImageProcessor](/docs/transformers/main/en/model_doc/idefics#transformers.IdeficsImageProcessor). The image processor is a required input.

tokenizer (`LlamaTokenizerFast`) : An instance of [LlamaTokenizerFast](/docs/transformers/main/en/model_doc/llama2#transformers.LlamaTokenizer). The tokenizer is a required input.

image_size (`int`, *optional*, defaults to 224) : Image size (assuming a square image)

add_end_of_utterance_token (`str`, *optional*) : The string representation of token representing end of utterance

**Returns:**

`a dict with entries`

`input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be
directly passed to `model.generate`
