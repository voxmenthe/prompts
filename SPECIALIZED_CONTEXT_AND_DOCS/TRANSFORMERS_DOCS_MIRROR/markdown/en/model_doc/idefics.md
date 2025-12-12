*This model was released on 2023-06-21 and added to Hugging Face Transformers on 2023-08-18.*

# IDEFICS

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The IDEFICS model was proposed in [OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents](https://huggingface.co/papers/2306.16527) by Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh

The abstract from the paper is the following:

*Large multimodal models trained on natural documents, which interleave images and text, outperform models trained on image-text pairs on various multimodal benchmarks that require reasoning over one or multiple images to generate a text. However, the datasets used to train these models have not been released, and the collection process has not been fully specified. We introduce the OBELICS dataset, an open web-scale filtered dataset of interleaved image-text documents comprising 141 million web pages extracted from Common Crawl, 353 million associated images, and 115 billion text tokens. We describe the dataset creation process, present comprehensive filtering rules, and provide an analysis of the dataset’s content. To show the viability of OBELISC, we train an 80 billion parameters vision and language model on the dataset and obtain competitive performance on various multimodal benchmarks. We release the code to reproduce the dataset along with the dataset itself.*

This model was contributed by [HuggingFaceM4](https://huggingface.co/HuggingFaceM4). The original code can be found [here](INSERT%20LINK%20TO%20GITHUB%20REPO%20HERE). (TODO: don’t have a public link yet).

IDEFICS modeling code in Transformers is for finetuning and inferencing the pre-trained IDEFICS models.

To train a new IDEFICS model from scratch use the m4 codebase (a link will be provided once it’s made public)

## IdeficsConfig

### class transformers.IdeficsConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics/configuration_idefics.py#L154)

( vocab\_size = 32000 additional\_vocab\_size = 0 hidden\_size = 4096 intermediate\_size = 11008 num\_hidden\_layers = 32 num\_attention\_heads = 32 dropout = 0.0 hidden\_act = 'silu' initializer\_range = 0.02 alpha\_initializer = 'zeros' alphas\_initializer\_range = 0.0 alpha\_type = 'float' rms\_norm\_eps = 1e-06 use\_cache = True pad\_token\_id = 0 bos\_token\_id = 1 eos\_token\_id = 2 tie\_word\_embeddings = False cross\_layer\_interval = 1 qk\_layer\_norms = False freeze\_text\_layers = True freeze\_text\_module\_exceptions = [] freeze\_lm\_head = False freeze\_vision\_layers = True freeze\_vision\_module\_exceptions = [] use\_resampler = False vision\_config = None perceiver\_config = None \*\*kwargs  )

Parameters

* **additional\_vocab\_size** (`int`, *optional*, defaults to 0) —
  Additional vocabulary size of the model, typically for the special ”![]()” token. Additional vocab tokens
  are always trainable whereas regular vocab tokens can be frozen or not.
* **vocab\_size** (`int`, *optional*, defaults to 32000) —
  Vocabulary size of the Idefics model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [~IdeficsModel](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsModel)
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 11008) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 32) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **alpha\_initializer** (`str`, *optional*, defaults to `"zeros"`) —
  Initialization type for the alphas.
* **alphas\_initializer\_range** (`float`, *optional*, defaults to 0.0) —
  The standard deviation of the truncated\_normal\_initializer for initializing the alphas in the Gated Cross
  Attention.
* **alpha\_type** (`str`, *optional*, defaults to `"float"`) —
  Whether the gating alphas should be vectors or single floats.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-6) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 1) —
  Beginning of stream token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  End of stream token id.
* **tie\_word\_embeddings(`bool`,** *optional*, defaults to `False`) —
  Whether to tie weight embeddings
* **cross\_layer\_interval** (`int`, *optional*, default to 1) —
  Interval for cross attention (from text to image) layers.
* **qk\_layer\_norms** (`bool`, *optional*, defaults to `False`) — Whether to add layer norm after q and k
* **freeze\_text\_layers** (`bool`, *optional*, defaults to `True`) — Whether to freeze text layers
* **freeze\_text\_module\_exceptions** (`bool`, *optional*, defaults to `[]`) —
  Exceptions to freezing text layers when `freeze_text_layers` is `True`
* **freeze\_lm\_head** (`bool`, *optional*, defaults to `False`) — Whether to freeze lm head
* **freeze\_vision\_layers** (`bool`, *optional*, defaults to `True`) — Whether to freeze vision layers
* **freeze\_vision\_module\_exceptions** (`bool`, *optional*, defaults to `[]`) —
  Exceptions to freezing vision layers when `freeze_vision_layers` is `True`
* **use\_resampler** (`bool`, *optional*, defaults to `False`) — Whether to use the Resampler
* **vision\_config** (`IdeficsVisionConfig`, *optional*) — Custom vision config or dict
* **perceiver\_config** (`IdeficsPerceiverConfig`, *optional*) — Custom perceiver config or dict

This is the configuration class to store the configuration of a [IdeficsModel](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsModel). It is used to instantiate an
Idefics model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Idefics-9B.

e.g. [HuggingFaceM4/idefics-9b](https://huggingface.co/HuggingFaceM4/idefics-9b)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import IdeficsModel, IdeficsConfig

>>> # Initializing a Idefics idefics-9b style configuration
>>> configuration = IdeficsConfig()

>>> # Initializing a model from the idefics-9b style configuration
>>> model = IdeficsModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## IdeficsModel

### class transformers.IdeficsModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics/modeling_idefics.py#L925)

( config: IdeficsConfig  )

Parameters

* **config** ([IdeficsConfig](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Idefics Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics/modeling_idefics.py#L1000)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None image\_encoder\_embeddings: typing.Optional[torch.FloatTensor] = None perceiver\_embeddings: typing.Optional[torch.FloatTensor] = None image\_attention\_mask: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: typing.Optional[bool] = False return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.idefics.modeling_idefics.IdeficsBaseModelOutputWithPast` or `tuple(torch.FloatTensor)`

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
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [IdeficsImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsImageProcessor). See [IdeficsImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([IdeficsProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsProcessor) uses
  [IdeficsImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsImageProcessor) for processing images).
* **image\_encoder\_embeddings** (`torch.FloatTensor`, *optional*) —
  The output of the image encoder.
* **perceiver\_embeddings** (`torch.FloatTensor`, *optional*) —
  The output of the perceiver resampler.
* **image\_attention\_mask** (`torch.LongTensor`, *optional*) —
  The attention mask for the image encoder.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, *optional*, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.models.idefics.modeling_idefics.IdeficsBaseModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.idefics.modeling_idefics.IdeficsBaseModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([IdeficsConfig](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
  `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images, sequence_length, hidden_size)`.

  image\_hidden\_states of the model produced by the vision encoder, and optionally by the perceiver

The [IdeficsModel](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## IdeficsForVisionText2Text

### class transformers.IdeficsForVisionText2Text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics/modeling_idefics.py#L1327)

( config vision\_model = None  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics/modeling_idefics.py#L1366)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None image\_encoder\_embeddings: typing.Optional[torch.FloatTensor] = None perceiver\_embeddings: typing.Optional[torch.FloatTensor] = None image\_attention\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: typing.Optional[bool] = False return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

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
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [IdeficsImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsImageProcessor). See [IdeficsImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([IdeficsProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsProcessor) uses
  [IdeficsImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsImageProcessor) for processing images).
* **image\_encoder\_embeddings** (`torch.FloatTensor`, *optional*) —
  The output of the image encoder.
* **perceiver\_embeddings** (`torch.FloatTensor`, *optional*) —
  The output of the perceiver resampler.
* **image\_attention\_mask** (`torch.LongTensor`, *optional*) —
  The attention mask for the image encoder.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, *optional*, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([IdeficsConfig](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images, sequence_length, hidden_size)`.

  image\_hidden\_states of the model produced by the vision encoder, and optionally by the perceiver

The [IdeficsForVisionText2Text](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsForVisionText2Text) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

## IdeficsImageProcessor

### class transformers.IdeficsImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics/image_processing_idefics.py#L51)

( image\_size: int = 224 image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None image\_num\_channels: typing.Optional[int] = 3 do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 \*\*kwargs  )

Parameters

* **image\_size** (`int`, *optional*, defaults to 224) —
  Resize to image size
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **image\_num\_channels** (`int`, *optional*, defaults to 3) —
  Number of image channels.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.

Constructs a Idefics image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics/image_processing_idefics.py#L97)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] image\_num\_channels: typing.Optional[int] = 3 image\_size: typing.Optional[dict[str, int]] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None transform: typing.Optional[typing.Callable] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = <TensorType.PYTORCH: 'pt'> \*\*kwargs  )

Parameters

* **images** (`ImageInput`) —
  A list of images to preprocess.
* **image\_size** (`int`, *optional*, defaults to `self.image_size`) —
  Resize to image size
* **image\_num\_channels** (`int`, *optional*, defaults to `self.image_num_channels`) —
  Number of image channels.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can
  be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess`
  method. Can be overridden by the `image_std` parameter in the `preprocess` method.
* **transform** (`Callable`, *optional*, defaults to `None`) —
  A custom transform function that accepts a single image can be passed for training. For example,
  `torchvision.Compose` can be used to compose multiple transforms. If `None` - an inference mode is
  assumed - and then a preset of inference-specific transforms will be applied to the images
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.

Preprocess a batch of images.

## IdeficsProcessor

### class transformers.IdeficsProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics/processing_idefics.py#L194)

( image\_processor tokenizer = None image\_size = 224 add\_end\_of\_utterance\_token = None \*\*kwargs  )

Parameters

* **image\_processor** (`IdeficsImageProcessor`) —
  An instance of [IdeficsImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsImageProcessor). The image processor is a required input.
* **tokenizer** (`LlamaTokenizerFast`) —
  An instance of [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast). The tokenizer is a required input.
* **image\_size** (`int`, *optional*, defaults to 224) —
  Image size (assuming a square image)
* **add\_end\_of\_utterance\_token** (`str`, *optional*) —
  The string representation of token representing end of utterance

Constructs a IDEFICS processor which wraps a LLama tokenizer and IDEFICS image processor into a single processor.

[IdeficsProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsProcessor) offers all the functionalities of [IdeficsImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsImageProcessor) and [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast). See
the docstring of [**call**()](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics/processing_idefics.py#L240)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]], str, list[str], list[list[str]]] = None text: typing.Union[str, list[str], list[list[str]], list[list[list[str]]]] = None audio = None videos = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.idefics.processing\_idefics.IdeficsProcessorKwargs]  ) → a dict with entries

Parameters

* **images** (`Union[ImageInput, list[ImageInput], str, list[str], list[list[str]]]`) —
  either a single image or a batched list of images - can be passed in when text contains only text prompts,
  in order to use the image-text-to-text behavior.
* **text** (`Union[list[TextInput], [list[list[TextInput]]]]`) —
  either a single prompt or a batched list of prompts - see the detailed description immediately after
  the end of the arguments doc section.
* **return\_tensors** (`str` or `TensorType`, *optional*, defaults to `TensorType.PYTORCH`) —
  The type of tensors to return. Can be one of:
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.

Returns

a dict with entries

`input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be
directly passed to `model.generate`

This method takes batched or non-batched prompts made of text and images and converts them into prompts that
the model was trained on and prepares the image pixel values for the model to process.

Detailed explanation:

Each entry in `text` is either a text to be passed as is or an image that will be processed.

An image can be either an image object (`PIL.Image`) or a url from which the image can be retrieved.

When the processor encounters an image it’ll inject `<fake_token_around_image><image><fake_token_around_image>`
entry into the prompt.

Example:


```
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
<s>User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
Assistant: An image of two kittens in grass.
User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
Assistant:'
```

and the two images will be massaged using [IdeficsImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) method and placed inside the
`pixel_values` dict entry of the return value.

This example also exemplifies that images can be passed as objects or as text urls. It can be seen that the
first image is passed as object and the second one as a url.

To do training do:


```
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

In order to help debug prompt generation enable `debug=True` which will show you what’s happening.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/idefics.md)
