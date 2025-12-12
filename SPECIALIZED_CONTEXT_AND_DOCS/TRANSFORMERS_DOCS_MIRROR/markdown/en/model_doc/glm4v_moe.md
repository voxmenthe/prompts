*This model was released on 2025-07-28 and added to Hugging Face Transformers on 2025-08-08.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# Glm4vMoe

## Overview

Vision-language models (VLMs) have become a key cornerstone of intelligent systems. As real-world AI tasks grow increasingly complex, VLMs urgently need to enhance reasoning capabilities beyond basic multimodal perception — improving accuracy, comprehensiveness, and intelligence — to enable complex problem solving, long-context understanding, and multimodal agents.

Through our open-source work, we aim to explore the technological frontier together with the community while empowering more developers to create exciting and innovative applications.

[GLM-4.5V](https://huggingface.co/papers/2508.06471) ([Github repo](https://github.com/zai-org/GLM-V)) is based on ZhipuAI’s next-generation flagship text foundation model GLM-4.5-Air (106B parameters, 12B active). It continues the technical approach of [GLM-4.1V-Thinking](https://huggingface.co/papers/2507.01006), achieving SOTA performance among models of the same scale on 42 public vision-language benchmarks. It covers common tasks such as image, video, and document understanding, as well as GUI agent operations.

![bench_45](https://raw.githubusercontent.com/zai-org/GLM-V/refs/heads/main/resources/bench_45v.jpeg)

Beyond benchmark performance, GLM-4.5V focuses on real-world usability. Through efficient hybrid training, it can handle diverse types of visual content, enabling full-spectrum vision reasoning, including:

* **Image reasoning** (scene understanding, complex multi-image analysis, spatial recognition)
* **Video understanding** (long video segmentation and event recognition)
* **GUI tasks** (screen reading, icon recognition, desktop operation assistance)
* **Complex chart & long document parsing** (research report analysis, information extraction)
* **Grounding** (precise visual element localization)

The model also introduces a **Thinking Mode** switch, allowing users to balance between quick responses and deep reasoning. This switch works the same as in the `GLM-4.5` language model.

## Glm4vMoeConfig

### class transformers.Glm4vMoeConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v_moe/configuration_glm4v_moe.py#L307)

( text\_config = None vision\_config = None image\_token\_id = 151363 video\_token\_id = 151364 image\_start\_token\_id = 151339 image\_end\_token\_id = 151340 video\_start\_token\_id = 151341 video\_end\_token\_id = 151342 \*\*kwargs  )

Parameters

* **text\_config** (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Glm4vMoeTextConfig`) —
  The config object or dictionary of the text backbone.
* **vision\_config** (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Glm4vMoeVisionConfig`) —
  The config object or dictionary of the vision backbone.
* **image\_token\_id** (`int`, *optional*, defaults to 151363) —
  The image token index to encode the image prompt.
* **video\_token\_id** (`int`, *optional*, defaults to 151364) —
  The video token index to encode the image prompt.
* **image\_start\_token\_id** (`int`, *optional*, defaults to 151339) —
  The image start token index to encode the start of image.
* **image\_end\_token\_id** (`int`, *optional*, defaults to 151340) —
  The image end token index to encode the end of image.
* **video\_start\_token\_id** (`int`, *optional*, defaults to 151341) —
  The video start token index to encode the start of video.
* **video\_end\_token\_id** (`int`, *optional*, defaults to 151342) —
  The video end token index to encode the end of video.

This is the configuration class to store the configuration of a [Glm4vMoeModel](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeModel). It is used to instantiate a
GLM-4.5V model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of
GLM-4.5V [zai-org/GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Glm4vMoeForConditionalGeneration, Glm4vMoeConfig

>>> # Initializing a GLM-4.5V style configuration
>>> configuration = Glm4vMoeConfig()

>>> # Initializing a model from the GLM-4.5V style configuration
>>> model = Glm4vMoeForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Glm4vMoeTextConfig

### class transformers.Glm4vMoeTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v_moe/configuration_glm4v_moe.py#L122)

( vocab\_size = 151424 hidden\_size = 4096 intermediate\_size = 10944 num\_hidden\_layers = 46 num\_attention\_heads = 96 partial\_rotary\_factor = 0.5 num\_key\_value\_heads = 8 hidden\_act = 'silu' max\_position\_embeddings = 65536 initializer\_range = 0.02 rms\_norm\_eps = 1e-05 use\_cache = True tie\_word\_embeddings = False rope\_theta = 10000.0 rope\_scaling = None attention\_bias = True attention\_dropout = 0.0 moe\_intermediate\_size = 1408 num\_experts\_per\_tok = 8 n\_shared\_experts = 1 n\_routed\_experts = 128 routed\_scaling\_factor = 1.0 n\_group = 1 topk\_group = 1 first\_k\_dense\_replace = 1 norm\_topk\_prob = True \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 151424) —
  Vocabulary size of the Glm4vMoe model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [Glm4vMoeModel](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeModel)
* **hidden\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 10944) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 46) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 96) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **partial\_rotary\_factor** (`float`, *optional*, defaults to 0.5) — The factor of the partial rotary position.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 8) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details checkout [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 65536) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether the model’s input and output word embeddings should be tied.
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (`Dict`, *optional*) —
  Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
  and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
  accordingly.
  Expected contents:
  `rope_type` (`str`):
  The sub-variant of RoPE to use. Can be one of [‘default’, ‘linear’, ‘dynamic’, ‘yarn’, ‘longrope’,
  ‘llama3’], with ‘default’ being the original RoPE implementation.
  `factor` (`float`, *optional*):
  Used with all rope types except ‘default’. The scaling factor to apply to the RoPE embeddings. In
  most scaling types, a `factor` of x will enable the model to handle sequences of length x *original maximum pre-trained length.
  `original_max_position_embeddings` (`int`,* optional*):
  Used with ‘dynamic’, ‘longrope’ and ‘llama3’. The original max position embeddings used during
  pretraining.
  `attention_factor` (`float`,* optional\*):
  Used with ‘yarn’ and ‘longrope’. The scaling factor to be applied on the attention
  computation. If unspecified, it defaults to value recommended by the implementation, using the
  `factor` field to infer the suggested value.
* **attention\_bias** (`bool`, defaults to `True`, *optional*, defaults to `True`) —
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **moe\_intermediate\_size** (`int`, *optional*, defaults to 1408) —
  Intermediate size of the routed expert.
* **num\_experts\_per\_tok** (`int`, *optional*, defaults to 8) —
  number of experts per token.
* **n\_shared\_experts** (`int`, *optional*, defaults to 1) —
  Number of shared experts.
* **n\_routed\_experts** (`int`, *optional*, defaults to 128) —
  Number of routed experts.
* **routed\_scaling\_factor** (`float`, *optional*, defaults to 1.0) —
  Scaling factor or routed experts.
* **n\_group** (`int`, *optional*, defaults to 1) —
  Number of groups for routed experts.
* **topk\_group** (`int`, *optional*, defaults to 1) —
  Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
* **first\_k\_dense\_replace** (`int`, *optional*, defaults to 1) —
  Number of dense layers in shallow layers(embed->dense->dense->…->dense->moe->moe…->lm\_head).
  --k dense layers—/
* **norm\_topk\_prob** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the topk probabilities.

This is the configuration class to store the configuration of a [Glm4vMoeModel](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeModel). It is used to instantiate a
GLM-4.5V model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of
GLM-4.5V [zai-org/GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Glm4vMoeTextModel, Glm4vMoeConfig

>>> # Initializing a GLM-4.5V style configuration
>>> configuration = Glm4vMoeConfig()

>>> # Initializing a model from the GLM-4.5V style configuration
>>> model = Glm4vMoeTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Glm4vMoeTextModel

### class transformers.Glm4vMoeTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v_moe/modeling_glm4v_moe.py#L920)

( config: Glm4vMoeTextConfig  )

Parameters

* **config** ([Glm4vMoeTextConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Glm4V Moe Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v_moe/modeling_glm4v_moe.py#L939)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
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
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Glm4vMoeTextModel](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Glm4vMoeModel

### class transformers.Glm4vMoeModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v_moe/modeling_glm4v_moe.py#L1009)

( config  )

Parameters

* **config** ([Glm4vMoeModel](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Glm4V Moe Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v_moe/modeling_glm4v_moe.py#L1309)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None pixel\_values: typing.Optional[torch.Tensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None rope\_deltas: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.glm4v_moe.modeling_glm4v_moe.Glm4vMoeModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
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
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
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
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  `video_processor_class`. See `video_processor_class.__call__` for details (`processor_class` uses
  `video_processor_class` for processing videos).
* **image\_grid\_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) —
  The temporal, height and width of feature shape of each image in LLM.
* **video\_grid\_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) —
  The temporal, height and width of feature shape of each video in LLM.
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) —
  The rope index difference between sequence length and multimodal rope.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.models.glm4v_moe.modeling_glm4v_moe.Glm4vMoeModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.glm4v_moe.modeling_glm4v_moe.Glm4vMoeModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **last\_hidden\_state** (`<class 'torch.FloatTensor'>.last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
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
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) — The rope index difference between sequence length and multimodal rope.

The [Glm4vMoeModel](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Glm4vMoeForConditionalGeneration

### class transformers.Glm4vMoeForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v_moe/modeling_glm4v_moe.py#L1447)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4v_moe/modeling_glm4v_moe.py#L1489)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.Tensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None rope\_deltas: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.glm4v_moe.modeling_glm4v_moe.Glm4vMoeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
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
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([Glm4vProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vProcessor) uses
  `image_processor_class` for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  `video_processor_class`. See `video_processor_class.__call__` for details ([Glm4vProcessor](/docs/transformers/v4.56.2/en/model_doc/glm4v#transformers.Glm4vProcessor) uses
  `video_processor_class` for processing videos).
* **image\_grid\_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) —
  The temporal, height and width of feature shape of each image in LLM.
* **video\_grid\_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) —
  The temporal, height and width of feature shape of each video in LLM.
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) —
  The rope index difference between sequence length and multimodal rope.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.glm4v_moe.modeling_glm4v_moe.Glm4vMoeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.glm4v_moe.modeling_glm4v_moe.Glm4vMoeCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Glm4vMoeConfig](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeConfig)) and inputs.

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
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) — The rope index difference between sequence length and multimodal rope.

The [Glm4vMoeForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/glm4v_moe#transformers.Glm4vMoeForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration

>>> model = Glm4vMoeForConditionalGeneration.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
>>> processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")

>>> messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
>>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm4v_moe.md)
