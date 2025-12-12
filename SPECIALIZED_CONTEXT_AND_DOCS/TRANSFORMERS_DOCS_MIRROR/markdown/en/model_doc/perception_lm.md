*This model was released on 2025-04-17 and added to Hugging Face Transformers on 2025-07-11.*

# PerceptionLM

## Overview

The [PerceptionLM](https://huggingface.co/papers/2504.13180) model was proposed in [PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding](https://ai.meta.com/research/publications/perceptionlm-open-access-data-and-models-for-detailed-visual-understanding/) by Jang Hyun Cho et al. It’s a fully open, reproducible model for transparent research in image and video understanding. PLM consists of
a vision encoder with a small scale (<8B parameters) LLM decoder.

The abstract from the paper is the following:

*Vision-language models are integral to computer vision research, yet many high-performing models
remain closed-source, obscuring their data, design and training recipe. The research community
has responded by using distillation from black-box models to label training data, achieving strong
benchmark results, at the cost of measurable scientific progress. However, without knowing the details
of the teacher model and its data sources, scientific progress remains difficult to measure. In this
paper, we study building a Perception Language Model (PLM) in a fully open and reproducible
framework for transparent research in image and video understanding. We analyze standard training
pipelines without distillation from proprietary models and explore large-scale synthetic data to identify
critical data gaps, particularly in detailed video understanding. To bridge these gaps, we release 2.8M
human-labeled instances of fine-grained video question-answer pairs and spatio-temporally grounded
video captions. Additionally, we introduce PLM–VideoBench, a suite for evaluating challenging video
understanding tasks focusing on the ability to reason about “what”, “where”, “when”, and “how” of a
video. We make our work fully reproducible by providing data, training recipes, code & models.*

This model was contributed by [shumingh](https://huggingface.co/shumingh).
The original code can be found [here](https://github.com/facebookresearch/perception_models).

## PerceptionLMConfig

### class transformers.PerceptionLMConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/configuration_perception_lm.py#L25)

( vision\_config = None text\_config = None vision\_use\_cls\_token = True projector\_pooling\_ratio = 1 image\_token\_id = 128002 video\_token\_id = 128003 \*\*kwargs  )

Parameters

* **vision\_config** (`Union[TimmWrapperConfig, dict]`, *optional*, defaults to `TimmWrapperConfig()`) —
  The config object or dictionary of the vision backbone.
* **text\_config** (`Union[PretrainedConfig, dict]`, *optional*, defaults to `LlamaConfig()`) —
  The config object or dictionary of the text backbone.
* **vision\_use\_cls\_token** (`bool`, *optional*, defaults to `True`) —
  Whether CLS token is used in the vision backbone. If used, we remove CLS token embedding from vision output.
* **projector\_pooling\_ratio** (`int`, *optional*, defaults to 1) —
  The pooling ratio used in the multimodal projector.
* **image\_token\_id** (`int`, *optional*, defaults to 128002) —
  The image token index to encode the image prompt.
* **video\_token\_id** (`int`, *optional*, defaults to 128003) —
  The video token index to encode the video prompt.

This is the configuration class to store the configuration of a [PerceptionLMForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMForConditionalGeneration). It is used to instantiate an
PerceptionLM model according to the specified arguments, defining the model architecture.

Example models:

* [facebook/Perception-LM-1B](https://huggingface.co/facebook/Perception-LM-1B).
* [facebook/Perception-LM-3B](https://huggingface.co/facebook/Perception-LM-3B).
* [facebook/Perception-LM-8B](https://huggingface.co/facebook/Perception-LM-8B).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## PerceptionLMProcessor

### class transformers.PerceptionLMProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/processing_perception_lm.py#L43)

( video\_processor = None image\_processor = None tokenizer = None patch\_size = None chat\_template = None pooling\_ratio = 2 \*\*kwargs  )

Parameters

* **video\_processor** ([PerceptionLMVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMVideoProcessor), *optional*) —
  The video processor to process video inputs.
* **image\_processor** ([PerceptionLMImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMImageProcessorFast), *optional*) —
  The image processor to process image inputs.
* **tokenizer** ([LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast) or similar, *optional*) —
  The tokenizer to process text inputs.
* **patch\_size** (`int`, *optional*) —
  Patch size from the vision tower.
* **chat\_template** (`str`, *optional*) —
  A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
* **pooling\_ratio** (`int`, *optional*, defaults to 2) —
  Pooling ratio for vision tokens. If not 1, 2D adaptive pooling is applied over projected vision tokens.

Constructs a PerceptionLM processor which wraps a PerceptionLM image processor, a PerceptionLM video processor, and a tokenizer into a single processor.

[PerceptionLMProcessor](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMProcessor) offers all the functionalities of [PerceptionLMImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMImageProcessorFast), [PerceptionLMVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMVideoProcessor), and the tokenizer (e.g. [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast)). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## PerceptionLMImageProcessorFast

### class transformers.PerceptionLMImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/image_processing_perception_lm_fast.py#L70)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.perception\_lm.image\_processing\_perception\_lm\_fast.PerceptionLMFastImageProcessorKwargs]  )

Constructs a fast Perception Lm image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/image_processing_perception_lm_fast.py#L85)

( images \*\*kwargs: typing\_extensions.Unpack[transformers.models.perception\_lm.image\_processing\_perception\_lm\_fast.PerceptionLMFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (``) -- Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set` do\_rescale=False`.
* **do\_resize** (`bool`, *optional*) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*) —
  Describes the maximum input dimensions to the model.
* **default\_to\_square** (`bool`, *optional*) —
  Whether to default to a square image when resizing, if size is an int.
* **resample** (`Union[PILImageResampling, F.InterpolationMode, NoneType]`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*) —
  Size of the output image after applying `center_crop`.
* **do\_rescale** (`bool`, *optional*) —
  Whether to rescale the image.
* **rescale\_factor** (`Union[int, float, NoneType]`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*) —
  Whether to normalize the image.
* **image\_mean** (`Union[float, list[float], NoneType]`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`Union[float, list[float], NoneType]`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_convert\_rgb** (`bool`, *optional*) —
  Whether to convert the image to RGB.
* **return\_tensors** (`Union[str, ~utils.generic.TensorType, NoneType]`) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`~image_utils.ChannelDimension`, *optional*) —
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
* **input\_data\_format** (`Union[str, ~image_utils.ChannelDimension, NoneType]`) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the images on. If unset, the device is inferred from the input images.
* **disable\_grouping** (`bool`, *optional*) —
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: <https://github.com/huggingface/transformers/pull/38157>
* **vision\_input\_type** (`str`, *optional*, defaults to `"thumb+tile"`) —
  Vision processing strategy. `"thumb+tile"` uses both thumbnails and multiple tiles for
  multi-scale processing, otherwise uses single tile for lower memory usage.
* **tile\_size** (`int`, *optional*, defaults to `448`) —
  Height and width dimension (in pixels) of each tile used for image processing.
* **max\_num\_tiles** (`int`, *optional*, defaults to `36`) —
  Maximum number of tiles an image can be split into based on its aspect ratio.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## PerceptionLMVideoProcessor

### class transformers.PerceptionLMVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/video_processing_perception_lm.py#L36)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.perception\_lm.video\_processing\_perception\_lm.PerceptionLMFastVideoProcessorInitKwargs]  )

## PerceptionLMModel

### class transformers.PerceptionLMModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/modeling_perception_lm.py#L167)

( config: PerceptionLMConfig  )

Parameters

* **config** ([PerceptionLMConfig](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Perception Lm Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/modeling_perception_lm.py#L250)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*lm\_kwargs  ) → `transformers.models.perception_lm.modeling_perception_lm.PerceptionLMModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([PerceptionLMProcessor](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMProcessor) uses
  `image_processor_class` for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  `video_processor_class`. See `video_processor_class.__call__` for details ([PerceptionLMProcessor](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMProcessor) uses
  `video_processor_class` for processing videos).
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
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
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

`transformers.models.perception_lm.modeling_perception_lm.PerceptionLMModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.perception_lm.modeling_perception_lm.PerceptionLMModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PerceptionLMConfig](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the model.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  Image hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.
* **video\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_videos, sequence_length, hidden_size)`.
  Video hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [PerceptionLMModel](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/modeling_perception_lm.py#L189)

( pixel\_values: FloatTensor \*\*kwargs  ) → image\_features (`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, num_tiles, channels, height, width)`) —
  The tensors corresponding to the input images.

Returns

image\_features (`torch.Tensor`)

Image feature tensor of shape `(num_tiles, num_patches, embed_dim)`).

Obtains image last hidden states from the vision tower and apply multimodal projection.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/modeling_perception_lm.py#L210)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor = None video\_features: FloatTensor = None  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

## PerceptionLMForConditionalGeneration

### class transformers.PerceptionLMForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/modeling_perception_lm.py#L324)

( config: PerceptionLMConfig  )

Parameters

* **config** ([PerceptionLMConfig](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Perception Lm Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/perception_lm/modeling_perception_lm.py#L349)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*lm\_kwargs  ) → `transformers.models.perception_lm.modeling_perception_lm.PerceptionLMCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([PerceptionLMProcessor](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMProcessor) uses
  `image_processor_class` for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  `video_processor_class`. See `video_processor_class.__call__` for details ([PerceptionLMProcessor](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMProcessor) uses
  `video_processor_class` for processing videos).
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
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
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

`transformers.models.perception_lm.modeling_perception_lm.PerceptionLMCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.perception_lm.modeling_perception_lm.PerceptionLMCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PerceptionLMConfig](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMConfig)) and inputs.

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
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  Image hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.
* **video\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_videos, sequence_length, hidden_size)`.
  Video hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [PerceptionLMForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/perception_lm#transformers.PerceptionLMForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, PerceptionLMForConditionalGeneration

>>> model = PerceptionLMForConditionalGeneration.from_pretrained("perception_lm-hf/perception_lm-1.5-7b-hf")
>>> processor = AutoProcessor.from_pretrained("perception_lm-hf/perception_lm-1.5-7b-hf")

>>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, text=prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs, max_new_tokens=15)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/perception_lm.md)
