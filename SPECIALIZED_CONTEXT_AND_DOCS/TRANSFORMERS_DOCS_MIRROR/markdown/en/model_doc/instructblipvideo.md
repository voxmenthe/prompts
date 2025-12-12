*This model was released on 2023-05-11 and added to Hugging Face Transformers on 2024-06-25.*

# InstructBlipVideo

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The InstructBLIPVideo is an extension of the models proposed in [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://huggingface.co/papers/2305.06500) by Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi.
InstructBLIPVideo uses the same architecture as [InstructBLIP](instructblip) and works with the same checkpoints as [InstructBLIP](instructblip). The only difference is the ability to process videos.

The abstract from the paper is the following:

*General-purpose language models that can solve various language-domain tasks have emerged driven by the pre-training and instruction-tuning pipeline. However, building general-purpose vision-language models is challenging due to the increased task discrepancy introduced by the additional visual input. Although vision-language pre-training has been widely studied, vision-language instruction tuning remains relatively less explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pre-trained BLIP-2 models. We gather a wide variety of 26 publicly available datasets, transform them into instruction tuning format and categorize them into two clusters for held-in instruction tuning and held-out zero-shot evaluation. Additionally, we introduce instruction-aware visual feature extraction, a crucial method that enables the model to extract informative features tailored to the given instruction. The resulting InstructBLIP models achieve state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and the larger Flamingo. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA IMG). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/instructblip_architecture.jpg) InstructBLIPVideo architecture. Taken from the [original paper.](https://huggingface.co/papers/2305.06500)

This model was contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip).

## Usage tips

* The model was trained by sampling 4 frames per video, so it’s recommended to sample 4 frames

> [!NOTE]
> BLIP models after release v4.46 will raise warnings about adding `processor.num_query_tokens = {{num_query_tokens}}` and expand model embeddings layer to add special `<image>` token. It is strongly recommended to add the attributes to the processor if you own the model checkpoint, or open a PR if it is not owned by you. Adding these attributes means that BLIP will add the number of query tokens required per image and expand the text with as many `<image>` placeholders as there will be query tokens. Usually it is around 500 tokens per image, so make sure that the text is not truncated as otherwise there will be failure when merging the embeddings.
> The attributes can be obtained from model config, as `model.config.num_query_tokens` and model embeddings expansion can be done by following [this link](https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042).

## InstructBlipVideoConfig

### class transformers.InstructBlipVideoConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/configuration_instructblipvideo.py#L220)

( vision\_config = None qformer\_config = None text\_config = None num\_query\_tokens = 32 video\_token\_index = None \*\*kwargs  )

Parameters

* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [InstructBlipVideoVisionConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoVisionConfig).
* **qformer\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [InstructBlipVideoQFormerConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoQFormerConfig).
* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize any [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig).
* **num\_query\_tokens** (`int`, *optional*, defaults to 32) —
  The number of query tokens passed through the Transformer.
* **video\_token\_index** (`int`, *optional*) —
  Token index of special video token.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig) is the configuration class to store the configuration of a
[InstructBlipVideoForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoForConditionalGeneration). It is used to instantiate a Instructblipvideo model according to the specified
arguments, defining the vision model, Q-Former model and language model configs. Instantiating a configuration with
the defaults will yield a similar configuration to that of the Instructblipvideo
[Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import (
...     InstructBlipVideoVisionConfig,
...     InstructBlipVideoQFormerConfig,
...     OPTConfig,
...     InstructBlipVideoConfig,
...     InstructBlipVideoForConditionalGeneration,
... )

>>> # Initializing a InstructBlipVideoConfig with Salesforce/instruct-blip-flan-t5 style configuration
>>> configuration = InstructBlipVideoConfig()

>>> # Initializing a InstructBlipVideoForConditionalGeneration (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
>>> model = InstructBlipVideoForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a InstructBlipVideoConfig from a InstructBlipVideoVisionConfig, InstructBlipVideoQFormerConfig and any PretrainedConfig

>>> # Initializing Instructblipvideo vision, Instructblipvideo Q-Former and language model configurations
>>> vision_config = InstructBlipVideoVisionConfig()
>>> qformer_config = InstructBlipVideoQFormerConfig()
>>> text_config = OPTConfig()

>>> config = InstructBlipVideoConfig.from_text_vision_configs(vision_config, qformer_config, text_config)
```

#### from\_vision\_qformer\_text\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/configuration_instructblipvideo.py#L321)

( vision\_config: InstructBlipVideoVisionConfig qformer\_config: InstructBlipVideoQFormerConfig text\_config: PretrainedConfig \*\*kwargs  ) → [InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig)

Returns

[InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig)

An instance of a configuration object

Instantiate a [InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig) (or a derived class) from a InstructBlipVideo vision model, Q-Former and
language model configurations.

## InstructBlipVideoVisionConfig

### class transformers.InstructBlipVideoVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/configuration_instructblipvideo.py#L32)

( hidden\_size = 1408 intermediate\_size = 6144 num\_hidden\_layers = 39 num\_attention\_heads = 16 image\_size = 224 patch\_size = 14 hidden\_act = 'gelu' layer\_norm\_eps = 1e-06 attention\_dropout = 0.0 initializer\_range = 1e-10 qkv\_bias = True \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 1408) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 6144) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 39) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The size (resolution) of each patch.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported. to 1e-5): The epsilon used by the layer
  normalization layers.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 1e-10) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries and values in the self-attention layers.

This is the configuration class to store the configuration of a [InstructBlipVideoVisionModel](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoVisionModel). It is used to
instantiate a InstructBlipVideo vision encoder according to the specified arguments, defining the model architecture.
Instantiating a configuration defaults will yield a similar configuration to that of the InstructBlipVideo
[Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import InstructBlipVideoVisionConfig, InstructBlipVideoVisionModel

>>> # Initializing a InstructBlipVideoVisionConfig with Salesforce/instruct-blip-flan-t5 style configuration
>>> configuration = InstructBlipVideoVisionConfig()

>>> # Initializing a InstructBlipVideoVisionModel (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
>>> model = InstructBlipVideoVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## InstructBlipVideoQFormerConfig

### class transformers.InstructBlipVideoQFormerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/configuration_instructblipvideo.py#L116)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 position\_embedding\_type = 'absolute' cross\_attention\_frequency = 2 encoder\_hidden\_size = 1408 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the Q-Former model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling the model.
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  Token id used for padding sequences.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"absolute"`) —
  Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
  positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
  [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
  For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
  with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
* **cross\_attention\_frequency** (`int`, *optional*, defaults to 2) —
  The frequency of adding cross-attention to the Transformer layers.
* **encoder\_hidden\_size** (`int`, *optional*, defaults to 1408) —
  The hidden size of the hidden states for cross-attention.

This is the configuration class to store the configuration of a [InstructBlipVideoQFormerModel](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoQFormerModel). It is used to
instantiate a InstructBlipVideo Querying Transformer (Q-Former) model according to the specified arguments, defining the
model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
the InstructBlipVideo [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5)
architecture. Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs.
Read the documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Note that [InstructBlipVideoQFormerModel](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoQFormerModel) is very similar to [BertLMHeadModel](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertLMHeadModel) with interleaved cross-attention.

Examples:


```
>>> from transformers import InstructBlipVideoQFormerConfig, InstructBlipVideoQFormerModel

>>> # Initializing a InstructBlipVideo Salesforce/instruct-blip-flan-t5 style configuration
>>> configuration = InstructBlipVideoQFormerConfig()

>>> # Initializing a model (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
>>> model = InstructBlipVideoQFormerModel(configuration)
>>> # Accessing the model configuration
>>> configuration = model.config
```

## InstructBlipVideoProcessor

### class transformers.InstructBlipVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/processing_instructblipvideo.py#L39)

( video\_processor tokenizer qformer\_tokenizer num\_query\_tokens = None \*\*kwargs  )

Parameters

* **video\_processor** (`InstructBlipVideoVideoProcessor`) —
  An instance of [InstructBlipVideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoVideoProcessor). The video processor is a required input.
* **tokenizer** (`AutoTokenizer`) —
  An instance of [‘PreTrainedTokenizer`]. The tokenizer is a required input.
* **qformer\_tokenizer** (`AutoTokenizer`) —
  An instance of [‘PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.
* **num\_query\_tokens** (`int`, *optional*) —
  Number of tokens used by the Qformer as queries, should be same as in model’s config.

Constructs an InstructBLIPVideo processor which wraps a InstructBLIP image processor and a LLaMa/T5 tokenizer into a single
processor.

[InstructBlipVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoProcessor) offers all the functionalities of [InstructBlipVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoImageProcessor) and [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See the
docstring of `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## InstructBlipVideoVideoProcessor

### class transformers.InstructBlipVideoVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/video_processing_instructblipvideo.py#L59)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.instructblipvideo.video\_processing\_instructblipvideo.InstructBlipVideoVideoProcessorInitKwargs]  )

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L355)

( videos: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]]] \*\*kwargs: typing\_extensions.Unpack[transformers.processing\_utils.VideosKwargs]  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the video’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `self.size`) —
  Size of the output video after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **size\_divisor** (`int`, *optional*, defaults to `self.size_divisor`) —
  The size by which to make sure both the height and width can be divided.
* **default\_to\_square** (`bool`, *optional*, defaults to `self.default_to_square`) —
  Whether to default to a square video when resizing, if size is an int.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the video to the specified `crop_size`. Can be overridden by `do_center_crop` in the
  `preprocess` method.
* **do\_pad** (`bool`, *optional*) —
  Whether to pad the video to the `(max_height, max_width)` of the videos in the batch.
* **crop\_size** (`dict[str, int]` *optional*, defaults to `self.crop_size`) —
  Size of the output video after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
  method.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `self.rescale_factor`) —
  Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Mean to use if normalizing the video. This is a float or list of floats the length of the number of
  channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
  number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.image_std`) —
  Whether to convert the video to RGB.
* **video\_metadata** (`VideoMetadata`, *optional*) —
  Metadata of the video containing information about total duration, fps and total number of frames.
* **do\_sample\_frames** (`int`, *optional*, defaults to `self.do_sample_frames`) —
  Whether to sample frames from the video before processing or to process the whole video.
* **num\_frames** (`int`, *optional*, defaults to `self.num_frames`) —
  Maximum number of frames to sample when `do_sample_frames=True`.
* **fps** (`int` or `float`, *optional*, defaults to `self.fps`) —
  Target frames to sample per second when `do_sample_frames=True`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input video.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input video. If unset, the channel dimension format is inferred
  from the input video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the videos on. If unset, the device is inferred from the input videos.
* **return\_metadata** (`bool`, *optional*) —
  Whether to return video metadata or not.

## InstructBlipVideoImageProcessor

### class transformers.InstructBlipVideoImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/image_processing_instructblipvideo.py#L47)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: bool = True \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `{"height" -- 384, "width": 384}`):
  Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) —
  Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.

Constructs a InstructBLIPVideo image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/image_processing_instructblipvideo.py#L162)

( images: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]]] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **videos** (`VideoInput`) —
  Video frames to preprocess. Expects a single or batch of videos as a list of frames with pixel values
  ranging from 0 to 255. If passing in video with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the video.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Controls the size of the video after `resize`. The shortest edge of the image is resized to
  `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
  is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
  edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the video values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the video by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the video.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to normalize the video by if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to normalize the video by if `do_normalize` is set to `True`.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess a video or batch of images/videos.

## InstructBlipVideoVisionModel

### class transformers.InstructBlipVideoVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py#L417)

( config: InstructBlipVideoVisionConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py#L432)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [InstructBlipVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoImageProcessor). See [InstructBlipVideoImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InstructBlipVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoProcessor) uses
  [InstructBlipVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [InstructBlipVideoVisionModel](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## InstructBlipVideoQFormerModel

### class transformers.InstructBlipVideoQFormerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py#L909)

( config: InstructBlipVideoQFormerConfig  )

Querying Transformer (Q-Former), used in InstructBlipVideo. Slightly modified from BLIP-2 as it also takes the
instruction as input.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py#L987)

( input\_ids: LongTensor attention\_mask: typing.Optional[torch.FloatTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None query\_embeds: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None encoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  )

encoder\_hidden\_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.
encoder\_attention\_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

* 1 for tokens that are **not masked**,
* 0 for tokens that are **masked**.
  past\_key\_values (`Cache` of length `config.n_layers` with each tuple having 4 tensors of:
  shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
  value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
  used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key
  value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
  `(batch_size, sequence_length)`.
  use\_cache (`bool`, *optional*):
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

## InstructBlipVideoModel

### class transformers.InstructBlipVideoModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py#L1140)

( config: InstructBlipVideoConfig  )

Parameters

* **config** ([InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

InstructBlipVideo base Model consisting of language model, qformer and vision encoder.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py#L1209)

( pixel\_values: FloatTensor qformer\_input\_ids: FloatTensor qformer\_attention\_mask: typing.Optional[torch.LongTensor] = None input\_ids: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.instructblipvideo.modeling_instructblipvideo.InstructBlipVideoForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [InstructBlipVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoImageProcessor). See [InstructBlipVideoImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InstructBlipVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoProcessor) uses
  [InstructBlipVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoImageProcessor) for processing images).
* **qformer\_input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary of the Q-Former. Input tokens can optionally be provided
  to serve as text prompt, which the Q-Former model will encode.

  Indices can be obtained using [InstructBlipVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoProcessor). See `InstructBlipVideoProcessor.__call__()` for
  details.

  [What are input IDs?](../glossary#input-ids)
* **qformer\_attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
  provided to serve as text prompt, which the language model can continue.

  Indices can be obtained using [InstructBlipVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoProcessor). See `InstructBlipVideoProcessor.__call__()` for
  details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  Only relevant in case an encoder-decoder language model (like T5) is used.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

`transformers.models.instructblipvideo.modeling_instructblipvideo.InstructBlipVideoForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.instructblipvideo.modeling_instructblipvideo.InstructBlipVideoForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig)) and inputs.

* **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Language modeling loss from the language model.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head of the language model.
* **vision\_outputs** (`torch.FloatTensor`, *optional*, defaults to `None`) — Outputs of the vision encoder.
* **qformer\_outputs** (`tuple[torch.FloatTensor]`, *optional*, defaults to `None`) — Outputs of the Q-Former (Querying Transformer).
* **language\_model\_outputs** (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`) — Outputs of the language model.

The [InstructBlipVideoModel](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## InstructBlipVideoForConditionalGeneration

### class transformers.InstructBlipVideoForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py#L1358)

( config: InstructBlipVideoConfig  )

Parameters

* **config** ([InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

InstructBlipVideo Model for generating text given an image and an optional text prompt. The model consists of a vision
encoder, Querying Transformer (Q-Former) and a language model.

One can optionally pass `input_ids` to the model, which serve as a text prompt, to make the language model continue
the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py#L1466)

( pixel\_values: FloatTensor qformer\_input\_ids: FloatTensor qformer\_attention\_mask: typing.Optional[torch.LongTensor] = None input\_ids: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None return\_dict: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.instructblipvideo.modeling_instructblipvideo.InstructBlipVideoForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [InstructBlipVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoImageProcessor). See [InstructBlipVideoImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([InstructBlipVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoProcessor) uses
  [InstructBlipVideoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoImageProcessor) for processing images).
* **qformer\_input\_ids** (`torch.LongTensor` of shape (batch\_size, sequence\_length)) —
  The sequence used as a prompt to be fed to the Q-Former module.
* **qformer\_attention\_mask** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  Mask to avoid performing attention on padding token indices.
* **input\_ids** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
  make sure the model can only look at previous inputs in order to predict the future.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

`transformers.models.instructblipvideo.modeling_instructblipvideo.InstructBlipVideoForConditionalGenerationModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.instructblipvideo.modeling_instructblipvideo.InstructBlipVideoForConditionalGenerationModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([InstructBlipVideoConfig](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoConfig)) and inputs.

* **loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Language modeling loss from the language model.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head of the language model.
* **vision\_outputs** (`torch.FloatTensor`, *optional*, defaults to `None`) — Outputs of the vision encoder.
* **qformer\_outputs** (`tuple[torch.FloatTensor]`, *optional*, defaults to `None`) — Outputs of the Q-Former (Querying Transformer).
* **language\_model\_outputs** (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`) — Outputs of the language model.

The [InstructBlipVideoForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/instructblipvideo#transformers.InstructBlipVideoForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration
>>> import torch
>>> from huggingface_hub import hf_hub_download
>>> import av
>>> import numpy as np

>>> def read_video_pyav(container, indices):
...     '''
...     Decode the video with PyAV decoder.
...     Args:
...         container (`av.container.input.InputContainer`): PyAV container.
...         indices (`list[int]`): List of frame indices to decode.
...     Returns:
...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
...     '''
...     frames = []
...     container.seek(0)
...     start_index = indices[0]
...     end_index = indices[-1]
...     for i, frame in enumerate(container.decode(video=0)):
...         if i > end_index:
...             break
...         if i >= start_index and i in indices:
...             frames.append(frame)
...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

>>> model = InstructBlipVideoForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map="auto")
>>> processor = InstructBlipVideoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

>>> file_path = hf_hub_download(
...       repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )
>>> container = av.open(file_path)

>>> # sample uniformly 4 frames from the videWhy is this video funny?o
>>> total_frames = container.streams.video[0].frames
>>> indices = np.arange(0, total_frames, total_frames / 4).astype(int)
>>> clip = read_video_pyav(container, indices)

>>> prompt = "What is happening in the video?"
>>> inputs = processor(text=prompt, images=clip, return_tensors="pt").to(model.device)

>>> outputs = model.generate(
...     **inputs,
...     do_sample=False,
...     num_beams=5,
...     max_length=256,
...     repetition_penalty=1.5,
...     length_penalty=1.0,
... )
>>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
"A person is eating a bowl of pasta, and they are using a fork to eat it. The person is sitting at a table, and the plate of pasta is on the table in front"
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py#L1612)

( pixel\_values: FloatTensor qformer\_input\_ids: typing.Optional[torch.LongTensor] = None qformer\_attention\_mask: typing.Optional[torch.LongTensor] = None input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None interpolate\_pos\_encoding: bool = False \*\*generate\_kwargs  ) → captions (list)

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape (batch\_size, num\_channels, height, width) or —
  (batch\_size, num\_frames, num\_channels, height, width)): Input images or videos to be processed.
* **qformer\_input\_ids** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  The sequence used as a prompt to be fed to the Q-Former module.
* **qformer\_attention\_mask** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  Mask to avoid performing attention on padding token indices.
* **input\_ids** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  The sequence used as a prompt for the generation.
* **attention\_mask** (`torch.LongTensor` of shape (batch\_size, sequence\_length), *optional*) —
  Mask to avoid performing attention on padding token indices.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) —
  Embedded representation of the inputs. Should be float, not int tokens.
* **interpolate\_pos\_encoding** (`bool`, *optional*, defaults to `False`) —
  Whether to interpolate the positional encoding of the image embeddings.

Returns

captions (list)

A list of strings of length batch\_size \* num\_captions.

Overrides `generate` function to be able to use the model as a conditional generator.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/instructblipvideo.md)
