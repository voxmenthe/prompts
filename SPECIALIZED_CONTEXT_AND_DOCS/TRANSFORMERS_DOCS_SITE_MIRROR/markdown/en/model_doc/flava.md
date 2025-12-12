# FLAVA

## Overview

The FLAVA model was proposed in [FLAVA: A Foundational Language And Vision Alignment Model](https://huggingface.co/papers/2112.04482) by Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela and is accepted at CVPR 2022.

The paper aims at creating a single unified foundation model which can work across vision, language
as well as vision-and-language multimodal tasks.

The abstract from the paper is the following:

*State-of-the-art vision and vision-and-language models rely on large-scale visio-linguistic pretraining for obtaining good performance on a variety
of downstream tasks. Generally, such models are often either cross-modal (contrastive) or multi-modal
(with earlier fusion) but not both; and they often only target specific modalities or tasks. A promising
direction would be to use a single holistic universal model, as a "foundation", that targets all modalities
at once -- a true vision and language foundation model should be good at vision tasks, language tasks, and
cross- and multi-modal vision and language tasks. We introduce FLAVA as such a model and demonstrate
impressive performance on a wide range of 35 tasks spanning these target modalities.*

This model was contributed by [aps](https://huggingface.co/aps). The original code can be found [here](https://github.com/facebookresearch/multimodal/tree/main/examples/flava).

## FlavaConfig[[transformers.FlavaConfig]]

#### transformers.FlavaConfig[[transformers.FlavaConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/configuration_flava.py#L387)

[FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig) is the configuration class to store the configuration of a [FlavaModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaModel). It is used to
instantiate FLAVA model according to the specified arguments, defining the text model, image model, image codebook
and multimodal model configs. Instantiating a configuration with the defaults will yield a similar configuration to
that of the FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import FlavaConfig, FlavaModel, FlavaForPreTraining

>>> # Initializing a FlavaConfig with style configuration
>>> configuration = FlavaConfig()

>>> # Initializing a FlavaModel and FlavaForPreTraining model (with random weights) from the style configuration
>>> model = FlavaModel(configuration)
>>> model_pre = FlavaForPreTraining(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
>>> configuration_pre = model_pre.config
```

**Parameters:**

text_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [FlavaTextConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextConfig).

image_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [FlavaImageConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageConfig).

multimodal_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [FlavaMultimodalConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaMultimodalConfig).

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

projection_dim (`int`, *optional*, defaults to 512) : Dimensionality of text and image projection layers.

logit_scale_init_value (`float`, *optional*, defaults to 2.6592) : The initial value of the *logit_scale* parameter. Default is used as per the original FLAVA/CLIP implementation.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

ce_ignore_index (`int`, *optional*, defaults to -100) : Cross entropy index to ignore.

mim_weight (`float`, *optional*, defaults to 1.0) : Weight to be assigned to MIM (Masked Image Modeling) unimodal loss

mlm_weight (`float`, *optional*, defaults to 1.0) : Weight to be assigned to MLM (Masked Language Modeling) unimodal loss

global_contrastive_weight (`float`, *optional*, defaults to 1.0) : Weight to be assigned to global contrastive cross-alignment loss.

itm_weight (`float`, *optional*, defaults to 1.0) : Weight to be assigned to image-text matching multimodal loss.

mmm_image_weight (`float`, *optional*, defaults to 1.0) : Weight to be assigned to MMM loss's image part.

mmm_text_weight (`float`, *optional*, defaults to 1.0) : Weight to be assigned to MMM loss's text part.

global_backprop_contrastive (`bool`, *optional*, defaults to `True`) : Whether to use global backpropgation through all workers in contrastive loss.

skip_unmasked_multimodal_encoder (`bool`, *optional*, defaults to `True`) : Whether to skip running unmasked multimodal encoder whose outputs are not used by FLAVA losses.

return_loss (`bool`, *optional*, defaults to `True`) : Whether to return loss or not 

kwargs (*optional*) : Dictionary of keyword arguments.

## FlavaTextConfig[[transformers.FlavaTextConfig]]

#### transformers.FlavaTextConfig[[transformers.FlavaTextConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/configuration_flava.py#L128)

This is the configuration class to store the configuration of a [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel). It is used to instantiate an
FLAVA model according to the specified arguments, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
[facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import FlavaTextConfig, FlavaTextModel

>>> # Initializing a FlavaTextModel with  style configuration
>>> configuration = FlavaTextConfig()

>>> # Initializing a FlavaTextModel model (with random weights) from the style configuration
>>> model = FlavaTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 30522) : Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).

type_vocab_size (`int`, *optional*, defaults to 2) : The vocabulary size of the `token_type_ids` passed when calling [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel). Note that even though text encoder allows `token_type_ids`'s value as 2, for text-only pretraining and fine-tuning, only 1 is used similar to RoBERTa.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048). For VL, max_length passed to model is 77.

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

image_size (`int`, *optional*, defaults to 224) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 16) : The size (resolution) of each patch.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries, keys and values.

## FlavaImageConfig[[transformers.FlavaImageConfig]]

#### transformers.FlavaImageConfig[[transformers.FlavaImageConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/configuration_flava.py#L26)

This is the configuration class to store the configuration of a [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel). It is used to instantiate an
FLAVA model according to the specified arguments, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
[facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import FlavaImageConfig, FlavaImageModel

>>> # Initializing a FlavaImageModel with  style configuration
>>> configuration = FlavaImageConfig()

>>> # Initializing a FlavaImageModel model (with random weights) from the style configuration
>>> model = FlavaImageModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

image_size (`int`, *optional*, defaults to 224) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 16) : The size (resolution) of each patch.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries, keys and values.

mask_token (`bool`, *optional*, defaults to `True`) : Whether to use a mask token or not. Used in MIM (Masked Image Modeling) loss for FLAVA.

vocab_size (`int`, *optional*, defaults to 8192) : Vocabulary size of the [FlavaImageCodebook](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageCodebook) used in conjunction with [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel) for MIM (Masked Image Modeling) loss for FLAVA.

## FlavaMultimodalConfig[[transformers.FlavaMultimodalConfig]]

#### transformers.FlavaMultimodalConfig[[transformers.FlavaMultimodalConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/configuration_flava.py#L233)

This is the configuration class to store the configuration of a [FlavaMultimodalModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaMultimodalModel). It is used to instantiate
an FLAVA model according to the specified arguments, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
[facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import FlavaMultimodalConfig, FlavaMultimodalModel

>>> # Initializing a FlavaMultimodalModel with  style configuration
>>> configuration = FlavaMultimodalConfig()

>>> # Initializing a FlavaMultimodalModel model (with random weights) from the style configuration
>>> model = FlavaMultimodalModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 6) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries, keys and values.

use_cls_token (`bool`, *optional*, defaults to `True`) : Whether to use an extra CLS token for multimodal settings. Usually needed by the FLAVA model.

## FlavaImageCodebookConfig[[transformers.FlavaImageCodebookConfig]]

#### transformers.FlavaImageCodebookConfig[[transformers.FlavaImageCodebookConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/configuration_flava.py#L319)

## FlavaProcessor[[transformers.FlavaProcessor]]

#### transformers.FlavaProcessor[[transformers.FlavaProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/processing_flava.py#L22)

Constructs a FLAVA processor which wraps a FLAVA image processor and a FLAVA tokenizer into a single processor.

[FlavaProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaProcessor) offers all the functionalities of [FlavaImageProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageProcessor) and [BertTokenizerFast](/docs/transformers/main/en/model_doc/electra#transformers.BertTokenizer). See the
[__call__()](/docs/transformers/main/en/model_doc/bros#transformers.BrosProcessor.__call__) and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

**Parameters:**

image_processor ([FlavaImageProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageProcessor), *optional*) : The image processor is a required input.

tokenizer ([BertTokenizerFast](/docs/transformers/main/en/model_doc/electra#transformers.BertTokenizer), *optional*) : The tokenizer is a required input.

## FlavaImageProcessor[[transformers.FlavaImageProcessor]]

#### transformers.FlavaImageProcessor[[transformers.FlavaImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/image_processing_flava.py#L223)

Constructs a Flava image processor.

preprocesstransformers.FlavaImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/image_processing_flava.py#L541[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "resample", "val": ": typing.Optional[PIL.Image.Resampling] = None"}, {"name": "do_center_crop", "val": ": typing.Optional[bool] = None"}, {"name": "crop_size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "return_image_mask", "val": ": typing.Optional[bool] = None"}, {"name": "input_size_patches", "val": ": typing.Optional[int] = None"}, {"name": "total_mask_patches", "val": ": typing.Optional[int] = None"}, {"name": "mask_group_min_patches", "val": ": typing.Optional[int] = None"}, {"name": "mask_group_max_patches", "val": ": typing.Optional[int] = None"}, {"name": "mask_group_min_aspect_ratio", "val": ": typing.Optional[float] = None"}, {"name": "mask_group_max_aspect_ratio", "val": ": typing.Optional[float] = None"}, {"name": "return_codebook_pixels", "val": ": typing.Optional[bool] = None"}, {"name": "codebook_do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "codebook_size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "codebook_resample", "val": ": typing.Optional[int] = None"}, {"name": "codebook_do_center_crop", "val": ": typing.Optional[bool] = None"}, {"name": "codebook_crop_size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "codebook_do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "codebook_rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "codebook_do_map_pixels", "val": ": typing.Optional[bool] = None"}, {"name": "codebook_do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "codebook_image_mean", "val": ": typing.Optional[collections.abc.Iterable[float]] = None"}, {"name": "codebook_image_std", "val": ": typing.Optional[collections.abc.Iterable[float]] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": ChannelDimension = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}]- **images** (`ImageInput`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the image.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  Size of the image.
- **resample** (`int`, *optional*, defaults to `self.resample`) --
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
- **do_center_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) --
  Whether to center crop the image.
- **crop_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) --
  Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the image values between [0 - 1].
- **rescale_factor** (`float`, *optional*, defaults to `self.rescale_factor`) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether to normalize the image.
- **image_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) --
  Image mean.
- **image_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) --
  Image standard deviation.
- **return_image_mask** (`bool`, *optional*, defaults to `self.return_image_mask`) --
  Whether to return the image mask.
- **input_size_patches** (`int`, *optional*, defaults to `self.input_size_patches`) --
  Size of the patches to extract from the image.
- **total_mask_patches** (`int`, *optional*, defaults to `self.total_mask_patches`) --
  Total number of patches to extract from the image.
- **mask_group_min_patches** (`int`, *optional*, defaults to `self.mask_group_min_patches`) --
  Minimum number of patches to extract from the image.
- **mask_group_max_patches** (`int`, *optional*, defaults to `self.mask_group_max_patches`) --
  Maximum number of patches to extract from the image.
- **mask_group_min_aspect_ratio** (`float`, *optional*, defaults to `self.mask_group_min_aspect_ratio`) --
  Minimum aspect ratio of the patches to extract from the image.
- **mask_group_max_aspect_ratio** (`float`, *optional*, defaults to `self.mask_group_max_aspect_ratio`) --
  Maximum aspect ratio of the patches to extract from the image.
- **return_codebook_pixels** (`bool`, *optional*, defaults to `self.return_codebook_pixels`) --
  Whether to return the codebook pixels.
- **codebook_do_resize** (`bool`, *optional*, defaults to `self.codebook_do_resize`) --
  Whether to resize the codebook pixels.
- **codebook_size** (`dict[str, int]`, *optional*, defaults to `self.codebook_size`) --
  Size of the codebook pixels.
- **codebook_resample** (`int`, *optional*, defaults to `self.codebook_resample`) --
  Resampling filter to use if resizing the codebook pixels. This can be one of the enum
  `PILImageResampling`, Only has an effect if `codebook_do_resize` is set to `True`.
- **codebook_do_center_crop** (`bool`, *optional*, defaults to `self.codebook_do_center_crop`) --
  Whether to center crop the codebook pixels.
- **codebook_crop_size** (`dict[str, int]`, *optional*, defaults to `self.codebook_crop_size`) --
  Size of the center crop of the codebook pixels. Only has an effect if `codebook_do_center_crop` is set
  to `True`.
- **codebook_do_rescale** (`bool`, *optional*, defaults to `self.codebook_do_rescale`) --
  Whether to rescale the codebook pixels values between [0 - 1].
- **codebook_rescale_factor** (`float`, *optional*, defaults to `self.codebook_rescale_factor`) --
  Rescale factor to rescale the codebook pixels by if `codebook_do_rescale` is set to `True`.
- **codebook_do_map_pixels** (`bool`, *optional*, defaults to `self.codebook_do_map_pixels`) --
  Whether to map the codebook pixels values.
- **codebook_do_normalize** (`bool`, *optional*, defaults to `self.codebook_do_normalize`) --
  Whether to normalize the codebook pixels.
- **codebook_image_mean** (`float` or `list[float]`, *optional*, defaults to `self.codebook_image_mean`) --
  Codebook pixels mean to normalize the codebook pixels by if `codebook_do_normalize` is set to `True`.
- **codebook_image_std** (`float` or `list[float]`, *optional*, defaults to `self.codebook_image_std`) --
  Codebook pixels standard deviation to normalize the codebook pixels by if `codebook_do_normalize` is
  set to `True`.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  The type of tensors to return. Can be one of:
  - Unset: Return a list of `np.ndarray`.
  - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
- **data_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) --
  The channel dimension format for the output image. Can be one of:
  - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.0

Preprocess an image or batch of images.

**Parameters:**

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the `do_resize` parameter in `preprocess`.

size (`dict[str, int]` *optional*, defaults to `{"height" : 224, "width": 224}`): Size of the image after resizing. Can be overridden by the `size` parameter in `preprocess`.

resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`) : Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in `preprocess`.

do_center_crop (`bool`, *optional*, defaults to `True`) : Whether to center crop the images. Can be overridden by the `do_center_crop` parameter in `preprocess`.

crop_size (`dict[str, int]` *optional*, defaults to `{"height" : 224, "width": 224}`): Size of image after the center crop `(crop_size["height"], crop_size["width"])`. Can be overridden by the `crop_size` parameter in `preprocess`.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in `preprocess`.

rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in `preprocess`.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether to normalize the image. Can be overridden by the `do_normalize` parameter in `preprocess`.

image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) : Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

return_image_mask (`bool`, *optional*, defaults to `False`) : Whether to return the image mask. Can be overridden by the `return_image_mask` parameter in `preprocess`.

input_size_patches (`int`, *optional*, defaults to 14) : Number of patches in the image in height and width direction. 14x14 = 196 total patches. Can be overridden by the `input_size_patches` parameter in `preprocess`.

total_mask_patches (`int`, *optional*, defaults to 75) : Total number of patches that should be masked. Can be overridden by the `total_mask_patches` parameter in `preprocess`.

mask_group_min_patches (`int`, *optional*, defaults to 16) : Minimum number of patches that should be masked. Can be overridden by the `mask_group_min_patches` parameter in `preprocess`.

mask_group_max_patches (`int`, *optional*) : Maximum number of patches that should be masked. Can be overridden by the `mask_group_max_patches` parameter in `preprocess`.

mask_group_min_aspect_ratio (`float`, *optional*, defaults to 0.3) : Minimum aspect ratio of the mask window. Can be overridden by the `mask_group_min_aspect_ratio` parameter in `preprocess`.

mask_group_max_aspect_ratio (`float`, *optional*) : Maximum aspect ratio of the mask window. Can be overridden by the `mask_group_max_aspect_ratio` parameter in `preprocess`.

codebook_do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the input for codebook to a certain. Can be overridden by the `codebook_do_resize` parameter in `preprocess`. `codebook_size`.

codebook_size (`dict[str, int]`, *optional*, defaults to `{"height" : 224, "width": 224}`): Resize the input for codebook to the given size. Can be overridden by the `codebook_size` parameter in `preprocess`.

codebook_resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.LANCZOS`) : Resampling filter to use if resizing the codebook image. Can be overridden by the `codebook_resample` parameter in `preprocess`.

codebook_do_center_crop (`bool`, *optional*, defaults to `True`) : Whether to crop the input for codebook at the center. If the input size is smaller than `codebook_crop_size` along any edge, the image is padded with 0's and then center cropped. Can be overridden by the `codebook_do_center_crop` parameter in `preprocess`.

codebook_crop_size (`dict[str, int]`, *optional*, defaults to `{"height" : 224, "width": 224}`): Desired output size for codebook input when applying center-cropping. Can be overridden by the `codebook_crop_size` parameter in `preprocess`.

codebook_do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the input for codebook by the specified scale `codebook_rescale_factor`. Can be overridden by the `codebook_do_rescale` parameter in `preprocess`.

codebook_rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Defines the scale factor to use if rescaling the codebook image. Can be overridden by the `codebook_rescale_factor` parameter in `preprocess`.

codebook_do_map_pixels (`bool`, *optional*, defaults to `True`) : Whether to map the pixel values of the codebook input to (1 - 2e)x + e. Can be overridden by the `codebook_do_map_pixels` parameter in `preprocess`.

codebook_do_normalize (`bool`, *optional*, defaults to `True`) : Whether or not to normalize the input for codebook with `codebook_image_mean` and `codebook_image_std`. Can be overridden by the `codebook_do_normalize` parameter in `preprocess`.

codebook_image_mean (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0, 0, 0]`) : The sequence of means for each channel, to be used when normalizing images for codebook. Can be overridden by the `codebook_image_mean` parameter in `preprocess`.

codebook_image_std (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0.5, 0.5, 0.5]`) : The sequence of standard deviations for each channel, to be used when normalizing images for codebook. Can be overridden by the `codebook_image_std` parameter in `preprocess`.

## FlavaImageProcessorFast[[transformers.FlavaImageProcessorFast]]

#### transformers.FlavaImageProcessorFast[[transformers.FlavaImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/image_processing_flava_fast.py#L124)

Constructs a fast Flava image processor.

preprocesstransformers.FlavaImageProcessorFast.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/image_processing_flava_fast.py#L162[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.flava.image_processing_flava.FlavaImageProcessorKwargs]"}]- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_convert_rgb** (`bool`, *optional*) --
  Whether to convert the image to RGB.
- **do_resize** (`bool`, *optional*) --
  Whether to resize the image.
- **size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  Describes the maximum input dimensions to the model.
- **crop_size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  Size of the output image after applying `center_crop`.
- **resample** (`Annotated[Union[PILImageResampling, int, NoneType], None]`) --
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
- **do_rescale** (`bool`, *optional*) --
  Whether to rescale the image.
- **rescale_factor** (`float`, *optional*) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*) --
  Whether to normalize the image.
- **image_mean** (`Union[float, list[float], tuple[float, ...], NoneType]`) --
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
- **image_std** (`Union[float, list[float], tuple[float, ...], NoneType]`) --
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
- **do_pad** (`bool`, *optional*) --
  Whether to pad the image. Padding is done either to the largest size in the batch
  or to a fixed square size per image. The exact padding strategy depends on the model.
- **pad_size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
  height and width in the batch. Applied only when `do_pad=True.`
- **do_center_crop** (`bool`, *optional*) --
  Whether to center crop the image.
- **data_format** (`Union[~image_utils.ChannelDimension, str, NoneType]`) --
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
- **input_data_format** (`Union[~image_utils.ChannelDimension, str, NoneType]`) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
- **device** (`Annotated[Union[str, torch.device, NoneType], None]`) --
  The device to process the images on. If unset, the device is inferred from the input images.
- **return_tensors** (`Annotated[Union[str, ~utils.generic.TensorType, NoneType], None]`) --
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
- **disable_grouping** (`bool`, *optional*) --
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157
- **image_seq_length** (`int`, *optional*) --
  The number of image tokens to be used for each image in the input.
  Added for backward compatibility but this should be set as a processor attribute in future models.
- **return_image_mask** (`bool`, *optional*, defaults to `False`) --
  Whether to return the image mask. Can be overridden by the `return_image_mask` parameter in `preprocess`.
- **input_size_patches** (`int`, *optional*, defaults to 14) --
  Number of patches in the image in height and width direction. 14x14 = 196 total patches. Can be overridden
  by the `input_size_patches` parameter in `preprocess`.
- **total_mask_patches** (`int`, *optional*, defaults to 75) --
  Total number of patches that should be masked. Can be overridden by the `total_mask_patches` parameter in
  `preprocess`.
- **mask_group_min_patches** (`int`, *optional*, defaults to 16) --
  Minimum number of patches that should be masked. Can be overridden by the `mask_group_min_patches`
  parameter in `preprocess`.
- **mask_group_max_patches** (`int`, *optional*) --
  Maximum number of patches that should be masked. Can be overridden by the `mask_group_max_patches`
  parameter in `preprocess`.
- **mask_group_min_aspect_ratio** (`float`, *optional*, defaults to 0.3) --
  Minimum aspect ratio of the mask window. Can be overridden by the `mask_group_min_aspect_ratio` parameter
  in `preprocess`.
- **mask_group_max_aspect_ratio** (`float`, *optional*) --
  Maximum aspect ratio of the mask window. Can be overridden by the `mask_group_max_aspect_ratio` parameter
  in `preprocess`.
- **return_codebook_pixels** (`bool`, *optional*, defaults to `False`) --
  Whether to return the codebook pixel values.
- **codebook_do_resize** (`bool`, *optional*, defaults to `True`) --
  Whether to resize the input for codebook to a certain. Can be overridden by the `codebook_do_resize`
  parameter in `preprocess`. `codebook_size`.
- **codebook_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Resize the input for codebook to the given size. Can be overridden by the `codebook_size` parameter in
  `preprocess`.
- **codebook_resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.LANCZOS`) --
  Resampling filter to use if resizing the codebook image. Can be overridden by the `codebook_resample`
  parameter in `preprocess`.
- **codebook_do_center_crop** (`bool`, *optional*, defaults to `True`) --
  Whether to crop the input for codebook at the center. If the input size is smaller than
  `codebook_crop_size` along any edge, the image is padded with 0's and then center cropped. Can be
  overridden by the `codebook_do_center_crop` parameter in `preprocess`.
- **codebook_crop_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Desired output size for codebook input when applying center-cropping. Can be overridden by the
  `codebook_crop_size` parameter in `preprocess`.
- **codebook_do_rescale** (`bool`, *optional*, defaults to `True`) --
  Whether to rescale the input for codebook by the specified scale `codebook_rescale_factor`. Can be
  overridden by the `codebook_do_rescale` parameter in `preprocess`.
- **codebook_rescale_factor** (`int` or `float`, *optional*, defaults to `1/255`) --
  Defines the scale factor to use if rescaling the codebook image. Can be overridden by the
  `codebook_rescale_factor` parameter in `preprocess`.
- **codebook_do_map_pixels** (`bool`, *optional*, defaults to `True`) --
  Whether to map the pixel values of the codebook input to (1 - 2e)x + e. Can be overridden by the
  `codebook_do_map_pixels` parameter in `preprocess`.
- **codebook_do_normalize** (`bool`, *optional*, defaults to `True`) --
  Whether or not to normalize the input for codebook with `codebook_image_mean` and `codebook_image_std`. Can
  be overridden by the `codebook_do_normalize` parameter in `preprocess`.
- **codebook_image_mean** (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0, 0, 0]`) --
  The sequence of means for each channel, to be used when normalizing images for codebook. Can be overridden
  by the `codebook_image_mean` parameter in `preprocess`.
- **codebook_image_std** (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0.5, 0.5, 0.5]`) --
  The sequence of standard deviations for each channel, to be used when normalizing images for codebook. Can
  be overridden by the `codebook_image_std` parameter in `preprocess`.0``- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

do_convert_rgb (`bool`, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *optional*) : Whether to resize the image.

size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : Describes the maximum input dimensions to the model.

crop_size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[PILImageResampling, int, NoneType], None]`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *optional*) : Whether to rescale the image.

rescale_factor (`float`, *optional*) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *optional*) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...], NoneType]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...], NoneType]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *optional*) : Whether to center crop the image.

data_format (`Union[~image_utils.ChannelDimension, str, NoneType]`) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[~image_utils.ChannelDimension, str, NoneType]`) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`) : The device to process the images on. If unset, the device is inferred from the input images.

return_tensors (`Annotated[Union[str, ~utils.generic.TensorType, NoneType], None]`) : Returns stacked tensors if set to `pt, otherwise returns a list of tensors.

disable_grouping (`bool`, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

return_image_mask (`bool`, *optional*, defaults to `False`) : Whether to return the image mask. Can be overridden by the `return_image_mask` parameter in `preprocess`.

input_size_patches (`int`, *optional*, defaults to 14) : Number of patches in the image in height and width direction. 14x14 = 196 total patches. Can be overridden by the `input_size_patches` parameter in `preprocess`.

total_mask_patches (`int`, *optional*, defaults to 75) : Total number of patches that should be masked. Can be overridden by the `total_mask_patches` parameter in `preprocess`.

mask_group_min_patches (`int`, *optional*, defaults to 16) : Minimum number of patches that should be masked. Can be overridden by the `mask_group_min_patches` parameter in `preprocess`.

mask_group_max_patches (`int`, *optional*) : Maximum number of patches that should be masked. Can be overridden by the `mask_group_max_patches` parameter in `preprocess`.

mask_group_min_aspect_ratio (`float`, *optional*, defaults to 0.3) : Minimum aspect ratio of the mask window. Can be overridden by the `mask_group_min_aspect_ratio` parameter in `preprocess`.

mask_group_max_aspect_ratio (`float`, *optional*) : Maximum aspect ratio of the mask window. Can be overridden by the `mask_group_max_aspect_ratio` parameter in `preprocess`.

return_codebook_pixels (`bool`, *optional*, defaults to `False`) : Whether to return the codebook pixel values.

codebook_do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the input for codebook to a certain. Can be overridden by the `codebook_do_resize` parameter in `preprocess`. `codebook_size`.

codebook_size (`dict[str, int]`, *optional*, defaults to `{"height" : 224, "width": 224}`): Resize the input for codebook to the given size. Can be overridden by the `codebook_size` parameter in `preprocess`.

codebook_resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.LANCZOS`) : Resampling filter to use if resizing the codebook image. Can be overridden by the `codebook_resample` parameter in `preprocess`.

codebook_do_center_crop (`bool`, *optional*, defaults to `True`) : Whether to crop the input for codebook at the center. If the input size is smaller than `codebook_crop_size` along any edge, the image is padded with 0's and then center cropped. Can be overridden by the `codebook_do_center_crop` parameter in `preprocess`.

codebook_crop_size (`dict[str, int]`, *optional*, defaults to `{"height" : 224, "width": 224}`): Desired output size for codebook input when applying center-cropping. Can be overridden by the `codebook_crop_size` parameter in `preprocess`.

codebook_do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the input for codebook by the specified scale `codebook_rescale_factor`. Can be overridden by the `codebook_do_rescale` parameter in `preprocess`.

codebook_rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Defines the scale factor to use if rescaling the codebook image. Can be overridden by the `codebook_rescale_factor` parameter in `preprocess`.

codebook_do_map_pixels (`bool`, *optional*, defaults to `True`) : Whether to map the pixel values of the codebook input to (1 - 2e)x + e. Can be overridden by the `codebook_do_map_pixels` parameter in `preprocess`.

codebook_do_normalize (`bool`, *optional*, defaults to `True`) : Whether or not to normalize the input for codebook with `codebook_image_mean` and `codebook_image_std`. Can be overridden by the `codebook_do_normalize` parameter in `preprocess`.

codebook_image_mean (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0, 0, 0]`) : The sequence of means for each channel, to be used when normalizing images for codebook. Can be overridden by the `codebook_image_mean` parameter in `preprocess`.

codebook_image_std (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0.5, 0.5, 0.5]`) : The sequence of standard deviations for each channel, to be used when normalizing images for codebook. Can be overridden by the `codebook_image_std` parameter in `preprocess`.

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## FlavaForPreTraining[[transformers.FlavaForPreTraining]]

#### transformers.FlavaForPreTraining[[transformers.FlavaForPreTraining]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L1513)

The FLAVA model for pretraining which outputs losses, embeddings, logits and transformer outputs.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlavaForPreTraining.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L1562[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "input_ids_masked", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "codebook_pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "bool_masked_pos", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "image_attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "skip_unmasked_multimodal_encoder", "val": ": typing.Optional[bool] = None"}, {"name": "mlm_labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "mim_labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "itm_labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": bool = True"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "return_loss", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, text_seq_len)`) --
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids)
- **input_ids_masked** (`torch.LongTensor` of shape `(batch_size, text_seq_len)`) --
  Indices of input sequence tokens in the vocabulary. These ones are the masked version of the original task
  to be used with MLM. Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer) along with
  `DataCollatorForMaskedLanguageModeling`. See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [FlavaImageProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageProcessor). See [FlavaImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([FlavaProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaProcessor) uses
  [FlavaImageProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageProcessor) for processing images).
- **codebook_pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_image_patches, patch_size, patch_size, 3)`, *optional*) --
  Pixel values for image patches that are used to compute the image codebook labels for masked image modeling.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, text_seq_len)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:
  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.
  [What are token type IDs?](../glossary#token-type-ids)
- **bool_masked_pos** (`torch.BoolTensor` of shape `(batch_size, image_num_patches)`) --
  Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **image_attention_mask** (`torch.FloatTensor` of shape `(batch_size, image_num_patches)`, *optional*) --
  Mask to avoid performing attention on padding token indices specifically for images. Mask values selected
  in `[0, 1]`:
  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.
  [What are attention masks?](../glossary#attention-mask)
- **skip_unmasked_multimodal_encoder** (`*bool*`, *optional*) --
  Skip any calculations for multimodal encoder for unmasked inputs. FLAVA pretraining doesn't need unmasked
  multimodal embeddings or outputs as of now.
- **mlm_labels** (`torch.LongTensor` of shape `(batch_size, text_seq_len)`, *optional*) --
  Labels for computing the left-to-right language and multimodal masked modeling loss (next word prediction).
  Indices should be in `[-100, 0, ..., text_config.vocab_size - 1]` (see `input_ids` docstring). Tokens with
  indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0,
  ..., text_config.vocab_size - 1]`.
- **mim_labels** (`torch.LongTensor` of shape `(batch_size, image_num_patches)`, *optional*) --
  Labels for computing the image and multimodal masked modeling loss. Indices should be in `[-100, 0, ...,
  image_config.vocab_size - 1]`. Tokens with indices set to `-100` are ignored (masked), the loss is only
  computed for the tokens with labels in `[0, ..., image_config.vocab_size - 1]`. If not passed, they are
  generated automatically using the image codebook assigned to the model. By default, it uses
  [FlavaImageCodebook](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageCodebook). See [FlavaImageCodebook](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageCodebook) to understand how to generate mim_labels.
- **itm_labels** (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*) --
  Labels for computing the image-text matching loss. 0 means the pairs don't match and 1 means they match.
  The pairs with 0 will be skipped for calculation of MMM and global contrastive losses as well.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, defaults to `True`) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **return_loss** (`bool`, *optional*, default to None) --
  Whether to return calculated loss or not.0`transformers.models.flava.modeling_flava.FlavaForPreTrainingOutput` or `tuple(torch.FloatTensor)`A `transformers.models.flava.modeling_flava.FlavaForPreTrainingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

- **loss** (`torch.FloatTensor`, *optional*, returned when `return_loss` is True) -- Total loss calculated for this model.
- **loss_info** (`.loss_info`, defaults to `None`) -- Detailed info for FLAVA Pretraining losses. Check `FlavaLosses` class description for the information on
  the keys.
- **image_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present) -- The image embeddings which are basically the pooled output of [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel).
- **image_output** (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present) -- The output of the [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel).
- **text_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` are present) -- The text embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **text_output** (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids` are present) -- The output of the [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **multimodal_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present and `skip_unmasked_multimodal_encoder` is `None` or `False`) -- The multimodal embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **multimodal_output** (`BaseModelOutputWithPooling`, returned when `input_ids` and `pixel_values` are present and `skip_unmasked_multimodal_encoder` is `None` or `False`) -- The output of the [FlavaMultimodalModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaMultimodalModel).
- **image_masked_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present) -- The image embeddings which are basically the pooled output of [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel). Uses `bool_masked_pos`
  to create masked images.
- **image_masked_output** (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present) -- The output of the [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel). Uses `bool_masked_pos` to create masked images.
- **text_masked_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids_masked` are present) -- The text embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **text_masked_output** (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids_masked` are present) -- The output of the [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **multimodal_masked_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present) -- The multimodal embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **multimodal_masked_output** (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids_masked` and `pixel_values` are present) -- The output of the [FlavaMultimodalModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaMultimodalModel).
- **mim_logits** (`torch.FloatTensor` of shape `(batch_size, num_image_patches, image_vocab_size)` or of shape `(total_masked_patches, image_vocab_size)` , *optional*, returned when `pixel_values` are present and `input_ids_masked` are not) -- The logits for MIM unimodal loss. Uses `book_masked_pos` to get masked patches. The flattened output is
  returned when `bool_masked_pos` has some of the patches masked.
- **mlm_logits** (`torch.FloatTensor` of shape `(batch_size, text_seq_length, text_vocab_size)` or of shape `(total_masked_seq_length, text_vocab_size)`, *optional*, returned when `input_ids_masked` are present and `pixel_values` are not) -- The logits for MLM unimodal loss. The flattened output is returned when `input_ids_masked` has some of
  the tokens masked.
- **itm_logits** (`torch.FloatTensor` of shape `(batch_size, 2)`, *optional*, returned when `input_ids_masked` and `pixel_values` are present) -- The logits for ITM loss. Note that ITM loss is calculated on masked pairs in FLAVA.
- **contrastive_logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeddings` and `text_embeddings` but passed through FLAVA's
  `image_projection` and `text_projection` layers respectively. This represents the image-text similarity
  scores. This is calculated on unmasked images and texts.
- **contrastive_logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeddings` and `image_embeddings` but passed through FLAVA's
  `text_projection` and `image_projection` layers respectively. This is calculated on unmasked images and
  texts.
- **mmm_image_logits** (`torch.FloatTensor` of shape `(batch_size, num_image_patches, image_vocab_size)` or of shape`(total_masked_patches, image_vocab_size)`, *optional*, returned when `pixel_values` and `input_ids_masked` are present) -- The logits for MMM image multimodal loss. Uses `book_masked_pos` to get masked patches. The flattened
  output is returned when `bool_masked_pos` has some of the patches masked.
- **mmm_text_logits** (`torch.FloatTensor` of shape `(batch_size, text_seq_length, text_vocab_size)` or of shape `(`(total_masked_seq_length, text_vocab_size)`), *optional*, returned when `pixel_values` and `input_ids_masked` are present) -- The logits for MMM text multimodal loss. The flattened output is returned when `input_ids_masked` has
  some of the tokens masked.
The [FlavaForPreTraining](/docs/transformers/main/en/model_doc/flava#transformers.FlavaForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:
```python
>>> from PIL import Image
>>> import requests
>>> from transformers import FlavaForPreTraining, AutoProcessor

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> model = FlavaForPreTraining.from_pretrained("facebook/flava-full")
>>> processor = AutoProcessor.from_pretrained("facebook/flava-full")

>>> text = ["a photo of a cat"]

>>> inputs = processor(
...     images=[image],
...     text=text,
...     return_masks=True,
...     return_codebook_pixels=True,
...     padding=True,
...     max_length=77,
...     return_tensors="pt",
... )

>>> output = model(**inputs)
```

**Parameters:**

config ([FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

image_codebook (`torch.nn.modules.module.Module`, *optional*) : If passed, the image codebook will be set to this. Otherwise, it will be initialized using the image_codebook_config defined in the config first as the first parameter.

**Returns:**

``transformers.models.flava.modeling_flava.FlavaForPreTrainingOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.flava.modeling_flava.FlavaForPreTrainingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

- **loss** (`torch.FloatTensor`, *optional*, returned when `return_loss` is True) -- Total loss calculated for this model.
- **loss_info** (`.loss_info`, defaults to `None`) -- Detailed info for FLAVA Pretraining losses. Check `FlavaLosses` class description for the information on
  the keys.
- **image_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present) -- The image embeddings which are basically the pooled output of [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel).
- **image_output** (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present) -- The output of the [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel).
- **text_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` are present) -- The text embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **text_output** (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids` are present) -- The output of the [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **multimodal_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present and `skip_unmasked_multimodal_encoder` is `None` or `False`) -- The multimodal embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **multimodal_output** (`BaseModelOutputWithPooling`, returned when `input_ids` and `pixel_values` are present and `skip_unmasked_multimodal_encoder` is `None` or `False`) -- The output of the [FlavaMultimodalModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaMultimodalModel).
- **image_masked_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present) -- The image embeddings which are basically the pooled output of [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel). Uses `bool_masked_pos`
  to create masked images.
- **image_masked_output** (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present) -- The output of the [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel). Uses `bool_masked_pos` to create masked images.
- **text_masked_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids_masked` are present) -- The text embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **text_masked_output** (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids_masked` are present) -- The output of the [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **multimodal_masked_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present) -- The multimodal embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
- **multimodal_masked_output** (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids_masked` and `pixel_values` are present) -- The output of the [FlavaMultimodalModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaMultimodalModel).
- **mim_logits** (`torch.FloatTensor` of shape `(batch_size, num_image_patches, image_vocab_size)` or of shape `(total_masked_patches, image_vocab_size)` , *optional*, returned when `pixel_values` are present and `input_ids_masked` are not) -- The logits for MIM unimodal loss. Uses `book_masked_pos` to get masked patches. The flattened output is
  returned when `bool_masked_pos` has some of the patches masked.
- **mlm_logits** (`torch.FloatTensor` of shape `(batch_size, text_seq_length, text_vocab_size)` or of shape `(total_masked_seq_length, text_vocab_size)`, *optional*, returned when `input_ids_masked` are present and `pixel_values` are not) -- The logits for MLM unimodal loss. The flattened output is returned when `input_ids_masked` has some of
  the tokens masked.
- **itm_logits** (`torch.FloatTensor` of shape `(batch_size, 2)`, *optional*, returned when `input_ids_masked` and `pixel_values` are present) -- The logits for ITM loss. Note that ITM loss is calculated on masked pairs in FLAVA.
- **contrastive_logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeddings` and `text_embeddings` but passed through FLAVA's
  `image_projection` and `text_projection` layers respectively. This represents the image-text similarity
  scores. This is calculated on unmasked images and texts.
- **contrastive_logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeddings` and `image_embeddings` but passed through FLAVA's
  `text_projection` and `image_projection` layers respectively. This is calculated on unmasked images and
  texts.
- **mmm_image_logits** (`torch.FloatTensor` of shape `(batch_size, num_image_patches, image_vocab_size)` or of shape`(total_masked_patches, image_vocab_size)`, *optional*, returned when `pixel_values` and `input_ids_masked` are present) -- The logits for MMM image multimodal loss. Uses `book_masked_pos` to get masked patches. The flattened
  output is returned when `bool_masked_pos` has some of the patches masked.
- **mmm_text_logits** (`torch.FloatTensor` of shape `(batch_size, text_seq_length, text_vocab_size)` or of shape `(`(total_masked_seq_length, text_vocab_size)`), *optional*, returned when `pixel_values` and `input_ids_masked` are present) -- The logits for MMM text multimodal loss. The flattened output is returned when `input_ids_masked` has
  some of the tokens masked.

## FlavaModel[[transformers.FlavaModel]]

#### transformers.FlavaModel[[transformers.FlavaModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L950)

The bare Flava Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlavaModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L1095[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "bool_masked_pos", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "image_attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "skip_multimodal_encoder", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": bool = True"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, image_num_patches + text_seq_len)`) --
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [FlavaImageProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageProcessor). See [FlavaImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([FlavaProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaProcessor) uses
  [FlavaImageProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageProcessor) for processing images).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, image_num_patches + text_seq_len)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:
  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.
  [What are token type IDs?](../glossary#token-type-ids)
- **bool_masked_pos** (`torch.BoolTensor` of shape `(batch_size, image_num_patches)`) --
  Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **image_attention_mask** (`torch.Tensor` of shape `(batch_size, image_num_patches)`, *optional*) --
  Mask to avoid performing attention on padding pixel values for image inputs. Mask values selected in `[0, 1]`:
  - 1 for pixel values that are real (i.e., **not masked**),
  - 0 for pixel values that are padding (i.e., **masked**).
- **skip_multimodal_encoder** (`*bool*`, *optional*) --
  Skip any calculations for multimodal encoder. Useful if multimodal encoding is not going to be used.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, defaults to `True`) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`Union[tuple, transformers.models.flava.modeling_flava.FlavaOutput]`
The [FlavaModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, FlavaModel

>>> model = FlavaModel.from_pretrained("facebook/flava-full")
>>> processor = AutoProcessor.from_pretrained("facebook/flava-full")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)

>>> image_embeddings = outputs.image_embeddings
>>> text_embeddings = outputs.text_embeddings
>>> multimodal_embeddings = outputs.multimodal_embeddings

>>> outputs.image_embeddings.shape
torch.Size([1, 197, 768])

>>> text_embeddings.shape
torch.Size([1, 7, 768])

>>> multimodal_embeddings.shape
torch.Size([1, 205, 768])
```

**Parameters:**

config ([FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`Union[tuple, transformers.models.flava.modeling_flava.FlavaOutput]`
#### get_text_features[[transformers.FlavaModel.get_text_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L996)

Examples:

```python
>>> import torch
>>> from transformers import AutoProcessor, FlavaModel

>>> model = FlavaModel.from_pretrained("{0}")
>>> processor = AutoProcessor.from_pretrained("{0}")

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], max_length=77, padding="max_length", return_tensors="pt"
... )
>>> with torch.inference_mode():
...     text_features = model.get_text_features(**inputs)
```

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, text_seq_length)`) : Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.LongTensor` of shape `(batch_size, text_seq_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`: - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token. [What are token type IDs?](../glossary#token-type-ids)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

**Returns:**

`text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The text embeddings obtained by
applying the projection layer to the pooled output of [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel).
#### get_image_features[[transformers.FlavaModel.get_image_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L1048)

Examples:

```python
>>> import torch
>>> from transformers import AutoProcessor, FlavaModel
>>> from transformers.image_utils import load_image

>>> model = FlavaModel.from_pretrained("{0}")
>>> processor = AutoProcessor.from_pretrained("{0}")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = load_image(url)

>>> inputs = processor(images=image, return_tensors="pt")

>>> with torch.inference_mode():
...     image_features = model.get_image_features(**inputs)
```

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [FlavaImageProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageProcessor). See [FlavaImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([FlavaProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaProcessor) uses [FlavaImageProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageProcessor) for processing images).

bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, image_num_patches)`) : Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

interpolate_pos_encoding (`bool`, *optional*) : Whether to interpolate the pre-trained position encodings.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

**Returns:**

`image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The image embeddings obtained by
applying the projection layer to the pooled output of [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel).

## FlavaImageCodebook[[transformers.FlavaImageCodebook]]

#### transformers.FlavaImageCodebook[[transformers.FlavaImageCodebook]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L1304)

The FLAVA's image codebook model inspired from DALL-E's original encoder. Outputs raw hidden states and can be used
to generate image tokens for an image based on DALL-E's vocab. Used to generate labels for MIM. Use
`get_codebook_indices` to get image tokens for an image.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlavaImageCodebook.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L1387[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "**kwargs", "val": ""}]

**Parameters:**

config ([FlavaImageCodebookConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageCodebookConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
#### get_codebook_indices[[transformers.FlavaImageCodebook.get_codebook_indices]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L1355)
#### get_codebook_probs[[transformers.FlavaImageCodebook.get_codebook_probs]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L1383)

## FlavaTextModel[[transformers.FlavaTextModel]]

#### transformers.FlavaTextModel[[transformers.FlavaTextModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L770)

The bare Flava Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlavaTextModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L798[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, text_seq_length)`) --
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, text_seq_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:
  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.
  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [FlavaTextModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([FlavaTextConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaTextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

add_pooling_layer (`bool`, *optional*, defaults to `True`) : Whether to add a pooling layer

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## FlavaImageModel[[transformers.FlavaImageModel]]

#### transformers.FlavaImageModel[[transformers.FlavaImageModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L688)

The bare Flava Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlavaImageModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L718[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "bool_masked_pos", "val": ": typing.Optional[torch.BoolTensor] = None"}, {"name": "interpolate_pos_encoding", "val": ": typing.Optional[bool] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [FlavaImageProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageProcessor). See [FlavaImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([FlavaProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaProcessor) uses
  [FlavaImageProcessor](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageProcessor) for processing images).
- **bool_masked_pos** (`torch.BoolTensor` of shape `(batch_size, image_num_patches)`) --
  Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
- **interpolate_pos_encoding** (`bool`, *optional*) --
  Whether to interpolate the pre-trained position encodings.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [FlavaImageModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([FlavaImageConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaImageConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

add_pooling_layer (`bool`, *optional*, defaults to `True`) : Whether to add a pooling layer

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## FlavaMultimodalModel[[transformers.FlavaMultimodalModel]]

#### transformers.FlavaMultimodalModel[[transformers.FlavaMultimodalModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L869)

The bare Flava Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.FlavaMultimodalModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/flava/modeling_flava.py#L893[{"name": "hidden_states", "val": ": Tensor"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **hidden_states** (`torch.FloatTensor` of shape `(batch_size, image_num_patches + text_seq_len, hidden_size)`) --
  The concatenated hidden states of unimodal encoders.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [FlavaMultimodalModel](/docs/transformers/main/en/model_doc/flava#transformers.FlavaMultimodalModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([FlavaMultimodalConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaMultimodalConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

add_pooling_layer (`bool`, *optional*, defaults to `True`) : Whether to add a pooling layer

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/main/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
