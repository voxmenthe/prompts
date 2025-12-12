*This model was released on 2021-12-08 and added to Hugging Face Transformers on 2022-05-11.*

# FLAVA

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The FLAVA model was proposed in [FLAVA: A Foundational Language And Vision Alignment Model](https://huggingface.co/papers/2112.04482) by Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela and is accepted at CVPR 2022.

The paper aims at creating a single unified foundation model which can work across vision, language
as well as vision-and-language multimodal tasks.

The abstract from the paper is the following:

*State-of-the-art vision and vision-and-language models rely on large-scale visio-linguistic pretraining for obtaining good performance on a variety
of downstream tasks. Generally, such models are often either cross-modal (contrastive) or multi-modal
(with earlier fusion) but not both; and they often only target specific modalities or tasks. A promising
direction would be to use a single holistic universal model, as a “foundation”, that targets all modalities
at once — a true vision and language foundation model should be good at vision tasks, language tasks, and
cross- and multi-modal vision and language tasks. We introduce FLAVA as such a model and demonstrate
impressive performance on a wide range of 35 tasks spanning these target modalities.*

This model was contributed by [aps](https://huggingface.co/aps). The original code can be found [here](https://github.com/facebookresearch/multimodal/tree/main/examples/flava).

## FlavaConfig

### class transformers.FlavaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/configuration_flava.py#L395)

( image\_config: typing.Optional[dict[str, typing.Any]] = None text\_config: typing.Optional[dict[str, typing.Any]] = None multimodal\_config: typing.Optional[dict[str, typing.Any]] = None image\_codebook\_config: typing.Optional[dict[str, typing.Any]] = None hidden\_size: int = 768 layer\_norm\_eps: float = 1e-12 projection\_dim: int = 768 init\_codebook: bool = True logit\_scale\_init\_value: float = 2.6592 initializer\_range: float = 0.02 ce\_ignore\_index: int = -100 mim\_weight: float = 1.0 mlm\_weight: float = 1.0 global\_contrastive\_weight: float = 1.0 itm\_weight: float = 1.0 mmm\_image\_weight: float = 1.0 mmm\_text\_weight: float = 1.0 global\_backprop\_contrastive: bool = True skip\_unmasked\_multimodal\_encoder: bool = True return\_loss: bool = True \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [FlavaTextConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextConfig).
* **image\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [FlavaImageConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageConfig).
* **multimodal\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [FlavaMultimodalConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaMultimodalConfig).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **projection\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of text and image projection layers.
* **logit\_scale\_init\_value** (`float`, *optional*, defaults to 2.6592) —
  The initial value of the *logit\_scale* parameter. Default is used as per the original FLAVA/CLIP
  implementation.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **ce\_ignore\_index** (`int`, *optional*, defaults to -100) —
  Cross entropy index to ignore.
* **mim\_weight** (`float`, *optional*, defaults to 1.0) —
  Weight to be assigned to MIM (Masked Image Modeling) unimodal loss
* **mlm\_weight** (`float`, *optional*, defaults to 1.0) —
  Weight to be assigned to MLM (Masked Language Modeling) unimodal loss
* **global\_contrastive\_weight** (`float`, *optional*, defaults to 1.0) —
  Weight to be assigned to global contrastive cross-alignment loss.
* **itm\_weight** (`float`, *optional*, defaults to 1.0) —
  Weight to be assigned to image-text matching multimodal loss.
* **mmm\_image\_weight** (`float`, *optional*, defaults to 1.0) —
  Weight to be assigned to MMM loss’s image part.
* **mmm\_text\_weight** (`float`, *optional*, defaults to 1.0) —
  Weight to be assigned to MMM loss’s text part.
* **global\_backprop\_contrastive** (`bool`, *optional*, defaults to `True`) —
  Whether to use global backpropgation through all workers in contrastive loss.
* **skip\_unmasked\_multimodal\_encoder** (`bool`, *optional*, defaults to `True`) —
  Whether to skip running unmasked multimodal encoder whose outputs are not used by FLAVA losses.
* **return\_loss** (`bool`, *optional*, defaults to `True`) —
  Whether to return loss or not
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig) is the configuration class to store the configuration of a [FlavaModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaModel). It is used to
instantiate FLAVA model according to the specified arguments, defining the text model, image model, image codebook
and multimodal model configs. Instantiating a configuration with the defaults will yield a similar configuration to
that of the FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
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

#### from\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/configuration_flava.py#L675)

( image\_config: FlavaImageConfig text\_config: FlavaTextConfig multimodal\_config: FlavaMultimodalConfig image\_codebook\_config: FlavaImageCodebookConfig \*\*kwargs  ) → [FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig)

Returns

[FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig)

An instance of a configuration object

Instantiate a [FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig) (or a derived class) from flava text model configuration, flava image model
configuration, flava multimodal model and flava codebook model configuration.

## FlavaTextConfig

### class transformers.FlavaTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/configuration_flava.py#L128)

( vocab\_size: int = 30522 type\_vocab\_size: int = 2 max\_position\_embeddings: int = 512 position\_embedding\_type: str = 'absolute' hidden\_size: int = 768 num\_hidden\_layers: int = 12 num\_attention\_heads: int = 12 intermediate\_size: int = 3072 hidden\_act: str = 'gelu' hidden\_dropout\_prob: float = 0.0 attention\_probs\_dropout\_prob: float = 0.0 initializer\_range: float = 0.02 layer\_norm\_eps: float = 1e-12 pad\_token\_id: int = 0 qkv\_bias: bool = True \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel).
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the `token_type_ids` passed when calling [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel). Note that even though
  text encoder allows `token_type_ids`’s value as 2, for text-only pretraining and fine-tuning, only 1 is
  used similar to RoBERTa.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048). For VL, max\_length passed to model is 77.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"absolute"`) —
  Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
  positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
  [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
  For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
  with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.

This is the configuration class to store the configuration of a [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel). It is used to instantiate an
FLAVA model according to the specified arguments, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
[facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import FlavaTextConfig, FlavaTextModel

>>> # Initializing a FlavaTextModel with  style configuration
>>> configuration = FlavaTextConfig()

>>> # Initializing a FlavaTextModel model (with random weights) from the style configuration
>>> model = FlavaTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## FlavaImageConfig

### class transformers.FlavaImageConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/configuration_flava.py#L26)

( hidden\_size: int = 768 num\_hidden\_layers: int = 12 num\_attention\_heads: int = 12 intermediate\_size: int = 3072 hidden\_act: int = 'gelu' hidden\_dropout\_prob: float = 0.0 attention\_probs\_dropout\_prob: float = 0.0 initializer\_range: float = 0.02 layer\_norm\_eps: float = 1e-12 image\_size: int = 224 patch\_size: int = 16 num\_channels: int = 3 qkv\_bias: bool = True mask\_token: bool = True vocab\_size: int = 8192 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.
* **mask\_token** (`bool`, *optional*, defaults to `True`) —
  Whether to use a mask token or not. Used in MIM (Masked Image Modeling) loss for FLAVA.
* **vocab\_size** (`int`, *optional*, defaults to 8192) —
  Vocabulary size of the [FlavaImageCodebook](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageCodebook) used in conjunction with [FlavaImageModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageModel) for MIM (Masked
  Image Modeling) loss for FLAVA.

This is the configuration class to store the configuration of a [FlavaImageModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageModel). It is used to instantiate an
FLAVA model according to the specified arguments, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
[facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import FlavaImageConfig, FlavaImageModel

>>> # Initializing a FlavaImageModel with  style configuration
>>> configuration = FlavaImageConfig()

>>> # Initializing a FlavaImageModel model (with random weights) from the style configuration
>>> model = FlavaImageModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## FlavaMultimodalConfig

### class transformers.FlavaMultimodalConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/configuration_flava.py#L241)

( hidden\_size: int = 768 num\_hidden\_layers: int = 6 num\_attention\_heads: int = 12 intermediate\_size: int = 3072 hidden\_act: int = 'gelu' hidden\_dropout\_prob: int = 0.0 attention\_probs\_dropout\_prob: int = 0.0 initializer\_range: float = 0.02 layer\_norm\_eps: float = 1e-12 qkv\_bias: bool = True use\_cls\_token: bool = True \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 6) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.
* **use\_cls\_token** (`bool`, *optional*, defaults to `True`) —
  Whether to use an extra CLS token for multimodal settings. Usually needed by the FLAVA model.

This is the configuration class to store the configuration of a [FlavaMultimodalModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaMultimodalModel). It is used to instantiate
an FLAVA model according to the specified arguments, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
[facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import FlavaMultimodalConfig, FlavaMultimodalModel

>>> # Initializing a FlavaMultimodalModel with  style configuration
>>> configuration = FlavaMultimodalConfig()

>>> # Initializing a FlavaMultimodalModel model (with random weights) from the style configuration
>>> model = FlavaMultimodalModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## FlavaImageCodebookConfig

### class transformers.FlavaImageCodebookConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/configuration_flava.py#L327)

( num\_groups: int = 4 input\_channels: int = 3 num\_blocks\_per\_group: int = 2 hidden\_size: int = 256 vocab\_size: int = 8192 freeze: int = True initializer\_range: float = 0.02 \*\*kwargs  )

## FlavaProcessor

### class transformers.FlavaProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/processing_flava.py#L28)

( image\_processor = None tokenizer = None \*\*kwargs  )

Parameters

* **image\_processor** ([FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor), *optional*) — The image processor is a required input.
* **tokenizer** ([BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast), *optional*) — The tokenizer is a required input.

Constructs a FLAVA processor which wraps a FLAVA image processor and a FLAVA tokenizer into a single processor.

[FlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaProcessor) offers all the functionalities of [FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor) and [BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## FlavaFeatureExtractor

### class transformers.FlavaFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/feature_extraction_flava.py#L28)

( \*args \*\*kwargs  )

## FlavaImageProcessor

### class transformers.FlavaImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/image_processing_flava.py#L139)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_center\_crop: bool = True crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, collections.abc.Iterable[float], NoneType] = None image\_std: typing.Union[float, collections.abc.Iterable[float], NoneType] = None return\_image\_mask: bool = False input\_size\_patches: int = 14 total\_mask\_patches: int = 75 mask\_group\_min\_patches: int = 16 mask\_group\_max\_patches: typing.Optional[int] = None mask\_group\_min\_aspect\_ratio: float = 0.3 mask\_group\_max\_aspect\_ratio: typing.Optional[float] = None return\_codebook\_pixels: bool = False codebook\_do\_resize: bool = True codebook\_size: typing.Optional[bool] = None codebook\_resample: int = <Resampling.LANCZOS: 1> codebook\_do\_center\_crop: bool = True codebook\_crop\_size: typing.Optional[int] = None codebook\_do\_rescale: bool = True codebook\_rescale\_factor: typing.Union[int, float] = 0.00392156862745098 codebook\_do\_map\_pixels: bool = True codebook\_do\_normalize: bool = True codebook\_image\_mean: typing.Union[float, collections.abc.Iterable[float], NoneType] = None codebook\_image\_std: typing.Union[float, collections.abc.Iterable[float], NoneType] = None \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in `preprocess`.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Size of the image after resizing. Can be overridden by the `size` parameter in `preprocess`.
* **resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`) —
  Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in
  `preprocess`.
* **do\_center\_crop** (`bool`, *optional*, defaults to `True`) —
  Whether to center crop the images. Can be overridden by the `do_center_crop` parameter in `preprocess`.
* **crop\_size** (`dict[str, int]` *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Size of image after the center crop `(crop_size["height"], crop_size["width"])`. Can be overridden by the
  `crop_size` parameter in `preprocess`.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in `preprocess`.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in
  `preprocess`.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in `preprocess`.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
* **return\_image\_mask** (`bool`, *optional*, defaults to `False`) —
  Whether to return the image mask. Can be overridden by the `return_image_mask` parameter in `preprocess`.
* **input\_size\_patches** (`int`, *optional*, defaults to 14) —
  Number of patches in the image in height and width direction. 14x14 = 196 total patches. Can be overridden
  by the `input_size_patches` parameter in `preprocess`.
* **total\_mask\_patches** (`int`, *optional*, defaults to 75) —
  Total number of patches that should be masked. Can be overridden by the `total_mask_patches` parameter in
  `preprocess`.
* **mask\_group\_min\_patches** (`int`, *optional*, defaults to 16) —
  Minimum number of patches that should be masked. Can be overridden by the `mask_group_min_patches`
  parameter in `preprocess`.
* **mask\_group\_max\_patches** (`int`, *optional*) —
  Maximum number of patches that should be masked. Can be overridden by the `mask_group_max_patches`
  parameter in `preprocess`.
* **mask\_group\_min\_aspect\_ratio** (`float`, *optional*, defaults to 0.3) —
  Minimum aspect ratio of the mask window. Can be overridden by the `mask_group_min_aspect_ratio` parameter
  in `preprocess`.
* **mask\_group\_max\_aspect\_ratio** (`float`, *optional*) —
  Maximum aspect ratio of the mask window. Can be overridden by the `mask_group_max_aspect_ratio` parameter
  in `preprocess`.
* **codebook\_do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the input for codebook to a certain. Can be overridden by the `codebook_do_resize`
  parameter in `preprocess`. `codebook_size`.
* **codebook\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Resize the input for codebook to the given size. Can be overridden by the `codebook_size` parameter in
  `preprocess`.
* **codebook\_resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.LANCZOS`) —
  Resampling filter to use if resizing the codebook image. Can be overridden by the `codebook_resample`
  parameter in `preprocess`.
* **codebook\_do\_center\_crop** (`bool`, *optional*, defaults to `True`) —
  Whether to crop the input for codebook at the center. If the input size is smaller than
  `codebook_crop_size` along any edge, the image is padded with 0’s and then center cropped. Can be
  overridden by the `codebook_do_center_crop` parameter in `preprocess`.
* **codebook\_crop\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Desired output size for codebook input when applying center-cropping. Can be overridden by the
  `codebook_crop_size` parameter in `preprocess`.
* **codebook\_do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the input for codebook by the specified scale `codebook_rescale_factor`. Can be
  overridden by the `codebook_do_rescale` parameter in `preprocess`.
* **codebook\_rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Defines the scale factor to use if rescaling the codebook image. Can be overridden by the
  `codebook_rescale_factor` parameter in `preprocess`.
* **codebook\_do\_map\_pixels** (`bool`, *optional*, defaults to `True`) —
  Whether to map the pixel values of the codebook input to (1 - 2e)x + e. Can be overridden by the
  `codebook_do_map_pixels` parameter in `preprocess`.
* **codebook\_do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether or not to normalize the input for codebook with `codebook_image_mean` and `codebook_image_std`. Can
  be overridden by the `codebook_do_normalize` parameter in `preprocess`.
* **codebook\_image\_mean** (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0, 0, 0]`) —
  The sequence of means for each channel, to be used when normalizing images for codebook. Can be overridden
  by the `codebook_image_mean` parameter in `preprocess`.
* **codebook\_image\_std** (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0.5, 0.5, 0.5]`) —
  The sequence of standard deviations for each channel, to be used when normalizing images for codebook. Can
  be overridden by the `codebook_image_std` parameter in `preprocess`.

Constructs a Flava image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/image_processing_flava.py#L456)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_center\_crop: typing.Optional[bool] = None crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_image\_mask: typing.Optional[bool] = None input\_size\_patches: typing.Optional[int] = None total\_mask\_patches: typing.Optional[int] = None mask\_group\_min\_patches: typing.Optional[int] = None mask\_group\_max\_patches: typing.Optional[int] = None mask\_group\_min\_aspect\_ratio: typing.Optional[float] = None mask\_group\_max\_aspect\_ratio: typing.Optional[float] = None return\_codebook\_pixels: typing.Optional[bool] = None codebook\_do\_resize: typing.Optional[bool] = None codebook\_size: typing.Optional[dict[str, int]] = None codebook\_resample: typing.Optional[int] = None codebook\_do\_center\_crop: typing.Optional[bool] = None codebook\_crop\_size: typing.Optional[dict[str, int]] = None codebook\_do\_rescale: typing.Optional[bool] = None codebook\_rescale\_factor: typing.Optional[float] = None codebook\_do\_map\_pixels: typing.Optional[bool] = None codebook\_do\_normalize: typing.Optional[bool] = None codebook\_image\_mean: typing.Optional[collections.abc.Iterable[float]] = None codebook\_image\_std: typing.Optional[collections.abc.Iterable[float]] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) —
  Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation.
* **return\_image\_mask** (`bool`, *optional*, defaults to `self.return_image_mask`) —
  Whether to return the image mask.
* **input\_size\_patches** (`int`, *optional*, defaults to `self.input_size_patches`) —
  Size of the patches to extract from the image.
* **total\_mask\_patches** (`int`, *optional*, defaults to `self.total_mask_patches`) —
  Total number of patches to extract from the image.
* **mask\_group\_min\_patches** (`int`, *optional*, defaults to `self.mask_group_min_patches`) —
  Minimum number of patches to extract from the image.
* **mask\_group\_max\_patches** (`int`, *optional*, defaults to `self.mask_group_max_patches`) —
  Maximum number of patches to extract from the image.
* **mask\_group\_min\_aspect\_ratio** (`float`, *optional*, defaults to `self.mask_group_min_aspect_ratio`) —
  Minimum aspect ratio of the patches to extract from the image.
* **mask\_group\_max\_aspect\_ratio** (`float`, *optional*, defaults to `self.mask_group_max_aspect_ratio`) —
  Maximum aspect ratio of the patches to extract from the image.
* **return\_codebook\_pixels** (`bool`, *optional*, defaults to `self.return_codebook_pixels`) —
  Whether to return the codebook pixels.
* **codebook\_do\_resize** (`bool`, *optional*, defaults to `self.codebook_do_resize`) —
  Whether to resize the codebook pixels.
* **codebook\_size** (`dict[str, int]`, *optional*, defaults to `self.codebook_size`) —
  Size of the codebook pixels.
* **codebook\_resample** (`int`, *optional*, defaults to `self.codebook_resample`) —
  Resampling filter to use if resizing the codebook pixels. This can be one of the enum
  `PILImageResampling`, Only has an effect if `codebook_do_resize` is set to `True`.
* **codebook\_do\_center\_crop** (`bool`, *optional*, defaults to `self.codebook_do_center_crop`) —
  Whether to center crop the codebook pixels.
* **codebook\_crop\_size** (`dict[str, int]`, *optional*, defaults to `self.codebook_crop_size`) —
  Size of the center crop of the codebook pixels. Only has an effect if `codebook_do_center_crop` is set
  to `True`.
* **codebook\_do\_rescale** (`bool`, *optional*, defaults to `self.codebook_do_rescale`) —
  Whether to rescale the codebook pixels values between [0 - 1].
* **codebook\_rescale\_factor** (`float`, *optional*, defaults to `self.codebook_rescale_factor`) —
  Rescale factor to rescale the codebook pixels by if `codebook_do_rescale` is set to `True`.
* **codebook\_do\_map\_pixels** (`bool`, *optional*, defaults to `self.codebook_do_map_pixels`) —
  Whether to map the codebook pixels values.
* **codebook\_do\_normalize** (`bool`, *optional*, defaults to `self.codebook_do_normalize`) —
  Whether to normalize the codebook pixels.
* **codebook\_image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.codebook_image_mean`) —
  Codebook pixels mean to normalize the codebook pixels by if `codebook_do_normalize` is set to `True`.
* **codebook\_image\_std** (`float` or `list[float]`, *optional*, defaults to `self.codebook_image_std`) —
  Codebook pixels standard deviation to normalize the codebook pixels by if `codebook_do_normalize` is
  set to `True`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

## FlavaImageProcessorFast

### class transformers.FlavaImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/image_processing_flava_fast.py#L221)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.flava.image\_processing\_flava\_fast.FlavaFastImageProcessorKwargs]  )

Constructs a fast Flava image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/image_processing_flava_fast.py#L259)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
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

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## FlavaForPreTraining

### class transformers.FlavaForPreTraining

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L1616)

( config: FlavaConfig image\_codebook: typing.Optional[torch.nn.modules.module.Module] = None  )

Parameters

* **config** ([FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **image\_codebook** (`torch.nn.modules.module.Module`, *optional*) —
  If passed, the image codebook will be set to this. Otherwise, it will be initialized using the
  image\_codebook\_config defined in the config first as the first parameter.

The FLAVA model for pretraining which outputs losses, embeddings, logits and transformer outputs.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L1665)

( input\_ids: typing.Optional[torch.LongTensor] = None input\_ids\_masked: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None codebook\_pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None bool\_masked\_pos: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None image\_attention\_mask: typing.Optional[torch.Tensor] = None skip\_unmasked\_multimodal\_encoder: typing.Optional[bool] = None mlm\_labels: typing.Optional[torch.Tensor] = None mim\_labels: typing.Optional[torch.Tensor] = None itm\_labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: bool = True return\_dict: typing.Optional[bool] = None return\_loss: typing.Optional[bool] = None  ) → `transformers.models.flava.modeling_flava.FlavaForPreTrainingOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, text_seq_len)`) —
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids)
* **input\_ids\_masked** (`torch.LongTensor` of shape `(batch_size, text_seq_len)`) —
  Indices of input sequence tokens in the vocabulary. These ones are the masked version of the original task
  to be used with MLM. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer) along with
  `DataCollatorForMaskedLanguageModeling`. See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor). See [FlavaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([FlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaProcessor) uses
  [FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor) for processing images).
* **codebook\_pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_image_patches, patch_size, patch_size, 3)`, *optional*) —
  Pixel values for image patches that are used to compute the image codebook labels for masked image modeling.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, text_seq_len)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    [What are token type IDs?](../glossary#token-type-ids)
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, image_num_patches)`) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **image\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, image_num_patches)`, *optional*) —
  Mask to avoid performing attention on padding token indices specifically for images. Mask values selected
  in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
    [What are attention masks?](../glossary#attention-mask)
* **skip\_unmasked\_multimodal\_encoder** (`*bool*`, *optional*) —
  Skip any calculations for multimodal encoder for unmasked inputs. FLAVA pretraining doesn’t need unmasked
  multimodal embeddings or outputs as of now.
* **mlm\_labels** (`torch.LongTensor` of shape `(batch_size, text_seq_len)`, *optional*) —
  Labels for computing the left-to-right language and multimodal masked modeling loss (next word prediction).
  Indices should be in `[-100, 0, ..., text_config.vocab_size - 1]` (see `input_ids` docstring). Tokens with
  indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., text_config.vocab_size - 1]`.
* **mim\_labels** (`torch.LongTensor` of shape `(batch_size, image_num_patches)`, *optional*) —
  Labels for computing the image and multimodal masked modeling loss. Indices should be in `[-100, 0, ..., image_config.vocab_size - 1]`. Tokens with indices set to `-100` are ignored (masked), the loss is only
  computed for the tokens with labels in `[0, ..., image_config.vocab_size - 1]`. If not passed, they are
  generated automatically using the image codebook assigned to the model. By default, it uses
  [FlavaImageCodebook](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageCodebook). See [FlavaImageCodebook](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageCodebook) to understand how to generate mim\_labels.
* **itm\_labels** (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*) —
  Labels for computing the image-text matching loss. 0 means the pairs don’t match and 1 means they match.
  The pairs with 0 will be skipped for calculation of MMM and global contrastive losses as well.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, defaults to `True`) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **return\_loss** (`bool`, *optional*, default to None) —
  Whether to return calculated loss or not.

Returns

`transformers.models.flava.modeling_flava.FlavaForPreTrainingOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.flava.modeling_flava.FlavaForPreTrainingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

* **loss** (`torch.FloatTensor`, *optional*, returned when `return_loss` is True) — Total loss calculated for this model.
* **loss\_info** (`<class '~models.flava.modeling_flava.FlavaLosses'>.loss_info`, defaults to `None`) — Detailed info for FLAVA Pretraining losses. Check `FlavaLosses` class description for the information on
  the keys.
* **image\_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present) — The image embeddings which are basically the pooled output of [FlavaImageModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageModel).
* **image\_output** (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present) — The output of the [FlavaImageModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageModel).
* **text\_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` are present) — The text embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel).
* **text\_output** (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids` are present) — The output of the [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel).
* **multimodal\_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present and `skip_unmasked_multimodal_encoder` is `None` or `False`) — The multimodal embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel).
* **multimodal\_output** (`BaseModelOutputWithPooling`, returned when `input_ids` and `pixel_values` are present and `skip_unmasked_multimodal_encoder` is `None` or `False`) — The output of the [FlavaMultimodalModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaMultimodalModel).
* **image\_masked\_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present) — The image embeddings which are basically the pooled output of [FlavaImageModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageModel). Uses `bool_masked_pos`
  to create masked images.
* **image\_masked\_output** (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present) — The output of the [FlavaImageModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageModel). Uses `bool_masked_pos` to create masked images.
* **text\_masked\_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids_masked` are present) — The text embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel).
* **text\_masked\_output** (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids_masked` are present) — The output of the [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel).
* **multimodal\_masked\_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present) — The multimodal embeddings which are basically the pooled output of [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel).
* **multimodal\_masked\_output** (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids_masked` and `pixel_values` are present) — The output of the [FlavaMultimodalModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaMultimodalModel).
* **mim\_logits** (`torch.FloatTensor` of shape `(batch_size, num_image_patches, image_vocab_size)` or of shape `(total_masked_patches, image_vocab_size)` , *optional*, returned when `pixel_values` are present and `input_ids_masked` are not) — The logits for MIM unimodal loss. Uses `book_masked_pos` to get masked patches. The flattened output is
  returned when `bool_masked_pos` has some of the patches masked.
* **mlm\_logits** (`torch.FloatTensor` of shape `(batch_size, text_seq_length, text_vocab_size)` or of shape `(total_masked_seq_length, text_vocab_size)`, *optional*, returned when `input_ids_masked` are present and `pixel_values` are not) — The logits for MLM unimodal loss. The flattened output is returned when `input_ids_masked` has some of
  the tokens masked.
* **itm\_logits** (`torch.FloatTensor` of shape `(batch_size, 2)`, *optional*, returned when `input_ids_masked` and `pixel_values` are present) — The logits for ITM loss. Note that ITM loss is calculated on masked pairs in FLAVA.
* **contrastive\_logits\_per\_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeddings` and `text_embeddings` but passed through FLAVA’s
  `image_projection` and `text_projection` layers respectively. This represents the image-text similarity
  scores. This is calculated on unmasked images and texts.
* **contrastive\_logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeddings` and `image_embeddings` but passed through FLAVA’s
  `text_projection` and `image_projection` layers respectively. This is calculated on unmasked images and
  texts.
* **mmm\_image\_logits** (`torch.FloatTensor` of shape `(batch_size, num_image_patches, image_vocab_size)` or of shape`(total_masked_patches, image_vocab_size)`, *optional*, returned when `pixel_values` and `input_ids_masked` are present) — The logits for MMM image multimodal loss. Uses `book_masked_pos` to get masked patches. The flattened
  output is returned when `bool_masked_pos` has some of the patches masked.
* **mmm\_text\_logits** (`torch.FloatTensor` of shape `(batch_size, text_seq_length, text_vocab_size)` or of shape `(`(total\_masked\_seq\_length, text\_vocab\_size)`), *optional*, returned when` pixel\_values`and`input\_ids\_masked`are present) -- The logits for MMM text multimodal loss. The flattened output is returned when`input\_ids\_masked` has
  some of the tokens masked.

The [FlavaForPreTraining](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
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

## FlavaModel

### class transformers.FlavaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L1038)

( config: FlavaConfig  )

Parameters

* **config** ([FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Flava Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L1194)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None bool\_masked\_pos: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None image\_attention\_mask: typing.Optional[torch.Tensor] = None skip\_multimodal\_encoder: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: bool = True return\_dict: typing.Optional[bool] = None  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, image_num_patches + text_seq_len)`) —
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor). See [FlavaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([FlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaProcessor) uses
  [FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, image_num_patches + text_seq_len)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    [What are token type IDs?](../glossary#token-type-ids)
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, image_num_patches)`) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **image\_attention\_mask** (`torch.Tensor` of shape `(batch_size, image_num_patches)`, *optional*) —
  Mask to avoid performing attention on padding pixel values for image inputs. Mask values selected in `[0, 1]`:
  + 1 for pixel values that are real (i.e., **not masked**),
  + 0 for pixel values that are padding (i.e., **masked**).
* **skip\_multimodal\_encoder** (`*bool*`, *optional*) —
  Skip any calculations for multimodal encoder. Useful if multimodal encoding is not going to be used.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, defaults to `True`) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

The [FlavaModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
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

#### get\_text\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L1084)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, text_seq_length)`) —
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, text_seq_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by
applying the projection layer to the pooled output of [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel).

Examples:


```
>>> from transformers import AutoProcessor, FlavaModel

>>> model = FlavaModel.from_pretrained("{0}")
>>> processor = AutoProcessor.from_pretrained("{0}")

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], max_length=77, padding="max_length", return_tensors="pt"
... )
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L1140)

( pixel\_values: typing.Optional[torch.Tensor] = None bool\_masked\_pos: typing.Optional[torch.BoolTensor] = None interpolate\_pos\_encoding: typing.Optional[bool] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor). See [FlavaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([FlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaProcessor) uses
  [FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor) for processing images).
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, image_num_patches)`) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).
* **interpolate\_pos\_encoding** (`bool`, *optional*) —
  Whether to interpolate the pre-trained position encodings.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by
applying the projection layer to the pooled output of [FlavaImageModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageModel).

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, FlavaModel

>>> model = FlavaModel.from_pretrained("{0}")
>>> processor = AutoProcessor.from_pretrained("{0}")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```

## FlavaImageCodebook

### class transformers.FlavaImageCodebook

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L1402)

( config: FlavaImageCodebookConfig \*\*kwargs: typing.Any  )

Parameters

* **config** ([FlavaImageCodebookConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageCodebookConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The FLAVA’s image codebook model inspired from DALL-E’s original encoder. Outputs raw hidden states and can be used
to generate image tokens for an image based on DALL-E’s vocab. Used to generate labels for MIM. Use
`get_codebook_indices` to get image tokens for an image.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L1484)

( pixel\_values: FloatTensor  )

#### get\_codebook\_indices

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L1452)

( pixel\_values: Tensor  )

#### get\_codebook\_probs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L1480)

( pixel\_values: Tensor  )

## FlavaTextModel

### class transformers.FlavaTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L829)

( config: FlavaTextConfig add\_pooling\_layer: bool = True  )

Parameters

* **config** ([FlavaTextConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The bare Flava Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L864)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, text_seq_length)`) —
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, text_seq_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.
    [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

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

The [FlavaTextModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## FlavaImageModel

### class transformers.FlavaImageModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L732)

( config: FlavaImageConfig add\_pooling\_layer: bool = True  )

Parameters

* **config** ([FlavaImageConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The bare Flava Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L769)

( pixel\_values: typing.Optional[torch.Tensor] = None bool\_masked\_pos: typing.Optional[torch.BoolTensor] = None interpolate\_pos\_encoding: typing.Optional[bool] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor). See [FlavaImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([FlavaProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaProcessor) uses
  [FlavaImageProcessor](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageProcessor) for processing images).
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, image_num_patches)`) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).
* **interpolate\_pos\_encoding** (`bool`, *optional*) —
  Whether to interpolate the pre-trained position encodings.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

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

The [FlavaImageModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaImageModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## FlavaMultimodalModel

### class transformers.FlavaMultimodalModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L942)

( config: FlavaMultimodalConfig add\_pooling\_layer = True  )

Parameters

* **config** ([FlavaMultimodalConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaMultimodalConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The bare Flava Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/flava/modeling_flava.py#L974)

( hidden\_states: Tensor attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **hidden\_states** (`torch.FloatTensor` of shape `(batch_size, image_num_patches + text_seq_len, hidden_size)`) —
  The concatenated hidden states of unimodal encoders.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([FlavaConfig](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaConfig)) and inputs.

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

The [FlavaMultimodalModel](/docs/transformers/v4.56.2/en/model_doc/flava#transformers.FlavaMultimodalModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/flava.md)
