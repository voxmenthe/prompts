*This model was released on 2021-03-24 and added to Hugging Face Transformers on 2022-03-28.*

# DPT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The DPT model was proposed in [Vision Transformers for Dense Prediction](https://huggingface.co/papers/2103.13413) by René Ranftl, Alexey Bochkovskiy, Vladlen Koltun.
DPT is a model that leverages the [Vision Transformer (ViT)](vit) as backbone for dense prediction tasks like semantic segmentation and depth estimation.

The abstract from the paper is the following:

*We introduce dense vision transformers, an architecture that leverages vision transformers in place of convolutional networks as a backbone for dense prediction tasks. We assemble tokens from various stages of the vision transformer into image-like representations at various resolutions and progressively combine them into full-resolution predictions using a convolutional decoder. The transformer backbone processes representations at a constant and relatively high resolution and has a global receptive field at every stage. These properties allow the dense vision transformer to provide finer-grained and more globally coherent predictions when compared to fully-convolutional networks. Our experiments show that this architecture yields substantial improvements on dense prediction tasks, especially when a large amount of training data is available. For monocular depth estimation, we observe an improvement of up to 28% in relative performance when compared to a state-of-the-art fully-convolutional network. When applied to semantic segmentation, dense vision transformers set a new state of the art on ADE20K with 49.02% mIoU. We further show that the architecture can be fine-tuned on smaller datasets such as NYUv2, KITTI, and Pascal Context where it also sets the new state of the art.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dpt_architecture.jpg) DPT architecture. Taken from the [original paper](https://huggingface.co/papers/2103.13413).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/isl-org/DPT).

## Usage tips

DPT is compatible with the [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone) class. This allows to use the DPT framework with various computer vision backbones available in the library, such as `VitDetBackbone` or `Dinov2Backbone`. One can create it as follows:


```
from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation

# initialize with a Transformer-based backbone such as DINOv2
# in that case, we also specify `reshape_hidden_states=False` to get feature maps of shape (batch_size, num_channels, height, width)
backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage1", "stage2", "stage3", "stage4"], reshape_hidden_states=False)

config = DPTConfig(backbone_config=backbone_config)
model = DPTForDepthEstimation(config=config)
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DPT.

* Demo notebooks for [DPTForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForDepthEstimation) can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DPT).
* [Semantic segmentation task guide](../tasks/semantic_segmentation)
* [Monocular depth estimation task guide](../tasks/monocular_depth_estimation)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DPTConfig

### class transformers.DPTConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/configuration_dpt.py#L29)

( hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 image\_size = 384 patch\_size = 16 num\_channels = 3 is\_hybrid = False qkv\_bias = True backbone\_out\_indices = [2, 5, 8, 11] readout\_type = 'project' reassemble\_factors = [4, 2, 1, 0.5] neck\_hidden\_sizes = [96, 192, 384, 768] fusion\_hidden\_size = 256 head\_in\_index = -1 use\_batch\_norm\_in\_fusion\_residual = False use\_bias\_in\_fusion\_residual = None add\_projection = False use\_auxiliary\_head = True auxiliary\_loss\_weight = 0.4 semantic\_loss\_ignore\_index = 255 semantic\_classifier\_dropout = 0.1 backbone\_featmap\_shape = [1, 1024, 24, 24] neck\_ignore\_stages = [0, 1] backbone\_config = None backbone = None use\_pretrained\_backbone = False use\_timm\_backbone = False backbone\_kwargs = None pooler\_output\_size = None pooler\_act = 'tanh' \*\*kwargs  )

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
* **image\_size** (`int`, *optional*, defaults to 384) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **is\_hybrid** (`bool`, *optional*, defaults to `False`) —
  Whether to use a hybrid backbone. Useful in the context of loading DPT-Hybrid models.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.
* **backbone\_out\_indices** (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`) —
  Indices of the intermediate hidden states to use from backbone.
* **readout\_type** (`str`, *optional*, defaults to `"project"`) —
  The readout type to use when processing the readout token (CLS token) of the intermediate hidden states of
  the ViT backbone. Can be one of [`"ignore"`, `"add"`, `"project"`].
  + “ignore” simply ignores the CLS token.
  + “add” passes the information from the CLS token to all other tokens by adding the representations.
  + “project” passes information to the other tokens by concatenating the readout to all other tokens before
    projecting the
    representation to the original feature dimension D using a linear layer followed by a GELU non-linearity.
* **reassemble\_factors** (`list[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`) —
  The up/downsampling factors of the reassemble layers.
* **neck\_hidden\_sizes** (`list[str]`, *optional*, defaults to `[96, 192, 384, 768]`) —
  The hidden sizes to project to for the feature maps of the backbone.
* **fusion\_hidden\_size** (`int`, *optional*, defaults to 256) —
  The number of channels before fusion.
* **head\_in\_index** (`int`, *optional*, defaults to -1) —
  The index of the features to use in the heads.
* **use\_batch\_norm\_in\_fusion\_residual** (`bool`, *optional*, defaults to `False`) —
  Whether to use batch normalization in the pre-activate residual units of the fusion blocks.
* **use\_bias\_in\_fusion\_residual** (`bool`, *optional*, defaults to `True`) —
  Whether to use bias in the pre-activate residual units of the fusion blocks.
* **add\_projection** (`bool`, *optional*, defaults to `False`) —
  Whether to add a projection layer before the depth estimation head.
* **use\_auxiliary\_head** (`bool`, *optional*, defaults to `True`) —
  Whether to use an auxiliary head during training.
* **auxiliary\_loss\_weight** (`float`, *optional*, defaults to 0.4) —
  Weight of the cross-entropy loss of the auxiliary head.
* **semantic\_loss\_ignore\_index** (`int`, *optional*, defaults to 255) —
  The index that is ignored by the loss function of the semantic segmentation model.
* **semantic\_classifier\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the semantic classification head.
* **backbone\_featmap\_shape** (`list[int]`, *optional*, defaults to `[1, 1024, 24, 24]`) —
  Used only for the `hybrid` embedding type. The shape of the feature maps of the backbone.
* **neck\_ignore\_stages** (`list[int]`, *optional*, defaults to `[0, 1]`) —
  Used only for the `hybrid` embedding type. The stages of the readout layers to ignore.
* **backbone\_config** (`Union[dict[str, Any], PretrainedConfig]`, *optional*) —
  The configuration of the backbone model. Only used in case `is_hybrid` is `True` or in case you want to
  leverage the [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone) API.
* **backbone** (`str`, *optional*) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to use pretrained weights for the backbone.
* **use\_timm\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
  library.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **pooler\_output\_size** (`int`, *optional*) —
  Dimensionality of the pooler layer. If None, defaults to `hidden_size`.
* **pooler\_act** (`str`, *optional*, defaults to `"tanh"`) —
  The activation function to be used by the pooler. Keys of ACT2FN are supported for Flax and
  Pytorch, and elements of <https://www.tensorflow.org/api_docs/python/tf/keras/activations> are
  supported for Tensorflow.

This is the configuration class to store the configuration of a [DPTModel](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTModel). It is used to instantiate an DPT
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the DPT
[Intel/dpt-large](https://huggingface.co/Intel/dpt-large) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import DPTModel, DPTConfig

>>> # Initializing a DPT dpt-large style configuration
>>> configuration = DPTConfig()

>>> # Initializing a model from the dpt-large style configuration
>>> model = DPTModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/configuration_dpt.py#L282)

( )

Serializes this instance to a Python dictionary. Override the default [to\_dict()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.to_dict). Returns:
`dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,

## DPTFeatureExtractor

### class transformers.DPTFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/feature_extraction_dpt.py#L28)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/image_processing_dpt.py#L431)

( images segmentation\_maps = None \*\*kwargs  )

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/image_processing_dpt.py#L594)

( outputs target\_sizes: typing.Optional[list[tuple]] = None  ) → semantic\_segmentation

Parameters

* **outputs** ([DPTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForSemanticSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[Tuple]` of length `batch_size`, *optional*) —
  List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
  predictions will not be resized.

Returns

semantic\_segmentation

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [DPTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## DPTImageProcessor

### class transformers.DPTImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/image_processing_dpt.py#L109)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> keep\_aspect\_ratio: bool = False ensure\_multiple\_of: int = 1 do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: bool = False size\_divisor: typing.Optional[int] = None do\_reduce\_labels: bool = False \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions. Can be overridden by `do_resize` in `preprocess`.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 384, "width": 384}`):
  Size of the image after resizing. Can be overridden by `size` in `preprocess`.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) —
  Defines the resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
* **keep\_aspect\_ratio** (`bool`, *optional*, defaults to `False`) —
  If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved. Can
  be overridden by `keep_aspect_ratio` in `preprocess`.
* **ensure\_multiple\_of** (`int`, *optional*, defaults to 1) —
  If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overridden
  by `ensure_multiple_of` in `preprocess`.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  `preprocess`.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in `preprocess`.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_pad** (`bool`, *optional*, defaults to `False`) —
  Whether to apply center padding. This was introduced in the DINOv2 paper, which uses the model in
  combination with DPT.
* **size\_divisor** (`int`, *optional*) —
  If `do_pad` is `True`, pads the image dimensions to be divisible by this value. This was introduced in the
  DINOv2 paper, which uses the model in combination with DPT.
* **do\_reduce\_labels** (`bool`, *optional*, defaults to `False`) —
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is
  used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The
  background label will be replaced by 255. Can be overridden by the `do_reduce_labels` parameter in the
  `preprocess` method.

Constructs a DPT image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/image_processing_dpt.py#L436)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[int] = None keep\_aspect\_ratio: typing.Optional[bool] = None ensure\_multiple\_of: typing.Optional[int] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: typing.Optional[bool] = None size\_divisor: typing.Optional[int] = None do\_reduce\_labels: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  Segmentation map to preprocess.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after reszing. If `keep_aspect_ratio` is `True`, the image is resized to the largest
  possible size such that the aspect ratio is preserved. If `ensure_multiple_of` is set, the image is
  resized to a size that is a multiple of this value.
* **keep\_aspect\_ratio** (`bool`, *optional*, defaults to `self.keep_aspect_ratio`) —
  Whether to keep the aspect ratio of the image. If False, the image will be resized to (size, size). If
  True, the image will be resized to keep the aspect ratio and the size will be the maximum possible.
* **ensure\_multiple\_of** (`int`, *optional*, defaults to `self.ensure_multiple_of`) —
  Ensure that the image size is a multiple of this value.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
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
* **do\_reduce\_labels** (`bool`, *optional*, defaults to `self.do_reduce_labels`) —
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
  is used for background, and background itself is not included in all classes of a dataset (e.g.
  ADE20k). The background label will be replaced by 255.
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

## DPTImageProcessorFast

### class transformers.DPTImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/image_processing_dpt_fast.py#L129)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.dpt.image\_processing\_dpt\_fast.DPTFastImageProcessorKwargs]  )

Constructs a fast Dpt image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/image_processing_dpt_fast.py#L161)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.dpt.image\_processing\_dpt\_fast.DPTFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  The segmentation maps to preprocess.
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
* **ensure\_multiple\_of** (`int`, *optional*, defaults to 1) —
  If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overidden
  by `ensure_multiple_of` in `preprocess`.
* **size\_divisor** (`int`, *optional*) —
  If `do_pad` is `True`, pads the image dimensions to be divisible by this value. This was introduced in the
  DINOv2 paper, which uses the model in combination with DPT.
* **do\_pad** (`bool`, *optional*, defaults to `False`) —
  Whether to apply center padding. This was introduced in the DINOv2 paper, which uses the model in
  combination with DPT.
* **keep\_aspect\_ratio** (`bool`, *optional*, defaults to `False`) —
  If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved. Can
  be overidden by `keep_aspect_ratio` in `preprocess`.
* **do\_reduce\_labels** (`bool`, *optional*, defaults to `self.do_reduce_labels`) —
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
  is used for background, and background itself is not included in all classes of a dataset (e.g.
  ADE20k). The background label will be replaced by 255.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/image_processing_dpt_fast.py#L269)

( outputs target\_sizes: typing.Optional[list[tuple]] = None  ) → semantic\_segmentation

Parameters

* **outputs** ([DPTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForSemanticSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[Tuple]` of length `batch_size`, *optional*) —
  List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
  predictions will not be resized.

Returns

semantic\_segmentation

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [DPTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

#### post\_process\_depth\_estimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/image_processing_dpt_fast.py#L380)

( outputs: DepthEstimatorOutput target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple[int, int]], NoneType] = None  ) → `List[Dict[str, TensorType]]`

Parameters

* **outputs** (`DepthEstimatorOutput`) —
  Raw outputs of the model.
* **target\_sizes** (`TensorType` or `List[Tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
  (height, width) of each image in the batch. If left to None, predictions will not be resized.

Returns

`List[Dict[str, TensorType]]`

A list of dictionaries of tensors representing the processed depth
predictions.

Converts the raw output of `DepthEstimatorOutput` into final depth predictions and depth PIL images.
Only supports PyTorch.

## DPTModel

### class transformers.DPTModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/modeling_dpt.py#L776)

( config: DPTConfig add\_pooling\_layer: bool = True  )

Parameters

* **config** ([DPTConfig](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The bare Dpt Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/modeling_dpt.py#L812)

( pixel\_values: FloatTensor head\_mask: typing.Optional[torch.FloatTensor] = None output\_hidden\_states: typing.Optional[bool] = None \*\*kwargs  ) → `transformers.models.dpt.modeling_dpt.BaseModelOutputWithPoolingAndIntermediateActivations` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessor). See [DPTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTFeatureExtractor.__call__) for details (`processor_class` uses
  [DPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessor) for processing images).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.models.dpt.modeling_dpt.BaseModelOutputWithPoolingAndIntermediateActivations` or `tuple(torch.FloatTensor)`

A `transformers.models.dpt.modeling_dpt.BaseModelOutputWithPoolingAndIntermediateActivations` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPTConfig](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **intermediate\_activations** (`tuple(torch.FloatTensor)`, *optional*) — Intermediate activations that can be used to compute hidden states of the model at various layers.

The [DPTModel](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## DPTForDepthEstimation

### class transformers.DPTForDepthEstimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/modeling_dpt.py#L969)

( config  )

Parameters

* **config** ([DPTForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForDepthEstimation)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DPT Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/modeling_dpt.py#L988)

( pixel\_values: FloatTensor head\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_hidden\_states: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessor). See [DPTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTFeatureExtractor.__call__) for details (`processor_class` uses
  [DPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessor) for processing images).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Ground truth depth estimation maps for computing the loss.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

[transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPTConfig](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **predicted\_depth** (`torch.FloatTensor` of shape `(batch_size, height, width)`) — Predicted depth for each pixel.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DPTForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForDepthEstimation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, DPTForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
>>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # interpolate to original size
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     target_sizes=[(image.height, image.width)],
... )

>>> # visualize the prediction
>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = predicted_depth * 255 / predicted_depth.max()
>>> depth = depth.detach().cpu().numpy()
>>> depth = Image.fromarray(depth.astype("uint8"))
```

## DPTForSemanticSegmentation

### class transformers.DPTForSemanticSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/modeling_dpt.py#L1122)

( config: DPTConfig  )

Parameters

* **config** ([DPTConfig](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Dpt Model with a semantic segmentation head on top e.g. for ADE20K, CityScapes.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dpt/modeling_dpt.py#L1138)

( pixel\_values: typing.Optional[torch.FloatTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_hidden\_states: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessor). See [DPTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTFeatureExtractor.__call__) for details (`processor_class` uses
  [DPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessor) for processing images).
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

[transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPTConfig](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`) — Classification scores for each pixel.

  The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
  to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
  original image size as post-processing. You should always check your logits shape and resize as needed.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DPTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, DPTForSemanticSegmentation
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large-ade")
>>> model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dpt.md)
