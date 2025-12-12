# DPT

## Overview

The DPT model was proposed in [Vision Transformers for Dense Prediction](https://huggingface.co/papers/2103.13413) by René Ranftl, Alexey Bochkovskiy, Vladlen Koltun.
DPT is a model that leverages the [Vision Transformer (ViT)](vit) as backbone for dense prediction tasks like semantic segmentation and depth estimation.

The abstract from the paper is the following:

*We introduce dense vision transformers, an architecture that leverages vision transformers in place of convolutional networks as a backbone for dense prediction tasks. We assemble tokens from various stages of the vision transformer into image-like representations at various resolutions and progressively combine them into full-resolution predictions using a convolutional decoder. The transformer backbone processes representations at a constant and relatively high resolution and has a global receptive field at every stage. These properties allow the dense vision transformer to provide finer-grained and more globally coherent predictions when compared to fully-convolutional networks. Our experiments show that this architecture yields substantial improvements on dense prediction tasks, especially when a large amount of training data is available. For monocular depth estimation, we observe an improvement of up to 28% in relative performance when compared to a state-of-the-art fully-convolutional network. When applied to semantic segmentation, dense vision transformers set a new state of the art on ADE20K with 49.02% mIoU. We further show that the architecture can be fine-tuned on smaller datasets such as NYUv2, KITTI, and Pascal Context where it also sets the new state of the art.*

 DPT architecture. Taken from the original paper. 

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/isl-org/DPT).

## Usage tips

DPT is compatible with the [AutoBackbone](/docs/transformers/main/en/main_classes/backbones#transformers.AutoBackbone) class. This allows to use the DPT framework with various computer vision backbones available in the library, such as `VitDetBackbone` or `Dinov2Backbone`. One can create it as follows:

```python
from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation

# initialize with a Transformer-based backbone such as DINOv2
# in that case, we also specify `reshape_hidden_states=False` to get feature maps of shape (batch_size, num_channels, height, width)
backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage1", "stage2", "stage3", "stage4"], reshape_hidden_states=False)

config = DPTConfig(backbone_config=backbone_config)
model = DPTForDepthEstimation(config=config)
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DPT.

- Demo notebooks for [DPTForDepthEstimation](/docs/transformers/main/en/model_doc/dpt#transformers.DPTForDepthEstimation) can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DPT).

- [Semantic segmentation task guide](../tasks/semantic_segmentation)
- [Monocular depth estimation task guide](../tasks/monocular_depth_estimation)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DPTConfig[[transformers.DPTConfig]]

#### transformers.DPTConfig[[transformers.DPTConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/configuration_dpt.py#L27)

This is the configuration class to store the configuration of a [DPTModel](/docs/transformers/main/en/model_doc/dpt#transformers.DPTModel). It is used to instantiate an DPT
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the DPT
[Intel/dpt-large](https://huggingface.co/Intel/dpt-large) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import DPTModel, DPTConfig

>>> # Initializing a DPT dpt-large style configuration
>>> configuration = DPTConfig()

>>> # Initializing a model from the dpt-large style configuration
>>> model = DPTModel(configuration)

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

image_size (`int`, *optional*, defaults to 384) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 16) : The size (resolution) of each patch.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

is_hybrid (`bool`, *optional*, defaults to `False`) : Whether to use a hybrid backbone. Useful in the context of loading DPT-Hybrid models.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries, keys and values.

backbone_out_indices (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`) : Indices of the intermediate hidden states to use from backbone.

readout_type (`str`, *optional*, defaults to `"project"`) : The readout type to use when processing the readout token (CLS token) of the intermediate hidden states of the ViT backbone. Can be one of [`"ignore"`, `"add"`, `"project"`].  - "ignore" simply ignores the CLS token. - "add" passes the information from the CLS token to all other tokens by adding the representations. - "project" passes information to the other tokens by concatenating the readout to all other tokens before projecting the representation to the original feature dimension D using a linear layer followed by a GELU non-linearity.

reassemble_factors (`list[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`) : The up/downsampling factors of the reassemble layers.

neck_hidden_sizes (`list[str]`, *optional*, defaults to `[96, 192, 384, 768]`) : The hidden sizes to project to for the feature maps of the backbone.

fusion_hidden_size (`int`, *optional*, defaults to 256) : The number of channels before fusion.

head_in_index (`int`, *optional*, defaults to -1) : The index of the features to use in the heads.

use_batch_norm_in_fusion_residual (`bool`, *optional*, defaults to `False`) : Whether to use batch normalization in the pre-activate residual units of the fusion blocks.

use_bias_in_fusion_residual (`bool`, *optional*, defaults to `True`) : Whether to use bias in the pre-activate residual units of the fusion blocks.

add_projection (`bool`, *optional*, defaults to `False`) : Whether to add a projection layer before the depth estimation head.

use_auxiliary_head (`bool`, *optional*, defaults to `True`) : Whether to use an auxiliary head during training.

auxiliary_loss_weight (`float`, *optional*, defaults to 0.4) : Weight of the cross-entropy loss of the auxiliary head.

semantic_loss_ignore_index (`int`, *optional*, defaults to 255) : The index that is ignored by the loss function of the semantic segmentation model.

semantic_classifier_dropout (`float`, *optional*, defaults to 0.1) : The dropout ratio for the semantic classification head.

backbone_featmap_shape (`list[int]`, *optional*, defaults to `[1, 1024, 24, 24]`) : Used only for the `hybrid` embedding type. The shape of the feature maps of the backbone.

neck_ignore_stages (`list[int]`, *optional*, defaults to `[0, 1]`) : Used only for the `hybrid` embedding type. The stages of the readout layers to ignore.

backbone_config (`Union[dict[str, Any], PreTrainedConfig]`, *optional*) : The configuration of the backbone model. Only used in case `is_hybrid` is `True` or in case you want to leverage the [AutoBackbone](/docs/transformers/main/en/main_classes/backbones#transformers.AutoBackbone) API.

backbone (`str`, *optional*) : Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone` is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.

use_pretrained_backbone (`bool`, *optional*, defaults to `False`) : Whether to use pretrained weights for the backbone.

use_timm_backbone (`bool`, *optional*, defaults to `False`) : Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers library.

backbone_kwargs (`dict`, *optional*) : Keyword arguments to be passed to AutoBackbone when loading from a checkpoint e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.

pooler_output_size (`int`, *optional*) : Dimensionality of the pooler layer. If None, defaults to `hidden_size`.

pooler_act (`str`, *optional*, defaults to `"tanh"`) : The activation function to be used by the pooler.

## DPTImageProcessor[[transformers.DPTImageProcessor]]

#### transformers.DPTImageProcessor[[transformers.DPTImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/image_processing_dpt.py#L130)

Constructs a DPT image processor.

preprocesstransformers.DPTImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/image_processing_dpt.py#L458[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "segmentation_maps", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[int] = None"}, {"name": "keep_aspect_ratio", "val": ": typing.Optional[bool] = None"}, {"name": "ensure_multiple_of", "val": ": typing.Optional[int] = None"}, {"name": "resample", "val": ": typing.Optional[PIL.Image.Resampling] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "do_pad", "val": ": typing.Optional[bool] = None"}, {"name": "size_divisor", "val": ": typing.Optional[int] = None"}, {"name": "do_reduce_labels", "val": ": typing.Optional[bool] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": ChannelDimension = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}]- **images** (`ImageInput`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **segmentation_maps** (`ImageInput`, *optional*) --
  Segmentation map to preprocess.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the image.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  Size of the image after reszing. If `keep_aspect_ratio` is `True`, the image is resized to the largest
  possible size such that the aspect ratio is preserved. If `ensure_multiple_of` is set, the image is
  resized to a size that is a multiple of this value.
- **keep_aspect_ratio** (`bool`, *optional*, defaults to `self.keep_aspect_ratio`) --
  Whether to keep the aspect ratio of the image. If False, the image will be resized to (size, size). If
  True, the image will be resized to keep the aspect ratio and the size will be the maximum possible.
- **ensure_multiple_of** (`int`, *optional*, defaults to `self.ensure_multiple_of`) --
  Ensure that the image size is a multiple of this value.
- **resample** (`int`, *optional*, defaults to `self.resample`) --
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
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
- **do_reduce_labels** (`bool`, *optional*, defaults to `self.do_reduce_labels`) --
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
  is used for background, and background itself is not included in all classes of a dataset (e.g.
  ADE20k). The background label will be replaced by 255.
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

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions. Can be overridden by `do_resize` in `preprocess`.

size (`dict[str, int]` *optional*, defaults to `{"height" : 384, "width": 384}`): Size of the image after resizing. Can be overridden by `size` in `preprocess`.

resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) : Defines the resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.

keep_aspect_ratio (`bool`, *optional*, defaults to `False`) : If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved. Can be overridden by `keep_aspect_ratio` in `preprocess`.

ensure_multiple_of (`int`, *optional*, defaults to 1) : If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overridden by `ensure_multiple_of` in `preprocess`.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in `preprocess`.

rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in `preprocess`.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method.

image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) : Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

do_pad (`bool`, *optional*, defaults to `False`) : Whether to apply center padding. This was introduced in the DINOv2 paper, which uses the model in combination with DPT.

size_divisor (`int`, *optional*) : If `do_pad` is `True`, pads the image dimensions to be divisible by this value. This was introduced in the DINOv2 paper, which uses the model in combination with DPT.

do_reduce_labels (`bool`, *optional*, defaults to `False`) : Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255. Can be overridden by the `do_reduce_labels` parameter in the `preprocess` method.

## DPTImageProcessorFast[[transformers.DPTImageProcessorFast]]

#### transformers.DPTImageProcessorFast[[transformers.DPTImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/image_processing_dpt_fast.py#L91)

Constructs a fast Dpt image processor.

preprocesstransformers.DPTImageProcessorFast.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/image_processing_dpt_fast.py#L123[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "segmentation_maps", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.dpt.image_processing_dpt.DPTImageProcessorKwargs]"}]- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **segmentation_maps** (`ImageInput`, *optional*) --
  The segmentation maps to preprocess.
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
- **ensure_multiple_of** (`int`, *optional*, defaults to 1) --
  If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overridden
  by `ensure_multiple_of` in `preprocess`.
- **size_divisor** (`.size_divisor`) --
  The size by which to make sure both the height and width can be divided.
- **keep_aspect_ratio** (`bool`, *optional*, defaults to `False`) --
  If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved. Can
  be overridden by `keep_aspect_ratio` in `preprocess`.
- **do_reduce_labels** (`bool`, *optional*, defaults to `self.do_reduce_labels`) --
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
  is used for background, and background itself is not included in all classes of a dataset (e.g.
  ADE20k). The background label will be replaced by 255.0``- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

segmentation_maps (`ImageInput`, *optional*) : The segmentation maps to preprocess.

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

ensure_multiple_of (`int`, *optional*, defaults to 1) : If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overridden by `ensure_multiple_of` in `preprocess`.

size_divisor (`.size_divisor`) : The size by which to make sure both the height and width can be divided.

keep_aspect_ratio (`bool`, *optional*, defaults to `False`) : If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved. Can be overridden by `keep_aspect_ratio` in `preprocess`.

do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`) : Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.
#### post_process_semantic_segmentation[[transformers.DPTImageProcessorFast.post_process_semantic_segmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/image_processing_dpt_fast.py#L231)

Converts the output of [DPTForSemanticSegmentation](/docs/transformers/main/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) into semantic segmentation maps.

**Parameters:**

outputs ([DPTForSemanticSegmentation](/docs/transformers/main/en/model_doc/dpt#transformers.DPTForSemanticSegmentation)) : Raw outputs of the model.

target_sizes (`list[Tuple]` of length `batch_size`, *optional*) : List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

**Returns:**

`semantic_segmentation`

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
#### post_process_depth_estimation[[transformers.DPTImageProcessorFast.post_process_depth_estimation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/image_processing_dpt_fast.py#L341)

Converts the raw output of `DepthEstimatorOutput` into final depth predictions and depth PIL images.
Only supports PyTorch.

**Parameters:**

outputs (`DepthEstimatorOutput`) : Raw outputs of the model.

target_sizes (`TensorType` or `List[Tuple[int, int]]`, *optional*) : Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the batch. If left to None, predictions will not be resized.

**Returns:**

``List[Dict[str, TensorType]]``

A list of dictionaries of tensors representing the processed depth
predictions.

## DPTModel[[transformers.DPTModel]]

#### transformers.DPTModel[[transformers.DPTModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/modeling_dpt.py#L746)

The bare Dpt Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.DPTModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/modeling_dpt.py#L774[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DPTImageProcessor](/docs/transformers/main/en/model_doc/dpt#transformers.DPTImageProcessor). See `DPTImageProcessor.__call__()` for details (`processor_class` uses
  [DPTImageProcessor](/docs/transformers/main/en/model_doc/dpt#transformers.DPTImageProcessor) for processing images).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.0`transformers.models.dpt.modeling_dpt.BaseModelOutputWithPoolingAndIntermediateActivations` or `tuple(torch.FloatTensor)`A `transformers.models.dpt.modeling_dpt.BaseModelOutputWithPoolingAndIntermediateActivations` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPTConfig](/docs/transformers/main/en/model_doc/dpt#transformers.DPTConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **intermediate_activations** (`tuple(torch.FloatTensor)`, *optional*) -- Intermediate activations that can be used to compute hidden states of the model at various layers.
The [DPTModel](/docs/transformers/main/en/model_doc/dpt#transformers.DPTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
```

**Parameters:**

config ([DPTConfig](/docs/transformers/main/en/model_doc/dpt#transformers.DPTConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

add_pooling_layer (`bool`, *optional*, defaults to `True`) : Whether to add a pooling layer

**Returns:**

``transformers.models.dpt.modeling_dpt.BaseModelOutputWithPoolingAndIntermediateActivations` or `tuple(torch.FloatTensor)``

A `transformers.models.dpt.modeling_dpt.BaseModelOutputWithPoolingAndIntermediateActivations` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPTConfig](/docs/transformers/main/en/model_doc/dpt#transformers.DPTConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **intermediate_activations** (`tuple(torch.FloatTensor)`, *optional*) -- Intermediate activations that can be used to compute hidden states of the model at various layers.

## DPTForDepthEstimation[[transformers.DPTForDepthEstimation]]

#### transformers.DPTForDepthEstimation[[transformers.DPTForDepthEstimation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/modeling_dpt.py#L923)

DPT Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.DPTForDepthEstimation.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/modeling_dpt.py#L942[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DPTImageProcessor](/docs/transformers/main/en/model_doc/dpt#transformers.DPTImageProcessor). See `DPTImageProcessor.__call__()` for details (`processor_class` uses
  [DPTImageProcessor](/docs/transformers/main/en/model_doc/dpt#transformers.DPTImageProcessor) for processing images).
- **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) --
  Ground truth depth estimation maps for computing the loss.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.0[transformers.modeling_outputs.DepthEstimatorOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.DepthEstimatorOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPTConfig](/docs/transformers/main/en/model_doc/dpt#transformers.DPTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **predicted_depth** (`torch.FloatTensor` of shape `(batch_size, height, width)`) -- Predicted depth for each pixel.

- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [DPTForDepthEstimation](/docs/transformers/main/en/model_doc/dpt#transformers.DPTForDepthEstimation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:
```python
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

**Parameters:**

config ([DPTForDepthEstimation](/docs/transformers/main/en/model_doc/dpt#transformers.DPTForDepthEstimation)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.DepthEstimatorOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.DepthEstimatorOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPTConfig](/docs/transformers/main/en/model_doc/dpt#transformers.DPTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **predicted_depth** (`torch.FloatTensor` of shape `(batch_size, height, width)`) -- Predicted depth for each pixel.

- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## DPTForSemanticSegmentation[[transformers.DPTForSemanticSegmentation]]

#### transformers.DPTForSemanticSegmentation[[transformers.DPTForSemanticSegmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/modeling_dpt.py#L1075)

The Dpt Model with a semantic segmentation head on top e.g. for ADE20K, CityScapes.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.DPTForSemanticSegmentation.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/modeling_dpt.py#L1091[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DPTImageProcessor](/docs/transformers/main/en/model_doc/dpt#transformers.DPTImageProcessor). See `DPTImageProcessor.__call__()` for details (`processor_class` uses
  [DPTImageProcessor](/docs/transformers/main/en/model_doc/dpt#transformers.DPTImageProcessor) for processing images).
- **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) --
  Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.0[transformers.modeling_outputs.SemanticSegmenterOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.SemanticSegmenterOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPTConfig](/docs/transformers/main/en/model_doc/dpt#transformers.DPTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`) -- Classification scores for each pixel.

  

  The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
  to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
  original image size as post-processing. You should always check your logits shape and resize as needed.

  

- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [DPTForSemanticSegmentation](/docs/transformers/main/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:
```python
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

**Parameters:**

config ([DPTConfig](/docs/transformers/main/en/model_doc/dpt#transformers.DPTConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.SemanticSegmenterOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.SemanticSegmenterOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DPTConfig](/docs/transformers/main/en/model_doc/dpt#transformers.DPTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`) -- Classification scores for each pixel.

  

  The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
  to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
  original image size as post-processing. You should always check your logits shape and resize as needed.

  

- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
