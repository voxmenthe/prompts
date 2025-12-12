# EfficientNet

## Overview

The EfficientNet model was proposed in [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://huggingface.co/papers/1905.11946)
by Mingxing Tan and Quoc V. Le. EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.

The abstract from the paper is the following:

*Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.
To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.*

This model was contributed by [adirik](https://huggingface.co/adirik).
The original code can be found [here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## EfficientNetConfig[[transformers.EfficientNetConfig]]

#### transformers.EfficientNetConfig[[transformers.EfficientNetConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/efficientnet/configuration_efficientnet.py#L24)

This is the configuration class to store the configuration of a [EfficientNetModel](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetModel). It is used to instantiate an
EfficientNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the EfficientNet
[google/efficientnet-b7](https://huggingface.co/google/efficientnet-b7) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:
```python
>>> from transformers import EfficientNetConfig, EfficientNetModel

>>> # Initializing a EfficientNet efficientnet-b7 style configuration
>>> configuration = EfficientNetConfig()

>>> # Initializing a model (with random weights) from the efficientnet-b7 style configuration
>>> model = EfficientNetModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

image_size (`int`, *optional*, defaults to 600) : The input image size.

width_coefficient (`float`, *optional*, defaults to 2.0) : Scaling coefficient for network width at each stage.

depth_coefficient (`float`, *optional*, defaults to 3.1) : Scaling coefficient for network depth at each stage.

depth_divisor `int`, *optional*, defaults to 8) : A unit of network width.

kernel_sizes (`list[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`) : List of kernel sizes to be used in each block.

in_channels (`list[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`) : List of input channel sizes to be used in each block for convolutional layers.

out_channels (`list[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`) : List of output channel sizes to be used in each block for convolutional layers.

depthwise_padding (`list[int]`, *optional*, defaults to `[]`) : List of block indices with square padding.

strides (`list[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`) : List of stride sizes to be used in each block for convolutional layers.

num_block_repeats (`list[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`) : List of the number of times each block is to repeated.

expand_ratios (`list[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`) : List of scaling coefficient of each block.

squeeze_expansion_ratio (`float`, *optional*, defaults to 0.25) : Squeeze expansion ratio.

hidden_act (`str` or `function`, *optional*, defaults to `"silu"`) : The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`, `"selu", `"gelu_new"`, `"silu"` and `"mish"` are supported.

hidden_dim (`int`, *optional*, defaults to 1280) : The hidden dimension of the layer before the classification head.

pooling_type (`str` or `function`, *optional*, defaults to `"mean"`) : Type of final pooling to be applied before the dense classification head. Available options are [`"mean"`, `"max"`]

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

batch_norm_eps (`float`, *optional*, defaults to 1e-3) : The epsilon used by the batch normalization layers.

batch_norm_momentum (`float`, *optional*, defaults to 0.99) : The momentum used by the batch normalization layers.

dropout_rate (`float`, *optional*, defaults to 0.5) : The dropout rate to be applied before final classifier layer.

drop_connect_rate (`float`, *optional*, defaults to 0.2) : The drop rate for skip connections.

## EfficientNetImageProcessor[[transformers.EfficientNetImageProcessor]]

#### transformers.EfficientNetImageProcessor[[transformers.EfficientNetImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/efficientnet/image_processing_efficientnet.py#L59)

Constructs a EfficientNet image processor.

preprocesstransformers.EfficientNetImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/efficientnet/image_processing_efficientnet.py#L225[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "resample", "val": " = None"}, {"name": "do_center_crop", "val": ": typing.Optional[bool] = None"}, {"name": "crop_size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "rescale_offset", "val": ": typing.Optional[bool] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "include_top", "val": ": typing.Optional[bool] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": ChannelDimension = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}]- **images** (`ImageInput`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the image.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  Size of the image after `resize`.
- **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) --
  PILImageResampling filter to use if resizing the image Only has an effect if `do_resize` is set to
  `True`.
- **do_center_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) --
  Whether to center crop the image.
- **crop_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) --
  Size of the image after center crop. If one edge the image is smaller than `crop_size`, it will be
  padded with zeros and then cropped
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the image values between [0 - 1].
- **rescale_factor** (`float`, *optional*, defaults to `self.rescale_factor`) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **rescale_offset** (`bool`, *optional*, defaults to `self.rescale_offset`) --
  Whether to rescale the image between [-scale_range, scale_range] instead of [0, scale_range].
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether to normalize the image.
- **image_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) --
  Image mean.
- **image_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) --
  Image standard deviation.
- **include_top** (`bool`, *optional*, defaults to `self.include_top`) --
  Rescales the image again for image classification if set to True.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  The type of tensors to return. Can be one of:
  - `None`: Return a list of `np.ndarray`.
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

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by `do_resize` in `preprocess`.

size (`dict[str, int]` *optional*, defaults to `{"height" : 346, "width": 346}`): Size of the image after `resize`. Can be overridden by `size` in `preprocess`.

resample (`PILImageResampling` filter, *optional*, defaults to 0) : Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.

do_center_crop (`bool`, *optional*, defaults to `False`) : Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image is padded with 0's and then center cropped. Can be overridden by `do_center_crop` in `preprocess`.

crop_size (`dict[str, int]`, *optional*, defaults to `{"height" : 289, "width": 289}`): Desired output size when applying center-cropping. Can be overridden by `crop_size` in `preprocess`.

rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.

rescale_offset (`bool`, *optional*, defaults to `False`) : Whether to rescale the image between [-scale_range, scale_range] instead of [0, scale_range]. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method.

image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) : Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

include_top (`bool`, *optional*, defaults to `True`) : Whether to rescale the image again. Should be set to True if the inputs are used for image classification.

## EfficientNetImageProcessorFast[[transformers.EfficientNetImageProcessorFast]]

#### transformers.EfficientNetImageProcessorFast[[transformers.EfficientNetImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/efficientnet/image_processing_efficientnet_fast.py#L35)

Constructs a fast Efficientnet image processor.

preprocesstransformers.EfficientNetImageProcessorFast.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/efficientnet/image_processing_efficientnet_fast.py#L185[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.efficientnet.image_processing_efficientnet.EfficientNetImageProcessorKwargs]"}]- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) --
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
- **rescale_offset** (`bool`, *optional*, defaults to `self.rescale_offset`) --
  Whether to rescale the image between [-max_range/2, scale_range/2] instead of [0, scale_range].
- **include_top** (`bool`, *optional*, defaults to `self.include_top`) --
  Normalize the image again with the standard deviation only for image classification if set to True.0``- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
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

rescale_offset (`bool`, *optional*, defaults to `self.rescale_offset`) : Whether to rescale the image between [-max_range/2, scale_range/2] instead of [0, scale_range].

include_top (`bool`, *optional*, defaults to `self.include_top`) : Normalize the image again with the standard deviation only for image classification if set to True.

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## EfficientNetModel[[transformers.EfficientNetModel]]

#### transformers.EfficientNetModel[[transformers.EfficientNetModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/efficientnet/modeling_efficientnet.py#L450)

The bare Efficientnet Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.EfficientNetModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/efficientnet/modeling_efficientnet.py#L468[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor). See [EfficientNetImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) for processing images).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientNetConfig](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state after a pooling operation on the spatial dimensions.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
The [EfficientNetModel](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
```

**Parameters:**

config ([EfficientNetConfig](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)``

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientNetConfig](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state after a pooling operation on the spatial dimensions.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

## EfficientNetForImageClassification[[transformers.EfficientNetForImageClassification]]

#### transformers.EfficientNetForImageClassification[[transformers.EfficientNetForImageClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/efficientnet/modeling_efficientnet.py#L513)

EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g.
for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.EfficientNetForImageClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/efficientnet/modeling_efficientnet.py#L526[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor). See [EfficientNetImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) for processing images).
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientNetConfig](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
The [EfficientNetForImageClassification](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, EfficientNetForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
>>> model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b7")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

**Parameters:**

config ([EfficientNetForImageClassification](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetForImageClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientNetConfig](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
