*This model was released on 2019-05-28 and added to Hugging Face Transformers on 2023-02-20.*

# EfficientNet

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The EfficientNet model was proposed in [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://huggingface.co/papers/1905.11946)
by Mingxing Tan and Quoc V. Le. EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.

The abstract from the paper is the following:

*Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.
To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.*

This model was contributed by [adirik](https://huggingface.co/adirik).
The original code can be found [here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## EfficientNetConfig

### class transformers.EfficientNetConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientnet/configuration_efficientnet.py#L30)

( num\_channels: int = 3 image\_size: int = 600 width\_coefficient: float = 2.0 depth\_coefficient: float = 3.1 depth\_divisor: int = 8 kernel\_sizes: list = [3, 3, 5, 3, 5, 5, 3] in\_channels: list = [32, 16, 24, 40, 80, 112, 192] out\_channels: list = [16, 24, 40, 80, 112, 192, 320] depthwise\_padding: list = [] strides: list = [1, 2, 2, 2, 1, 2, 1] num\_block\_repeats: list = [1, 2, 2, 3, 3, 4, 1] expand\_ratios: list = [1, 6, 6, 6, 6, 6, 6] squeeze\_expansion\_ratio: float = 0.25 hidden\_act: str = 'swish' hidden\_dim: int = 2560 pooling\_type: str = 'mean' initializer\_range: float = 0.02 batch\_norm\_eps: float = 0.001 batch\_norm\_momentum: float = 0.99 dropout\_rate: float = 0.5 drop\_connect\_rate: float = 0.2 \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **image\_size** (`int`, *optional*, defaults to 600) —
  The input image size.
* **width\_coefficient** (`float`, *optional*, defaults to 2.0) —
  Scaling coefficient for network width at each stage.
* **depth\_coefficient** (`float`, *optional*, defaults to 3.1) —
  Scaling coefficient for network depth at each stage.
* **depth\_divisor** `int`, *optional*, defaults to 8) —
  A unit of network width.
* **kernel\_sizes** (`list[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`) —
  List of kernel sizes to be used in each block.
* **in\_channels** (`list[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`) —
  List of input channel sizes to be used in each block for convolutional layers.
* **out\_channels** (`list[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`) —
  List of output channel sizes to be used in each block for convolutional layers.
* **depthwise\_padding** (`list[int]`, *optional*, defaults to `[]`) —
  List of block indices with square padding.
* **strides** (`list[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`) —
  List of stride sizes to be used in each block for convolutional layers.
* **num\_block\_repeats** (`list[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`) —
  List of the number of times each block is to repeated.
* **expand\_ratios** (`list[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`) —
  List of scaling coefficient of each block.
* **squeeze\_expansion\_ratio** (`float`, *optional*, defaults to 0.25) —
  Squeeze expansion ratio.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
  `"selu",` “gelu\_new”`,` “silu”`and`“mish”` are supported.
* **hidden\_dim** (`int`, *optional*, defaults to 1280) —
  The hidden dimension of the layer before the classification head.
* **pooling\_type** (`str` or `function`, *optional*, defaults to `"mean"`) —
  Type of final pooling to be applied before the dense classification head. Available options are [`"mean"`,
  `"max"`]
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **batch\_norm\_eps** (`float`, *optional*, defaults to 1e-3) —
  The epsilon used by the batch normalization layers.
* **batch\_norm\_momentum** (`float`, *optional*, defaults to 0.99) —
  The momentum used by the batch normalization layers.
* **dropout\_rate** (`float`, *optional*, defaults to 0.5) —
  The dropout rate to be applied before final classifier layer.
* **drop\_connect\_rate** (`float`, *optional*, defaults to 0.2) —
  The drop rate for skip connections.

This is the configuration class to store the configuration of a [EfficientNetModel](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetModel). It is used to instantiate an
EfficientNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the EfficientNet
[google/efficientnet-b7](https://huggingface.co/google/efficientnet-b7) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import EfficientNetConfig, EfficientNetModel

>>> # Initializing a EfficientNet efficientnet-b7 style configuration
>>> configuration = EfficientNetConfig()

>>> # Initializing a model (with random weights) from the efficientnet-b7 style configuration
>>> model = EfficientNetModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## EfficientNetImageProcessor

### class transformers.EfficientNetImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientnet/image_processing_efficientnet.py#L46)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = 0 do\_center\_crop: bool = False crop\_size: typing.Optional[dict[str, int]] = None rescale\_factor: typing.Union[int, float] = 0.00392156862745098 rescale\_offset: bool = False do\_rescale: bool = True do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None include\_top: bool = True \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by
  `do_resize` in `preprocess`.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 346, "width": 346}`):
  Size of the image after `resize`. Can be overridden by `size` in `preprocess`.
* **resample** (`PILImageResampling` filter, *optional*, defaults to 0) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
* **do\_center\_crop** (`bool`, *optional*, defaults to `False`) —
  Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image
  is padded with 0’s and then center cropped. Can be overridden by `do_center_crop` in `preprocess`.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 289, "width": 289}`):
  Desired output size when applying center-cropping. Can be overridden by `crop_size` in `preprocess`.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
  `preprocess` method.
* **rescale\_offset** (`bool`, *optional*, defaults to `False`) —
  Whether to rescale the image between [-scale\_range, scale\_range] instead of [0, scale\_range]. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
* **include\_top** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image again. Should be set to True if the inputs are used for image classification.

Constructs a EfficientNet image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientnet/image_processing_efficientnet.py#L211)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample = None do\_center\_crop: typing.Optional[bool] = None crop\_size: typing.Optional[dict[str, int]] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None rescale\_offset: typing.Optional[bool] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None include\_top: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after `resize`.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  PILImageResampling filter to use if resizing the image Only has an effect if `do_resize` is set to
  `True`.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) —
  Size of the image after center crop. If one edge the image is smaller than `crop_size`, it will be
  padded with zeros and then cropped
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **rescale\_offset** (`bool`, *optional*, defaults to `self.rescale_offset`) —
  Whether to rescale the image between [-scale\_range, scale\_range] instead of [0, scale\_range].
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation.
* **include\_top** (`bool`, *optional*, defaults to `self.include_top`) —
  Rescales the image again for image classification if set to True.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + `None`: Return a list of `np.ndarray`.
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

## EfficientNetImageProcessorFast

### class transformers.EfficientNetImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientnet/image_processing_efficientnet_fast.py#L57)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.efficientnet.image\_processing\_efficientnet\_fast.EfficientNetFastImageProcessorKwargs]  )

Constructs a fast Efficientnet image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientnet/image_processing_efficientnet_fast.py#L207)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.efficientnet.image\_processing\_efficientnet\_fast.EfficientNetFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **rescale\_offset** (`bool`, *optional*, defaults to `self.rescale_offset`) —
  Whether to rescale the image between [-max\_range/2, scale\_range/2] instead of [0, scale\_range].
* **include\_top** (`bool`, *optional*, defaults to `self.include_top`) —
  Normalize the image again with the standard deviation only for image classification if set to True.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## EfficientNetModel

### class transformers.EfficientNetModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientnet/modeling_efficientnet.py#L451)

( config: EfficientNetConfig  )

Parameters

* **config** ([EfficientNetConfig](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Efficientnet Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientnet/modeling_efficientnet.py#L469)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor). See [EfficientNetImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientNetConfig](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state after a pooling operation on the spatial dimensions.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [EfficientNetModel](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## EfficientNetForImageClassification

### class transformers.EfficientNetForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientnet/modeling_efficientnet.py#L513)

( config  )

Parameters

* **config** ([EfficientNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetForImageClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g.
for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientnet/modeling_efficientnet.py#L526)

( pixel\_values: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor). See [EfficientNetImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientNetConfig](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.

The [EfficientNetForImageClassification](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/efficientnet.md)
