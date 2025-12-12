*This model was released on 2021-10-05 and added to Hugging Face Transformers on 2022-06-29.*

# MobileViT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

[MobileViT](https://huggingface.co/papers/2110.02178) is a lightweight vision transformer for mobile devices that merges CNNs’s efficiency and inductive biases with transformers global context modeling. It treats transformers as convolutions, enabling global information processing without the heavy computational cost of standard ViTs.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/MobileViT.png)

You can find all the original MobileViT checkpoints under the [Apple](https://huggingface.co/apple/models?search=mobilevit) organization.

* This model was contributed by [matthijs](https://huggingface.co/Matthijs).

Click on the MobileViT models in the right sidebar for more examples of how to apply MobileViT to different vision tasks.

The example below demonstrates how to do [Image Classification] with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) and the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

classifier = pipeline(
   task="image-classification",
   model="apple/mobilevit-small",
   dtype=torch.float16, device=0,
)

preds = classifier("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
print(f"Prediction: {preds}\n")
```

## Notes

* Does **not** operate on sequential data, it’s purely designed for image tasks.
* Feature maps are used directly instead of token embeddings.
* Use [MobileViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) to preprocess images.
* If using custom preprocessing, ensure that images are in **BGR** format (not RGB), as expected by the pretrained weights.
* The classification models are pretrained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k).
* The segmentation models use a [DeepLabV3](https://huggingface.co/papers/1706.05587) head and are pretrained on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

## MobileViTConfig

### class transformers.MobileViTConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/configuration_mobilevit.py#L30)

( num\_channels = 3 image\_size = 256 patch\_size = 2 hidden\_sizes = [144, 192, 240] neck\_hidden\_sizes = [16, 32, 64, 96, 128, 160, 640] num\_attention\_heads = 4 mlp\_ratio = 2.0 expand\_ratio = 4.0 hidden\_act = 'silu' conv\_kernel\_size = 3 output\_stride = 32 hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.0 classifier\_dropout\_prob = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 qkv\_bias = True aspp\_out\_channels = 256 atrous\_rates = [6, 12, 18] aspp\_dropout\_prob = 0.1 semantic\_loss\_ignore\_index = 255 \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **image\_size** (`int`, *optional*, defaults to 256) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 2) —
  The size (resolution) of each patch.
* **hidden\_sizes** (`list[int]`, *optional*, defaults to `[144, 192, 240]`) —
  Dimensionality (hidden size) of the Transformer encoders at each stage.
* **neck\_hidden\_sizes** (`list[int]`, *optional*, defaults to `[16, 32, 64, 96, 128, 160, 640]`) —
  The number of channels for the feature maps of the backbone.
* **num\_attention\_heads** (`int`, *optional*, defaults to 4) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **mlp\_ratio** (`float`, *optional*, defaults to 2.0) —
  The ratio of the number of channels in the output of the MLP to the number of channels in the input.
* **expand\_ratio** (`float`, *optional*, defaults to 4.0) —
  Expansion factor for the MobileNetv2 layers.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
* **conv\_kernel\_size** (`int`, *optional*, defaults to 3) —
  The size of the convolutional kernel in the MobileViT layer.
* **output\_stride** (`int`, *optional*, defaults to 32) —
  The ratio of the spatial resolution of the output to the resolution of the input image.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the Transformer encoder.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **classifier\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for attached classifiers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.
* **aspp\_out\_channels** (`int`, *optional*, defaults to 256) —
  Number of output channels used in the ASPP layer for semantic segmentation.
* **atrous\_rates** (`list[int]`, *optional*, defaults to `[6, 12, 18]`) —
  Dilation (atrous) factors used in the ASPP layer for semantic segmentation.
* **aspp\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the ASPP layer for semantic segmentation.
* **semantic\_loss\_ignore\_index** (`int`, *optional*, defaults to 255) —
  The index that is ignored by the loss function of the semantic segmentation model.

This is the configuration class to store the configuration of a [MobileViTModel](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTModel). It is used to instantiate a
MobileViT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the MobileViT
[apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MobileViTConfig, MobileViTModel

>>> # Initializing a mobilevit-small style configuration
>>> configuration = MobileViTConfig()

>>> # Initializing a model from the mobilevit-small style configuration
>>> model = MobileViTModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MobileViTFeatureExtractor

### class transformers.MobileViTFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/feature_extraction_mobilevit.py#L28)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/image_processing_mobilevit.py#L202)

( images segmentation\_maps = None \*\*kwargs  )

Preprocesses a batch of images and optionally segmentation maps.

Overrides the `__call__` method of the `Preprocessor` class so that both images and segmentation maps can be
passed in as positional arguments.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/image_processing_mobilevit.py#L474)

( outputs target\_sizes: typing.Optional[list[tuple]] = None  ) → semantic\_segmentation

Parameters

* **outputs** ([MobileViTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[Tuple]` of length `batch_size`, *optional*) —
  List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
  predictions will not be resized.

Returns

semantic\_segmentation

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [MobileViTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## MobileViTImageProcessor

### class transformers.MobileViTImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/image_processing_mobilevit.py#L56)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_center\_crop: bool = True crop\_size: typing.Optional[dict[str, int]] = None do\_flip\_channel\_order: bool = True do\_reduce\_labels: bool = False \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"shortest_edge" -- 224}`):
  Controls the size of the output image after resizing. Can be overridden by the `size` parameter in the
  `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Defines the resampling filter to use if resizing the image. Can be overridden by the `resample` parameter
  in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
  `preprocess` method.
* **do\_center\_crop** (`bool`, *optional*, defaults to `True`) —
  Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
  image is padded with 0’s and then center cropped. Can be overridden by the `do_center_crop` parameter in
  the `preprocess` method.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `{"height" -- 256, "width": 256}`):
  Desired output size `(size["height"], size["width"])` when applying center-cropping. Can be overridden by
  the `crop_size` parameter in the `preprocess` method.
* **do\_flip\_channel\_order** (`bool`, *optional*, defaults to `True`) —
  Whether to flip the color channels from RGB to BGR. Can be overridden by the `do_flip_channel_order`
  parameter in the `preprocess` method.
* **do\_reduce\_labels** (`bool`, *optional*, defaults to `False`) —
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is
  used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The
  background label will be replaced by 255. Can be overridden by the `do_reduce_labels` parameter in the
  `preprocess` method.

Constructs a MobileViT image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/image_processing_mobilevit.py#L325)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_center\_crop: typing.Optional[bool] = None crop\_size: typing.Optional[dict[str, int]] = None do\_flip\_channel\_order: typing.Optional[bool] = None do\_reduce\_labels: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  Segmentation map to preprocess.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image by rescale factor.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) —
  Size of the center crop if `do_center_crop` is set to `True`.
* **do\_flip\_channel\_order** (`bool`, *optional*, defaults to `self.do_flip_channel_order`) —
  Whether to flip the channel order of the image.
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

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/image_processing_mobilevit.py#L474)

( outputs target\_sizes: typing.Optional[list[tuple]] = None  ) → semantic\_segmentation

Parameters

* **outputs** ([MobileViTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[Tuple]` of length `batch_size`, *optional*) —
  List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
  predictions will not be resized.

Returns

semantic\_segmentation

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [MobileViTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## MobileViTImageProcessorFast

### class transformers.MobileViTImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/image_processing_mobilevit_fast.py#L68)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.mobilevit.image\_processing\_mobilevit\_fast.MobileVitFastImageProcessorKwargs]  )

Constructs a fast Mobilevit image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/image_processing_mobilevit_fast.py#L96)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.mobilevit.image\_processing\_mobilevit\_fast.MobileVitFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **do\_flip\_channel\_order** (`bool`, *optional*, defaults to `self.do_flip_channel_order`) —
  Whether to flip the color channels from RGB to BGR or vice versa.
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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/image_processing_mobilevit_fast.py#L215)

( outputs target\_sizes: typing.Optional[list[tuple]] = None  ) → semantic\_segmentation

Parameters

* **outputs** ([MobileNetV2ForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2ForSemanticSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[Tuple]` of length `batch_size`, *optional*) —
  List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
  predictions will not be resized.

Returns

semantic\_segmentation

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [MobileNetV2ForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2ForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## MobileViTModel

### class transformers.MobileViTModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/modeling_mobilevit.py#L646)

( config: MobileViTConfig expand\_output: bool = True  )

Parameters

* **config** ([MobileViTConfig](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **expand\_output** (`bool`, *optional*, defaults to `True`) —
  Whether to expand the output of the model using a 1x1 convolution. If `True`, the model will apply an additional
  1x1 convolution to expand the output channels from `config.neck_hidden_sizes[5]` to `config.neck_hidden_sizes[6]`.

The bare Mobilevit Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/modeling_mobilevit.py#L688)

( pixel\_values: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MobileViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessor). See [MobileViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTFeatureExtractor.__call__) for details (`processor_class` uses
  [MobileViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MobileViTConfig](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state after a pooling operation on the spatial dimensions.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [MobileViTModel](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## MobileViTForImageClassification

### class transformers.MobileViTForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/modeling_mobilevit.py#L737)

( config: MobileViTConfig  )

Parameters

* **config** ([MobileViTConfig](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/modeling_mobilevit.py#L753)

( pixel\_values: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MobileViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessor). See [MobileViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTFeatureExtractor.__call__) for details (`processor_class` uses
  [MobileViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MobileViTConfig](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.

The [MobileViTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, MobileViTForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
>>> model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

## MobileViTForSemanticSegmentation

### class transformers.MobileViTForSemanticSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/modeling_mobilevit.py#L925)

( config: MobileViTConfig  )

Parameters

* **config** ([MobileViTConfig](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MobileViT model with a semantic segmentation head on top, e.g. for Pascal VOC.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevit/modeling_mobilevit.py#L936)

( pixel\_values: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MobileViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessor). See [MobileViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTFeatureExtractor.__call__) for details (`processor_class` uses
  [MobileViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MobileViTConfig](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

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

The [MobileViTForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import requests
>>> import torch
>>> from PIL import Image
>>> from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
>>> model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # logits are of shape (batch_size, num_labels, height, width)
>>> logits = outputs.logits
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mobilevit.md)
