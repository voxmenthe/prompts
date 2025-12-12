*This model was released on 2023-02-23 and added to Hugging Face Transformers on 2024-07-08.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# ZoeDepth

[ZoeDepth](https://huggingface.co/papers/2302.12288) is a depth estimation model that combines the generalization performance of relative depth estimation (how far objects are from each other) and metric depth estimation (precise depth measurement on metric scale) from a single image. It is pre-trained on 12 datasets using relative depth and 2 datasets (NYU Depth v2 and KITTI) for metric accuracy. A lightweight head with a metric bin module for each domain is used, and during inference, it automatically selects the appropriate head for each input image with a latent classifier.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/zoedepth_architecture_bis.png)

You can find all the original ZoeDepth checkpoints under the [Intel](https://huggingface.co/Intel?search=zoedepth) organization.

The example below demonstrates how to estimate depth with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import requests
import torch
from transformers import pipeline
from PIL import Image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
pipeline = pipeline(
    task="depth-estimation",
    model="Intel/zoedepth-nyu-kitti",
    dtype=torch.float16,
    device=0
)
results = pipeline(image)
results["depth"]
```

## Notes

* In the [original implementation](https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/models/depth_model.py#L131) ZoeDepth performs inference on both the original and flipped images and averages the results. The `post_process_depth_estimation` function handles this by passing the flipped outputs to the optional `outputs_flipped` argument as shown below.


  ```
   with torch.no_grad():
       outputs = model(pixel_values)
       outputs_flipped = model(pixel_values=torch.flip(inputs.pixel_values, dims=[3]))
       post_processed_output = image_processor.post_process_depth_estimation(
           outputs,
           source_sizes=[(image.height, image.width)],
           outputs_flipped=outputs_flipped,
       )
  ```

## Resources

* Refer to this [notebook](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ZoeDepth) for an inference example.

## ZoeDepthConfig

### class transformers.ZoeDepthConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/zoedepth/configuration_zoedepth.py#L29)

( backbone\_config = None backbone = None use\_pretrained\_backbone = False backbone\_kwargs = None hidden\_act = 'gelu' initializer\_range = 0.02 batch\_norm\_eps = 1e-05 readout\_type = 'project' reassemble\_factors = [4, 2, 1, 0.5] neck\_hidden\_sizes = [96, 192, 384, 768] fusion\_hidden\_size = 256 head\_in\_index = -1 use\_batch\_norm\_in\_fusion\_residual = False use\_bias\_in\_fusion\_residual = None num\_relative\_features = 32 add\_projection = False bottleneck\_features = 256 num\_attractors = [16, 8, 4, 1] bin\_embedding\_dim = 128 attractor\_alpha = 1000 attractor\_gamma = 2 attractor\_kind = 'mean' min\_temp = 0.0212 max\_temp = 50.0 bin\_centers\_type = 'softplus' bin\_configurations = [{'n\_bins': 64, 'min\_depth': 0.001, 'max\_depth': 10.0}] num\_patch\_transformer\_layers = None patch\_transformer\_hidden\_size = None patch\_transformer\_intermediate\_size = None patch\_transformer\_num\_attention\_heads = None \*\*kwargs  )

Parameters

* **backbone\_config** (`Union[dict[str, Any], PretrainedConfig]`, *optional*, defaults to `BeitConfig()`) —
  The configuration of the backbone model.
* **backbone** (`str`, *optional*) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to use pretrained weights for the backbone.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **batch\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the batch normalization layers.
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
* **num\_relative\_features** (`int`, *optional*, defaults to 32) —
  The number of features to use in the relative depth estimation head.
* **add\_projection** (`bool`, *optional*, defaults to `False`) —
  Whether to add a projection layer before the depth estimation head.
* **bottleneck\_features** (`int`, *optional*, defaults to 256) —
  The number of features in the bottleneck layer.
* **num\_attractors** (`list[int], *optional*, defaults to` [16, 8, 4, 1]`) —
  The number of attractors to use in each stage.
* **bin\_embedding\_dim** (`int`, *optional*, defaults to 128) —
  The dimension of the bin embeddings.
* **attractor\_alpha** (`int`, *optional*, defaults to 1000) —
  The alpha value to use in the attractor.
* **attractor\_gamma** (`int`, *optional*, defaults to 2) —
  The gamma value to use in the attractor.
* **attractor\_kind** (`str`, *optional*, defaults to `"mean"`) —
  The kind of attractor to use. Can be one of [`"mean"`, `"sum"`].
* **min\_temp** (`float`, *optional*, defaults to 0.0212) —
  The minimum temperature value to consider.
* **max\_temp** (`float`, *optional*, defaults to 50.0) —
  The maximum temperature value to consider.
* **bin\_centers\_type** (`str`, *optional*, defaults to `"softplus"`) —
  Activation type used for bin centers. Can be “normed” or “softplus”. For “normed” bin centers, linear normalization trick
  is applied. This results in bounded bin centers. For “softplus”, softplus activation is used and thus are unbounded.
* **bin\_configurations** (`list[dict]`, *optional*, defaults to `[{'n_bins' -- 64, 'min_depth': 0.001, 'max_depth': 10.0}]`):
  Configuration for each of the bin heads.
  Each configuration should consist of the following keys:
  + name (`str`): The name of the bin head - only required in case of multiple bin configurations.
  + `n_bins` (`int`): The number of bins to use.
  + `min_depth` (`float`): The minimum depth value to consider.
  + `max_depth` (`float`): The maximum depth value to consider.
    In case only a single configuration is passed, the model will use a single head with the specified configuration.
    In case multiple configurations are passed, the model will use multiple heads with the specified configurations.
* **num\_patch\_transformer\_layers** (`int`, *optional*) —
  The number of transformer layers to use in the patch transformer. Only used in case of multiple bin configurations.
* **patch\_transformer\_hidden\_size** (`int`, *optional*) —
  The hidden size to use in the patch transformer. Only used in case of multiple bin configurations.
* **patch\_transformer\_intermediate\_size** (`int`, *optional*) —
  The intermediate size to use in the patch transformer. Only used in case of multiple bin configurations.
* **patch\_transformer\_num\_attention\_heads** (`int`, *optional*) —
  The number of attention heads to use in the patch transformer. Only used in case of multiple bin configurations.

This is the configuration class to store the configuration of a [ZoeDepthForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthForDepthEstimation). It is used to instantiate an ZoeDepth
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the ZoeDepth
[Intel/zoedepth-nyu](https://huggingface.co/Intel/zoedepth-nyu) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import ZoeDepthConfig, ZoeDepthForDepthEstimation

>>> # Initializing a ZoeDepth zoedepth-large style configuration
>>> configuration = ZoeDepthConfig()

>>> # Initializing a model from the zoedepth-large style configuration
>>> model = ZoeDepthForDepthEstimation(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ZoeDepthImageProcessor

### class transformers.ZoeDepthImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/zoedepth/image_processing_zoedepth.py#L103)

( do\_pad: bool = True do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> keep\_aspect\_ratio: bool = True ensure\_multiple\_of: int = 32 \*\*kwargs  )

Parameters

* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to apply pad the input.
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
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions. Can be overridden by `do_resize` in `preprocess`.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 384, "width": 512}`):
  Size of the image after resizing. Size of the image after resizing. If `keep_aspect_ratio` is `True`,
  the image is resized by choosing the smaller of the height and width scaling factors and using it for both dimensions.
  If `ensure_multiple_of` is also set, the image is further resized to a size that is a multiple of this value.
  Can be overridden by `size` in `preprocess`.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Defines the resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
* **keep\_aspect\_ratio** (`bool`, *optional*, defaults to `True`) —
  If `True`, the image is resized by choosing the smaller of the height and width scaling factors and using it
  for both dimensions. This ensures that the image is scaled down as little as possible while still fitting
  within the desired output size. In case `ensure_multiple_of` is also set, the image is further resized to a
  size that is a multiple of this value by flooring the height and width to the nearest multiple of this value.
  Can be overridden by `keep_aspect_ratio` in `preprocess`.
* **ensure\_multiple\_of** (`int`, *optional*, defaults to 32) —
  If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Works by flooring
  the height and width to the nearest multiple of this value.

  Works both with and without `keep_aspect_ratio` being set to `True`. Can be overridden by `ensure_multiple_of`
  in `preprocess`.

Constructs a ZoeDepth image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/zoedepth/image_processing_zoedepth.py#L298)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_pad: typing.Optional[bool] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[int] = None keep\_aspect\_ratio: typing.Optional[bool] = None ensure\_multiple\_of: typing.Optional[int] = None resample: Resampling = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether to pad the input image.
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
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing. If `keep_aspect_ratio` is `True`, he image is resized by choosing the
  smaller of the height and width scaling factors and using it for both dimensions. If `ensure_multiple_of`
  is also set, the image is further resized to a size that is a multiple of this value.
* **keep\_aspect\_ratio** (`bool`, *optional*, defaults to `self.keep_aspect_ratio`) —
  If `True` and `do_resize=True`, the image is resized by choosing the smaller of the height and width
  scaling factors and using it for both dimensions. This ensures that the image is scaled down as little
  as possible while still fitting within the desired output size. In case `ensure_multiple_of` is also
  set, the image is further resized to a size that is a multiple of this value by flooring the height and
  width to the nearest multiple of this value.
* **ensure\_multiple\_of** (`int`, *optional*, defaults to `self.ensure_multiple_of`) —
  If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Works by
  flooring the height and width to the nearest multiple of this value.

  Works both with and without `keep_aspect_ratio` being set to `True`. Can be overridden by
  `ensure_multiple_of` in `preprocess`.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
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

## ZoeDepthImageProcessorFast

### class transformers.ZoeDepthImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/zoedepth/image_processing_zoedepth_fast.py#L94)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.zoedepth.image\_processing\_zoedepth\_fast.ZoeDepthFastImageProcessorKwargs]  )

Constructs a fast Zoedepth image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/zoedepth/image_processing_zoedepth_fast.py#L110)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.zoedepth.image\_processing\_zoedepth\_fast.ZoeDepthFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to apply pad the input.
* **keep\_aspect\_ratio** (`bool`, *optional*, defaults to `True`) —
  If `True`, the image is resized by choosing the smaller of the height and width scaling factors and using it
  for both dimensions. This ensures that the image is scaled down as little as possible while still fitting
  within the desired output size. In case `ensure_multiple_of` is also set, the image is further resized to a
  size that is a multiple of this value by flooring the height and width to the nearest multiple of this value.
  Can be overridden by `keep_aspect_ratio` in `preprocess`.
* **ensure\_multiple\_of** (`int`, *optional*, defaults to 32) —
  If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Works by flooring
  the height and width to the nearest multiple of this value.
  Works both with and without `keep_aspect_ratio` being set to `True`.
  Can be overridden by `ensure_multiple_of` in `preprocess`.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## ZoeDepthForDepthEstimation

### class transformers.ZoeDepthForDepthEstimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/zoedepth/modeling_zoedepth.py#L1232)

( config  )

Parameters

* **config** ([ZoeDepthForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthForDepthEstimation)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ZoeDepth model with one or multiple metric depth estimation head(s) on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/zoedepth/modeling_zoedepth.py#L1258)

( pixel\_values: FloatTensor labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ZoeDepthImageProcessor](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthImageProcessor). See [ZoeDepthImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ZoeDepthImageProcessor](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Ground truth depth estimation maps for computing the loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ZoeDepthConfig](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **predicted\_depth** (`torch.FloatTensor` of shape `(batch_size, height, width)`) — Predicted depth for each pixel.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ZoeDepthForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/zoedepth#transformers.ZoeDepthForDepthEstimation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
>>> model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # interpolate to original size
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     source_sizes=[(image.height, image.width)],
... )

>>> # visualize the prediction
>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = predicted_depth * 255 / predicted_depth.max()
>>> depth = depth.detach().cpu().numpy()
>>> depth = Image.fromarray(depth.astype("uint8"))
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/zoedepth.md)
