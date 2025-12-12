*This model was released on 2024-12-18 and added to Hugging Face Transformers on 2025-03-21.*

# Prompt Depth Anything

## Overview

The Prompt Depth Anything model was introduced in [Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](https://huggingface.co/papers/2412.14015) by Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, Bingyi Kang.

The abstract from the paper is as follows:

*Prompts play a critical role in unleashing the power of language and vision foundation models for specific tasks. For the first time, we introduce prompting into depth foundation models, creating a new paradigm for metric depth estimation termed Prompt Depth Anything. Specifically, we use a low-cost LiDAR as the prompt to guide the Depth Anything model for accurate metric depth output, achieving up to 4K resolution. Our approach centers on a concise prompt fusion design that integrates the LiDAR at multiple scales within the depth decoder. To address training challenges posed by limited datasets containing both LiDAR depth and precise GT depth, we propose a scalable data pipeline that includes synthetic data LiDAR simulation and real data pseudo GT depth generation. Our approach sets new state-of-the-arts on the ARKitScenes and ScanNet++ datasets and benefits downstream applications, including 3D reconstruction and generalized robotic grasping.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/prompt_depth_anything_architecture.jpg) Prompt Depth Anything overview. Taken from the [original paper](https://huggingface.co/papers/2412.14015).

## Usage example

The Transformers library allows you to use the model with just a few lines of code:


```
>>> import torch
>>> import requests
>>> import numpy as np

>>> from PIL import Image
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation

>>> url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")
>>> model = AutoModelForDepthEstimation.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")

>>> prompt_depth_url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/arkit_depth.png?raw=true"
>>> prompt_depth = Image.open(requests.get(prompt_depth_url, stream=True).raw)
>>> # the prompt depth can be None, and the model will output a monocular relative depth.

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt", prompt_depth=prompt_depth)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # interpolate to original size
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     target_sizes=[(image.height, image.width)],
... )

>>> # visualize the prediction
>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = predicted_depth * 1000 
>>> depth = depth.detach().cpu().numpy()
>>> depth = Image.fromarray(depth.astype("uint16")) # mm
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Prompt Depth Anything.

* [Prompt Depth Anything Demo](https://huggingface.co/spaces/depth-anything/PromptDA)
* [Prompt Depth Anything Interactive Results](https://promptda.github.io/interactive.html)

If you are interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## PromptDepthAnythingConfig

### class transformers.PromptDepthAnythingConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prompt_depth_anything/configuration_prompt_depth_anything.py#L31)

( backbone\_config = None backbone = None use\_pretrained\_backbone = False use\_timm\_backbone = False backbone\_kwargs = None patch\_size = 14 initializer\_range = 0.02 reassemble\_hidden\_size = 384 reassemble\_factors = [4, 2, 1, 0.5] neck\_hidden\_sizes = [48, 96, 192, 384] fusion\_hidden\_size = 64 head\_in\_index = -1 head\_hidden\_size = 32 depth\_estimation\_type = 'relative' max\_depth = None \*\*kwargs  )

Parameters

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
  Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone)
  API.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The size of the patches to extract from the backbone features.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **reassemble\_hidden\_size** (`int`, *optional*, defaults to 384) —
  The number of input channels of the reassemble layers.
* **reassemble\_factors** (`list[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`) —
  The up/downsampling factors of the reassemble layers.
* **neck\_hidden\_sizes** (`list[str]`, *optional*, defaults to `[48, 96, 192, 384]`) —
  The hidden sizes to project to for the feature maps of the backbone.
* **fusion\_hidden\_size** (`int`, *optional*, defaults to 64) —
  The number of channels before fusion.
* **head\_in\_index** (`int`, *optional*, defaults to -1) —
  The index of the features to use in the depth estimation head.
* **head\_hidden\_size** (`int`, *optional*, defaults to 32) —
  The number of output channels in the second convolution of the depth estimation head.
* **depth\_estimation\_type** (`str`, *optional*, defaults to `"relative"`) —
  The type of depth estimation to use. Can be one of `["relative", "metric"]`.
* **max\_depth** (`float`, *optional*) —
  The maximum depth to use for the “metric” depth estimation head. 20 should be used for indoor models
  and 80 for outdoor models. For “relative” depth estimation, this value is ignored.

This is the configuration class to store the configuration of a `PromptDepthAnythingModel`. It is used to instantiate a PromptDepthAnything
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the PromptDepthAnything
[LiheYoung/depth-anything-small-hf](https://huggingface.co/LiheYoung/depth-anything-small-hf) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import PromptDepthAnythingConfig, PromptDepthAnythingForDepthEstimation

>>> # Initializing a PromptDepthAnything small style configuration
>>> configuration = PromptDepthAnythingConfig()

>>> # Initializing a model from the PromptDepthAnything small style configuration
>>> model = PromptDepthAnythingForDepthEstimation(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prompt_depth_anything/configuration_prompt_depth_anything.py#L165)

( )

Serializes this instance to a Python dictionary. Override the default [to\_dict()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.to_dict). Returns:
`dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,

## PromptDepthAnythingForDepthEstimation

### class transformers.PromptDepthAnythingForDepthEstimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prompt_depth_anything/modeling_prompt_depth_anything.py#L373)

( config  )

Parameters

* **config** ([PromptDepthAnythingForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/prompt_depth_anything#transformers.PromptDepthAnythingForDepthEstimation)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Prompt Depth Anything Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prompt_depth_anything/modeling_prompt_depth_anything.py#L386)

( pixel\_values: FloatTensor prompt\_depth: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PromptDepthAnythingImageProcessor](/docs/transformers/v4.56.2/en/model_doc/prompt_depth_anything#transformers.PromptDepthAnythingImageProcessor). See [PromptDepthAnythingImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [PromptDepthAnythingImageProcessor](/docs/transformers/v4.56.2/en/model_doc/prompt_depth_anything#transformers.PromptDepthAnythingImageProcessor) for processing images).
* **prompt\_depth** (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`, *optional*) —
  Prompt depth is the sparse or low-resolution depth obtained from multi-view geometry or a
  low-resolution depth sensor. It generally has shape (height, width), where height
  and width can be smaller than those of the images. It is optional and can be None, which means no prompt depth
  will be used. If it is None, the output will be a monocular relative depth.
  The values are recommended to be in meters, but this is not necessary.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
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
elements depending on the configuration ([PromptDepthAnythingConfig](/docs/transformers/v4.56.2/en/model_doc/prompt_depth_anything#transformers.PromptDepthAnythingConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **predicted\_depth** (`torch.FloatTensor` of shape `(batch_size, height, width)`) — Predicted depth for each pixel.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [PromptDepthAnythingForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/prompt_depth_anything#transformers.PromptDepthAnythingForDepthEstimation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")
>>> model = AutoModelForDepthEstimation.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")

>>> prompt_depth_url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/arkit_depth.png?raw=true"
>>> prompt_depth = Image.open(requests.get(prompt_depth_url, stream=True).raw)

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt", prompt_depth=prompt_depth)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # interpolate to original size
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     target_sizes=[(image.height, image.width)],
... )

>>> # visualize the prediction
>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = predicted_depth * 1000.
>>> depth = depth.detach().cpu().numpy()
>>> depth = Image.fromarray(depth.astype("uint16")) # mm
```

## PromptDepthAnythingImageProcessor

### class transformers.PromptDepthAnythingImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prompt_depth_anything/image_processing_prompt_depth_anything.py#L100)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BICUBIC: 3> keep\_aspect\_ratio: bool = False ensure\_multiple\_of: int = 1 do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: bool = False size\_divisor: typing.Optional[int] = None prompt\_scale\_to\_meter: float = 0.001 \*\*kwargs  )

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
* **prompt\_scale\_to\_meter** (`float`, *optional*, defaults to 0.001) —
  Scale factor to convert the prompt depth to meters.

Constructs a PromptDepthAnything image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prompt_depth_anything/image_processing_prompt_depth_anything.py#L277)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] prompt\_depth: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[int] = None keep\_aspect\_ratio: typing.Optional[bool] = None ensure\_multiple\_of: typing.Optional[int] = None resample: typing.Optional[PIL.Image.Resampling] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: typing.Optional[bool] = None size\_divisor: typing.Optional[int] = None prompt\_scale\_to\_meter: typing.Optional[float] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **prompt\_depth** (`ImageInput`, *optional*) —
  Prompt depth to preprocess, which can be sparse depth obtained from multi-view geometry or
  low-resolution depth from a depth sensor. Generally has shape (height, width), where height
  and width can be smaller than those of the images. It’s optional and can be None, which means no prompt depth
  is used. If it is None, the output depth will be a monocular relative depth.
  It is recommended to provide a prompt\_scale\_to\_meter value, which is the scale factor to convert the prompt depth
  to meters. This is useful when the prompt depth is not in meters.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing. If `keep_aspect_ratio` is `True`, the image is resized to the largest
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
* **prompt\_scale\_to\_meter** (`float`, *optional*, defaults to `self.prompt_scale_to_meter`) —
  Scale factor to convert the prompt depth to meters.
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

#### post\_process\_depth\_estimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prompt_depth_anything/image_processing_prompt_depth_anything.py#L463)

( outputs: DepthEstimatorOutput target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple[int, int]], NoneType] = None  ) → `list[dict[str, TensorType]]`

Parameters

* **outputs** (`DepthEstimatorOutput`) —
  Raw outputs of the model.
* **target\_sizes** (`TensorType` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  (height, width) of each image in the batch. If left to None, predictions will not be resized.

Returns

`list[dict[str, TensorType]]`

A list of dictionaries of tensors representing the processed depth
predictions.

Converts the raw output of `DepthEstimatorOutput` into final depth predictions and depth PIL images.
Only supports PyTorch.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/prompt_depth_anything.md)
