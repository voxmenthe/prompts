*This model was released on 2022-01-19 and added to Hugging Face Transformers on 2022-03-22.*

# GLPN

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

This is a recently introduced model so the API hasn’t been tested extensively. There may be some bugs or slight
breaking changes to fix it in the future. If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title).

## Overview

The GLPN model was proposed in [Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth](https://huggingface.co/papers/2201.07436) by Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, Junmo Kim.
GLPN combines [SegFormer](segformer)’s hierarchical mix-Transformer with a lightweight decoder for monocular depth estimation. The proposed decoder shows better performance than the previously proposed decoders, with considerably
less computational complexity.

The abstract from the paper is the following:

*Depth estimation from a single image is an important task that can be applied to various fields in computer vision, and has grown rapidly with the development of convolutional neural networks. In this paper, we propose a novel structure and training strategy for monocular depth estimation to further improve the prediction accuracy of the network. We deploy a hierarchical transformer encoder to capture and convey the global context, and design a lightweight yet powerful decoder to generate an estimated depth map while considering local connectivity. By constructing connected paths between multi-scale local features and the global decoding stream with our proposed selective feature fusion module, the network can integrate both representations and recover fine details. In addition, the proposed decoder shows better performance than the previously proposed decoders, with considerably less computational complexity. Furthermore, we improve the depth-specific augmentation method by utilizing an important observation in depth estimation to enhance the model. Our network achieves state-of-the-art performance over the challenging depth dataset NYU Depth V2. Extensive experiments have been conducted to validate and show the effectiveness of the proposed approach. Finally, our model shows better generalisation ability and robustness than other comparative models.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/glpn_architecture.jpg) Summary of the approach. Taken from the [original paper](https://huggingface.co/papers/2201.07436).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/vinvino02/GLPDepth).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with GLPN.

* Demo notebooks for [GLPNForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNForDepthEstimation) can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/GLPN).
* [Monocular depth estimation task guide](../tasks/monocular_depth_estimation)

## GLPNConfig

### class transformers.GLPNConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glpn/configuration_glpn.py#L24)

( num\_channels = 3 num\_encoder\_blocks = 4 depths = [2, 2, 2, 2] sr\_ratios = [8, 4, 2, 1] hidden\_sizes = [32, 64, 160, 256] patch\_sizes = [7, 3, 3, 3] strides = [4, 2, 2, 2] num\_attention\_heads = [1, 2, 5, 8] mlp\_ratios = [4, 4, 4, 4] hidden\_act = 'gelu' hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 initializer\_range = 0.02 drop\_path\_rate = 0.1 layer\_norm\_eps = 1e-06 decoder\_hidden\_size = 64 max\_depth = 10 head\_in\_index = -1 \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **num\_encoder\_blocks** (`int`, *optional*, defaults to 4) —
  The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
* **depths** (`list[int]`, *optional*, defaults to `[2, 2, 2, 2]`) —
  The number of layers in each encoder block.
* **sr\_ratios** (`list[int]`, *optional*, defaults to `[8, 4, 2, 1]`) —
  Sequence reduction ratios in each encoder block.
* **hidden\_sizes** (`list[int]`, *optional*, defaults to `[32, 64, 160, 256]`) —
  Dimension of each of the encoder blocks.
* **patch\_sizes** (`list[int]`, *optional*, defaults to `[7, 3, 3, 3]`) —
  Patch size before each encoder block.
* **strides** (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`) —
  Stride before each encoder block.
* **num\_attention\_heads** (`list[int]`, *optional*, defaults to `[1, 2, 5, 8]`) —
  Number of attention heads for each attention layer in each block of the Transformer encoder.
* **mlp\_ratios** (`list[int]`, *optional*, defaults to `[4, 4, 4, 4]`) —
  Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
  encoder blocks.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **decoder\_hidden\_size** (`int`, *optional*, defaults to 64) —
  The dimension of the decoder.
* **max\_depth** (`int`, *optional*, defaults to 10) —
  The maximum depth of the decoder.
* **head\_in\_index** (`int`, *optional*, defaults to -1) —
  The index of the features to use in the head.

This is the configuration class to store the configuration of a [GLPNModel](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNModel). It is used to instantiate an GLPN
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the GLPN
[vinvino02/glpn-kitti](https://huggingface.co/vinvino02/glpn-kitti) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import GLPNModel, GLPNConfig

>>> # Initializing a GLPN vinvino02/glpn-kitti style configuration
>>> configuration = GLPNConfig()

>>> # Initializing a model from the vinvino02/glpn-kitti style configuration
>>> model = GLPNModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## GLPNFeatureExtractor

### class transformers.GLPNFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glpn/feature_extraction_glpn.py#L28)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

## GLPNImageProcessor

### class transformers.GLPNImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glpn/image_processing_glpn.py#L53)

( do\_resize: bool = True size\_divisor: int = 32 resample = <Resampling.BILINEAR: 2> do\_rescale: bool = True \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions, rounding them down to the closest multiple of
  `size_divisor`. Can be overridden by `do_resize` in `preprocess`.
* **size\_divisor** (`int`, *optional*, defaults to 32) —
  When `do_resize` is `True`, images are resized so their height and width are rounded down to the closest
  multiple of `size_divisor`. Can be overridden by `size_divisor` in `preprocess`.
* **resample** (`PIL.Image` resampling filter, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Can be
  overridden by `do_rescale` in `preprocess`.

Constructs a GLPN image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glpn/image_processing_glpn.py#L137)

( images: typing.Union[ForwardRef('PIL.Image.Image'), transformers.utils.generic.TensorType, list['PIL.Image.Image'], list[transformers.utils.generic.TensorType]] do\_resize: typing.Optional[bool] = None size\_divisor: typing.Optional[int] = None resample = None do\_rescale: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`PIL.Image.Image` or `TensorType` or `list[np.ndarray]` or `list[TensorType]`) —
  Images to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_normalize=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the input such that the (height, width) dimensions are a multiple of `size_divisor`.
* **size\_divisor** (`int`, *optional*, defaults to `self.size_divisor`) —
  When `do_resize` is `True`, images are resized so their height and width are rounded down to the
  closest multiple of `size_divisor`.
* **resample** (`PIL.Image` resampling filter, *optional*, defaults to `self.resample`) —
  `PIL.Image` resampling filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
  an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.).
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

Preprocess the given images.

## GLPNModel

### class transformers.GLPNModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glpn/modeling_glpn.py#L436)

( config  )

Parameters

* **config** ([GLPNModel](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Glpn Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glpn/modeling_glpn.py#L456)

( pixel\_values: FloatTensor output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [GLPNImageProcessor](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNImageProcessor). See [GLPNImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [GLPNImageProcessor](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GLPNConfig](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [GLPNModel](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## GLPNForDepthEstimation

### class transformers.GLPNForDepthEstimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glpn/modeling_glpn.py#L631)

( config  )

Parameters

* **config** ([GLPNForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNForDepthEstimation)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

GLPN Model transformer with a lightweight depth estimation head on top e.g. for KITTI, NYUv2.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glpn/modeling_glpn.py#L642)

( pixel\_values: FloatTensor labels: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [GLPNImageProcessor](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNImageProcessor). See [GLPNImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [GLPNImageProcessor](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNImageProcessor) for processing images).
* **labels** (`torch.FloatTensor` of shape `(batch_size, height, width)`, *optional*) —
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
elements depending on the configuration ([GLPNConfig](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **predicted\_depth** (`torch.FloatTensor` of shape `(batch_size, height, width)`) — Predicted depth for each pixel.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [GLPNForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/glpn#transformers.GLPNForDepthEstimation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, GLPNForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")
>>> model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")

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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glpn.md)
