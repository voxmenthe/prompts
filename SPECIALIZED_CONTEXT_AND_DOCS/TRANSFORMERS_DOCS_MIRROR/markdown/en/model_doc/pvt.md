*This model was released on 2021-02-24 and added to Hugging Face Transformers on 2023-07-24.*

# Pyramid Vision Transformer (PVT)

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The PVT model was proposed in
[Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://huggingface.co/papers/2102.12122)
by Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao. The PVT is a type of
vision transformer that utilizes a pyramid structure to make it an effective backbone for dense prediction tasks. Specifically
it allows for more fine-grained inputs (4 x 4 pixels per patch) to be used, while simultaneously shrinking the sequence length
of the Transformer as it deepens - reducing the computational cost. Additionally, a spatial-reduction attention (SRA) layer
is used to further reduce the resource consumption when learning high-resolution features.

The abstract from the paper is the following:

*Although convolutional neural networks (CNNs) have achieved great success in computer vision, this work investigates a
simpler, convolution-free backbone network useful for many dense prediction tasks. Unlike the recently proposed Vision
Transformer (ViT) that was designed for image classification specifically, we introduce the Pyramid Vision Transformer
(PVT), which overcomes the difficulties of porting Transformer to various dense prediction tasks. PVT has several
merits compared to current state of the arts. Different from ViT that typically yields low resolution outputs and
incurs high computational and memory costs, PVT not only can be trained on dense partitions of an image to achieve high
output resolution, which is important for dense prediction, but also uses a progressive shrinking pyramid to reduce the
computations of large feature maps. PVT inherits the advantages of both CNN and Transformer, making it a unified
backbone for various vision tasks without convolutions, where it can be used as a direct replacement for CNN backbones.
We validate PVT through extensive experiments, showing that it boosts the performance of many downstream tasks, including
object detection, instance and semantic segmentation. For example, with a comparable number of parameters, PVT+RetinaNet
achieves 40.4 AP on the COCO dataset, surpassing ResNet50+RetinNet (36.3 AP) by 4.1 absolute AP (see Figure 2). We hope
that PVT could serve as an alternative and useful backbone for pixel-level predictions and facilitate future research.*

This model was contributed by [Xrenya](https://huggingface.co/Xrenya). The original code can be found [here](https://github.com/whai362/PVT).

* PVTv1 on ImageNet-1K

| **Model variant** | **Size** | **Acc@1** | **Params (M)** |
| --- | --- | --- | --- |
| PVT-Tiny | 224 | 75.1 | 13.2 |
| PVT-Small | 224 | 79.8 | 24.5 |
| PVT-Medium | 224 | 81.2 | 44.2 |
| PVT-Large | 224 | 81.7 | 61.4 |

## PvtConfig

### class transformers.PvtConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt/configuration_pvt.py#L33)

( image\_size: int = 224 num\_channels: int = 3 num\_encoder\_blocks: int = 4 depths: list = [2, 2, 2, 2] sequence\_reduction\_ratios: list = [8, 4, 2, 1] hidden\_sizes: list = [64, 128, 320, 512] patch\_sizes: list = [4, 2, 2, 2] strides: list = [4, 2, 2, 2] num\_attention\_heads: list = [1, 2, 5, 8] mlp\_ratios: list = [8, 8, 4, 4] hidden\_act: Mapping = 'gelu' hidden\_dropout\_prob: float = 0.0 attention\_probs\_dropout\_prob: float = 0.0 initializer\_range: float = 0.02 drop\_path\_rate: float = 0.0 layer\_norm\_eps: float = 1e-06 qkv\_bias: bool = True num\_labels: int = 1000 \*\*kwargs  )

Parameters

* **image\_size** (`int`, *optional*, defaults to 224) —
  The input image size
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **num\_encoder\_blocks** (`int`, *optional*, defaults to 4) —
  The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
* **depths** (`list[int]`, *optional*, defaults to `[2, 2, 2, 2]`) —
  The number of layers in each encoder block.
* **sequence\_reduction\_ratios** (`list[int]`, *optional*, defaults to `[8, 4, 2, 1]`) —
  Sequence reduction ratios in each encoder block.
* **hidden\_sizes** (`list[int]`, *optional*, defaults to `[64, 128, 320, 512]`) —
  Dimension of each of the encoder blocks.
* **patch\_sizes** (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`) —
  Patch size before each encoder block.
* **strides** (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`) —
  Stride before each encoder block.
* **num\_attention\_heads** (`list[int]`, *optional*, defaults to `[1, 2, 5, 8]`) —
  Number of attention heads for each attention layer in each block of the Transformer encoder.
* **mlp\_ratios** (`list[int]`, *optional*, defaults to `[8, 8, 4, 4]`) —
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
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not a learnable bias should be added to the queries, keys and values.
* **num\_labels** (‘int’, *optional*, defaults to 1000) —
  The number of classes.

This is the configuration class to store the configuration of a [PvtModel](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtModel). It is used to instantiate an Pvt
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Pvt
[Xrenya/pvt-tiny-224](https://huggingface.co/Xrenya/pvt-tiny-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import PvtModel, PvtConfig

>>> # Initializing a PVT Xrenya/pvt-tiny-224 style configuration
>>> configuration = PvtConfig()

>>> # Initializing a model from the Xrenya/pvt-tiny-224 style configuration
>>> model = PvtModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## PvtImageProcessor

### class transformers.PvtImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt/image_processing_pvt.py#L42)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `(size["height"], size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
  `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
  `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a PVT image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt/image_processing_pvt.py#L148)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
  resizing.
* **resample** (`PILImageResampling` filter, *optional*, defaults to `self.resample`) —
  `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
  an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use if `do_normalize` is set to `True`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

## PvtImageProcessorFast

### class transformers.PvtImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt/image_processing_pvt_fast.py#L23)

( \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  )

Constructs a fast Pvt image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L639)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*args \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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

## PvtForImageClassification

### class transformers.PvtForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt/modeling_pvt.py#L534)

( config: PvtConfig  )

Parameters

* **config** ([PvtConfig](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Pvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
the [CLS] token) e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt/modeling_pvt.py#L549)

( pixel\_values: typing.Optional[torch.Tensor] labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PvtImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtImageProcessor). See [PvtImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [PvtImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PvtConfig](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [PvtForImageClassification](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, PvtForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("Xrenya/pvt-tiny-224")
>>> model = PvtForImageClassification.from_pretrained("Xrenya/pvt-tiny-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

## PvtModel

### class transformers.PvtModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt/modeling_pvt.py#L477)

( config: PvtConfig  )

Parameters

* **config** ([PvtConfig](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Pvt Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt/modeling_pvt.py#L496)

( pixel\_values: FloatTensor output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PvtImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtImageProcessor). See [PvtImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [PvtImageProcessor](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtImageProcessor) for processing images).
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
elements depending on the configuration ([PvtConfig](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [PvtModel](/docs/transformers/v4.56.2/en/model_doc/pvt#transformers.PvtModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/pvt.md)
