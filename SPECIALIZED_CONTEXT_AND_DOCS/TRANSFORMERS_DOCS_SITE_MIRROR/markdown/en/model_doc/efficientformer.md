# EfficientFormer

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

## Overview

The EfficientFormer model was proposed in [EfficientFormer: Vision Transformers at MobileNet Speed](https://huggingface.co/papers/2206.01191)
by Yanyu Li, Geng Yuan, Yang Wen, Eric Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren.  EfficientFormer proposes a
dimension-consistent pure transformer that can be run on mobile devices for dense prediction tasks like image classification, object
detection and semantic segmentation.

The abstract from the paper is the following:

*Vision Transformers (ViT) have shown rapid progress in computer vision tasks, achieving promising results on various benchmarks.
However, due to the massive number of parameters and model design, e.g., attention mechanism, ViT-based models are generally
times slower than lightweight convolutional networks. Therefore, the deployment of ViT for real-time applications is particularly
challenging, especially on resource-constrained hardware such as mobile devices. Recent efforts try to reduce the computation
complexity of ViT through network architecture search or hybrid design with MobileNet block, yet the inference speed is still
unsatisfactory. This leads to an important question: can transformers run as fast as MobileNet while obtaining high performance?
To answer this, we first revisit the network architecture and operators used in ViT-based models and identify inefficient designs.
Then we introduce a dimension-consistent pure transformer (without MobileNet blocks) as a design paradigm.
Finally, we perform latency-driven slimming to get a series of final models dubbed EfficientFormer.
Extensive experiments show the superiority of EfficientFormer in performance and speed on mobile devices.
Our fastest model, EfficientFormer-L1, achieves 79.2% top-1 accuracy on ImageNet-1K with only 1.6 ms inference latency on
iPhone 12 (compiled with CoreML), which { runs as fast as MobileNetV2×1.4 (1.6 ms, 74.7% top-1),} and our largest model,
EfficientFormer-L7, obtains 83.3% accuracy with only 7.0 ms latency. Our work proves that properly designed transformers can
reach extremely low latency on mobile devices while maintaining high performance.*

This model was contributed by [novice03](https://huggingface.co/novice03) and [Bearnardd](https://huggingface.co/Bearnardd).
The original code can be found [here](https://github.com/snap-research/EfficientFormer).

## Documentation resources

- [Image classification task guide](../tasks/image_classification)

## EfficientFormerConfig[[transformers.EfficientFormerConfig]]

#### transformers.EfficientFormerConfig[[transformers.EfficientFormerConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/efficientformer/configuration_efficientformer.py#L24)

This is the configuration class to store the configuration of an [EfficientFormerModel](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerModel). It is used to
instantiate an EfficientFormer model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the EfficientFormer
[snap-research/efficientformer-l1](https://huggingface.co/snap-research/efficientformer-l1) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import EfficientFormerConfig, EfficientFormerModel

>>> # Initializing a EfficientFormer efficientformer-l1 style configuration
>>> configuration = EfficientFormerConfig()

>>> # Initializing a EfficientFormerModel (with random weights) from the efficientformer-l3 style configuration
>>> model = EfficientFormerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

depths (`List(int)`, *optional*, defaults to `[3, 2, 6, 4]`) : Depth of each stage.

hidden_sizes (`List(int)`, *optional*, defaults to `[48, 96, 224, 448]`) : Dimensionality of each stage.

downsamples (`List(bool)`, *optional*, defaults to `[True, True, True, True]`) : Whether or not to downsample inputs between two stages.

dim (`int`, *optional*, defaults to 448) : Number of channels in Meta3D layers

key_dim (`int`, *optional*, defaults to 32) : The size of the key in meta3D block.

attention_ratio (`int`, *optional*, defaults to 4) : Ratio of the dimension of the query and value to the dimension of the key in MSHA block

resolution (`int`, *optional*, defaults to 7) : Size of each patch

num_hidden_layers (`int`, *optional*, defaults to 5) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 8) : Number of attention heads for each attention layer in the 3D MetaBlock.

mlp_expansion_ratio (`int`, *optional*, defaults to 4) : Ratio of size of the hidden dimensionality of an MLP to the dimensionality of its input.

hidden_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings and encoder.

patch_size (`int`, *optional*, defaults to 16) : The size (resolution) of each patch.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

pool_size (`int`, *optional*, defaults to 3) : Kernel size of pooling layers.

downsample_patch_size (`int`, *optional*, defaults to 3) : The size of patches in downsampling layers.

downsample_stride (`int`, *optional*, defaults to 2) : The stride of convolution kernels in downsampling layers.

downsample_pad (`int`, *optional*, defaults to 1) : Padding in downsampling layers.

drop_path_rate (`int`, *optional*, defaults to 0) : Rate at which to increase dropout probability in DropPath.

num_meta3d_blocks (`int`, *optional*, defaults to 1) : The number of 3D MetaBlocks in the last stage.

distillation (`bool`, *optional*, defaults to `True`) : Whether to add a distillation head.

use_layer_scale (`bool`, *optional*, defaults to `True`) : Whether to scale outputs from token mixers.

layer_scale_init_value (`float`, *optional*, defaults to 1e-5) : Factor by which outputs from token mixers are scaled.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

image_size (`int`, *optional*, defaults to `224`) : The size (resolution) of each image.

## EfficientFormerImageProcessor[[transformers.EfficientFormerImageProcessor]]

#### transformers.EfficientFormerImageProcessor[[transformers.EfficientFormerImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/efficientformer/image_processing_efficientformer.py#L47)

Constructs a EfficientFormer image processor.

preprocesstransformers.EfficientFormerImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/efficientformer/image_processing_efficientformer.py#L178[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "resample", "val": ": typing.Optional[PIL.Image.Resampling] = None"}, {"name": "do_center_crop", "val": ": typing.Optional[bool] = None"}, {"name": "crop_size", "val": ": typing.Optional[int] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension] = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}, {"name": "**kwargs", "val": ""}]- **images** (`ImageInput`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the image.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
  resizing.
- **resample** (`PILImageResampling` filter, *optional*, defaults to `self.resample`) --
  `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
  an effect if `do_resize` is set to `True`.
- **do_center_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) --
  Whether to center crop the image.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the image values between [0 - 1].
- **rescale_factor** (`float`, *optional*, defaults to `self.rescale_factor`) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **crop_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) --
  Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether to normalize the image.
- **image_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) --
  Image mean to use if `do_normalize` is set to `True`.
- **image_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) --
  Image standard deviation to use if `do_normalize` is set to `True`.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  The type of tensors to return. Can be one of:
  - Unset: Return a list of `np.ndarray`.
  - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
- **data_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) --
  The channel dimension format for the output image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - Unset: Use the channel dimension format of the input image.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.0

Preprocess an image or batch of images.

**Parameters:**

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions to the specified `(size["height"], size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.

size (`dict`, *optional*, defaults to `{"height" : 224, "width": 224}`): Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess` method.

resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`) : Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the `preprocess` method.

do_center_crop (`bool`, *optional*, defaults to `True`) : Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the `preprocess` method.

crop_size (`dict[str, int]` *optional*, defaults to 224) : Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess` method.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.

rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.

do_normalize : Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method.

image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) : Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

## EfficientFormerModel[[transformers.EfficientFormerModel]]

#### transformers.EfficientFormerModel[[transformers.EfficientFormerModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/efficientformer/modeling_efficientformer.py#L532)

The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top.
This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a
regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

forwardtransformers.EfficientFormerModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/efficientformer/modeling_efficientformer.py#L545[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Pixel values. Pixel values can be obtained using [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See
  [ViTImageProcessor.preprocess()](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor.preprocess) for details.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [EfficientFormerModel](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, EfficientFormerModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
>>> model = EfficientFormerModel.from_pretrained("snap-research/efficientformer-l1-300")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 49, 448]
```

**Parameters:**

config ([EfficientFormerConfig](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## EfficientFormerForImageClassification[[transformers.EfficientFormerForImageClassification]]

#### transformers.EfficientFormerForImageClassification[[transformers.EfficientFormerForImageClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/efficientformer/modeling_efficientformer.py#L595)

EfficientFormer Model transformer with an image classification head on top (a linear layer on top of the final
hidden state of the [CLS] token) e.g. for ImageNet.

This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a
regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

forwardtransformers.EfficientFormerForImageClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/efficientformer/modeling_efficientformer.py#L610[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Pixel values. Pixel values can be obtained using [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See
  [ViTImageProcessor.preprocess()](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor.preprocess) for details.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).0[transformers.modeling_outputs.ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [EfficientFormerForImageClassification](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, EfficientFormerForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
>>> model = EfficientFormerForImageClassification.from_pretrained("snap-research/efficientformer-l1-300")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
Egyptian cat
```

**Parameters:**

config ([EfficientFormerConfig](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## EfficientFormerForImageClassificationWithTeacher[[transformers.EfficientFormerForImageClassificationWithTeacher]]

#### transformers.EfficientFormerForImageClassificationWithTeacher[[transformers.EfficientFormerForImageClassificationWithTeacher]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/efficientformer/modeling_efficientformer.py#L706)

EfficientFormer Model transformer with image classification heads on top (a linear layer on top of the final hidden
state of the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for
ImageNet.

This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
supported.

This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a
regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

forwardtransformers.EfficientFormerForImageClassificationWithTeacher.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/efficientformer/modeling_efficientformer.py#L723[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Pixel values. Pixel values can be obtained using [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See
  [ViTImageProcessor.preprocess()](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor.preprocess) for details.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.deprecated.efficientformer.modeling_efficientformer.EfficientFormerForImageClassificationWithTeacherOutput` or `tuple(torch.FloatTensor)`A `transformers.models.deprecated.efficientformer.modeling_efficientformer.EfficientFormerForImageClassificationWithTeacherOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Prediction scores as the average of the cls_logits and distillation logits.
- **cls_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
  class token).
- **distillation_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
  distillation token).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
  plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
The [EfficientFormerForImageClassificationWithTeacher](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerForImageClassificationWithTeacher) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, EfficientFormerForImageClassificationWithTeacher
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
>>> model = EfficientFormerForImageClassificationWithTeacher.from_pretrained("snap-research/efficientformer-l1-300")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
Egyptian cat
```

**Parameters:**

config ([EfficientFormerConfig](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.deprecated.efficientformer.modeling_efficientformer.EfficientFormerForImageClassificationWithTeacherOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.deprecated.efficientformer.modeling_efficientformer.EfficientFormerForImageClassificationWithTeacherOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/main/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Prediction scores as the average of the cls_logits and distillation logits.
- **cls_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
  class token).
- **distillation_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
  distillation token).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
  plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
  the self-attention heads.
