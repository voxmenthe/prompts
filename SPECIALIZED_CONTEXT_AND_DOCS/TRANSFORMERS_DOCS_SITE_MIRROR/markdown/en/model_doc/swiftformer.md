# SwiftFormer

## Overview

The SwiftFormer model was proposed in [SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications](https://huggingface.co/papers/2303.15446) by Abdelrahman Shaker, Muhammad Maaz, Hanoona Rasheed, Salman Khan, Ming-Hsuan Yang, Fahad Shahbaz Khan.

The SwiftFormer paper introduces a novel efficient additive attention mechanism that effectively replaces the quadratic matrix multiplication operations in the self-attention computation with linear element-wise multiplications. A series of models called 'SwiftFormer' is built based on this, which achieves state-of-the-art performance in terms of both accuracy and mobile inference speed. Even their small variant achieves 78.5% top-1 ImageNet1K accuracy with only 0.8 ms latency on iPhone 14, which is more accurate and 2× faster compared to MobileViT-v2.

The abstract from the paper is the following:

*Self-attention has become a defacto choice for capturing global context in various vision applications. However, its quadratic computational complexity with respect to image resolution limits its use in real-time applications, especially for deployment on resource-constrained mobile devices. Although hybrid approaches have been proposed to combine the advantages of convolutions and self-attention for a better speed-accuracy trade-off, the expensive matrix multiplication operations in self-attention remain a bottleneck. In this work, we introduce a novel efficient additive attention mechanism that effectively replaces the quadratic matrix multiplication operations with linear element-wise multiplications. Our design shows that the key-value interaction can be replaced with a linear layer without sacrificing any accuracy. Unlike previous state-of-the-art methods, our efficient formulation of self-attention enables its usage at all stages of the network. Using our proposed efficient additive attention, we build a series of models called "SwiftFormer" which achieves state-of-the-art performance in terms of both accuracy and mobile inference speed. Our small variant achieves 78.5% top-1 ImageNet-1K accuracy with only 0.8 ms latency on iPhone 14, which is more accurate and 2x faster compared to MobileViT-v2.*

This model was contributed by [shehan97](https://huggingface.co/shehan97). The original code can be found [here](https://github.com/Amshaker/SwiftFormer).

## SwiftFormerConfig[[transformers.SwiftFormerConfig]]

#### transformers.SwiftFormerConfig[[transformers.SwiftFormerConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/swiftformer/configuration_swiftformer.py#L24)

This is the configuration class to store the configuration of a [SwiftFormerModel](/docs/transformers/main/en/model_doc/swiftformer#transformers.SwiftFormerModel). It is used to instantiate an
SwiftFormer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the SwiftFormer
[MBZUAI/swiftformer-xs](https://huggingface.co/MBZUAI/swiftformer-xs) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import SwiftFormerConfig, SwiftFormerModel

>>> # Initializing a SwiftFormer swiftformer-base-patch16-224 style configuration
>>> configuration = SwiftFormerConfig()

>>> # Initializing a model (with random weights) from the swiftformer-base-patch16-224 style configuration
>>> model = SwiftFormerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

image_size (`int`, *optional*, defaults to 224) : The size (resolution) of each image

num_channels (`int`, *optional*, defaults to 3) : The number of input channels

depths (`list[int]`, *optional*, defaults to `[3, 3, 6, 4]`) : Depth of each stage

embed_dims (`list[int]`, *optional*, defaults to `[48, 56, 112, 220]`) : The embedding dimension at each stage

mlp_ratio (`int`, *optional*, defaults to 4) : Ratio of size of the hidden dimensionality of an MLP to the dimensionality of its input.

downsamples (`list[bool]`, *optional*, defaults to `[True, True, True, True]`) : Whether or not to downsample inputs between two stages.

hidden_act (`str`, *optional*, defaults to `"gelu"`) : The non-linear activation function (string). `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

down_patch_size (`int`, *optional*, defaults to 3) : The size of patches in downsampling layers.

down_stride (`int`, *optional*, defaults to 2) : The stride of convolution kernels in downsampling layers.

down_pad (`int`, *optional*, defaults to 1) : Padding in downsampling layers.

drop_path_rate (`float`, *optional*, defaults to 0.0) : Rate at which to increase dropout probability in DropPath.

drop_mlp_rate (`float`, *optional*, defaults to 0.0) : Dropout rate for the MLP component of SwiftFormer.

drop_conv_encoder_rate (`float`, *optional*, defaults to 0.0) : Dropout rate for the ConvEncoder component of SwiftFormer.

use_layer_scale (`bool`, *optional*, defaults to `True`) : Whether to scale outputs from token mixers.

layer_scale_init_value (`float`, *optional*, defaults to 1e-05) : Factor by which outputs from token mixers are scaled.

batch_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the batch normalization layers.

## SwiftFormerModel[[transformers.SwiftFormerModel]]

#### transformers.SwiftFormerModel[[transformers.SwiftFormerModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/swiftformer/modeling_swiftformer.py#L414)

The bare Swiftformer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.SwiftFormerModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/swiftformer/modeling_swiftformer.py#L425[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.modeling_outputs.BaseModelOutputWithNoAttention` or `tuple(torch.FloatTensor)`A `transformers.modeling_outputs.BaseModelOutputWithNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SwiftFormerConfig](/docs/transformers/main/en/model_doc/swiftformer#transformers.SwiftFormerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
The [SwiftFormerModel](/docs/transformers/main/en/model_doc/swiftformer#transformers.SwiftFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
```

**Parameters:**

config ([SwiftFormerConfig](/docs/transformers/main/en/model_doc/swiftformer#transformers.SwiftFormerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.modeling_outputs.BaseModelOutputWithNoAttention` or `tuple(torch.FloatTensor)``

A `transformers.modeling_outputs.BaseModelOutputWithNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SwiftFormerConfig](/docs/transformers/main/en/model_doc/swiftformer#transformers.SwiftFormerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

## SwiftFormerForImageClassification[[transformers.SwiftFormerForImageClassification]]

#### transformers.SwiftFormerForImageClassification[[transformers.SwiftFormerForImageClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/swiftformer/modeling_swiftformer.py#L458)

The Swiftformer Model with an image classification head on top e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.SwiftFormerForImageClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/swiftformer/modeling_swiftformer.py#L475[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
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
elements depending on the configuration ([SwiftFormerConfig](/docs/transformers/main/en/model_doc/swiftformer#transformers.SwiftFormerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
The [SwiftFormerForImageClassification](/docs/transformers/main/en/model_doc/swiftformer#transformers.SwiftFormerForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, SwiftFormerForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("MBZUAI/swiftformer-xs")
>>> model = SwiftFormerForImageClassification.from_pretrained("MBZUAI/swiftformer-xs")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

**Parameters:**

config ([SwiftFormerConfig](/docs/transformers/main/en/model_doc/swiftformer#transformers.SwiftFormerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SwiftFormerConfig](/docs/transformers/main/en/model_doc/swiftformer#transformers.SwiftFormerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
