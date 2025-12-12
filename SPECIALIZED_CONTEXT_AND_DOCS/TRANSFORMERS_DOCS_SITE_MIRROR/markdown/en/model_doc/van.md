# VAN

This model is in maintenance mode only, we don't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

## Overview

The VAN model was proposed in [Visual Attention Network](https://huggingface.co/papers/2202.09741) by Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu.

This paper introduces a new attention layer based on convolution operations able to capture both local and distant relationships. This is done by combining normal and large kernel convolution layers. The latter uses a dilated convolution to capture distant correlations.

The abstract from the paper is the following:

*While originally designed for natural language processing tasks, the self-attention mechanism has recently taken various computer vision areas by storm. However, the 2D nature of images brings three challenges for applying self-attention in computer vision. (1) Treating images as 1D sequences neglects their 2D structures. (2) The quadratic complexity is too expensive for high-resolution images. (3) It only captures spatial adaptability but ignores channel adaptability. In this paper, we propose a novel large kernel attention (LKA) module to enable self-adaptive and long-range correlations in self-attention while avoiding the above issues. We further introduce a novel neural network based on LKA, namely Visual Attention Network (VAN). While extremely simple, VAN outperforms the state-of-the-art vision transformers and convolutional neural networks with a large margin in extensive experiments, including image classification, object detection, semantic segmentation, instance segmentation, etc. Code is available at [this https URL](https://github.com/Visual-Attention-Network/VAN-Classification).*

Tips:

- VAN does not have an embedding layer, thus the `hidden_states` will have a length equal to the number of stages.

The figure below illustrates the architecture of a Visual Attention Layer. Taken from the [original paper](https://huggingface.co/papers/2202.09741).

This model was contributed by [Francesco](https://huggingface.co/Francesco). The original code can be found [here](https://github.com/Visual-Attention-Network/VAN-Classification).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with VAN.

- [VanForImageClassification](/docs/transformers/main/en/model_doc/van#transformers.VanForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## VanConfig[[transformers.VanConfig]]

#### transformers.VanConfig[[transformers.VanConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/van/configuration_van.py#L24)

This is the configuration class to store the configuration of a [VanModel](/docs/transformers/main/en/model_doc/van#transformers.VanModel). It is used to instantiate a VAN model
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the VAN
[Visual-Attention-Network/van-base](https://huggingface.co/Visual-Attention-Network/van-base) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:
```python
>>> from transformers import VanModel, VanConfig

>>> # Initializing a VAN van-base style configuration
>>> configuration = VanConfig()
>>> # Initializing a model from the van-base style configuration
>>> model = VanModel(configuration)
>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

image_size (`int`, *optional*, defaults to 224) : The size (resolution) of each image.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

patch_sizes (`list[int]`, *optional*, defaults to `[7, 3, 3, 3]`) : Patch size to use in each stage's embedding layer.

strides (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`) : Stride size to use in each stage's embedding layer to downsample the input.

hidden_sizes (`list[int]`, *optional*, defaults to `[64, 128, 320, 512]`) : Dimensionality (hidden size) at each stage.

depths (`list[int]`, *optional*, defaults to `[3, 3, 12, 3]`) : Depth (number of layers) for each stage.

mlp_ratios (`list[int]`, *optional*, defaults to `[8, 8, 4, 4]`) : The expansion ratio for mlp layer at each stage.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in each layer. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-06) : The epsilon used by the layer normalization layers.

layer_scale_init_value (`float`, *optional*, defaults to 0.01) : The initial value for layer scaling.

drop_path_rate (`float`, *optional*, defaults to 0.0) : The dropout probability for stochastic depth.

dropout_rate (`float`, *optional*, defaults to 0.0) : The dropout probability for dropout.

## VanModel[[transformers.VanModel]]

#### transformers.VanModel[[transformers.VanModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/van/modeling_van.py#L411)

The bare VAN model outputting raw features without any specific head on top. Note, VAN does not have an embedding layer.
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.VanModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/van/modeling_van.py#L421[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor]"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoImageProcessor). See
  [ConvNextImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.

- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all stages. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VanConfig](/docs/transformers/main/en/model_doc/van#transformers.VanConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state after a pooling operation on the spatial dimensions.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
The [VanModel](/docs/transformers/main/en/model_doc/van#transformers.VanModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, VanModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("Visual-Attention-Network/van-base")
>>> model = VanModel.from_pretrained("Visual-Attention-Network/van-base")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 512, 7, 7]
```

**Parameters:**

config ([VanConfig](/docs/transformers/main/en/model_doc/van#transformers.VanConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)``

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VanConfig](/docs/transformers/main/en/model_doc/van#transformers.VanConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state after a pooling operation on the spatial dimensions.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

## VanForImageClassification[[transformers.VanForImageClassification]]

#### transformers.VanForImageClassification[[transformers.VanForImageClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/van/modeling_van.py#L466)

VAN Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.

forwardtransformers.VanForImageClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/van/modeling_van.py#L478[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) --
  Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoImageProcessor). See
  [ConvNextImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details.

- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all stages. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).0[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VanConfig](/docs/transformers/main/en/model_doc/van#transformers.VanConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
The [VanForImageClassification](/docs/transformers/main/en/model_doc/van#transformers.VanForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, VanForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("Visual-Attention-Network/van-base")
>>> model = VanForImageClassification.from_pretrained("Visual-Attention-Network/van-base")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

**Parameters:**

config ([VanConfig](/docs/transformers/main/en/model_doc/van#transformers.VanConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VanConfig](/docs/transformers/main/en/model_doc/van#transformers.VanConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
