*This model was released on 2022-06-06 and added to Hugging Face Transformers on 2023-06-02.*

# MobileViTV2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The MobileViTV2 model was proposed in [Separable Self-attention for Mobile Vision Transformers](https://huggingface.co/papers/2206.02680) by Sachin Mehta and Mohammad Rastegari.

MobileViTV2 is the second version of MobileViT, constructed by replacing the multi-headed self-attention in MobileViT with separable self-attention.

The abstract from the paper is the following:

*Mobile vision transformers (MobileViT) can achieve state-of-the-art performance across several mobile vision tasks, including classification and detection. Though these models have fewer parameters, they have high latency as compared to convolutional neural network-based models. The main efficiency bottleneck in MobileViT is the multi-headed self-attention (MHA) in transformers, which requires O(k2) time complexity with respect to the number of tokens (or patches) k. Moreover, MHA requires costly operations (e.g., batch-wise matrix multiplication) for computing self-attention, impacting latency on resource-constrained devices. This paper introduces a separable self-attention method with linear complexity, i.e. O(k). A simple yet effective characteristic of the proposed method is that it uses element-wise operations for computing self-attention, making it a good choice for resource-constrained devices. The improved model, MobileViTV2, is state-of-the-art on several mobile vision tasks, including ImageNet object classification and MS-COCO object detection. With about three million parameters, MobileViTV2 achieves a top-1 accuracy of 75.6% on the ImageNet dataset, outperforming MobileViT by about 1% while running 3.2× faster on a mobile device.*

This model was contributed by [shehan97](https://huggingface.co/shehan97).
The original code can be found [here](https://github.com/apple/ml-cvnets).

## Usage tips

* MobileViTV2 is more like a CNN than a Transformer model. It does not work on sequence data but on batches of images. Unlike ViT, there are no embeddings. The backbone model outputs a feature map.
* One can use [MobileViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) to prepare images for the model. Note that if you do your own preprocessing, the pretrained checkpoints expect images to be in BGR pixel order (not RGB).
* The available image classification checkpoints are pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes).
* The segmentation model uses a [DeepLabV3](https://huggingface.co/papers/1706.05587) head. The available semantic segmentation checkpoints are pre-trained on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

## MobileViTV2Config

### class transformers.MobileViTV2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevitv2/configuration_mobilevitv2.py#L30)

( num\_channels = 3 image\_size = 256 patch\_size = 2 expand\_ratio = 2.0 hidden\_act = 'swish' conv\_kernel\_size = 3 output\_stride = 32 classifier\_dropout\_prob = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 aspp\_out\_channels = 512 atrous\_rates = [6, 12, 18] aspp\_dropout\_prob = 0.1 semantic\_loss\_ignore\_index = 255 n\_attn\_blocks = [2, 4, 3] base\_attn\_unit\_dims = [128, 192, 256] width\_multiplier = 1.0 ffn\_multiplier = 2 attn\_dropout = 0.0 ffn\_dropout = 0.0 \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **image\_size** (`int`, *optional*, defaults to 256) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 2) —
  The size (resolution) of each patch.
* **expand\_ratio** (`float`, *optional*, defaults to 2.0) —
  Expansion factor for the MobileNetv2 layers.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"swish"`) —
  The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
* **conv\_kernel\_size** (`int`, *optional*, defaults to 3) —
  The size of the convolutional kernel in the MobileViTV2 layer.
* **output\_stride** (`int`, *optional*, defaults to 32) —
  The ratio of the spatial resolution of the output to the resolution of the input image.
* **classifier\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for attached classifiers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **aspp\_out\_channels** (`int`, *optional*, defaults to 512) —
  Number of output channels used in the ASPP layer for semantic segmentation.
* **atrous\_rates** (`list[int]`, *optional*, defaults to `[6, 12, 18]`) —
  Dilation (atrous) factors used in the ASPP layer for semantic segmentation.
* **aspp\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the ASPP layer for semantic segmentation.
* **semantic\_loss\_ignore\_index** (`int`, *optional*, defaults to 255) —
  The index that is ignored by the loss function of the semantic segmentation model.
* **n\_attn\_blocks** (`list[int]`, *optional*, defaults to `[2, 4, 3]`) —
  The number of attention blocks in each MobileViTV2Layer
* **base\_attn\_unit\_dims** (`list[int]`, *optional*, defaults to `[128, 192, 256]`) —
  The base multiplier for dimensions of attention blocks in each MobileViTV2Layer
* **width\_multiplier** (`float`, *optional*, defaults to 1.0) —
  The width multiplier for MobileViTV2.
* **ffn\_multiplier** (`int`, *optional*, defaults to 2) —
  The FFN multiplier for MobileViTV2.
* **attn\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout in the attention layer.
* **ffn\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout between FFN layers.

This is the configuration class to store the configuration of a [MobileViTV2Model](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Model). It is used to instantiate a
MobileViTV2 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the MobileViTV2
[apple/mobilevitv2-1.0](https://huggingface.co/apple/mobilevitv2-1.0) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MobileViTV2Config, MobileViTV2Model

>>> # Initializing a mobilevitv2-small style configuration
>>> configuration = MobileViTV2Config()

>>> # Initializing a model from the mobilevitv2-small style configuration
>>> model = MobileViTV2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MobileViTV2Model

### class transformers.MobileViTV2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py#L593)

( config: MobileViTV2Config expand\_output: bool = True  )

Parameters

* **config** ([MobileViTV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **expand\_output** (`bool`, *optional*, defaults to `True`) —
  Whether to expand the output of the model. If `True`, the model will output pooled features in addition to
  hidden states. If `False`, only the hidden states will be returned.

The bare Mobilevitv2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py#L632)

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
elements depending on the configuration ([MobileViTV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state after a pooling operation on the spatial dimensions.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [MobileViTV2Model](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## MobileViTV2ForImageClassification

### class transformers.MobileViTV2ForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py#L681)

( config: MobileViTV2Config  )

Parameters

* **config** ([MobileViTV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MobileViTV2 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py#L699)

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
elements depending on the configuration ([MobileViTV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.

The [MobileViTV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, MobileViTV2ForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevitv2-1.0")
>>> model = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

## MobileViTV2ForSemanticSegmentation

### class transformers.MobileViTV2ForSemanticSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py#L874)

( config: MobileViTV2Config  )

Parameters

* **config** ([MobileViTV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MobileViTV2 model with a semantic segmentation head on top, e.g. for Pascal VOC.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py#L885)

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
elements depending on the configuration ([MobileViTV2Config](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2Config)) and inputs.

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

The [MobileViTV2ForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/mobilevitv2#transformers.MobileViTV2ForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import requests
>>> import torch
>>> from PIL import Image
>>> from transformers import AutoImageProcessor, MobileViTV2ForSemanticSegmentation

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
>>> model = MobileViTV2ForSemanticSegmentation.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # logits are of shape (batch_size, num_labels, height, width)
>>> logits = outputs.logits
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mobilevitv2.md)
