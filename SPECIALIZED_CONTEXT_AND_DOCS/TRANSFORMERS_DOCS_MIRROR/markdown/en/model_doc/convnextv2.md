*This model was released on 2023-01-02 and added to Hugging Face Transformers on 2023-03-14.*

# ConvNeXt V2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The ConvNeXt V2 model was proposed in [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://huggingface.co/papers/2301.00808) by Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
ConvNeXt V2 is a pure convolutional model (ConvNet), inspired by the design of Vision Transformers, and a successor of [ConvNeXT](convnext).

The abstract from the paper is the following:

*Driven by improved architectures and better representation learning frameworks, the field of visual recognition has enjoyed rapid modernization and performance boost in the early 2020s. For example, modern ConvNets, represented by ConvNeXt, have demonstrated strong performance in various scenarios. While these models were originally designed for supervised learning with ImageNet labels, they can also potentially benefit from self-supervised learning techniques such as masked autoencoders (MAE). However, we found that simply combining these two approaches leads to subpar performance. In this paper, we propose a fully convolutional masked autoencoder framework and a new Global Response Normalization (GRN) layer that can be added to the ConvNeXt architecture to enhance inter-channel feature competition. This co-design of self-supervised learning techniques and architectural improvement results in a new model family called ConvNeXt V2, which significantly improves the performance of pure ConvNets on various recognition benchmarks, including ImageNet classification, COCO detection, and ADE20K segmentation. We also provide pre-trained ConvNeXt V2 models of various sizes, ranging from an efficient 3.7M-parameter Atto model with 76.7% top-1 accuracy on ImageNet, to a 650M Huge model that achieves a state-of-the-art 88.9% accuracy using only public training data.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnextv2_architecture.png) ConvNeXt V2 architecture. Taken from the [original paper](https://huggingface.co/papers/2301.00808).

This model was contributed by [adirik](https://huggingface.co/adirik). The original code can be found [here](https://github.com/facebookresearch/ConvNeXt-V2).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ConvNeXt V2.

Image Classification

* [ConvNextV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2ForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## ConvNextV2Config

### class transformers.ConvNextV2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/convnextv2/configuration_convnextv2.py#L25)

( num\_channels = 3 patch\_size = 4 num\_stages = 4 hidden\_sizes = None depths = None hidden\_act = 'gelu' initializer\_range = 0.02 layer\_norm\_eps = 1e-12 drop\_path\_rate = 0.0 image\_size = 224 out\_features = None out\_indices = None \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **patch\_size** (`int`, *optional*, defaults to 4) —
  Patch size to use in the patch embedding layer.
* **num\_stages** (`int`, *optional*, defaults to 4) —
  The number of stages in the model.
* **hidden\_sizes** (`list[int]`, *optional*, defaults to `[96, 192, 384, 768]`) —
  Dimensionality (hidden size) at each stage.
* **depths** (`list[int]`, *optional*, defaults to `[3, 3, 9, 3]`) —
  Depth (number of blocks) for each stage.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
  `"selu"` and `"gelu_new"` are supported.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.0) —
  The drop rate for stochastic depth.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **out\_features** (`list[str]`, *optional*) —
  If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
  (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
  corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
  same order as defined in the `stage_names` attribute.
* **out\_indices** (`list[int]`, *optional*) —
  If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
  many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
  If unset and `out_features` is unset, will default to the last stage. Must be in the
  same order as defined in the `stage_names` attribute.

This is the configuration class to store the configuration of a [ConvNextV2Model](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2Model). It is used to instantiate an
ConvNeXTV2 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the ConvNeXTV2
[facebook/convnextv2-tiny-1k-224](https://huggingface.co/facebook/convnextv2-tiny-1k-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import ConvNeXTV2Config, ConvNextV2Model

>>> # Initializing a ConvNeXTV2 convnextv2-tiny-1k-224 style configuration
>>> configuration = ConvNeXTV2Config()

>>> # Initializing a model (with random weights) from the convnextv2-tiny-1k-224 style configuration
>>> model = ConvNextV2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ConvNextV2Model

### class transformers.ConvNextV2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/convnextv2/modeling_convnextv2.py#L286)

( config  )

Parameters

* **config** ([ConvNextV2Model](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2Model)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Convnextv2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/convnextv2/modeling_convnextv2.py#L300)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_hidden\_states: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor). See [ConvNextImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor) for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ConvNextV2Config](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state after a pooling operation on the spatial dimensions.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [ConvNextV2Model](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## ConvNextV2ForImageClassification

### class transformers.ConvNextV2ForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/convnextv2/modeling_convnextv2.py#L334)

( config  )

Parameters

* **config** ([ConvNextV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2ForImageClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ConvNextV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/convnextv2/modeling_convnextv2.py#L350)

( pixel\_values: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None \*\*kwargs  ) → [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor). See [ConvNextImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

[transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ConvNextV2Config](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.

The [ConvNextV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/convnextv2#transformers.ConvNextV2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
>>> model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/convnextv2.md)
