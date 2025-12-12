# Backbone

A backbone is a model used for feature extraction for higher level computer vision tasks such as object detection and image classification. Transformers provides an [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone) class for initializing a Transformers backbone from pretrained model weights, and two utility classes:

* [BackboneMixin](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.utils.BackboneMixin) enables initializing a backbone from Transformers or [timm](https://hf.co/docs/timm/index) and includes functions for returning the output features and indices.
* [BackboneConfigMixin](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.utils.BackboneConfigMixin) sets the output features and indices of the backbone configuration.

[timm](https://hf.co/docs/timm/index) models are loaded with the [TimmBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.TimmBackbone) and [TimmBackboneConfig](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.TimmBackboneConfig) classes.

Backbones are supported for the following models:

* [BEiT](../model_doc/beit)
* [BiT](../model_doc/bit)
* [ConvNext](../model_doc/convnext)
* [ConvNextV2](../model_doc/convnextv2)
* [DiNAT](../model_doc/dinat)
* [DINOV2](../model_doc/dinov2)
* [FocalNet](../model_doc/focalnet)
* [MaskFormer](../model_doc/maskformer)
* [NAT](../model_doc/nat)
* [ResNet](../model_doc/resnet)
* [Swin Transformer](../model_doc/swin)
* [Swin Transformer v2](../model_doc/swinv2)
* [ViTDet](../model_doc/vitdet)

## AutoBackbone

### class transformers.AutoBackbone

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/auto/modeling_auto.py#L2185)

( \*args \*\*kwargs  )

## BackboneMixin

### class transformers.utils.BackboneMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/backbone_utils.py#L140)

( )

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/backbone_utils.py#L253)

( )

Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig` to
include the `out_features` and `out_indices` attributes.

## BackboneConfigMixin

### class transformers.utils.BackboneConfigMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/backbone_utils.py#L264)

( )

A Mixin to support handling the `out_features` and `out_indices` attributes for the backbone configurations.

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/backbone_utils.py#L295)

( )

Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig` to
include the `out_features` and `out_indices` attributes.

## TimmBackbone

### class transformers.TimmBackbone

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timm_backbone/modeling_timm_backbone.py#L35)

( config \*\*kwargs  )

Wrapper class for timm models to be used as backbones. This enables using the timm models interchangeably with the
other models in the library keeping the same API.

## TimmBackboneConfig

### class transformers.TimmBackboneConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timm_backbone/configuration_timm_backbone.py#L25)

( backbone = None num\_channels = 3 features\_only = True use\_pretrained\_backbone = True out\_indices = None freeze\_batch\_norm\_2d = False \*\*kwargs  )

Parameters

* **backbone** (`str`, *optional*) —
  The timm checkpoint to load.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **features\_only** (`bool`, *optional*, defaults to `True`) —
  Whether to output only the features or also the logits.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `True`) —
  Whether to use a pretrained backbone.
* **out\_indices** (`list[int]`, *optional*) —
  If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
  many stages the model has). Will default to the last stage if unset.
* **freeze\_batch\_norm\_2d** (`bool`, *optional*, defaults to `False`) —
  Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`.

This is the configuration class to store the configuration for a timm backbone [TimmBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.TimmBackbone).

It is used to instantiate a timm backbone model according to the specified arguments, defining the model.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import TimmBackboneConfig, TimmBackbone

>>> # Initializing a timm backbone
>>> configuration = TimmBackboneConfig("resnet50")

>>> # Initializing a model from the configuration
>>> model = TimmBackbone(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/backbones.md)
