# Backbones

Higher-level computer visions tasks, such as object detection or image segmentation, use several models together to generate a prediction. A separate model is used for the *backbone*, neck, and head. The backbone extracts useful features from an input image into a feature map, the neck combines and processes the feature maps, and the head uses them to make a prediction.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Backbone.png)

Load a backbone with [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained) and use the `out_indices` parameter to determine which layer, given by the index, to extract a feature map from.


```
from transformers import AutoBackbone

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
```

This guide describes the backbone class, backbones from the [timm](https://hf.co/docs/timm/index) library, and how to extract features with them.

## Backbone classes

There are two backbone classes.

* [BackboneMixin](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.utils.BackboneMixin) allows you to load a backbone and includes functions for extracting the feature maps and indices.
* [BackboneConfigMixin](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.utils.BackboneConfigMixin) allows you to set the feature map and indices of a backbone configuration.

Refer to the [Backbone](./main_classes/backbones) API documentation to check which models support a backbone.

There are two ways to load a Transformers backbone, [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone) and a model-specific backbone class.

AutoBackbone

model-specific backbone

The [AutoClass](./model_doc/auto) API automatically loads a pretrained vision model with [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained) as a backbone if it’s supported.

Set the `out_indices` parameter to the layer you’d like to get the feature map from. If you know the name of the layer, you could also use `out_features`. These parameters can be used interchangeably, but if you use both, make sure they refer to the same layer.

When `out_indices` or `out_features` isn’t used, the backbone returns the feature map from the last layer. The example code below uses `out_indices=(1,)` to get the feature map from the first layer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Swin%20Stage%201.png)


```
from transformers import AutoImageProcessor, AutoBackbone

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
```

## timm backbones

[timm](https://hf.co/docs/timm/index) is a collection of vision models for training and inference. Transformers supports timm models as backbones with the [TimmBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.TimmBackbone) and [TimmBackboneConfig](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.TimmBackboneConfig) classes.

Set `use_timm_backbone=True` to load pretrained timm weights, and `use_pretrained_backbone` to use pretrained or randomly initialized weights.


```
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="resnet50", use_timm_backbone=True, use_pretrained_backbone=True)
model = MaskFormerForInstanceSegmentation(config)
```

You could also explicitly call the [TimmBackboneConfig](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.TimmBackboneConfig) class to load and create a pretrained timm backbone.


```
from transformers import TimmBackboneConfig

backbone_config = TimmBackboneConfig("resnet50", use_pretrained_backbone=True)
```

Pass the backbone configuration to the model configuration and instantiate the model head, [MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation), with the backbone.


```
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)
```

## Feature extraction

The backbone is used to extract image features. Pass an image through the backbone to get the feature maps.

Load and preprocess an image and pass it to the backbone. The example below extracts the feature maps from the first layer.


```
from transformers import AutoImageProcessor, AutoBackbone
import torch
from PIL import Image
import requests

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)
```

The features are stored and accessed from the outputs `feature_maps` attribute.


```
feature_maps = outputs.feature_maps
list(feature_maps[0].shape)
[1, 96, 56, 56]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/backbones.md)
