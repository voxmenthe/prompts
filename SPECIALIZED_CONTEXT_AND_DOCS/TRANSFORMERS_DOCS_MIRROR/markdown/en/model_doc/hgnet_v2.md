*This model was released on 2024-07-01 and added to Hugging Face Transformers on 2025-04-29.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# HGNet-V2

[HGNetV2](https://github.com/PaddlePaddle/PaddleClas/blob/v2.6.0/docs/zh_CN/models/ImageNet1k/PP-HGNetV2.md) is a next-generation convolutional neural network (CNN) backbone built for optimal accuracy-latency tradeoff on NVIDIA GPUs. Building on the original[HGNet](https://github.com/PaddlePaddle/PaddleClas/blob/v2.6.0/docs/en/models/PP-HGNet_en.md), HGNetV2 delivers high accuracy at fast inference speeds and performs strongly on tasks like image classification, object detection, and segmentation, making it a practical choice for GPU-based computer vision applications.

You can find all the original HGNet V2 models under the [USTC](https://huggingface.co/ustc-community/models?search=hgnet) organization.

This model was contributed by [VladOS95-cyber](https://github.com/VladOS95-cyber).
Click on the HGNet V2 models in the right sidebar for more examples of how to apply HGNet V2 to different computer vision tasks.

The example below demonstrates how to classify an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="ustc-community/hgnet-v2",
    dtype=torch.float16,
    device=0
)
pipeline("http://images.cocodataset.org/val2017/000000039769.jpg")
```

## HGNetV2Config

### class transformers.HGNetV2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hgnet_v2/configuration_hgnet_v2.py#L29)

( num\_channels = 3 embedding\_size = 64 depths = [3, 4, 6, 3] hidden\_sizes = [256, 512, 1024, 2048] hidden\_act = 'relu' out\_features = None out\_indices = None stem\_channels = [3, 32, 48] stage\_in\_channels = [48, 128, 512, 1024] stage\_mid\_channels = [48, 96, 192, 384] stage\_out\_channels = [128, 512, 1024, 2048] stage\_num\_blocks = [1, 1, 3, 1] stage\_downsample = [False, True, True, True] stage\_light\_block = [False, False, True, True] stage\_kernel\_size = [3, 3, 5, 5] stage\_numb\_of\_layers = [6, 6, 6, 6] use\_learnable\_affine\_block = False initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **embedding\_size** (`int`, *optional*, defaults to 64) —
  Dimensionality (hidden size) for the embedding layer.
* **depths** (`list[int]`, *optional*, defaults to `[3, 4, 6, 3]`) —
  Depth (number of layers) for each stage.
* **hidden\_sizes** (`list[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`) —
  Dimensionality (hidden size) at each stage.
* **hidden\_act** (`str`, *optional*, defaults to `"relu"`) —
  The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
  are supported.
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
* **stem\_channels** (`list[int]`, *optional*, defaults to `[3, 32, 48]`) —
  Channel dimensions for the stem layers:
  + First number (3) is input image channels
  + Second number (32) is intermediate stem channels
  + Third number (48) is output stem channels
* **stage\_in\_channels** (`list[int]`, *optional*, defaults to `[48, 128, 512, 1024]`) —
  Input channel dimensions for each stage of the backbone.
  This defines how many channels the input to each stage will have.
* **stage\_mid\_channels** (`list[int]`, *optional*, defaults to `[48, 96, 192, 384]`) —
  Mid-channel dimensions for each stage of the backbone.
  This defines the number of channels used in the intermediate layers of each stage.
* **stage\_out\_channels** (`list[int]`, *optional*, defaults to `[128, 512, 1024, 2048]`) —
  Output channel dimensions for each stage of the backbone.
  This defines how many channels the output of each stage will have.
* **stage\_num\_blocks** (`list[int]`, *optional*, defaults to `[1, 1, 3, 1]`) —
  Number of blocks to be used in each stage of the backbone.
  This controls the depth of each stage by specifying how many convolutional blocks to stack.
* **stage\_downsample** (`list[bool]`, *optional*, defaults to `[False, True, True, True]`) —
  Indicates whether to downsample the feature maps at each stage.
  If `True`, the spatial dimensions of the feature maps will be reduced.
* **stage\_light\_block** (`list[bool]`, *optional*, defaults to `[False, False, True, True]`) —
  Indicates whether to use light blocks in each stage.
  Light blocks are a variant of convolutional blocks that may have fewer parameters.
* **stage\_kernel\_size** (`list[int]`, *optional*, defaults to `[3, 3, 5, 5]`) —
  Kernel sizes for the convolutional layers in each stage.
* **stage\_numb\_of\_layers** (`list[int]`, *optional*, defaults to `[6, 6, 6, 6]`) —
  Number of layers to be used in each block of the stage.
* **use\_learnable\_affine\_block** (`bool`, *optional*, defaults to `False`) —
  Whether to use Learnable Affine Blocks (LAB) in the network.
  LAB adds learnable scale and bias parameters after certain operations.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a [HGNetV2Backbone](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2Backbone). It is used to instantiate a HGNet-V2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of D-FINE-X-COCO B4 ”[ustc-community/dfine\_x\_coco”](https://huggingface.co/ustc-community/dfine_x_coco%22).
Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## HGNetV2Backbone

### class transformers.HGNetV2Backbone

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hgnet_v2/modeling_hgnet_v2.py#L334)

( config: HGNetV2Config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hgnet_v2/modeling_hgnet_v2.py#L348)

( pixel\_values: Tensor output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.BackboneOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.BackboneOutput` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BackboneOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HGNetV2Config](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2Config)) and inputs.

* **feature\_maps** (`tuple(torch.FloatTensor)` of shape `(batch_size, num_channels, height, width)`) — Feature maps of the stages.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, num_channels, height, width)`,
  depending on the backbone.

  Hidden-states of the model at the output of each stage plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Only applicable if the backbone uses attention.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [HGNetV2Backbone](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2Backbone) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import HGNetV2Config, HGNetV2Backbone
>>> import torch

>>> config = HGNetV2Config()
>>> model = HGNetV2Backbone(config)

>>> pixel_values = torch.randn(1, 3, 224, 224)

>>> with torch.no_grad():
...     outputs = model(pixel_values)

>>> feature_maps = outputs.feature_maps
>>> list(feature_maps[-1].shape)
[1, 2048, 7, 7]
```

## HGNetV2ForImageClassification

### class transformers.HGNetV2ForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hgnet_v2/modeling_hgnet_v2.py#L406)

( config: HGNetV2Config  )

Parameters

* **config** ([HGNetV2Config](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

HGNetV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hgnet_v2/modeling_hgnet_v2.py#L422)

( pixel\_values: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([HGNetV2Config](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.

The [HGNetV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/hgnet_v2#transformers.HGNetV2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> import requests
>>> from transformers import HGNetV2ForImageClassification, AutoImageProcessor
>>> from PIL import Image

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> model = HGNetV2ForImageClassification.from_pretrained("ustc-community/hgnet-v2")
>>> processor = AutoImageProcessor.from_pretrained("ustc-community/hgnet-v2")

>>> inputs = processor(images=image, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> outputs.logits.shape
torch.Size([1, 2])
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/hgnet_v2.md)
