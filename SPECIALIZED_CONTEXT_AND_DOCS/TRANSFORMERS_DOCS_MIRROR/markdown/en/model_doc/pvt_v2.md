*This model was released on 2021-06-25 and added to Hugging Face Transformers on 2024-03-13.*

# Pyramid Vision Transformer V2 (PVTv2)

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The PVTv2 model was proposed in
[PVT v2: Improved Baselines with Pyramid Vision Transformer](https://huggingface.co/papers/2106.13797) by Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. As an improved variant of PVT, it eschews position embeddings, relying instead on positional information encoded through zero-padding and overlapping patch embeddings. This lack of reliance on position embeddings simplifies the architecture, and enables running inference at any resolution without needing to interpolate them.

The PVTv2 encoder structure has been successfully deployed to achieve state-of-the-art scores in [Segformer](https://huggingface.co/papers/2105.15203) for semantic segmentation, [GLPN](https://huggingface.co/papers/2201.07436) for monocular depth, and [Panoptic Segformer](https://huggingface.co/papers/2109.03814) for panoptic segmentation.

PVTv2 belongs to a family of models called [hierarchical transformers](https://natecibik.medium.com/the-rise-of-vision-transformers-f623c980419f) , which make adaptations to transformer layers in order to generate multi-scale feature maps. Unlike the columnal structure of Vision Transformer ([ViT](https://huggingface.co/papers/2010.11929)) which loses fine-grained detail, multi-scale feature maps are known preserve this detail and aid performance in dense prediction tasks. In the case of PVTv2, this is achieved by generating image patch tokens using 2D convolution with overlapping kernels in each encoder layer.

The multi-scale features of hierarchical transformers allow them to be easily swapped in for traditional workhorse computer vision backbone models like ResNet in larger architectures. Both Segformer and Panoptic Segformer demonstrated that configurations using PVTv2 for a backbone consistently outperformed those with similarly sized ResNet backbones.

Another powerful feature of the PVTv2 is the complexity reduction in the self-attention layers called Spatial Reduction Attention (SRA), which uses 2D convolution layers to project hidden states to a smaller resolution before attending to them with the queries, improving the $O(n^2)$ complexity of self-attention to $O(n^2/R)$, with $R$ being the spatial reduction ratio (`sr_ratio`, aka kernel size and stride in the 2D convolution).

SRA was introduced in PVT, and is the default attention complexity reduction method used in PVTv2. However, PVTv2 also introduced the option of using a self-attention mechanism with linear complexity related to image size, which they called “Linear SRA”. This method uses average pooling to reduce the hidden states to a fixed size that is invariant to their original resolution (although this is inherently more lossy than regular SRA). This option can be enabled by setting `linear_attention` to `True` in the PVTv2Config.

### Abstract from the paper:

*Transformer recently has presented encouraging progress in computer vision. In this work, we present new baselines by improving the original Pyramid Vision Transformer (PVT v1) by adding three designs, including (1) linear complexity attention layer, (2) overlapping patch embedding, and (3) convolutional feed-forward network. With these modifications, PVT v2 reduces the computational complexity of PVT v1 to linear and achieves significant improvements on fundamental vision tasks such as classification, detection, and segmentation. Notably, the proposed PVT v2 achieves comparable or better performances than recent works such as Swin Transformer. We hope this work will facilitate state-of-the-art Transformer researches in computer vision. Code is available at <https://github.com/whai362/PVT>.*

This model was contributed by [FoamoftheSea](https://huggingface.co/FoamoftheSea). The original code can be found [here](https://github.com/whai362/PVT).

## Usage tips

* [PVTv2](https://huggingface.co/papers/2106.13797) is a hierarchical transformer model which has demonstrated powerful performance in image classification and multiple other tasks, used as a backbone for semantic segmentation in [Segformer](https://huggingface.co/papers/2105.15203), monocular depth estimation in [GLPN](https://huggingface.co/papers/2201.07436), and panoptic segmentation in [Panoptic Segformer](https://huggingface.co/papers/2109.03814), consistently showing higher performance than similar ResNet configurations.
* Hierarchical transformers like PVTv2 achieve superior data and parameter efficiency on image data compared with pure transformer architectures by incorporating design elements of convolutional neural networks (CNNs) into their encoders. This creates a best-of-both-worlds architecture that infuses the useful inductive biases of CNNs like translation equivariance and locality into the network while still enjoying the benefits of dynamic data response and global relationship modeling provided by the self-attention mechanism of [transformers](https://huggingface.co/papers/1706.03762).
* PVTv2 uses overlapping patch embeddings to create multi-scale feature maps, which are infused with location information using zero-padding and depth-wise convolutions.
* To reduce the complexity in the attention layers, PVTv2 performs a spatial reduction on the hidden states using either strided 2D convolution (SRA) or fixed-size average pooling (Linear SRA). Although inherently more lossy, Linear SRA provides impressive performance with a linear complexity with respect to image size. To use Linear SRA in the self-attention layers, set `linear_attention=True` in the `PvtV2Config`.
* [PvtV2Model](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Model) is the hierarchical transformer encoder (which is also often referred to as Mix Transformer or MiT in the literature). [PvtV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2ForImageClassification) adds a simple classifier head on top to perform Image Classification. `PvtV2Backbone` can be used with the [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone) system in larger architectures like Deformable DETR.
* ImageNet pretrained weights for all model sizes can be found on the [hub](https://huggingface.co/models?other=pvt_v2).

  The best way to get started with the PVTv2 is to load the pretrained checkpoint with the size of your choosing using `AutoModelForImageClassification`:


```
import requests
import torch

from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image

model = AutoModelForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0")
image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
processed = image_processor(image)
outputs = model(torch.tensor(processed["pixel_values"]))
```

To use the PVTv2 as a backbone for more complex architectures like DeformableDETR, you can use AutoBackbone (this model would need fine-tuning as you’re replacing the backbone in the pretrained model):


```
import requests
import torch

from transformers import AutoConfig, AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image

model = AutoModelForObjectDetection.from_config(
    config=AutoConfig.from_pretrained(
        "SenseTime/deformable-detr",
        backbone_config=AutoConfig.from_pretrained("OpenGVLab/pvt_v2_b5"),
        use_timm_backbone=False
    ),
)

image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
processed = image_processor(image)
outputs = model(torch.tensor(processed["pixel_values"]))
```

[PVTv2](https://github.com/whai362/PVT/tree/v2) performance on ImageNet-1K by model size (B0-B5):

| Method | Size | Acc@1 | #Params (M) |
| --- | --- | --- | --- |
| PVT-V2-B0 | 224 | 70.5 | 3.7 |
| PVT-V2-B1 | 224 | 78.7 | 14.0 |
| PVT-V2-B2-Linear | 224 | 82.1 | 22.6 |
| PVT-V2-B2 | 224 | 82.0 | 25.4 |
| PVT-V2-B3 | 224 | 83.1 | 45.2 |
| PVT-V2-B4 | 224 | 83.6 | 62.6 |
| PVT-V2-B5 | 224 | 83.8 | 82.0 |

## PvtV2Config

### class transformers.PvtV2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt_v2/configuration_pvt_v2.py#L29)

( image\_size: typing.Union[int, tuple[int, int]] = 224 num\_channels: int = 3 num\_encoder\_blocks: int = 4 depths: list = [2, 2, 2, 2] sr\_ratios: list = [8, 4, 2, 1] hidden\_sizes: list = [32, 64, 160, 256] patch\_sizes: list = [7, 3, 3, 3] strides: list = [4, 2, 2, 2] num\_attention\_heads: list = [1, 2, 5, 8] mlp\_ratios: list = [8, 8, 4, 4] hidden\_act: typing.Union[str, typing.Callable] = 'gelu' hidden\_dropout\_prob: float = 0.0 attention\_probs\_dropout\_prob: float = 0.0 initializer\_range: float = 0.02 drop\_path\_rate: float = 0.0 layer\_norm\_eps: float = 1e-06 qkv\_bias: bool = True linear\_attention: bool = False out\_features = None out\_indices = None \*\*kwargs  )

Parameters

* **image\_size** (`Union[int, tuple[int, int]]`, *optional*, defaults to 224) —
  The input image size. Pass int value for square image, or tuple of (height, width).
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **num\_encoder\_blocks** (`[int]`, *optional*, defaults to 4) —
  The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
* **depths** (`list[int]`, *optional*, defaults to `[2, 2, 2, 2]`) —
  The number of layers in each encoder block.
* **sr\_ratios** (`list[int]`, *optional*, defaults to `[8, 4, 2, 1]`) —
  Spatial reduction ratios in each encoder block.
* **hidden\_sizes** (`list[int]`, *optional*, defaults to `[32, 64, 160, 256]`) —
  Dimension of each of the encoder blocks.
* **patch\_sizes** (`list[int]`, *optional*, defaults to `[7, 3, 3, 3]`) —
  Patch size for overlapping patch embedding before each encoder block.
* **strides** (`list[int]`, *optional*, defaults to `[4, 2, 2, 2]`) —
  Stride for overlapping patch embedding before each encoder block.
* **num\_attention\_heads** (`list[int]`, *optional*, defaults to `[1, 2, 5, 8]`) —
  Number of attention heads for each attention layer in each block of the Transformer encoder.
* **mlp\_ratios** (`list[int]`, *optional*, defaults to `[8, 8, 4, 4]`) —
  Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
  encoder blocks.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"gelu"`) —
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
* **linear\_attention** (`bool`, *optional*, defaults to `False`) —
  Use linear attention complexity. If set to True, `sr_ratio` is ignored and average pooling is used for
  dimensionality reduction in the attention layers rather than strided convolution.
* **out\_features** (`list[str]`, *optional*) —
  If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
  (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
  corresponding stages. If unset and `out_indices` is unset, will default to the last stage.
* **out\_indices** (`list[int]`, *optional*) —
  If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
  many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
  If unset and `out_features` is unset, will default to the last stage.

This is the configuration class to store the configuration of a [PvtV2Model](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Model). It is used to instantiate a Pvt V2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Pvt V2 B0
[OpenGVLab/pvt\_v2\_b0](https://huggingface.co/OpenGVLab/pvt_v2_b0) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import PvtV2Model, PvtV2Config

>>> # Initializing a pvt_v2_b0 style configuration
>>> configuration = PvtV2Config()

>>> # Initializing a model from the OpenGVLab/pvt_v2_b0 style configuration
>>> model = PvtV2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## PvtForImageClassification

### class transformers.PvtV2ForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt_v2/modeling_pvt_v2.py#L473)

( config: PvtV2Config  )

Parameters

* **config** ([PvtV2Config](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Pvt-v2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state
of the [CLS] token) e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt_v2/modeling_pvt_v2.py#L488)

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
elements depending on the configuration ([PvtV2Config](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [PvtV2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, PvtV2ForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
>>> model = PvtV2ForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

## PvtModel

### class transformers.PvtV2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt_v2/modeling_pvt_v2.py#L416)

( config: PvtV2Config  )

Parameters

* **config** ([PvtV2Config](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Pvt V2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pvt_v2/modeling_pvt_v2.py#L435)

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
elements depending on the configuration ([PvtV2Config](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [PvtV2Model](/docs/transformers/v4.56.2/en/model_doc/pvt_v2#transformers.PvtV2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/pvt_v2.md)
