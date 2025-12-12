*This model was released on 2018-07-26 and added to Hugging Face Transformers on 2023-01-16.*

# UPerNet

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The UPerNet model was proposed in [Unified Perceptual Parsing for Scene Understanding](https://huggingface.co/papers/1807.10221)
by Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, Jian Sun. UPerNet is a general framework to effectively segment
a wide range of concepts from images, leveraging any vision backbone like [ConvNeXt](convnext) or [Swin](swin).

The abstract from the paper is the following:

*Humans recognize the visual world at multiple levels: we effortlessly categorize scenes and detect objects inside, while also identifying the textures and surfaces of the objects along with their different compositional parts. In this paper, we study a new task called Unified Perceptual Parsing, which requires the machine vision systems to recognize as many visual concepts as possible from a given image. A multi-task framework called UPerNet and a training strategy are developed to learn from heterogeneous image annotations. We benchmark our framework on Unified Perceptual Parsing and show that it is able to effectively segment a wide range of concepts from images. The trained networks are further applied to discover visual knowledge in natural scenes.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/upernet_architecture.jpg) UPerNet framework. Taken from the [original paper](https://huggingface.co/papers/1807.10221).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code is based on OpenMMLab’s mmsegmentation [here](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/uper_head.py).

## Usage examples

UPerNet is a general framework for semantic segmentation. It can be used with any vision backbone, like so:


```
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation

backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

config = UperNetConfig(backbone_config=backbone_config)
model = UperNetForSemanticSegmentation(config)
```

To use another vision backbone, like [ConvNeXt](convnext), simply instantiate the model with the appropriate backbone:


```
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation

backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

config = UperNetConfig(backbone_config=backbone_config)
model = UperNetForSemanticSegmentation(config)
```

Note that this will randomly initialize all the weights of the model.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with UPerNet.

* Demo notebooks for UPerNet can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/UPerNet).
* [UperNetForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb).
* See also: [Semantic segmentation task guide](../tasks/semantic_segmentation)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## UperNetConfig

### class transformers.UperNetConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/upernet/configuration_upernet.py#L26)

( backbone\_config = None backbone = None use\_pretrained\_backbone = False use\_timm\_backbone = False backbone\_kwargs = None hidden\_size = 512 initializer\_range = 0.02 pool\_scales = [1, 2, 3, 6] use\_auxiliary\_head = True auxiliary\_loss\_weight = 0.4 auxiliary\_in\_channels = None auxiliary\_channels = 256 auxiliary\_num\_convs = 1 auxiliary\_concat\_input = False loss\_ignore\_index = 255 \*\*kwargs  )

Parameters

* **backbone\_config** (`PretrainedConfig` or `dict`, *optional*, defaults to `ResNetConfig()`) —
  The configuration of the backbone model.
* **backbone** (`str`, *optional*) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, `False`) —
  Whether to use pretrained weights for the backbone.
* **use\_timm\_backbone** (`bool`, *optional*, `False`) —
  Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
  library.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **hidden\_size** (`int`, *optional*, defaults to 512) —
  The number of hidden units in the convolutional layers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **pool\_scales** (`tuple[int]`, *optional*, defaults to `[1, 2, 3, 6]`) —
  Pooling scales used in Pooling Pyramid Module applied on the last feature map.
* **use\_auxiliary\_head** (`bool`, *optional*, defaults to `True`) —
  Whether to use an auxiliary head during training.
* **auxiliary\_loss\_weight** (`float`, *optional*, defaults to 0.4) —
  Weight of the cross-entropy loss of the auxiliary head.
* **auxiliary\_channels** (`int`, *optional*, defaults to 256) —
  Number of channels to use in the auxiliary head.
* **auxiliary\_num\_convs** (`int`, *optional*, defaults to 1) —
  Number of convolutional layers to use in the auxiliary head.
* **auxiliary\_concat\_input** (`bool`, *optional*, defaults to `False`) —
  Whether to concatenate the output of the auxiliary head with the input before the classification layer.
* **loss\_ignore\_index** (`int`, *optional*, defaults to 255) —
  The index that is ignored by the loss function.

This is the configuration class to store the configuration of an [UperNetForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation). It is used to
instantiate an UperNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the UperNet
[openmmlab/upernet-convnext-tiny](https://huggingface.co/openmmlab/upernet-convnext-tiny) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import UperNetConfig, UperNetForSemanticSegmentation

>>> # Initializing a configuration
>>> configuration = UperNetConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = UperNetForSemanticSegmentation(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## UperNetForSemanticSegmentation

### class transformers.UperNetForSemanticSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/upernet/modeling_upernet.py#L289)

( config  )

Parameters

* **config** ([UperNetForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UperNet framework leveraging any vision backbone e.g. for ADE20k, CityScapes.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/upernet/modeling_upernet.py#L304)

( pixel\_values: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SegformerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessor). See [SegformerImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerFeatureExtractor.__call__) for details (`processor_class` uses
  [SegformerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/segformer#transformers.SegformerImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([UperNetConfig](/docs/transformers/v4.56.2/en/model_doc/upernet#transformers.UperNetConfig)) and inputs.

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

The [UperNetForSemanticSegmentation](/docs/transformers/v4.56.2/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
>>> from PIL import Image
>>> from huggingface_hub import hf_hub_download

>>> image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
>>> model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

>>> filepath = hf_hub_download(
...     repo_id="hf-internal-testing/fixtures_ade20k", filename="ADE_val_00000001.jpg", repo_type="dataset"
... )
>>> image = Image.open(filepath).convert("RGB")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)

>>> logits = outputs.logits  # shape (batch_size, num_labels, height, width)
>>> list(logits.shape)
[1, 150, 512, 512]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/upernet.md)
