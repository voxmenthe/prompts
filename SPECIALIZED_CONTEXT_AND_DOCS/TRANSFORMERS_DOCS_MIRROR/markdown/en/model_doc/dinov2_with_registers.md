*This model was released on 2023-09-28 and added to Hugging Face Transformers on 2024-12-24.*

# DINOv2 with Registers

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The DINOv2 with Registers model was proposed in [Vision Transformers Need Registers](https://huggingface.co/papers/2309.16588) by Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski.

The [Vision Transformer](vit) (ViT) is a transformer encoder model (BERT-like) originally introduced to do supervised image classification on ImageNet.

Next, people figured out ways to make ViT work really well on self-supervised image feature extraction (i.e. learning meaningful features, also called embeddings) on images without requiring any labels. Some example papers here include [DINOv2](dinov2) and [MAE](vit_mae).

The authors of DINOv2 noticed that ViTs have artifacts in attention maps. It’s due to the model using some image patches as “registers”. The authors propose a fix: just add some new tokens (called “register” tokens), which you only use during pre-training (and throw away afterwards). This results in:

* no artifacts
* interpretable attention maps
* and improved performances.

The abstract from the paper is the following:

*Transformers have recently emerged as a powerful tool for learning visual representations. In this paper, we identify and characterize artifacts in feature maps of both supervised and self-supervised ViT networks. The artifacts correspond to high-norm tokens appearing during inference primarily in low-informative background areas of images, that are repurposed for internal computations. We propose a simple yet effective solution based on providing additional tokens to the input sequence of the Vision Transformer to fill that role. We show that this solution fixes that problem entirely for both supervised and self-supervised models, sets a new state of the art for self-supervised visual models on dense visual prediction tasks, enables object discovery methods with larger models, and most importantly leads to smoother feature maps and attention maps for downstream visual processing.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dinov2_with_registers_visualization.png) Visualization of attention maps of various models trained with vs. without registers. Taken from the [original paper](https://huggingface.co/papers/2309.16588).

Tips:

* Usage of DINOv2 with Registers is identical to DINOv2 without, you’ll just get better performance.

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/facebookresearch/dinov2).

## Dinov2WithRegistersConfig

### class transformers.Dinov2WithRegistersConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov2_with_registers/configuration_dinov2_with_registers.py#L27)

( hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 mlp\_ratio = 4 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 initializer\_range = 0.02 layer\_norm\_eps = 1e-06 image\_size = 224 patch\_size = 16 num\_channels = 3 qkv\_bias = True layerscale\_value = 1.0 drop\_path\_rate = 0.0 use\_swiglu\_ffn = False num\_register\_tokens = 4 out\_features = None out\_indices = None apply\_layernorm = True reshape\_hidden\_states = True \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **mlp\_ratio** (`int`, *optional*, defaults to 4) —
  Ratio of the hidden size of the MLPs relative to the `hidden_size`.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.
* **layerscale\_value** (`float`, *optional*, defaults to 1.0) —
  Initial value to use for layer scale.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.0) —
  Stochastic depth rate per sample (when applied in the main path of residual layers).
* **use\_swiglu\_ffn** (`bool`, *optional*, defaults to `False`) —
  Whether to use the SwiGLU feedforward neural network.
* **num\_register\_tokens** (`int`, *optional*, defaults to 4) —
  Number of register tokens to use.
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
* **apply\_layernorm** (`bool`, *optional*, defaults to `True`) —
  Whether to apply layer normalization to the feature maps in case the model is used as backbone.
* **reshape\_hidden\_states** (`bool`, *optional*, defaults to `True`) —
  Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
  case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size, seq_len, hidden_size)`.

This is the configuration class to store the configuration of a [Dinov2WithRegistersModel](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersModel). It is used to instantiate an
Dinov2WithRegisters model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the DINOv2 with Registers
[facebook/dinov2-with-registers-base](https://huggingface.co/facebook/dinov2-with-registers-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

>>> # Initializing a Dinov2WithRegisters base style configuration
>>> configuration = Dinov2WithRegistersConfig()

>>> # Initializing a model (with random weights) from the base style configuration
>>> model = Dinov2WithRegistersModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Dinov2WithRegistersModel

### class transformers.Dinov2WithRegistersModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov2_with_registers/modeling_dinov2_with_registers.py#L497)

( config: Dinov2WithRegistersConfig  )

Parameters

* **config** ([Dinov2WithRegistersConfig](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Dinov2 With Registers Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov2_with_registers/modeling_dinov2_with_registers.py#L521)

( pixel\_values: typing.Optional[torch.Tensor] = None bool\_masked\_pos: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0). Only relevant for
  pre-training.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dinov2WithRegistersConfig](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Dinov2WithRegistersModel](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## Dinov2WithRegistersForImageClassification

### class transformers.Dinov2WithRegistersForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov2_with_registers/modeling_dinov2_with_registers.py#L571)

( config: Dinov2WithRegistersConfig  )

Parameters

* **config** ([Dinov2WithRegistersConfig](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Dinov2WithRegisters Model transformer with an image classification head on top (a linear layer on top of the final hidden state
of the [CLS] token) e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dinov2_with_registers/modeling_dinov2_with_registers.py#L586)

( pixel\_values: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

[transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dinov2WithRegistersConfig](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Dinov2WithRegistersForImageClassification](/docs/transformers/v4.56.2/en/model_doc/dinov2_with_registers#transformers.Dinov2WithRegistersForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, Dinov2WithRegistersForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
>>> model = Dinov2WithRegistersForImageClassification.from_pretrained("facebook/dinov2-with-registers-base")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dinov2_with_registers.md)
