*This model was released on 2021-03-29 and added to Hugging Face Transformers on 2022-05-18.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# Convolutional Vision Transformer (CvT)

[Convolutional Vision Transformer (CvT)](https://huggingface.co/papers/2103.15808) is a model that combines the strengths of convolutional neural networks (CNNs) and Vision transformers for the computer vision tasks. It introduces convolutional layers into the vision transformer architecture, allowing it to capture local patterns in images while maintaining the global context provided by self-attention mechanisms.

You can find all the CvT checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=cvt) organization.

This model was contributed by [anujunj](https://huggingface.co/anugunj).

Click on the CvT models in the right sidebar for more examples of how to apply CvT to different computer vision tasks.

The example below demonstrates how to classify an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="microsoft/cvt-13",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

## Resources

Refer to this set of ViT [notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) for examples of inference and fine-tuning on custom datasets. Replace [ViTFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTFeatureExtractor) and [ViTForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTForImageClassification) in these notebooks with [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor) and [CvtForImageClassification](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtForImageClassification).

## CvtConfig

### class transformers.CvtConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cvt/configuration_cvt.py#L24)

( num\_channels = 3 patch\_sizes = [7, 3, 3] patch\_stride = [4, 2, 2] patch\_padding = [2, 1, 1] embed\_dim = [64, 192, 384] num\_heads = [1, 3, 6] depth = [1, 2, 10] mlp\_ratio = [4.0, 4.0, 4.0] attention\_drop\_rate = [0.0, 0.0, 0.0] drop\_rate = [0.0, 0.0, 0.0] drop\_path\_rate = [0.0, 0.0, 0.1] qkv\_bias = [True, True, True] cls\_token = [False, False, True] qkv\_projection\_method = ['dw\_bn', 'dw\_bn', 'dw\_bn'] kernel\_qkv = [3, 3, 3] padding\_kv = [1, 1, 1] stride\_kv = [2, 2, 2] padding\_q = [1, 1, 1] stride\_q = [1, 1, 1] initializer\_range = 0.02 layer\_norm\_eps = 1e-12 \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **patch\_sizes** (`list[int]`, *optional*, defaults to `[7, 3, 3]`) —
  The kernel size of each encoder’s patch embedding.
* **patch\_stride** (`list[int]`, *optional*, defaults to `[4, 2, 2]`) —
  The stride size of each encoder’s patch embedding.
* **patch\_padding** (`list[int]`, *optional*, defaults to `[2, 1, 1]`) —
  The padding size of each encoder’s patch embedding.
* **embed\_dim** (`list[int]`, *optional*, defaults to `[64, 192, 384]`) —
  Dimension of each of the encoder blocks.
* **num\_heads** (`list[int]`, *optional*, defaults to `[1, 3, 6]`) —
  Number of attention heads for each attention layer in each block of the Transformer encoder.
* **depth** (`list[int]`, *optional*, defaults to `[1, 2, 10]`) —
  The number of layers in each encoder block.
* **mlp\_ratios** (`list[float]`, *optional*, defaults to `[4.0, 4.0, 4.0, 4.0]`) —
  Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
  encoder blocks.
* **attention\_drop\_rate** (`list[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`) —
  The dropout ratio for the attention probabilities.
* **drop\_rate** (`list[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`) —
  The dropout ratio for the patch embeddings probabilities.
* **drop\_path\_rate** (`list[float]`, *optional*, defaults to `[0.0, 0.0, 0.1]`) —
  The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
* **qkv\_bias** (`list[bool]`, *optional*, defaults to `[True, True, True]`) —
  The bias bool for query, key and value in attentions
* **cls\_token** (`list[bool]`, *optional*, defaults to `[False, False, True]`) —
  Whether or not to add a classification token to the output of each of the last 3 stages.
* **qkv\_projection\_method** (`list[string]`, *optional*, defaults to [“dw\_bn”, “dw\_bn”, “dw\_bn”]`) —
  The projection method for query, key and value Default is depth-wise convolutions with batch norm. For
  Linear projection use “avg”.
* **kernel\_qkv** (`list[int]`, *optional*, defaults to `[3, 3, 3]`) —
  The kernel size for query, key and value in attention layer
* **padding\_kv** (`list[int]`, *optional*, defaults to `[1, 1, 1]`) —
  The padding size for key and value in attention layer
* **stride\_kv** (`list[int]`, *optional*, defaults to `[2, 2, 2]`) —
  The stride size for key and value in attention layer
* **padding\_q** (`list[int]`, *optional*, defaults to `[1, 1, 1]`) —
  The padding size for query in attention layer
* **stride\_q** (`list[int]`, *optional*, defaults to `[1, 1, 1]`) —
  The stride size for query in attention layer
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-6) —
  The epsilon used by the layer normalization layers.

This is the configuration class to store the configuration of a [CvtModel](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtModel). It is used to instantiate a CvT model
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the CvT
[microsoft/cvt-13](https://huggingface.co/microsoft/cvt-13) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import CvtConfig, CvtModel

>>> # Initializing a Cvt msft/cvt style configuration
>>> configuration = CvtConfig()

>>> # Initializing a model (with random weights) from the msft/cvt style configuration
>>> model = CvtModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## CvtModel

### class transformers.CvtModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cvt/modeling_cvt.py#L535)

( config add\_pooling\_layer = True  )

Parameters

* **config** ([CvtModel](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The bare Cvt Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cvt/modeling_cvt.py#L554)

( pixel\_values: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.cvt.modeling_cvt.BaseModelOutputWithCLSToken` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor). See [ConvNextImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor) for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.cvt.modeling_cvt.BaseModelOutputWithCLSToken` or `tuple(torch.FloatTensor)`

A `transformers.models.cvt.modeling_cvt.BaseModelOutputWithCLSToken` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CvtConfig](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **cls\_token\_value** (`torch.FloatTensor` of shape `(batch_size, 1, hidden_size)`) — Classification token at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [CvtModel](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## CvtForImageClassification

### class transformers.CvtForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cvt/modeling_cvt.py#L592)

( config  )

Parameters

* **config** ([CvtForImageClassification](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtForImageClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
the [CLS] token) e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cvt/modeling_cvt.py#L607)

( pixel\_values: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor). See [ConvNextImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ConvNextImageProcessor](/docs/transformers/v4.56.2/en/model_doc/convnext#transformers.ConvNextImageProcessor) for processing images).
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
elements depending on the configuration ([CvtConfig](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.

The [CvtForImageClassification](/docs/transformers/v4.56.2/en/model_doc/cvt#transformers.CvtForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, CvtForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
>>> model = CvtForImageClassification.from_pretrained("microsoft/cvt-13")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/cvt.md)
