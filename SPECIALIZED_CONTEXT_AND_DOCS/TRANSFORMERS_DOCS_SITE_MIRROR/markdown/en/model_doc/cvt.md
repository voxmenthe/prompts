# Convolutional Vision Transformer (CvT)

[Convolutional Vision Transformer (CvT)](https://huggingface.co/papers/2103.15808) is a model that combines the strengths of convolutional neural networks (CNNs) and Vision transformers for the computer vision tasks. It introduces convolutional layers into the vision transformer architecture, allowing it to capture local patterns in images while maintaining the global context provided by self-attention mechanisms.

You can find all the CvT checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=cvt) organization.

> [!TIP]
> This model was contributed by [anujunj](https://huggingface.co/anugunj).
>
> Click on the CvT models in the right sidebar for more examples of how to apply CvT to different computer vision tasks.

The example below demonstrates how to classify an image with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

```py
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

```py
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/cvt-13",
    dtype=torch.float16,
    device_map="auto"
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to(model.device)

with torch.no_grad():
  logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
```

## Resources

Refer to this set of ViT [notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) for examples of inference and fine-tuning on custom datasets. Replace `ViTFeatureExtractor` and [ViTForImageClassification](/docs/transformers/main/en/model_doc/vit#transformers.ViTForImageClassification) in these notebooks with [AutoImageProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoImageProcessor) and [CvtForImageClassification](/docs/transformers/main/en/model_doc/cvt#transformers.CvtForImageClassification).

## CvtConfig[[transformers.CvtConfig]]

#### transformers.CvtConfig[[transformers.CvtConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cvt/configuration_cvt.py#L24)

This is the configuration class to store the configuration of a [CvtModel](/docs/transformers/main/en/model_doc/cvt#transformers.CvtModel). It is used to instantiate a CvT model
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the CvT
[microsoft/cvt-13](https://huggingface.co/microsoft/cvt-13) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import CvtConfig, CvtModel

>>> # Initializing a Cvt msft/cvt style configuration
>>> configuration = CvtConfig()

>>> # Initializing a model (with random weights) from the msft/cvt style configuration
>>> model = CvtModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

patch_sizes (`list[int]`, *optional*, defaults to `[7, 3, 3]`) : The kernel size of each encoder's patch embedding.

patch_stride (`list[int]`, *optional*, defaults to `[4, 2, 2]`) : The stride size of each encoder's patch embedding.

patch_padding (`list[int]`, *optional*, defaults to `[2, 1, 1]`) : The padding size of each encoder's patch embedding.

embed_dim (`list[int]`, *optional*, defaults to `[64, 192, 384]`) : Dimension of each of the encoder blocks.

num_heads (`list[int]`, *optional*, defaults to `[1, 3, 6]`) : Number of attention heads for each attention layer in each block of the Transformer encoder.

depth (`list[int]`, *optional*, defaults to `[1, 2, 10]`) : The number of layers in each encoder block.

mlp_ratios (`list[float]`, *optional*, defaults to `[4.0, 4.0, 4.0, 4.0]`) : Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the encoder blocks.

attention_drop_rate (`list[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`) : The dropout ratio for the attention probabilities.

drop_rate (`list[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`) : The dropout ratio for the patch embeddings probabilities.

drop_path_rate (`list[float]`, *optional*, defaults to `[0.0, 0.0, 0.1]`) : The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.

qkv_bias (`list[bool]`, *optional*, defaults to `[True, True, True]`) : The bias bool for query, key and value in attentions

cls_token (`list[bool]`, *optional*, defaults to `[False, False, True]`) : Whether or not to add a classification token to the output of each of the last 3 stages.

qkv_projection_method (`list[string]`, *optional*, defaults to ["dw_bn", "dw_bn", "dw_bn"]`) : The projection method for query, key and value Default is depth-wise convolutions with batch norm. For Linear projection use "avg".

kernel_qkv (`list[int]`, *optional*, defaults to `[3, 3, 3]`) : The kernel size for query, key and value in attention layer

padding_kv (`list[int]`, *optional*, defaults to `[1, 1, 1]`) : The padding size for key and value in attention layer

stride_kv (`list[int]`, *optional*, defaults to `[2, 2, 2]`) : The stride size for key and value in attention layer

padding_q (`list[int]`, *optional*, defaults to `[1, 1, 1]`) : The padding size for query in attention layer

stride_q (`list[int]`, *optional*, defaults to `[1, 1, 1]`) : The stride size for query in attention layer

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-6) : The epsilon used by the layer normalization layers.

## CvtModel[[transformers.CvtModel]]

#### transformers.CvtModel[[transformers.CvtModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cvt/modeling_cvt.py#L509)

The bare Cvt Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.CvtModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/cvt/modeling_cvt.py#L520[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ConvNextImageProcessor](/docs/transformers/main/en/model_doc/convnext#transformers.ConvNextImageProcessor). See [ConvNextImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ConvNextImageProcessor](/docs/transformers/main/en/model_doc/convnext#transformers.ConvNextImageProcessor) for processing images).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.cvt.modeling_cvt.BaseModelOutputWithCLSToken` or `tuple(torch.FloatTensor)`A `transformers.models.cvt.modeling_cvt.BaseModelOutputWithCLSToken` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CvtConfig](/docs/transformers/main/en/model_doc/cvt#transformers.CvtConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **cls_token_value** (`torch.FloatTensor` of shape `(batch_size, 1, hidden_size)`) -- Classification token at the output of the last layer of the model.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
The [CvtModel](/docs/transformers/main/en/model_doc/cvt#transformers.CvtModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([CvtModel](/docs/transformers/main/en/model_doc/cvt#transformers.CvtModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

add_pooling_layer (`bool`, *optional*, defaults to `True`) : Whether to add a pooling layer

**Returns:**

``transformers.models.cvt.modeling_cvt.BaseModelOutputWithCLSToken` or `tuple(torch.FloatTensor)``

A `transformers.models.cvt.modeling_cvt.BaseModelOutputWithCLSToken` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CvtConfig](/docs/transformers/main/en/model_doc/cvt#transformers.CvtConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **cls_token_value** (`torch.FloatTensor` of shape `(batch_size, 1, hidden_size)`) -- Classification token at the output of the last layer of the model.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

## CvtForImageClassification[[transformers.CvtForImageClassification]]

#### transformers.CvtForImageClassification[[transformers.CvtForImageClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cvt/modeling_cvt.py#L559)

Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
the [CLS] token) e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.CvtForImageClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/cvt/modeling_cvt.py#L574[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ConvNextImageProcessor](/docs/transformers/main/en/model_doc/convnext#transformers.ConvNextImageProcessor). See [ConvNextImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ConvNextImageProcessor](/docs/transformers/main/en/model_doc/convnext#transformers.ConvNextImageProcessor) for processing images).
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CvtConfig](/docs/transformers/main/en/model_doc/cvt#transformers.CvtConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
The [CvtForImageClassification](/docs/transformers/main/en/model_doc/cvt#transformers.CvtForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
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

**Parameters:**

config ([CvtForImageClassification](/docs/transformers/main/en/model_doc/cvt#transformers.CvtForImageClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CvtConfig](/docs/transformers/main/en/model_doc/cvt#transformers.CvtConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
