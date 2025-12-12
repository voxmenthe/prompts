# DINOv2

[DINOv2](https://huggingface.co/papers/2304.07193) is a vision foundation model that uses [ViT](./vit) as a feature extractor for multiple downstream tasks like image classification and depth estimation. It focuses on stabilizing and accelerating training through techniques like a faster memory-efficient attention, sequence packing, improved stochastic depth, Fully Sharded Data Parallel (FSDP), and model distillation.

You can find all the original DINOv2 checkpoints under the [Dinov2](https://huggingface.co/collections/facebook/dinov2-6526c98554b3d2576e071ce3) collection.

> [!TIP]
> Click on the DINOv2 models in the right sidebar for more examples of how to apply DINOv2 to different vision tasks.

The example below demonstrates how to obtain an image embedding with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="image-classification",
    model="facebook/dinov2-small-imagenet1k-1-layer",
    dtype=torch.float16,
    device=0
)

pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

```py
import requests
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer")
model = AutoModelForImageClassification.from_pretrained(
    "facebook/dinov2-small-imagenet1k-1-layer",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

inputs = processor(images=image, return_tensors="pt")
logits = model(**inputs).logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```py
# pip install torchao
import requests
from transformers import TorchAoConfig, AutoImageProcessor, AutoModelForImageClassification
from torchao.quantization import Int4WeightOnlyConfig
from PIL import Image

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant-imagenet1k-1-layer')

quant_config = Int4WeightOnlyConfig(group_size=128)
quantization_config = TorchAoConfig(quant_type=quant_config)

model = AutoModelForImageClassification.from_pretrained(
    'facebook/dinov2-giant-imagenet1k-1-layer',
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

## Notes

- The example below shows how to split the output tensor into:
  - one embedding for the whole image, commonly referred to as a `CLS` token,
    useful for classification and retrieval
  - a set of local embeddings, one for each `14x14` patch of the input image,
    useful for dense tasks, such as semantic segmentation

  ```py
  from transformers import AutoImageProcessor, AutoModel
  from PIL import Image
  import requests

  url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
  image = Image.open(requests.get(url, stream=True).raw)
  print(image.height, image.width)  # [480, 640]

  processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
  model = AutoModel.from_pretrained('facebook/dinov2-base')
  patch_size = model.config.patch_size

  inputs = processor(images=image, return_tensors="pt")
  print(inputs.pixel_values.shape)  # [1, 3, 224, 224]
  batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
  num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
  num_patches_flat = num_patches_height * num_patches_width

  outputs = model(**inputs)
  last_hidden_states = outputs[0]
  print(last_hidden_states.shape)  # [1, 1 + 256, 768]
  assert last_hidden_states.shape == (batch_size, 1 + num_patches_flat, model.config.hidden_size)

  cls_token = last_hidden_states[:, 0, :]
  patch_features = last_hidden_states[:, 1:, :].unflatten(1, (num_patches_height, num_patches_width))
  ```

- Use [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) to speedup inference.
  However, it will produce some mismatched elements. The difference between the original and traced model is 1e-4.

  ```py
  import torch
  from transformers import AutoImageProcessor, AutoModel
  from PIL import Image
  import requests

  url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
  image = Image.open(requests.get(url, stream=True).raw)

  processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
  model = AutoModel.from_pretrained('facebook/dinov2-base')

  inputs = processor(images=image, return_tensors="pt")
  outputs = model(**inputs)
  last_hidden_states = outputs[0]

  # We have to force return_dict=False for tracing
  model.config.return_dict = False

  with torch.no_grad():
      traced_model = torch.jit.trace(model, [inputs.pixel_values])
      traced_outputs = traced_model(inputs.pixel_values)

  print((last_hidden_states - traced_outputs[0]).abs().max())
  ```

## Dinov2Config[[transformers.Dinov2Config]]

#### transformers.Dinov2Config[[transformers.Dinov2Config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/configuration_dinov2.py#L25)

This is the configuration class to store the configuration of a [Dinov2Model](/docs/transformers/main/en/model_doc/dinov2#transformers.Dinov2Model). It is used to instantiate an
Dinov2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Dinov2
[google/dinov2-base-patch16-224](https://huggingface.co/google/dinov2-base-patch16-224) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Dinov2Config, Dinov2Model

>>> # Initializing a Dinov2 dinov2-base-patch16-224 style configuration
>>> configuration = Dinov2Config()

>>> # Initializing a model (with random weights) from the dinov2-base-patch16-224 style configuration
>>> model = Dinov2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

mlp_ratio (`int`, *optional*, defaults to 4) : Ratio of the hidden size of the MLPs relative to the `hidden_size`.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-06) : The epsilon used by the layer normalization layers.

image_size (`int`, *optional*, defaults to 224) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 14) : The size (resolution) of each patch.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries, keys and values.

layerscale_value (`float`, *optional*, defaults to 1.0) : Initial value to use for layer scale.

drop_path_rate (`float`, *optional*, defaults to 0.0) : Stochastic depth rate per sample (when applied in the main path of residual layers).

use_swiglu_ffn (`bool`, *optional*, defaults to `False`) : Whether to use the SwiGLU feedforward neural network.

out_features (`list[str]`, *optional*) : If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc. (depending on how many stages the model has). If unset and `out_indices` is set, will default to the corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the same order as defined in the `stage_names` attribute.

out_indices (`list[int]`, *optional*) : If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how many stages the model has). If unset and `out_features` is set, will default to the corresponding stages. If unset and `out_features` is unset, will default to the last stage. Must be in the same order as defined in the `stage_names` attribute.

apply_layernorm (`bool`, *optional*, defaults to `True`) : Whether to apply layer normalization to the feature maps in case the model is used as backbone.

reshape_hidden_states (`bool`, *optional*, defaults to `True`) : Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size, seq_len, hidden_size)`.

use_mask_token (`bool`, *optional*, defaults to `True`) : Whether to use mask_token in embeddings.

## Dinov2Model[[transformers.Dinov2Model]]

#### transformers.Dinov2Model[[transformers.Dinov2Model]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L438)

The bare Dinov2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Dinov2Model.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L454[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "bool_masked_pos", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BitImageProcessor](/docs/transformers/main/en/model_doc/bit#transformers.BitImageProcessor). See [BitImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [BitImageProcessor](/docs/transformers/main/en/model_doc/bit#transformers.BitImageProcessor) for processing images).
- **bool_masked_pos** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`) --
  Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
  pre-training.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dinov2Config](/docs/transformers/main/en/model_doc/dinov2#transformers.Dinov2Config)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [Dinov2Model](/docs/transformers/main/en/model_doc/dinov2#transformers.Dinov2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
```

**Parameters:**

config ([Dinov2Config](/docs/transformers/main/en/model_doc/dinov2#transformers.Dinov2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dinov2Config](/docs/transformers/main/en/model_doc/dinov2#transformers.Dinov2Config)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Dinov2ForImageClassification[[transformers.Dinov2ForImageClassification]]

#### transformers.Dinov2ForImageClassification[[transformers.Dinov2ForImageClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L494)

Dinov2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state
of the [CLS] token) e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Dinov2ForImageClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L509[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [BitImageProcessor](/docs/transformers/main/en/model_doc/bit#transformers.BitImageProcessor). See [BitImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [BitImageProcessor](/docs/transformers/main/en/model_doc/bit#transformers.BitImageProcessor) for processing images).
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).0[transformers.modeling_outputs.ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dinov2Config](/docs/transformers/main/en/model_doc/dinov2#transformers.Dinov2Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [Dinov2ForImageClassification](/docs/transformers/main/en/model_doc/dinov2#transformers.Dinov2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, Dinov2ForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/dinov2-base-patch16-224")
>>> model = Dinov2ForImageClassification.from_pretrained("google/dinov2-base-patch16-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

**Parameters:**

config ([Dinov2Config](/docs/transformers/main/en/model_doc/dinov2#transformers.Dinov2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.ImageClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dinov2Config](/docs/transformers/main/en/model_doc/dinov2#transformers.Dinov2Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
