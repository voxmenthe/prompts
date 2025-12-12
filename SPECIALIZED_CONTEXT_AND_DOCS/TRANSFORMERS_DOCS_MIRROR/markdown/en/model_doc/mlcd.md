*This model was released on 2024-07-24 and added to Hugging Face Transformers on 2025-04-15.*

# MLCD

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The [MLCD](https://huggingface.co/papers/2407.17331) models were released by the DeepGlint-AI team in [unicom](https://github.com/deepglint/unicom), which focuses on building foundational visual models for large multimodal language models using large-scale datasets such as LAION400M and COYO700M, and employs sample-to-cluster contrastive learning to optimize performance. MLCD models are primarily used for multimodal visual large language models, such as LLaVA.

🔥**MLCD-ViT-bigG**🔥 series is the state-of-the-art vision transformer model enhanced with 2D Rotary Position Embedding (RoPE2D), achieving superior performance on document understanding and visual question answering tasks. Developed by DeepGlint AI, this model demonstrates exceptional capabilities in processing complex visual-language interactions.

Tips:

* We adopted the official [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and the official training dataset [LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) for evaluating the foundational visual models.
* The language model is [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

Result:

| Vision Tower | RoPE2D | ChartQA | DocVQA | InfoVQA | OCRBench | MMMU |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP (ViT-L-14-336px) | × | 66.52 | 75.21 | 38.88 | 525.00 | 44.20 |
| SigLIP (ViT-SO400M-384px) | × | 69.28 | 76.71 | 41.38 | 554.00 | 46.78 |
| DFN5B (ViT-H-14-378px) | × | 64.36 | 70.87 | 38.59 | 473.00 | **48.00** |
| **[MLCD (ViT-L-14-336px)](https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336)** | × | 67.84 | 76.46 | 43.48 | 531.00 | 44.30 |
| **[MLCD (ViT-bigG-14-336px)](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-336)** | √ | 71.07 | 79.63 | 44.38 | 572.00 | 46.78 |
| **[MLCD (ViT-bigG-14-448px)](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-448)** | √ | **73.80** | **83.34** | **46.59** | **582.00** | 46.00 |

## Usage


```
import requests
from PIL import Image
from transformers import AutoProcessor, MLCDVisionModel

# Load model and processor
model = MLCDVisionModel.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-448")
processor = AutoProcessor.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-448")

# Process single image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

# Generate outputs
with torch.no_grad():
    outputs = model(**inputs)

# Get visual features
features = outputs.last_hidden_state

print(f"Extracted features shape: {features.shape}")
```

## MLCDVisionConfig

### class transformers.MLCDVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mlcd/configuration_mlcd.py#L25)

( hidden\_size = 1664 intermediate\_size = 8192 num\_hidden\_layers = 48 num\_attention\_heads = 16 num\_key\_value\_groups = 1 num\_channels = 3 image\_size = 336 patch\_size = 14 hidden\_act = 'gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 1664) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 8192) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **projection\_dim** (`int`, *optional*, defaults to 1024) —
  Dimensionality of text and vision projection layers.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 48) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **image\_size** (`int`, *optional*, defaults to 336) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The size (resolution) of each patch.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).

This is the configuration class to store the configuration of a [MLCDVisionModel](/docs/transformers/v4.56.2/en/model_doc/mlcd#transformers.MLCDVisionModel). It is used to instantiate a MLCD
vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the vision encoder of the MLCD
[DeepGlint-AI/mlcd-vit-bigG-patch14-336](https://huggingface.co/DeepGlint-AI/mlcd-vit-bigG-patch14-336) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MLCDVisionConfig, MLCDVisionModel

>>> # Initializing a MLCDVisionConfig with DeepGlint-AI/mlcd-vit-bigG-patch14-336 style configuration
>>> configuration = MLCDVisionConfig()

>>> # Initializing a MLCDVisionModel (with random weights) from the DeepGlint-AI/mlcd-vit-bigG-patch14-336 style configuration
>>> model = MLCDVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MLCDVisionModel

### class transformers.MLCDVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mlcd/modeling_mlcd.py#L553)

( config: MLCDVisionConfig  )

Parameters

* **config** ([MLCDVisionConfig](/docs/transformers/v4.56.2/en/model_doc/mlcd#transformers.MLCDVisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The vision model from M\_L\_C\_D without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mlcd/modeling_mlcd.py#L567)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MLCDVisionConfig](/docs/transformers/v4.56.2/en/model_doc/mlcd#transformers.MLCDVisionConfig)) and inputs.

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

The [MLCDVisionModel](/docs/transformers/v4.56.2/en/model_doc/mlcd#transformers.MLCDVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import requests
>>> from PIL import Image
>>> from transformers import AutoProcessor, MLCDVisionModel
>>> model = MLCDVisionModel.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-448")
>>> processor = AutoProcessor.from_pretrained("DeepGlint-AI/mlcd-vit-bigG-patch14-448")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs, output_attentions=True)

>>> features = outputs.last_hidden_state
>>> print(f"Extracted features shape: {features.shape}")
>>> print(f"Number of attention layers: {len(outputs.attentions)}")
>>> print(f"Attention shape: {outputs.attentions[0].shape}")
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mlcd.md)
