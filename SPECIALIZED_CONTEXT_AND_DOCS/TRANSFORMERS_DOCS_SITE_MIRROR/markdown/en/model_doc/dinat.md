# Dilated Neighborhood Attention Transformer

## Overview

DiNAT was proposed in [Dilated Neighborhood Attention Transformer](https://huggingface.co/papers/2209.15001)
by Ali Hassani and Humphrey Shi.

It extends [NAT](nat) by adding a Dilated Neighborhood Attention pattern to capture global context,
and shows significant performance improvements over it.

The abstract from the paper is the following:

*Transformers are quickly becoming one of the most heavily applied deep learning architectures across modalities,
domains, and tasks. In vision, on top of ongoing efforts into plain transformers, hierarchical transformers have
also gained significant attention, thanks to their performance and easy integration into existing frameworks.
These models typically employ localized attention mechanisms, such as the sliding-window Neighborhood Attention (NA)
or Swin Transformer's Shifted Window Self Attention. While effective at reducing self attention's quadratic complexity,
local attention weakens two of the most desirable properties of self attention: long range inter-dependency modeling,
and global receptive field. In this paper, we introduce Dilated Neighborhood Attention (DiNA), a natural, flexible and
efficient extension to NA that can capture more global context and expand receptive fields exponentially at no
additional cost. NA's local attention and DiNA's sparse global attention complement each other, and therefore we
introduce Dilated Neighborhood Attention Transformer (DiNAT), a new hierarchical vision transformer built upon both.
DiNAT variants enjoy significant improvements over strong baselines such as NAT, Swin, and ConvNeXt.
Our large model is faster and ahead of its Swin counterpart by 1.5% box AP in COCO object detection,
1.3% mask AP in COCO instance segmentation, and 1.1% mIoU in ADE20K semantic segmentation.
Paired with new frameworks, our large variant is the new state of the art panoptic segmentation model on COCO (58.2 PQ)
and ADE20K (48.5 PQ), and instance segmentation model on Cityscapes (44.5 AP) and ADE20K (35.4 AP) (no extra data).
It also matches the state of the art specialized semantic segmentation models on ADE20K (58.2 mIoU),
and ranks second on Cityscapes (84.5 mIoU) (no extra data).*

 Neighborhood Attention with different dilation values.
Taken from the original paper.

This model was contributed by [Ali Hassani](https://huggingface.co/alihassanijr).
The original code can be found [here](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer).

## Usage tips

DiNAT can be used as a *backbone*. When `output_hidden_states = True`,
it will output both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, height, width, num_channels)`.

Notes:

- DiNAT depends on [NATTEN](https://github.com/SHI-Labs/NATTEN/)'s implementation of Neighborhood Attention and Dilated Neighborhood Attention.
You can install it with pre-built wheels for Linux by referring to [shi-labs.com/natten](https://shi-labs.com/natten), or build on your system by running `pip install natten`.
Note that the latter will likely take time to compile. NATTEN does not support Windows devices yet.
- Patch size of 4 is only supported at the moment.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DiNAT.

- [DinatForImageClassification](/docs/transformers/main/en/model_doc/dinat#transformers.DinatForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DinatConfig[[transformers.DinatConfig]]

#### transformers.DinatConfig[[transformers.DinatConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinat/configuration_dinat.py#L25)

This is the configuration class to store the configuration of a [DinatModel](/docs/transformers/main/en/model_doc/dinat#transformers.DinatModel). It is used to instantiate a Dinat
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Dinat
[shi-labs/dinat-mini-in1k-224](https://huggingface.co/shi-labs/dinat-mini-in1k-224) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import DinatConfig, DinatModel

>>> # Initializing a Dinat shi-labs/dinat-mini-in1k-224 style configuration
>>> configuration = DinatConfig()

>>> # Initializing a model (with random weights) from the shi-labs/dinat-mini-in1k-224 style configuration
>>> model = DinatModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

patch_size (`int`, *optional*, defaults to 4) : The size (resolution) of each patch. NOTE: Only patch size of 4 is supported at the moment.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

embed_dim (`int`, *optional*, defaults to 64) : Dimensionality of patch embedding.

depths (`list[int]`, *optional*, defaults to `[3, 4, 6, 5]`) : Number of layers in each level of the encoder.

num_heads (`list[int]`, *optional*, defaults to `[2, 4, 8, 16]`) : Number of attention heads in each layer of the Transformer encoder.

kernel_size (`int`, *optional*, defaults to 7) : Neighborhood Attention kernel size.

dilations (`list[list[int]]`, *optional*, defaults to `[[1, 8, 1], [1, 4, 1, 4], [1, 2, 1, 2, 1, 2], [1, 1, 1, 1, 1]]`) : Dilation value of each NA layer in the Transformer encoder.

mlp_ratio (`float`, *optional*, defaults to 3.0) : Ratio of MLP hidden dimensionality to embedding dimensionality.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether or not a learnable bias should be added to the queries, keys and values.

hidden_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout probability for all fully connected layers in the embeddings and encoder.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

drop_path_rate (`float`, *optional*, defaults to 0.1) : Stochastic depth rate.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

layer_scale_init_value (`float`, *optional*, defaults to 0.0) : The initial value for the layer scale. Disabled if  1` a classification loss is computed (Cross-Entropy).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.dinat.modeling_dinat.DinatImageClassifierOutput` or `tuple(torch.FloatTensor)`A `transformers.models.dinat.modeling_dinat.DinatImageClassifierOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DinatConfig](/docs/transformers/main/en/model_doc/dinat#transformers.DinatConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **reshaped_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.
The [DinatForImageClassification](/docs/transformers/main/en/model_doc/dinat#transformers.DinatForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, DinatForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("shi-labs/dinat-mini-in1k-224")
>>> model = DinatForImageClassification.from_pretrained("shi-labs/dinat-mini-in1k-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

**Parameters:**

config ([DinatForImageClassification](/docs/transformers/main/en/model_doc/dinat#transformers.DinatForImageClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.dinat.modeling_dinat.DinatImageClassifierOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.dinat.modeling_dinat.DinatImageClassifierOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DinatConfig](/docs/transformers/main/en/model_doc/dinat#transformers.DinatConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **reshaped_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, hidden_size, height, width)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
  include the spatial dimensions.
