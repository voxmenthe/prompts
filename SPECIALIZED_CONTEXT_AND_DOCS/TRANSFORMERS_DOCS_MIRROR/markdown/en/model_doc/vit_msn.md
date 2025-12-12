*This model was released on 2022-04-14 and added to Hugging Face Transformers on 2022-09-22.*

# ViTMSN

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The ViTMSN model was proposed in [Masked Siamese Networks for Label-Efficient Learning](https://huggingface.co/papers/2204.07141) by Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes,
Pascal Vincent, Armand Joulin, Michael Rabbat, Nicolas Ballas. The paper presents a joint-embedding architecture to match the prototypes
of masked patches with that of the unmasked patches. With this setup, their method yields excellent performance in the low-shot and extreme low-shot
regimes.

The abstract from the paper is the following:

*We propose Masked Siamese Networks (MSN), a self-supervised learning framework for learning image representations. Our
approach matches the representation of an image view containing randomly masked patches to the representation of the original
unmasked image. This self-supervised pre-training strategy is particularly scalable when applied to Vision Transformers since only the
unmasked patches are processed by the network. As a result, MSNs improve the scalability of joint-embedding architectures,
while producing representations of a high semantic level that perform competitively on low-shot image classification. For instance,
on ImageNet-1K, with only 5,000 annotated images, our base MSN model achieves 72.4% top-1 accuracy,
and with 1% of ImageNet-1K labels, we achieve 75.7% top-1 accuracy, setting a new state-of-the-art for self-supervised learning on this benchmark.*

![drawing](https://i.ibb.co/W6PQMdC/Screenshot-2022-09-13-at-9-08-40-AM.png) MSN architecture. Taken from the [original paper.](https://huggingface.co/papers/2204.07141)

This model was contributed by [sayakpaul](https://huggingface.co/sayakpaul). The original code can be found [here](https://github.com/facebookresearch/msn).

## Usage tips

* MSN (masked siamese networks) is a method for self-supervised pre-training of Vision Transformers (ViTs). The pre-training
  objective is to match the prototypes assigned to the unmasked views of the images to that of the masked views of the same images.
* The authors have only released pre-trained weights of the backbone (ImageNet-1k pre-training). So, to use that on your own image classification dataset,
  use the [ViTMSNForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNForImageClassification) class which is initialized from [ViTMSNModel](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNModel). Follow
  [this notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb) for a detailed tutorial on fine-tuning.
* MSN is particularly useful in the low-shot and extreme low-shot regimes. Notably, it achieves 75.7% top-1 accuracy with only 1% of ImageNet-1K
  labels when fine-tuned.

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.


```
from transformers import ViTMSNForImageClassification
model = ViTMSNForImageClassification.from_pretrained("facebook/vit-msn-base", attn_implementation="sdpa", dtype=torch.float16)
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `facebook/vit-msn-base` model, we saw the following speedups during inference.

| Batch size | Average inference time (ms), eager mode | Average inference time (ms), sdpa model | Speed up, Sdpa / Eager (x) |
| --- | --- | --- | --- |
| 1 | 7 | 6 | 1.17 |
| 2 | 8 | 6 | 1.33 |
| 4 | 8 | 6 | 1.33 |
| 8 | 8 | 6 | 1.33 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViT MSN.

Image Classification

* [ViTMSNForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
* See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## ViTMSNConfig

### class transformers.ViTMSNConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vit_msn/configuration_vit_msn.py#L24)

( hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 initializer\_range = 0.02 layer\_norm\_eps = 1e-06 image\_size = 224 patch\_size = 16 num\_channels = 3 qkv\_bias = True \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
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

This is the configuration class to store the configuration of a [ViTMSNModel](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNModel). It is used to instantiate an ViT
MSN model according to the specified arguments, defining the model architecture. Instantiating a configuration with
the defaults will yield a similar configuration to that of the ViT
[facebook/vit\_msn\_base](https://huggingface.co/facebook/vit_msn_base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import ViTMSNModel, ViTMSNConfig

>>> # Initializing a ViT MSN vit-msn-base style configuration
>>> configuration = ViTConfig()

>>> # Initializing a model from the vit-msn-base style configuration
>>> model = ViTMSNModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ViTMSNModel

### class transformers.ViTMSNModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vit_msn/modeling_vit_msn.py#L415)

( config: ViTMSNConfig use\_mask\_token: bool = False  )

Parameters

* **config** ([ViTMSNConfig](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **use\_mask\_token** (`bool`, *optional*, defaults to `False`) —
  Whether to use a mask token for masked image modeling.

The bare Vit Msn Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vit_msn/modeling_vit_msn.py#L443)

( pixel\_values: typing.Optional[torch.Tensor] = None bool\_masked\_pos: typing.Optional[torch.BoolTensor] = None head\_mask: typing.Optional[torch.Tensor] = None interpolate\_pos\_encoding: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
* **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*) —
  Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **interpolate\_pos\_encoding** (`bool`, *optional*) —
  Whether to interpolate the pre-trained position encodings.

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ViTMSNConfig](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ViTMSNModel](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, ViTMSNModel
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
>>> model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")
>>> inputs = image_processor(images=image, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## ViTMSNForImageClassification

### class transformers.ViTMSNForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vit_msn/modeling_vit_msn.py#L499)

( config: ViTMSNConfig  )

Parameters

* **config** ([ViTMSNConfig](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Vit Msn Model with an image classification head on top e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vit_msn/modeling_vit_msn.py#L512)

( pixel\_values: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None interpolate\_pos\_encoding: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **interpolate\_pos\_encoding** (`bool`, *optional*) —
  Whether to interpolate the pre-trained position encodings.

Returns

[transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ViTMSNConfig](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ViTMSNForImageClassification](/docs/transformers/v4.56.2/en/model_doc/vit_msn#transformers.ViTMSNForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, ViTMSNForImageClassification
>>> import torch
>>> from PIL import Image
>>> import requests

>>> torch.manual_seed(2)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
>>> model = ViTMSNForImageClassification.from_pretrained("facebook/vit-msn-small")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tusker
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/vit_msn.md)
