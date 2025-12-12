# TimmWrapper

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

Helper class to enable loading timm models to be used with the transformers library and its autoclasses.


```
>>> import torch
>>> from PIL import Image
>>> from urllib.request import urlopen
>>> from transformers import AutoModelForImageClassification, AutoImageProcessor

>>> # Load image
>>> image = Image.open(urlopen(
...     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
... ))

>>> # Load model and image processor
>>> checkpoint = "timm/resnet50.a1_in1k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModelForImageClassification.from_pretrained(checkpoint).eval()

>>> # Preprocess image
>>> inputs = image_processor(image)

>>> # Forward pass
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # Get top 5 predictions
>>> top5_probabilities, top5_class_indices = torch.topk(logits.softmax(dim=1) * 100, k=5)
```

## Resources:

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with TimmWrapper.

Image Classification

* [Collection of Example Notebook](https://github.com/ariG23498/timm-wrapper-examples) 🌎

For a more detailed overview please read the [official blog post](https://huggingface.co/blog/timm-transformers) on the timm integration.

## TimmWrapperConfig

### class transformers.TimmWrapperConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timm_wrapper/configuration_timm_wrapper.py#L31)

( initializer\_range: float = 0.02 do\_pooling: bool = True model\_args: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Parameters

* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **do\_pooling** (`bool`, *optional*, defaults to `True`) —
  Whether to do pooling for the last\_hidden\_state in `TimmWrapperModel` or not.
* **model\_args** (`dict[str, Any]`, *optional*) —
  Additional keyword arguments to pass to the `timm.create_model` function. e.g. `model_args={"depth": 3}`
  for `timm/vit_base_patch32_clip_448.laion2b_ft_in12k_in1k` to create a model with 3 blocks. Defaults to `None`.

This is the configuration class to store the configuration for a timm backbone `TimmWrapper`.

It is used to instantiate a timm model according to the specified arguments, defining the model.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Config loads imagenet label descriptions and stores them in `id2label` attribute, `label2id` attribute for default
imagenet models is set to `None` due to occlusions in the label descriptions.

Example:


```
>>> from transformers import TimmWrapperModel

>>> # Initializing a timm model
>>> model = TimmWrapperModel.from_pretrained("timm/resnet18.a1_in1k")

>>> # Accessing the model configuration
>>> configuration = model.config
```

## TimmWrapperImageProcessor

### class transformers.TimmWrapperImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timm_wrapper/image_processing_timm_wrapper.py#L39)

( pretrained\_cfg: dict architecture: typing.Optional[str] = None \*\*kwargs  )

Parameters

* **pretrained\_cfg** (`dict[str, Any]`) —
  The configuration of the pretrained model used to resolve evaluation and
  training transforms.
* **architecture** (`Optional[str]`, *optional*) —
  Name of the architecture of the model.

Wrapper class for timm models to be used within transformers.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timm_wrapper/image_processing_timm_wrapper.py#L97)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = 'pt'  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return.

Preprocess an image or batch of images.

## TimmWrapperModel

### class transformers.TimmWrapperModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timm_wrapper/modeling_timm_wrapper.py#L133)

( config: TimmWrapperConfig  )

Wrapper class for timm models to be used in transformers.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timm_wrapper/modeling_timm_wrapper.py#L145)

( pixel\_values: FloatTensor output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Union[bool, list[int], NoneType] = None return\_dict: typing.Optional[bool] = None do\_pooling: typing.Optional[bool] = None \*\*kwargs  ) → `transformers.models.timm_wrapper.modeling_timm_wrapper.TimmWrapperModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [TimmWrapperImageProcessor](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperImageProcessor). See [TimmWrapperImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [TimmWrapperImageProcessor](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. Not compatible with timm wrapped models.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. Not compatible with timm wrapped models.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **do\_pooling** (`bool`, *optional*) —
  Whether to do pooling for the last\_hidden\_state in `TimmWrapperModel` or not. If `None` is passed, the
  `do_pooling` value from the config is used.

Returns

`transformers.models.timm_wrapper.modeling_timm_wrapper.TimmWrapperModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.timm_wrapper.modeling_timm_wrapper.TimmWrapperModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TimmWrapperConfig](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperConfig)) and inputs.

* **last\_hidden\_state** (`<class 'torch.FloatTensor'>.last_hidden_state`) — The last hidden state of the model, output before applying the classification head.
* **pooler\_output** (`torch.FloatTensor`, *optional*) — The pooled output derived from the last hidden state, if applicable.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_hidden_states=True` is set or if `config.output_hidden_states=True`) — A tuple containing the intermediate hidden states of the model at the output of each layer or specified layers.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_attentions=True` is set or if `config.output_attentions=True`.) — A tuple containing the intermediate attention weights of the model at the output of each layer.
  Note: Currently, Timm models do not support attentions output.

The [TimmWrapperModel](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from PIL import Image
>>> from urllib.request import urlopen
>>> from transformers import AutoModel, AutoImageProcessor

>>> # Load image
>>> image = Image.open(urlopen(
...     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
... ))

>>> # Load model and image processor
>>> checkpoint = "timm/resnet50.a1_in1k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModel.from_pretrained(checkpoint).eval()

>>> # Preprocess image
>>> inputs = image_processor(image)

>>> # Forward pass
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Get pooled output
>>> pooled_output = outputs.pooler_output

>>> # Get last hidden state
>>> last_hidden_state = outputs.last_hidden_state
```

## TimmWrapperForImageClassification

### class transformers.TimmWrapperForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timm_wrapper/modeling_timm_wrapper.py#L242)

( config: TimmWrapperConfig  )

Wrapper class for timm models to be used in transformers for image classification.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/timm_wrapper/modeling_timm_wrapper.py#L264)

( pixel\_values: FloatTensor labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Union[bool, list[int], NoneType] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [TimmWrapperImageProcessor](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperImageProcessor). See [TimmWrapperImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [TimmWrapperImageProcessor](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. Not compatible with timm wrapped models.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. Not compatible with timm wrapped models.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
  \*\*kwargs:
  Additional keyword arguments passed along to the `timm` model forward.

Returns

[transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TimmWrapperConfig](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [TimmWrapperForImageClassification](/docs/transformers/v4.56.2/en/model_doc/timm_wrapper#transformers.TimmWrapperForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import torch
>>> from PIL import Image
>>> from urllib.request import urlopen
>>> from transformers import AutoModelForImageClassification, AutoImageProcessor

>>> # Load image
>>> image = Image.open(urlopen(
...     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
... ))

>>> # Load model and image processor
>>> checkpoint = "timm/resnet50.a1_in1k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModelForImageClassification.from_pretrained(checkpoint).eval()

>>> # Preprocess image
>>> inputs = image_processor(image)

>>> # Forward pass
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # Get top 5 predictions
>>> top5_probabilities, top5_class_indices = torch.topk(logits.softmax(dim=1) * 100, k=5)
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/timm_wrapper.md)
