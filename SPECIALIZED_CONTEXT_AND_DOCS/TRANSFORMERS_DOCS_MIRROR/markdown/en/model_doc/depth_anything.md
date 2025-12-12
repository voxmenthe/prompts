*This model was released on 2024-01-19 and added to Hugging Face Transformers on 2024-01-25.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# Depth Anything

[Depth Anything](https://huggingface.co/papers/2401.10891) is designed to be a foundation model for monocular depth estimation (MDE). It is jointly trained on labeled and ~62M unlabeled images to enhance the dataset. It uses a pretrained [DINOv2](./dinov2) model as an image encoder to inherit its existing rich semantic priors, and [DPT](./dpt) as the decoder. A teacher model is trained on unlabeled images to create pseudo-labels. The student model is trained on a combination of the pseudo-labels and labeled images. To improve the student model’s performance, strong perturbations are added to the unlabeled images to challenge the student model to learn more visual knowledge from the image.

You can find all the original Depth Anything checkpoints under the [Depth Anything](https://huggingface.co/collections/LiheYoung/depth-anything-release-65b317de04eec72abf6b55aa) collection.

Click on the Depth Anything models in the right sidebar for more examples of how to apply Depth Anything to different vision tasks.

The example below demonstrates how to obtain a depth map with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", dtype=torch.bfloat16, device=0)
pipe("http://images.cocodataset.org/val2017/000000039769.jpg")["depth"]
```

## Notes

* [DepthAnythingV2](./depth_anything_v2), released in June 2024, uses the same architecture as Depth Anything and is compatible with all code examples and existing workflows. It uses synthetic data and a larger capacity teacher model to achieve much finer and robust depth predictions.

## DepthAnythingConfig

### class transformers.DepthAnythingConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_anything/configuration_depth_anything.py#L28)

( backbone\_config = None backbone = None use\_pretrained\_backbone = False use\_timm\_backbone = False backbone\_kwargs = None patch\_size = 14 initializer\_range = 0.02 reassemble\_hidden\_size = 384 reassemble\_factors = [4, 2, 1, 0.5] neck\_hidden\_sizes = [48, 96, 192, 384] fusion\_hidden\_size = 64 head\_in\_index = -1 head\_hidden\_size = 32 depth\_estimation\_type = 'relative' max\_depth = None \*\*kwargs  )

Parameters

* **backbone\_config** (`Union[dict[str, Any], PretrainedConfig]`, *optional*) —
  The configuration of the backbone model. Only used in case `is_hybrid` is `True` or in case you want to
  leverage the [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone) API.
* **backbone** (`str`, *optional*) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to use pretrained weights for the backbone.
* **use\_timm\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone)
  API.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The size of the patches to extract from the backbone features.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **reassemble\_hidden\_size** (`int`, *optional*, defaults to 384) —
  The number of input channels of the reassemble layers.
* **reassemble\_factors** (`list[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`) —
  The up/downsampling factors of the reassemble layers.
* **neck\_hidden\_sizes** (`list[str]`, *optional*, defaults to `[48, 96, 192, 384]`) —
  The hidden sizes to project to for the feature maps of the backbone.
* **fusion\_hidden\_size** (`int`, *optional*, defaults to 64) —
  The number of channels before fusion.
* **head\_in\_index** (`int`, *optional*, defaults to -1) —
  The index of the features to use in the depth estimation head.
* **head\_hidden\_size** (`int`, *optional*, defaults to 32) —
  The number of output channels in the second convolution of the depth estimation head.
* **depth\_estimation\_type** (`str`, *optional*, defaults to `"relative"`) —
  The type of depth estimation to use. Can be one of `["relative", "metric"]`.
* **max\_depth** (`float`, *optional*) —
  The maximum depth to use for the “metric” depth estimation head. 20 should be used for indoor models
  and 80 for outdoor models. For “relative” depth estimation, this value is ignored.

This is the configuration class to store the configuration of a `DepthAnythingModel`. It is used to instantiate a DepthAnything
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the DepthAnything
[LiheYoung/depth-anything-small-hf](https://huggingface.co/LiheYoung/depth-anything-small-hf) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation

>>> # Initializing a DepthAnything small style configuration
>>> configuration = DepthAnythingConfig()

>>> # Initializing a model from the DepthAnything small style configuration
>>> model = DepthAnythingForDepthEstimation(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_anything/configuration_depth_anything.py#L162)

( )

Serializes this instance to a Python dictionary. Override the default [to\_dict()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.to_dict). Returns:
`dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,

## DepthAnythingForDepthEstimation

### class transformers.DepthAnythingForDepthEstimation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_anything/modeling_depth_anything.py#L331)

( config  )

Parameters

* **config** ([DepthAnythingForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/depth_anything#transformers.DepthAnythingForDepthEstimation)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Depth Anything Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/depth_anything/modeling_depth_anything.py#L344)

( pixel\_values: FloatTensor labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessor). See [DPTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTFeatureExtractor.__call__) for details (`processor_class` uses
  [DPTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/dpt#transformers.DPTImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Ground truth depth estimation maps for computing the loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DepthAnythingConfig](/docs/transformers/v4.56.2/en/model_doc/depth_anything#transformers.DepthAnythingConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **predicted\_depth** (`torch.FloatTensor` of shape `(batch_size, height, width)`) — Predicted depth for each pixel.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DepthAnythingForDepthEstimation](/docs/transformers/v4.56.2/en/model_doc/depth_anything#transformers.DepthAnythingForDepthEstimation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
>>> model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # interpolate to original size
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     target_sizes=[(image.height, image.width)],
... )

>>> # visualize the prediction
>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = predicted_depth * 255 / predicted_depth.max()
>>> depth = depth.detach().cpu().numpy()
>>> depth = Image.fromarray(depth.astype("uint8"))
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/depth_anything.md)
