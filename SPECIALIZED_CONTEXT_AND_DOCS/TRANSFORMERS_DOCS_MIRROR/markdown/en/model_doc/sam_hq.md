*This model was released on 2023-06-02 and added to Hugging Face Transformers on 2025-04-28.*

# SAM-HQ

## Overview

SAM-HQ (High-Quality Segment Anything Model) was proposed in [Segment Anything in High Quality](https://huggingface.co/papers/2306.01567) by Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu.

The model is an enhancement to the original SAM model that produces significantly higher quality segmentation masks while maintaining SAM’s original promptable design, efficiency, and zero-shot generalizability.

![example image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-output.png)

SAM-HQ introduces several key improvements over the original SAM model:

1. High-Quality Output Token: A learnable token injected into SAM’s mask decoder for higher quality mask prediction
2. Global-local Feature Fusion: Combines features from different stages of the model for improved mask details
3. Training Data: Uses a carefully curated dataset of 44K high-quality masks instead of SA-1B
4. Efficiency: Adds only 0.5% additional parameters while significantly improving mask quality
5. Zero-shot Capability: Maintains SAM’s strong zero-shot performance while improving accuracy

The abstract from the paper is the following:

*The recent Segment Anything Model (SAM) represents a big leap in scaling up segmentation models, allowing for powerful zero-shot capabilities and flexible prompting. Despite being trained with 1.1 billion masks, SAM’s mask prediction quality falls short in many cases, particularly when dealing with objects that have intricate structures. We propose HQ-SAM, equipping SAM with the ability to accurately segment any object, while maintaining SAM’s original promptable design, efficiency, and zero-shot generalizability. Our careful design reuses and preserves the pre-trained model weights of SAM, while only introducing minimal additional parameters and computation. We design a learnable High-Quality Output Token, which is injected into SAM’s mask decoder and is responsible for predicting the high-quality mask. Instead of only applying it on mask-decoder features, we first fuse them with early and final ViT features for improved mask details. To train our introduced learnable parameters, we compose a dataset of 44K fine-grained masks from several sources. HQ-SAM is only trained on the introduced dataset of 44k masks, which takes only 4 hours on 8 GPUs.*

Tips:

* SAM-HQ produces higher quality masks than the original SAM model, particularly for objects with intricate structures and fine details
* The model predicts binary masks with more accurate boundaries and better handling of thin structures
* Like SAM, the model performs better with input 2D points and/or input bounding boxes
* You can prompt multiple points for the same image and predict a single high-quality mask
* The model maintains SAM’s zero-shot generalization capabilities
* SAM-HQ only adds ~0.5% additional parameters compared to SAM
* Fine-tuning the model is not supported yet

This model was contributed by [sushmanth](https://huggingface.co/sushmanth).
The original code can be found [here](https://github.com/SysCV/SAM-HQ).

Below is an example on how to run mask generation given an image and a 2D point:


```
import torch
from PIL import Image
import requests
from transformers import infer_device, SamHQModel, SamHQProcessor

device = infer_device()
model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base").to(device)
processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

You can also process your own masks alongside the input images in the processor to be passed to the model:


```
import torch
from PIL import Image
import requests
from transformers import infer_device, SamHQModel, SamHQProcessor

device = infer_device()
model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base").to(device)
processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
mask_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
segmentation_map = Image.open(requests.get(mask_url, stream=True).raw).convert("1")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, segmentation_maps=segmentation_map, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SAM-HQ:

* Demo notebook for using the model (coming soon)
* Paper implementation and code: [SAM-HQ GitHub Repository](https://github.com/SysCV/SAM-HQ)

## SamHQConfig

### class transformers.SamHQConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam_hq/configuration_sam_hq.py#L259)

( vision\_config = None prompt\_encoder\_config = None mask\_decoder\_config = None initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **vision\_config** (Union[`dict`, `SamHQVisionConfig`], *optional*) —
  Dictionary of configuration options used to initialize [SamHQVisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQVisionConfig).
* **prompt\_encoder\_config** (Union[`dict`, `SamHQPromptEncoderConfig`], *optional*) —
  Dictionary of configuration options used to initialize [SamHQPromptEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQPromptEncoderConfig).
* **mask\_decoder\_config** (Union[`dict`, `SamHQMaskDecoderConfig`], *optional*) —
  Dictionary of configuration options used to initialize [SamHQMaskDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQMaskDecoderConfig).
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[SamHQConfig](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQConfig) is the configuration class to store the configuration of a [SamHQModel](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQModel). It is used to instantiate a
SAM-HQ model according to the specified arguments, defining the vision model, prompt-encoder model and mask decoder
configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the
SAM-HQ-ViT-H [sushmanth/sam\_hq\_vit\_h](https://huggingface.co/sushmanth/sam_hq_vit_h) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## SamHQVisionConfig

### class transformers.SamHQVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam_hq/configuration_sam_hq.py#L75)

( hidden\_size = 768 output\_channels = 256 num\_hidden\_layers = 12 num\_attention\_heads = 12 num\_channels = 3 image\_size = 1024 patch\_size = 16 hidden\_act = 'gelu' layer\_norm\_eps = 1e-06 attention\_dropout = 0.0 initializer\_range = 1e-10 qkv\_bias = True mlp\_ratio = 4.0 use\_abs\_pos = True use\_rel\_pos = True window\_size = 14 global\_attn\_indexes = [2, 5, 8, 11] num\_pos\_feats = 128 mlp\_dim = None \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **output\_channels** (`int`, *optional*, defaults to 256) —
  Dimensionality of the output channels in the Patch Encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  Number of channels in the input image.
* **image\_size** (`int`, *optional*, defaults to 1024) —
  Expected resolution. Target size of the resized input image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  Size of the patches to be extracted from the input image.
* **hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string)
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 1e-10) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to query, key, value projections.
* **mlp\_ratio** (`float`, *optional*, defaults to 4.0) —
  Ratio of mlp hidden dim to embedding dim.
* **use\_abs\_pos** (`bool`, *optional*, defaults to `True`) —
  Whether to use absolute position embedding.
* **use\_rel\_pos** (`bool`, *optional*, defaults to `True`) —
  Whether to use relative position embedding.
* **window\_size** (`int`, *optional*, defaults to 14) —
  Window size for relative position.
* **global\_attn\_indexes** (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`) —
  The indexes of the global attention layers.
* **num\_pos\_feats** (`int`, *optional*, defaults to 128) —
  The dimensionality of the position embedding.
* **mlp\_dim** (`int`, *optional*) —
  The dimensionality of the MLP layer in the Transformer encoder. If `None`, defaults to `mlp_ratio * hidden_size`.

This is the configuration class to store the configuration of a [SamHQVisionModel](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQVisionModel). It is used to instantiate a SAM\_HQ
vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
defaults will yield a similar configuration to that of the SAM\_HQ ViT-h
[facebook/sam\_hq-vit-huge](https://huggingface.co/facebook/sam_hq-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import (
...     SamHQVisionConfig,
...     SamHQVisionModel,
... )

>>> # Initializing a SamHQVisionConfig with `"facebook/sam_hq-vit-huge"` style configuration
>>> configuration = SamHQVisionConfig()

>>> # Initializing a SamHQVisionModel (with random weights) from the `"facebook/sam_hq-vit-huge"` style configuration
>>> model = SamHQVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## SamHQMaskDecoderConfig

### class transformers.SamHQMaskDecoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam_hq/configuration_sam_hq.py#L193)

( hidden\_size = 256 hidden\_act = 'relu' mlp\_dim = 2048 num\_hidden\_layers = 2 num\_attention\_heads = 8 attention\_downsample\_rate = 2 num\_multimask\_outputs = 3 iou\_head\_depth = 3 iou\_head\_hidden\_dim = 256 layer\_norm\_eps = 1e-06 vit\_dim = 768 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the hidden states.
* **hidden\_act** (`str`, *optional*, defaults to `"relu"`) —
  The non-linear activation function used inside the `SamHQMaskDecoder` module.
* **mlp\_dim** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 2) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **attention\_downsample\_rate** (`int`, *optional*, defaults to 2) —
  The downsampling rate of the attention layer.
* **num\_multimask\_outputs** (`int`, *optional*, defaults to 3) —
  The number of outputs from the `SamHQMaskDecoder` module. In the Segment Anything paper, this is set to 3.
* **iou\_head\_depth** (`int`, *optional*, defaults to 3) —
  The number of layers in the IoU head module.
* **iou\_head\_hidden\_dim** (`int`, *optional*, defaults to 256) —
  The dimensionality of the hidden states in the IoU head module.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **vit\_dim** (`int`, *optional*, defaults to 768) —
  Dimensionality of the Vision Transformer (ViT) used in the `SamHQMaskDecoder` module.

This is the configuration class to store the configuration of a `SamHQMaskDecoder`. It is used to instantiate a SAM\_HQ
mask decoder to the specified arguments, defining the model architecture. Instantiating a configuration defaults
will yield a similar configuration to that of the SAM\_HQ-vit-h
[facebook/sam\_hq-vit-huge](https://huggingface.co/facebook/sam_hq-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## SamHQPromptEncoderConfig

### class transformers.SamHQPromptEncoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam_hq/configuration_sam_hq.py#L26)

( hidden\_size = 256 image\_size = 1024 patch\_size = 16 mask\_input\_channels = 16 num\_point\_embeddings = 4 hidden\_act = 'gelu' layer\_norm\_eps = 1e-06 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the hidden states.
* **image\_size** (`int`, *optional*, defaults to 1024) —
  The expected output resolution of the image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **mask\_input\_channels** (`int`, *optional*, defaults to 16) —
  The number of channels to be fed to the `MaskDecoder` module.
* **num\_point\_embeddings** (`int`, *optional*, defaults to 4) —
  The number of point embeddings to be used.
* **hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function in the encoder and pooler.

This is the configuration class to store the configuration of a `SamHQPromptEncoderModel`.The `SamHQPromptEncoderModel`
module is used to encode the input 2D points and bounding boxes. Instantiating a configuration defaults will yield a
similar configuration to that of the SAM\_HQ model. The configuration is used to store the configuration of the model.
[Uminosachi/sam-hq](https://huggingface.co/Uminosachi/sam-hq) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model’s output.Read the documentation from
[PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## SamHQProcessor

### class transformers.SamHQProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam_hq/processing_samhq.py#L52)

( image\_processor  )

Parameters

* **image\_processor** (`SamImageProcessor`) —
  An instance of [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor). The image processor is a required input.

Constructs a SAM HQ processor which wraps a SAM image processor and an 2D points & Bounding boxes processor into a
single processor.

[SamHQProcessor](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQProcessor) offers all the functionalities of [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor). See the docstring of
`__call__()` for more information.

## SamHQVisionModel

### class transformers.SamHQVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam_hq/modeling_sam_hq.py#L1054)

( config: SamHQVisionConfig  )

Parameters

* **config** ([SamHQVisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQVisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The vision model from SamHQ without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam_hq/modeling_sam_hq.py#L1066)

( pixel\_values: typing.Optional[torch.FloatTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.sam_hq.modeling_sam_hq.SamHQVisionEncoderOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor). See `SamImageProcessor.__call__()` for details ([SamHQProcessor](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQProcessor) uses
  [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor) for processing images).

Returns

`transformers.models.sam_hq.modeling_sam_hq.SamHQVisionEncoderOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.sam_hq.modeling_sam_hq.SamHQVisionEncoderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SamHQConfig](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQConfig)) and inputs.

* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) — The image embeddings obtained by applying the projection layer to the pooler\_output.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **intermediate\_embeddings** (`list(torch.FloatTensor)`, *optional*) — A list of intermediate embeddings collected from certain blocks within the model, typically those without
  windowed attention. Each element in the list is of shape `(batch_size, sequence_length, hidden_size)`.
  This is specific to SAM-HQ and not present in base SAM.

The [SamHQVisionModel](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## SamHQModel

### class transformers.SamHQModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam_hq/modeling_sam_hq.py#L1235)

( config  )

Parameters

* **config** ([SamHQModel](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Segment Anything Model HQ (SAM-HQ) for generating masks, given an input image and optional 2D location and bounding boxes.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam_hq/modeling_sam_hq.py#L1323)

( pixel\_values: typing.Optional[torch.FloatTensor] = None input\_points: typing.Optional[torch.FloatTensor] = None input\_labels: typing.Optional[torch.LongTensor] = None input\_boxes: typing.Optional[torch.FloatTensor] = None input\_masks: typing.Optional[torch.LongTensor] = None image\_embeddings: typing.Optional[torch.FloatTensor] = None multimask\_output: bool = True hq\_token\_only: bool = False attention\_similarity: typing.Optional[torch.FloatTensor] = None target\_embedding: typing.Optional[torch.FloatTensor] = None intermediate\_embeddings: typing.Optional[list[torch.FloatTensor]] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  )

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor). See `SamImageProcessor.__call__()` for details ([SamHQProcessor](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQProcessor) uses
  [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor) for processing images).
* **input\_points** (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`) —
  Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
  better results. The points can be obtained by passing a list of list of list to the processor that will
  create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
  second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
  per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
  multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
  coordinates of the point. If a different number of points is passed either for each image, or for each
  mask, the processor will create “PAD” points that will correspond to the (0, 0) coordinate, and the
  computation of the embedding will be skipped for these points using the labels.
* **input\_labels** (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`) —
  Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
  official implementation, there are 3 types of labels
  + `1`: the point is a point that contains the object of interest
  + `0`: the point is a point that does not contain the object of interest
  + `-1`: the point corresponds to the background

  We added the label:

  + `-10`: the point is a padding point, thus should be ignored by the prompt encoder

  The padding labels should be automatically done by the processor.
* **input\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`) —
  Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
  much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
  that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
  size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
  In the order (`x1`, `y1`, `x2`, `y2`):
  + `x1`: the x coordinate of the top left point of the input box
  + `y1`: the y coordinate of the top left point of the input box
  + `x2`: the x coordinate of the bottom right point of the input box
  + `y2`: the y coordinate of the bottom right point of the input box
* **input\_masks** (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`) —
  SAM\_HQ model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
  generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
  manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).
* **image\_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`) —
  Image embeddings, this is used by the mask decder to generate masks and iou scores. For more memory
  efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
  method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
* **multimask\_output** (`bool`, *optional*) —
  In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
  bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
  “best” mask, by specifying `multimask_output=False`.
* **hq\_token\_only** (`bool`, *optional*, defaults to `False`) —
  Whether to use only the HQ token path for mask generation. When False, combines both standard and HQ paths.
  This is specific to SAM-HQ’s architecture.
* **attention\_similarity** (`torch.FloatTensor`, *optional*) —
  Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
  model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
* **target\_embedding** (`torch.FloatTensor`, *optional*) —
  Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
  the model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
* **intermediate\_embeddings** (`List[torch.FloatTensor]`, *optional*) —
  Intermediate embeddings from vision encoder’s non-windowed blocks, used by SAM-HQ for enhanced mask quality.
  Required when providing pre-computed image\_embeddings instead of pixel\_values.

The [SamHQModel](/docs/transformers/v4.56.2/en/model_doc/sam_hq#transformers.SamHQModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoModel, AutoProcessor

>>> model = AutoModel.from_pretrained("sushmanth/sam_hq_vit_b")
>>> processor = AutoProcessor.from_pretrained("sushmanth/sam_hq_vit_b")

>>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
>>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
>>> input_points = [[[400, 650]]]  # 2D location of a window on the car
>>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

>>> # Get high-quality segmentation mask
>>> outputs = model(**inputs)

>>> # For high-quality mask only
>>> outputs = model(**inputs, hq_token_only=True)

>>> # Postprocess masks
>>> masks = processor.post_process_masks(
...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
... )
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/sam_hq.md)
