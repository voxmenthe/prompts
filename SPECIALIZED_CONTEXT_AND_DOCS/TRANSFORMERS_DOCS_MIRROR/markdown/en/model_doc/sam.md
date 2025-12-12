*This model was released on 2023-04-05 and added to Hugging Face Transformers on 2023-04-19.*

# SAM

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

SAM (Segment Anything Model) was proposed in [Segment Anything](https://huggingface.co/papers/2304.02643) by Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick.

The model can be used to predict segmentation masks of any object of interest given an input image.

![example image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-output.png)

The abstract from the paper is the following:

*We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive — often competitive with or even superior to prior fully supervised results. We are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at <https://segment-anything.com> to foster research into foundation models for computer vision.*

Tips:

* The model predicts binary masks that states the presence or not of the object of interest given an image.
* The model predicts much better results if input 2D points and/or input bounding boxes are provided
* You can prompt multiple points for the same image, and predict a single mask.
* Fine-tuning the model is not supported yet
* According to the paper, textual input should be also supported. However, at this time of writing this seems not to be supported according to [the official repository](https://github.com/facebookresearch/segment-anything/issues/4#issuecomment-1497626844).

This model was contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).
The original code can be found [here](https://github.com/facebookresearch/segment-anything).

Below is an example on how to run mask generation given an image and a 2D point:


```
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor, infer_device

device = infer_device()
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

You can also process your own masks alongside the input images in the processor to be passed to the model.


```
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor, infer_device

device = infer_device()
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
mask_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
segmentation_map = Image.open(requests.get(mask_url, stream=True).raw).convert("1")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, segmentation_maps=segmentation_map, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SAM.

* [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb) for using the model.
* [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb) for using the automatic mask generation pipeline.
* [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Run_inference_with_MedSAM_using_HuggingFace_Transformers.ipynb) for inference with MedSAM, a fine-tuned version of SAM on the medical domain. 🌎
* [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb) for fine-tuning the model on custom data. 🌎

## SlimSAM

SlimSAM, a pruned version of SAM, was proposed in [0.1% Data Makes Segment Anything Slim](https://huggingface.co/papers/2312.05284) by Zigeng Chen et al. SlimSAM reduces the size of the SAM models considerably while maintaining the same performance.

Checkpoints can be found on the [hub](https://huggingface.co/models?other=slimsam), and they can be used as a drop-in replacement of SAM.

## Grounded SAM

One can combine [Grounding DINO](grounding-dino) with SAM for text-based mask generation as introduced in [Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks](https://huggingface.co/papers/2401.14159). You can refer to this [demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb) 🌍 for details.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/grounded_sam.png) Grounded SAM overview. Taken from the [original repository](https://github.com/IDEA-Research/Grounded-Segment-Anything).

## SamConfig

### class transformers.SamConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/configuration_sam.py#L254)

( vision\_config = None prompt\_encoder\_config = None mask\_decoder\_config = None initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **vision\_config** (Union[`dict`, `SamVisionConfig`], *optional*) —
  Dictionary of configuration options used to initialize [SamVisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamVisionConfig).
* **prompt\_encoder\_config** (Union[`dict`, `SamPromptEncoderConfig`], *optional*) —
  Dictionary of configuration options used to initialize [SamPromptEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamPromptEncoderConfig).
* **mask\_decoder\_config** (Union[`dict`, `SamMaskDecoderConfig`], *optional*) —
  Dictionary of configuration options used to initialize [SamMaskDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamMaskDecoderConfig).
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[SamConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamConfig) is the configuration class to store the configuration of a [SamModel](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamModel). It is used to instantiate a
SAM model according to the specified arguments, defining the vision model, prompt-encoder model and mask decoder
configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the
SAM-ViT-H [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import (
...     SamVisionConfig,
...     SamPromptEncoderConfig,
...     SamMaskDecoderConfig,
...     SamModel,
... )

>>> # Initializing a SamConfig with `"facebook/sam-vit-huge"` style configuration
>>> configuration = SamConfig()

>>> # Initializing a SamModel (with random weights) from the `"facebook/sam-vit-huge"` style configuration
>>> model = SamModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a SamConfig from a SamVisionConfig, SamPromptEncoderConfig, and SamMaskDecoderConfig

>>> # Initializing SAM vision, SAM Q-Former and language model configurations
>>> vision_config = SamVisionConfig()
>>> prompt_encoder_config = SamPromptEncoderConfig()
>>> mask_decoder_config = SamMaskDecoderConfig()

>>> config = SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)
```

## SamVisionConfig

### class transformers.SamVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/configuration_sam.py#L136)

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

This is the configuration class to store the configuration of a [SamVisionModel](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamVisionModel). It is used to instantiate a SAM
vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
defaults will yield a similar configuration to that of the SAM ViT-h
[facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import (
...     SamVisionConfig,
...     SamVisionModel,
... )

>>> # Initializing a SamVisionConfig with `"facebook/sam-vit-huge"` style configuration
>>> configuration = SamVisionConfig()

>>> # Initializing a SamVisionModel (with random weights) from the `"facebook/sam-vit-huge"` style configuration
>>> model = SamVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## SamMaskDecoderConfig

### class transformers.SamMaskDecoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/configuration_sam.py#L73)

( hidden\_size = 256 hidden\_act = 'relu' mlp\_dim = 2048 num\_hidden\_layers = 2 num\_attention\_heads = 8 attention\_downsample\_rate = 2 num\_multimask\_outputs = 3 iou\_head\_depth = 3 iou\_head\_hidden\_dim = 256 layer\_norm\_eps = 1e-06 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the hidden states.
* **hidden\_act** (`str`, *optional*, defaults to `"relu"`) —
  The non-linear activation function used inside the `SamMaskDecoder` module.
* **mlp\_dim** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 2) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **attention\_downsample\_rate** (`int`, *optional*, defaults to 2) —
  The downsampling rate of the attention layer.
* **num\_multimask\_outputs** (`int`, *optional*, defaults to 3) —
  The number of outputs from the `SamMaskDecoder` module. In the Segment Anything paper, this is set to 3.
* **iou\_head\_depth** (`int`, *optional*, defaults to 3) —
  The number of layers in the IoU head module.
* **iou\_head\_hidden\_dim** (`int`, *optional*, defaults to 256) —
  The dimensionality of the hidden states in the IoU head module.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.

This is the configuration class to store the configuration of a `SamMaskDecoder`. It is used to instantiate a SAM
mask decoder to the specified arguments, defining the model architecture. Instantiating a configuration defaults
will yield a similar configuration to that of the SAM-vit-h
[facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## SamPromptEncoderConfig

### class transformers.SamPromptEncoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/configuration_sam.py#L24)

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

This is the configuration class to store the configuration of a `SamPromptEncoder`. The `SamPromptEncoder`
module is used to encode the input 2D points and bounding boxes. Instantiating a configuration defaults will yield
a similar configuration to that of the SAM-vit-h
[facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## SamProcessor

### class transformers.SamProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/processing_sam.py#L55)

( image\_processor  )

Parameters

* **image\_processor** (`SamImageProcessor`) —
  An instance of [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor). The image processor is a required input.

Constructs a SAM processor which wraps a SAM image processor and an 2D points & Bounding boxes processor into a
single processor.

[SamProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamProcessor) offers all the functionalities of [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor). See the docstring of
`__call__()` for more information.

## SamImageProcessor

### class transformers.SamImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam.py#L67)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None mask\_size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: bool = True pad\_size: typing.Optional[int] = None mask\_pad\_size: typing.Optional[int] = None do\_convert\_rgb: bool = True \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `{"longest_edge" -- 1024}`):
  Size of the output image after resizing. Resizes the longest edge of the image to match
  `size["longest_edge"]` while maintaining the aspect ratio. Can be overridden by the `size` parameter in the
  `preprocess` method.
* **mask\_size** (`dict`, *optional*, defaults to `{"longest_edge" -- 256}`):
  Size of the output segmentation map after resizing. Resizes the longest edge of the image to match
  `size["longest_edge"]` while maintaining the aspect ratio. Can be overridden by the `mask_size` parameter
  in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
  `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Wwhether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the image to the specified `pad_size`. Can be overridden by the `do_pad` parameter in the
  `preprocess` method.
* **pad\_size** (`dict`, *optional*, defaults to `{"height" -- 1024, "width": 1024}`):
  Size of the output image after padding. Can be overridden by the `pad_size` parameter in the `preprocess`
  method.
* **mask\_pad\_size** (`dict`, *optional*, defaults to `{"height" -- 256, "width": 256}`):
  Size of the output segmentation map after padding. Can be overridden by the `mask_pad_size` parameter in
  the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.

Constructs a SAM image processor.

#### filter\_masks

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam.py#L811)

( masks iou\_scores original\_size cropped\_box\_image pred\_iou\_thresh = 0.88 stability\_score\_thresh = 0.95 mask\_threshold = 0 stability\_score\_offset = 1 return\_tensors = 'pt'  )

Parameters

* **masks** (`Union[torch.Tensor, tf.Tensor]`) —
  Input masks.
* **iou\_scores** (`Union[torch.Tensor, tf.Tensor]`) —
  List of IoU scores.
* **original\_size** (`tuple[int,int]`) —
  Size of the original image.
* **cropped\_box\_image** (`np.array`) —
  The cropped image.
* **pred\_iou\_thresh** (`float`, *optional*, defaults to 0.88) —
  The threshold for the iou scores.
* **stability\_score\_thresh** (`float`, *optional*, defaults to 0.95) —
  The threshold for the stability score.
* **mask\_threshold** (`float`, *optional*, defaults to 0) —
  The threshold for the predicted masks.
* **stability\_score\_offset** (`float`, *optional*, defaults to 1) —
  The offset for the stability score used in the `_compute_stability_score` method.
* **return\_tensors** (`str`, *optional*, defaults to `pt`) —
  If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.

Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
bounding boxes and pad the predicted masks if necessary.

#### generate\_crop\_boxes

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam.py#L746)

( image target\_size crop\_n\_layers: int = 0 overlap\_ratio: float = 0.3413333333333333 points\_per\_crop: typing.Optional[int] = 32 crop\_n\_points\_downscale\_factor: typing.Optional[list[int]] = 1 device: typing.Optional[ForwardRef('torch.device')] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None return\_tensors: str = 'pt'  )

Parameters

* **image** (`np.array`) —
  Input original image
* **target\_size** (`int`) —
  Target size of the resized image
* **crop\_n\_layers** (`int`, *optional*, defaults to 0) —
  If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where
  each layer has 2\*\*i\_layer number of image crops.
* **overlap\_ratio** (`float`, *optional*, defaults to 512/1500) —
  Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
  the image length. Later layers with more crops scale down this overlap.
* **points\_per\_crop** (`int`, *optional*, defaults to 32) —
  Number of points to sample from each crop.
* **crop\_n\_points\_downscale\_factor** (`list[int]`, *optional*, defaults to 1) —
  The number of points-per-side sampled in layer n is scaled down by crop\_n\_points\_downscale\_factor\*\*n.
* **device** (`torch.device`, *optional*, defaults to None) —
  Device to use for the computation. If None, cpu will be used.
* **input\_data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format of the input image. If not provided, it will be inferred.
* **return\_tensors** (`str`, *optional*, defaults to `pt`) —
  If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.

Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

#### pad\_image

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam.py#L166)

( image: ndarray pad\_size: dict data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  )

Parameters

* **image** (`np.ndarray`) —
  Image to pad.
* **pad\_size** (`dict[str, int]`) —
  Size of the output image after padding.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The data format of the image. Can be either “channels\_first” or “channels\_last”. If `None`, the
  `data_format` of the `image` will be used.
* **input\_data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format of the input image. If not provided, it will be inferred.

Pad an image to `(pad_size["height"], pad_size["width"])` with zeros to the right and bottom.

#### post\_process\_for\_mask\_generation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam.py#L723)

( all\_masks all\_scores all\_boxes crops\_nms\_thresh return\_tensors = 'pt'  )

Parameters

* **all\_masks** (`Union[list[torch.Tensor], list[tf.Tensor]]`) —
  List of all predicted segmentation masks
* **all\_scores** (`Union[list[torch.Tensor], list[tf.Tensor]]`) —
  List of all predicted iou scores
* **all\_boxes** (`Union[list[torch.Tensor], list[tf.Tensor]]`) —
  List of all bounding boxes of the predicted masks
* **crops\_nms\_thresh** (`float`) —
  Threshold for NMS (Non Maximum Suppression) algorithm.
* **return\_tensors** (`str`, *optional*, defaults to `pt`) —
  If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.

Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.

#### post\_process\_masks

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam.py#L579)

( masks original\_sizes reshaped\_input\_sizes mask\_threshold = 0.0 binarize = True pad\_size = None return\_tensors = 'pt'  ) → (`Union[torch.Tensor, tf.Tensor]`)

Parameters

* **masks** (`Union[list[torch.Tensor], list[np.ndarray], list[tf.Tensor]]`) —
  Batched masks from the mask\_decoder in (batch\_size, num\_channels, height, width) format.
* **original\_sizes** (`Union[torch.Tensor, tf.Tensor, list[tuple[int,int]]]`) —
  The original sizes of each image before it was resized to the model’s expected input shape, in (height,
  width) format.
* **reshaped\_input\_sizes** (`Union[torch.Tensor, tf.Tensor, list[tuple[int,int]]]`) —
  The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
* **mask\_threshold** (`float`, *optional*, defaults to 0.0) —
  The threshold to use for binarizing the masks.
* **binarize** (`bool`, *optional*, defaults to `True`) —
  Whether to binarize the masks.
* **pad\_size** (`int`, *optional*, defaults to `self.pad_size`) —
  The target size the images were padded to before being passed to the model. If None, the target size is
  assumed to be the processor’s `pad_size`.
* **return\_tensors** (`str`, *optional*, defaults to `"pt"`) —
  If `"pt"`, return PyTorch tensors. If `"tf"`, return TensorFlow tensors.

Returns

(`Union[torch.Tensor, tf.Tensor]`)

Batched masks in batch\_size, num\_channels, height, width) format, where
(height, width) is given by original\_size.

Remove padding and upscale masks to the original image size.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam.py#L395)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None mask\_size: typing.Optional[dict[str, int]] = None resample: typing.Optional[ForwardRef('PILImageResampling')] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Union[int, float, NoneType] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: typing.Optional[bool] = None pad\_size: typing.Optional[dict[str, int]] = None mask\_pad\_size: typing.Optional[dict[str, int]] = None do\_convert\_rgb: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  Segmentation map to preprocess.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Controls the size of the image after `resize`. The longest edge of the image is resized to
  `size["longest_edge"]` whilst preserving the aspect ratio.
* **mask\_size** (`dict[str, int]`, *optional*, defaults to `self.mask_size`) —
  Controls the size of the segmentation map after `resize`. The longest edge of the image is resized to
  `size["longest_edge"]` whilst preserving the aspect ratio.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image pixel values by rescaling factor.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to apply to the image pixel values.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to normalize the image by if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether to pad the image.
* **pad\_size** (`dict[str, int]`, *optional*, defaults to `self.pad_size`) —
  Controls the size of the padding applied to the image. The image is padded to `pad_size["height"]` and
  `pad_size["width"]` if `do_pad` is set to `True`.
* **mask\_pad\_size** (`dict[str, int]`, *optional*, defaults to `self.mask_pad_size`) —
  Controls the size of the padding applied to the segmentation map. The image is padded to
  `mask_pad_size["height"]` and `mask_pad_size["width"]` if `do_pad` is set to `True`.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

#### resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam.py#L214)

( image: ndarray size: dict resample: Resampling = <Resampling.BICUBIC: 3> data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  Image to resize.
* **size** (`dict[str, int]`) —
  Dictionary in the format `{"longest_edge": int}` specifying the size of the output image. The longest
  edge of the image will be resized to the specified size, while the other edge will be resized to
  maintain the aspect ratio.
* **resample** —
  `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
* **data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the output image. If unset, the channel dimension format of the input
  image is used. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.

Returns

`np.ndarray`

The resized image.

Resize an image to `(size["height"], size["width"])`.

## SamImageProcessorFast

### class transformers.SamImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam_fast.py#L85)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.sam.image\_processing\_sam\_fast.SamFastImageProcessorKwargs]  )

Constructs a fast Sam image processor.

#### filter\_masks

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam_fast.py#L366)

( masks iou\_scores original\_size cropped\_box\_image pred\_iou\_thresh = 0.88 stability\_score\_thresh = 0.95 mask\_threshold = 0 stability\_score\_offset = 1  )

Parameters

* **masks** (`torch.Tensor`) —
  Input masks.
* **iou\_scores** (`torch.Tensor`) —
  List of IoU scores.
* **original\_size** (`tuple[int,int]`) —
  Size of the original image.
* **cropped\_box\_image** (`torch.Tensor`) —
  The cropped image.
* **pred\_iou\_thresh** (`float`, *optional*, defaults to 0.88) —
  The threshold for the iou scores.
* **stability\_score\_thresh** (`float`, *optional*, defaults to 0.95) —
  The threshold for the stability score.
* **mask\_threshold** (`float`, *optional*, defaults to 0) —
  The threshold for the predicted masks.
* **stability\_score\_offset** (`float`, *optional*, defaults to 1) —
  The offset for the stability score used in the `_compute_stability_score` method.

Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
bounding boxes and pad the predicted masks if necessary.

#### generate\_crop\_boxes

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam_fast.py#L313)

( image: torch.Tensor target\_size crop\_n\_layers: int = 0 overlap\_ratio: float = 0.3413333333333333 points\_per\_crop: typing.Optional[int] = 32 crop\_n\_points\_downscale\_factor: typing.Optional[list[int]] = 1 device: typing.Optional[ForwardRef('torch.device')] = None  )

Parameters

* **image** (`torch.Tensor`) —
  Input original image
* **target\_size** (`int`) —
  Target size of the resized image
* **crop\_n\_layers** (`int`, *optional*, defaults to 0) —
  If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where
  each layer has 2\*\*i\_layer number of image crops.
* **overlap\_ratio** (`float`, *optional*, defaults to 512/1500) —
  Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
  the image length. Later layers with more crops scale down this overlap.
* **points\_per\_crop** (`int`, *optional*, defaults to 32) —
  Number of points to sample from each crop.
* **crop\_n\_points\_downscale\_factor** (`list[int]`, *optional*, defaults to 1) —
  The number of points-per-side sampled in layer n is scaled down by crop\_n\_points\_downscale\_factor\*\*n.
* **device** (`torch.device`, *optional*, defaults to None) —
  Device to use for the computation. If None, cpu will be used.
* **input\_data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format of the input image. If not provided, it will be inferred.
* **return\_tensors** (`str`, *optional*, defaults to `pt`) —
  If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.

Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

#### pad\_image

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam_fast.py#L105)

( images: torch.Tensor pad\_size: SizeDict  )

Pad images to the specified size.

#### post\_process\_for\_mask\_generation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam_fast.py#L498)

( all\_masks all\_scores all\_boxes crops\_nms\_thresh  )

Parameters

* **all\_masks** (`torch.Tensor`) —
  List of all predicted segmentation masks
* **all\_scores** (`torch.Tensor`) —
  List of all predicted iou scores
* **all\_boxes** (`torch.Tensor`) —
  List of all bounding boxes of the predicted masks
* **crops\_nms\_thresh** (`float`) —
  Threshold for NMS (Non Maximum Suppression) algorithm.

Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.

#### post\_process\_masks

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam_fast.py#L445)

( masks original\_sizes reshaped\_input\_sizes mask\_threshold = 0.0 binarize = True pad\_size = None  ) → (`torch.Tensor`)

Parameters

* **masks** (`Union[List[torch.Tensor], List[np.ndarray]]`) —
  Batched masks from the mask\_decoder in (batch\_size, num\_channels, height, width) format.
* **original\_sizes** (`Union[torch.Tensor, List[Tuple[int,int]]]`) —
  The original sizes of each image before it was resized to the model’s expected input shape, in (height,
  width) format.
* **reshaped\_input\_sizes** (`Union[torch.Tensor, List[Tuple[int,int]]]`) —
  The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
* **mask\_threshold** (`float`, *optional*, defaults to 0.0) —
  The threshold to use for binarizing the masks.
* **binarize** (`bool`, *optional*, defaults to `True`) —
  Whether to binarize the masks.
* **pad\_size** (`int`, *optional*, defaults to `self.pad_size`) —
  The target size the images were padded to before being passed to the model. If None, the target size is
  assumed to be the processor’s `pad_size`.

Returns

(`torch.Tensor`)

Batched masks in batch\_size, num\_channels, height, width) format, where (height, width)
is given by original\_size.

Remove padding and upscale masks to the original image size.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam_fast.py#L204)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.sam.image\_processing\_sam\_fast.SamFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  The segmentation maps to preprocess.
* **do\_resize** (`bool`, *optional*) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*) —
  Describes the maximum input dimensions to the model.
* **default\_to\_square** (`bool`, *optional*) —
  Whether to default to a square image when resizing, if size is an int.
* **resample** (`Union[PILImageResampling, F.InterpolationMode, NoneType]`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*) —
  Size of the output image after applying `center_crop`.
* **do\_rescale** (`bool`, *optional*) —
  Whether to rescale the image.
* **rescale\_factor** (`Union[int, float, NoneType]`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*) —
  Whether to normalize the image.
* **image\_mean** (`Union[float, list[float], NoneType]`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`Union[float, list[float], NoneType]`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_convert\_rgb** (`bool`, *optional*) —
  Whether to convert the image to RGB.
* **return\_tensors** (`Union[str, ~utils.generic.TensorType, NoneType]`) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`~image_utils.ChannelDimension`, *optional*) —
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
* **input\_data\_format** (`Union[str, ~image_utils.ChannelDimension, NoneType]`) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the images on. If unset, the device is inferred from the input images.
* **disable\_grouping** (`bool`, *optional*) —
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: <https://github.com/huggingface/transformers/pull/38157>
* **mask\_size** (`dict[str, int]`, *optional*) —
  The size `{"longest_edge": int}` to resize the segmentation maps to.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
  method. If `True`, padding will be applied to the bottom and right of the image with zeros.
* **pad\_size** (`dict[str, int]`, *optional*) —
  The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing.
* **mask\_pad\_size** (`dict[str, int]`, *optional*) —
  The size `{"height": int, "width": int}` to pad the segmentation maps to. Must be larger than any segmentation
  map size provided for preprocessing.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

#### resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/image_processing_sam_fast.py#L125)

( image: torch.Tensor size: SizeDict interpolation: typing.Optional[ForwardRef('F\_t.InterpolationMode')] \*\*kwargs  ) → `torch.Tensor`

Parameters

* **image** (`np.ndarray`) —
  Image to resize.
* **size** (`dict[str, int]`) —
  Dictionary in the format `{"longest_edge": int}` specifying the size of the output image. The longest
  edge of the image will be resized to the specified size, while the other edge will be resized to
  maintain the aspect ratio.
* **interpolation** —
  `F_t.InterpolationMode` filter to use when resizing the image e.g. `F_t.InterpolationMode.BICUBIC`.

Returns

`torch.Tensor`

The resized image.

Resize an image to `(size["height"], size["width"])`.

## SamVisionModel

### class transformers.SamVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/modeling_sam.py#L1085)

( config: SamVisionConfig  )

Parameters

* **config** ([SamVisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamVisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The vision model from Sam without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/modeling_sam.py#L1097)

( pixel\_values: typing.Optional[torch.FloatTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.sam.modeling_sam.SamVisionEncoderOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor). See `SamImageProcessor.__call__()` for details ([SamProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamProcessor) uses
  [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor) for processing images).

Returns

`transformers.models.sam.modeling_sam.SamVisionEncoderOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.sam.modeling_sam.SamVisionEncoderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SamConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamConfig)) and inputs.

* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) — The image embeddings obtained by applying the projection layer to the pooler\_output.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [SamVisionModel](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## SamModel

### class transformers.SamModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/modeling_sam.py#L1112)

( config: SamConfig  )

Parameters

* **config** ([SamConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Segment Anything Model (SAM) for generating segmentation masks, given an input image and
input points and labels, boxes, or masks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam/modeling_sam.py#L1200)

( pixel\_values: typing.Optional[torch.FloatTensor] = None input\_points: typing.Optional[torch.FloatTensor] = None input\_labels: typing.Optional[torch.LongTensor] = None input\_boxes: typing.Optional[torch.FloatTensor] = None input\_masks: typing.Optional[torch.LongTensor] = None image\_embeddings: typing.Optional[torch.FloatTensor] = None multimask\_output: bool = True attention\_similarity: typing.Optional[torch.FloatTensor] = None target\_embedding: typing.Optional[torch.FloatTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.sam.modeling_sam.SamImageSegmentationOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SamImageProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamImageProcessor). See `SamImageProcessor.__call__()` for details ([SamProcessor](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamProcessor) uses
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
  SAM model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
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
* **attention\_similarity** (`torch.FloatTensor`, *optional*) —
  Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
  model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
* **target\_embedding** (`torch.FloatTensor`, *optional*) —
  Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
  the model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).

Returns

`transformers.models.sam.modeling_sam.SamImageSegmentationOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.sam.modeling_sam.SamImageSegmentationOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SamConfig](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamConfig)) and inputs.

* **iou\_scores** (`torch.FloatTensor` of shape `(batch_size, num_masks)`) — The iou scores of the predicted masks.
* **pred\_masks** (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`) — The predicted low resolutions masks. Needs to be post-processed by the processor
* **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
* **vision\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **mask\_decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [SamModel](/docs/transformers/v4.56.2/en/model_doc/sam#transformers.SamModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoModel, AutoProcessor

>>> model = AutoModel.from_pretrained("facebook/sam-vit-base")
>>> processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

>>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
>>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
>>> input_points = [[[400, 650]]]  # 2D location of a window on the car
>>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

>>> # Get segmentation mask
>>> outputs = model(**inputs)

>>> # Postprocess masks
>>> masks = processor.post_process_masks(
...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
... )
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/sam.md)
