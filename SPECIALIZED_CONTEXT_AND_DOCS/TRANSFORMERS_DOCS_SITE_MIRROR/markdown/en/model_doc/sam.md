# SAM

## Overview

SAM (Segment Anything Model) was proposed in [Segment Anything](https://huggingface.co/papers/2304.02643) by Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick.

The model can be used to predict segmentation masks of any object of interest given an input image.

![example image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-output.png)

The abstract from the paper is the following:

*We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive -- often competitive with or even superior to prior fully supervised results. We are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at [https://segment-anything.com](https://segment-anything.com) to foster research into foundation models for computer vision.*

Tips:

- The model predicts binary masks that states the presence or not of the object of interest given an image.
- The model predicts much better results if input 2D points and/or input bounding boxes are provided
- You can prompt multiple points for the same image, and predict a single mask.
- Fine-tuning the model is not supported yet
- According to the paper, textual input should be also supported. However, at this time of writing this seems not to be supported according to [the official repository](https://github.com/facebookresearch/segment-anything/issues/4#issuecomment-1497626844).

This model was contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).
The original code can be found [here](https://github.com/facebookresearch/segment-anything).

Below is an example on how to run mask generation given an image and a 2D point:

```python
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
from accelerate import Accelerator

device = Accelerator().device
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

```python
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
from accelerate import Accelerator

device = Accelerator().device
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

- [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb) for using the model.
- [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb) for using the automatic mask generation pipeline.
- [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Run_inference_with_MedSAM_using_HuggingFace_Transformers.ipynb) for inference with MedSAM, a fine-tuned version of SAM on the medical domain. 🌎
- [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb) for fine-tuning the model on custom data. 🌎

## SlimSAM

SlimSAM, a pruned version of SAM, was proposed in [0.1% Data Makes Segment Anything Slim](https://huggingface.co/papers/2312.05284) by Zigeng Chen et al. SlimSAM reduces the size of the SAM models considerably while maintaining the same performance.

Checkpoints can be found on the [hub](https://huggingface.co/models?other=slimsam), and they can be used as a drop-in replacement of SAM.

## Grounded SAM

One can combine [Grounding DINO](grounding-dino) with SAM for text-based mask generation as introduced in [Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks](https://huggingface.co/papers/2401.14159). You can refer to this [demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb) 🌍 for details.

 Grounded SAM overview. Taken from the original repository. 

## SamConfig[[transformers.SamConfig]]

#### transformers.SamConfig[[transformers.SamConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/configuration_sam.py#L254)

[SamConfig](/docs/transformers/main/en/model_doc/sam#transformers.SamConfig) is the configuration class to store the configuration of a [SamModel](/docs/transformers/main/en/model_doc/sam#transformers.SamModel). It is used to instantiate a
SAM model according to the specified arguments, defining the vision model, prompt-encoder model and mask decoder
configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the
SAM-ViT-H [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
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

**Parameters:**

vision_config (Union[`dict`, `SamVisionConfig`], *optional*) : Dictionary of configuration options used to initialize [SamVisionConfig](/docs/transformers/main/en/model_doc/sam#transformers.SamVisionConfig).

prompt_encoder_config (Union[`dict`, `SamPromptEncoderConfig`], *optional*) : Dictionary of configuration options used to initialize [SamPromptEncoderConfig](/docs/transformers/main/en/model_doc/sam#transformers.SamPromptEncoderConfig).

mask_decoder_config (Union[`dict`, `SamMaskDecoderConfig`], *optional*) : Dictionary of configuration options used to initialize [SamMaskDecoderConfig](/docs/transformers/main/en/model_doc/sam#transformers.SamMaskDecoderConfig). 

kwargs (*optional*) : Dictionary of keyword arguments.

## SamVisionConfig[[transformers.SamVisionConfig]]

#### transformers.SamVisionConfig[[transformers.SamVisionConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/configuration_sam.py#L136)

This is the configuration class to store the configuration of a [SamVisionModel](/docs/transformers/main/en/model_doc/sam#transformers.SamVisionModel). It is used to instantiate a SAM
vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
defaults will yield a similar configuration to that of the SAM ViT-h
[facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
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

**Parameters:**

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

output_channels (`int`, *optional*, defaults to 256) : Dimensionality of the output channels in the Patch Encoder.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

num_channels (`int`, *optional*, defaults to 3) : Number of channels in the input image.

image_size (`int`, *optional*, defaults to 1024) : Expected resolution. Target size of the resized input image.

patch_size (`int`, *optional*, defaults to 16) : Size of the patches to be extracted from the input image.

hidden_act (`str`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string)

layer_norm_eps (`float`, *optional*, defaults to 1e-06) : The epsilon used by the layer normalization layers.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 1e-10) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to query, key, value projections.

mlp_ratio (`float`, *optional*, defaults to 4.0) : Ratio of mlp hidden dim to embedding dim.

use_abs_pos (`bool`, *optional*, defaults to `True`) : Whether to use absolute position embedding.

use_rel_pos (`bool`, *optional*, defaults to `True`) : Whether to use relative position embedding.

window_size (`int`, *optional*, defaults to 14) : Window size for relative position.

global_attn_indexes (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`) : The indexes of the global attention layers.

num_pos_feats (`int`, *optional*, defaults to 128) : The dimensionality of the position embedding.

mlp_dim (`int`, *optional*) : The dimensionality of the MLP layer in the Transformer encoder. If `None`, defaults to `mlp_ratio * hidden_size`.

## SamMaskDecoderConfig[[transformers.SamMaskDecoderConfig]]

#### transformers.SamMaskDecoderConfig[[transformers.SamMaskDecoderConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/configuration_sam.py#L73)

This is the configuration class to store the configuration of a `SamMaskDecoder`. It is used to instantiate a SAM
mask decoder to the specified arguments, defining the model architecture. Instantiating a configuration defaults
will yield a similar configuration to that of the SAM-vit-h
[facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

**Parameters:**

hidden_size (`int`, *optional*, defaults to 256) : Dimensionality of the hidden states.

hidden_act (`str`, *optional*, defaults to `"relu"`) : The non-linear activation function used inside the `SamMaskDecoder` module.

mlp_dim (`int`, *optional*, defaults to 2048) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

num_hidden_layers (`int`, *optional*, defaults to 2) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 8) : Number of attention heads for each attention layer in the Transformer encoder.

attention_downsample_rate (`int`, *optional*, defaults to 2) : The downsampling rate of the attention layer.

num_multimask_outputs (`int`, *optional*, defaults to 3) : The number of outputs from the `SamMaskDecoder` module. In the Segment Anything paper, this is set to 3.

iou_head_depth (`int`, *optional*, defaults to 3) : The number of layers in the IoU head module.

iou_head_hidden_dim (`int`, *optional*, defaults to 256) : The dimensionality of the hidden states in the IoU head module.

layer_norm_eps (`float`, *optional*, defaults to 1e-06) : The epsilon used by the layer normalization layers.

## SamPromptEncoderConfig[[transformers.SamPromptEncoderConfig]]

#### transformers.SamPromptEncoderConfig[[transformers.SamPromptEncoderConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/configuration_sam.py#L24)

This is the configuration class to store the configuration of a `SamPromptEncoder`. The `SamPromptEncoder`
module is used to encode the input 2D points and bounding boxes. Instantiating a configuration defaults will yield
a similar configuration to that of the SAM-vit-h
[facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

**Parameters:**

hidden_size (`int`, *optional*, defaults to 256) : Dimensionality of the hidden states.

image_size (`int`, *optional*, defaults to 1024) : The expected output resolution of the image.

patch_size (`int`, *optional*, defaults to 16) : The size (resolution) of each patch.

mask_input_channels (`int`, *optional*, defaults to 16) : The number of channels to be fed to the `MaskDecoder` module.

num_point_embeddings (`int`, *optional*, defaults to 4) : The number of point embeddings to be used.

hidden_act (`str`, *optional*, defaults to `"gelu"`) : The non-linear activation function in the encoder and pooler.

## SamProcessor[[transformers.SamProcessor]]

#### transformers.SamProcessor[[transformers.SamProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/processing_sam.py#L55)

Constructs a SAM processor which wraps a SAM image processor and an 2D points & Bounding boxes processor into a
single processor.

[SamProcessor](/docs/transformers/main/en/model_doc/sam#transformers.SamProcessor) offers all the functionalities of [SamImageProcessor](/docs/transformers/main/en/model_doc/sam#transformers.SamImageProcessor). See the docstring of
`__call__()` for more information.

**Parameters:**

image_processor (`SamImageProcessor`) : An instance of [SamImageProcessor](/docs/transformers/main/en/model_doc/sam#transformers.SamImageProcessor). The image processor is a required input.

## SamImageProcessor[[transformers.SamImageProcessor]]

#### transformers.SamImageProcessor[[transformers.SamImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam.py#L74)

Constructs a SAM image processor.

filter_maskstransformers.SamImageProcessor.filter_maskshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam.py#L751[{"name": "masks", "val": ""}, {"name": "iou_scores", "val": ""}, {"name": "original_size", "val": ""}, {"name": "cropped_box_image", "val": ""}, {"name": "pred_iou_thresh", "val": " = 0.88"}, {"name": "stability_score_thresh", "val": " = 0.95"}, {"name": "mask_threshold", "val": " = 0"}, {"name": "stability_score_offset", "val": " = 1"}, {"name": "return_tensors", "val": " = 'pt'"}]- **masks** (`torch.Tensor`) --
  Input masks.
- **iou_scores** (`torch.Tensor`) --
  List of IoU scores.
- **original_size** (`tuple[int,int]`) --
  Size of the original image.
- **cropped_box_image** (`np.ndarray`) --
  The cropped image.
- **pred_iou_thresh** (`float`, *optional*, defaults to 0.88) --
  The threshold for the iou scores.
- **stability_score_thresh** (`float`, *optional*, defaults to 0.95) --
  The threshold for the stability score.
- **mask_threshold** (`float`, *optional*, defaults to 0) --
  The threshold for the predicted masks.
- **stability_score_offset** (`float`, *optional*, defaults to 1) --
  The offset for the stability score used in the `_compute_stability_score` method.
- **return_tensors** (`str`, *optional*, defaults to `pt`) --
  If `pt`, returns `torch.Tensor`.0

Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
bounding boxes and pad the predicted masks if necessary.

**Parameters:**

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the `do_resize` parameter in the `preprocess` method.

size (`dict`, *optional*, defaults to `{"longest_edge" : 1024}`): Size of the output image after resizing. Resizes the longest edge of the image to match `size["longest_edge"]` while maintaining the aspect ratio. Can be overridden by the `size` parameter in the `preprocess` method.

mask_size (`dict`, *optional*, defaults to `{"longest_edge" : 256}`): Size of the output segmentation map after resizing. Resizes the longest edge of the image to match `size["longest_edge"]` while maintaining the aspect ratio. Can be overridden by the `mask_size` parameter in the `preprocess` method.

resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) : Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the `preprocess` method.

do_rescale (`bool`, *optional*, defaults to `True`) : Wwhether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.

rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.

image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`) : Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method. Can be overridden by the `image_std` parameter in the `preprocess` method.

do_pad (`bool`, *optional*, defaults to `True`) : Whether to pad the image to the specified `pad_size`. Can be overridden by the `do_pad` parameter in the `preprocess` method.

pad_size (`dict`, *optional*, defaults to `{"height" : 1024, "width": 1024}`): Size of the output image after padding. Can be overridden by the `pad_size` parameter in the `preprocess` method.

mask_pad_size (`dict`, *optional*, defaults to `{"height" : 256, "width": 256}`): Size of the output segmentation map after padding. Can be overridden by the `mask_pad_size` parameter in the `preprocess` method.

do_convert_rgb (`bool`, *optional*, defaults to `True`) : Whether to convert the image to RGB.
#### generate_crop_boxes[[transformers.SamImageProcessor.generate_crop_boxes]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam.py#L693)

Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

**Parameters:**

image (`np.ndarray`) : Input original image

target_size (`int`) : Target size of the resized image

crop_n_layers (`int`, *optional*, defaults to 0) : If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.

overlap_ratio (`float`, *optional*, defaults to 512/1500) : Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.

points_per_crop (`int`, *optional*, defaults to 32) : Number of points to sample from each crop.

crop_n_points_downscale_factor (`list[int]`, *optional*, defaults to 1) : The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.

device (`torch.device`, *optional*, defaults to None) : Device to use for the computation. If None, cpu will be used.

input_data_format (`str` or `ChannelDimension`, *optional*) : The channel dimension format of the input image. If not provided, it will be inferred.

return_tensors (`str`, *optional*, defaults to `pt`) : If `pt`, returns `torch.Tensor`.
#### pad_image[[transformers.SamImageProcessor.pad_image]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam.py#L174)

Pad an image to `(pad_size["height"], pad_size["width"])` with zeros to the right and bottom.

**Parameters:**

image (`np.ndarray`) : Image to pad.

pad_size (`dict[str, int]`) : Size of the output image after padding.

data_format (`str` or `ChannelDimension`, *optional*) : The data format of the image. Can be either "channels_first" or "channels_last". If `None`, the `data_format` of the `image` will be used.

input_data_format (`str` or `ChannelDimension`, *optional*) : The channel dimension format of the input image. If not provided, it will be inferred.
#### post_process_for_mask_generation[[transformers.SamImageProcessor.post_process_for_mask_generation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam.py#L672)

Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.

**Parameters:**

all_masks (`list[torch.Tensor]`) : List of all predicted segmentation masks

all_scores (`list[torch.Tensor]`) : List of all predicted iou scores

all_boxes (`list[torch.Tensor]`) : List of all bounding boxes of the predicted masks

crops_nms_thresh (`float`) : Threshold for NMS (Non Maximum Suppression) algorithm.

return_tensors (`str`, *optional*, defaults to `pt`) : If `pt`, returns `torch.Tensor`.
#### post_process_masks[[transformers.SamImageProcessor.post_process_masks]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam.py#L579)

Remove padding and upscale masks to the original image size.

**Parameters:**

masks (`Union[list[torch.Tensor], list[np.ndarray]]`) : Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.

original_sizes (`Union[torch.Tensor, list[tuple[int,int]]]`) : The original sizes of each image before it was resized to the model's expected input shape, in (height, width) format.

reshaped_input_sizes (`Union[torch.Tensor, list[tuple[int,int]]]`) : The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.

mask_threshold (`float`, *optional*, defaults to 0.0) : The threshold to use for binarizing the masks.

binarize (`bool`, *optional*, defaults to `True`) : Whether to binarize the masks.

pad_size (`int`, *optional*, defaults to `self.pad_size`) : The target size the images were padded to before being passed to the model. If None, the target size is assumed to be the processor's `pad_size`.

return_tensors (`str`, *optional*, defaults to `"pt"`) : If `"pt"`, return PyTorch tensors.

**Returns:**

`(`torch.Tensor`)`

Batched masks in batch_size, num_channels, height, width) format, where
(height, width) is given by original_size.
#### preprocess[[transformers.SamImageProcessor.preprocess]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam.py#L403)

Preprocess an image or batch of images.

**Parameters:**

images (`ImageInput`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

segmentation_maps (`ImageInput`, *optional*) : Segmentation map to preprocess.

do_resize (`bool`, *optional*, defaults to `self.do_resize`) : Whether to resize the image.

size (`dict[str, int]`, *optional*, defaults to `self.size`) : Controls the size of the image after `resize`. The longest edge of the image is resized to `size["longest_edge"]` whilst preserving the aspect ratio.

mask_size (`dict[str, int]`, *optional*, defaults to `self.mask_size`) : Controls the size of the segmentation map after `resize`. The longest edge of the image is resized to `size["longest_edge"]` whilst preserving the aspect ratio.

resample (`PILImageResampling`, *optional*, defaults to `self.resample`) : `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.

do_rescale (`bool`, *optional*, defaults to `self.do_rescale`) : Whether to rescale the image pixel values by rescaling factor.

rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`) : Rescale factor to apply to the image pixel values.

do_normalize (`bool`, *optional*, defaults to `self.do_normalize`) : Whether to normalize the image.

image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) : Image mean to normalize the image by if `do_normalize` is set to `True`.

image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`) : Image standard deviation to normalize the image by if `do_normalize` is set to `True`.

do_pad (`bool`, *optional*, defaults to `self.do_pad`) : Whether to pad the image.

pad_size (`dict[str, int]`, *optional*, defaults to `self.pad_size`) : Controls the size of the padding applied to the image. The image is padded to `pad_size["height"]` and `pad_size["width"]` if `do_pad` is set to `True`.

mask_pad_size (`dict[str, int]`, *optional*, defaults to `self.mask_pad_size`) : Controls the size of the padding applied to the segmentation map. The image is padded to `mask_pad_size["height"]` and `mask_pad_size["width"]` if `do_pad` is set to `True`.

do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`) : Whether to convert the image to RGB.

return_tensors (`str` or `TensorType`, *optional*) : The type of tensors to return. Can be one of: - Unset: Return a list of `np.ndarray`. - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`. - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.

data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) : The channel dimension format for the output image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - Unset: Use the channel dimension format of the input image.

input_data_format (`ChannelDimension` or `str`, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
#### resize[[transformers.SamImageProcessor.resize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam.py#L222)

Resize an image to `(size["height"], size["width"])`.

**Parameters:**

image (`np.ndarray`) : Image to resize.

size (`dict[str, int]`) : Dictionary in the format `{"longest_edge": int}` specifying the size of the output image. The longest edge of the image will be resized to the specified size, while the other edge will be resized to maintain the aspect ratio.

resample : `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.

data_format (`ChannelDimension` or `str`, *optional*) : The channel dimension format for the output image. If unset, the channel dimension format of the input image is used. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

input_data_format (`ChannelDimension` or `str`, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

**Returns:**

``np.ndarray``

The resized image.

## SamImageProcessorFast[[transformers.SamImageProcessorFast]]

#### transformers.SamImageProcessorFast[[transformers.SamImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam_fast.py#L52)

Constructs a fast Sam image processor.

filter_maskstransformers.SamImageProcessorFast.filter_maskshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam_fast.py#L329[{"name": "masks", "val": ""}, {"name": "iou_scores", "val": ""}, {"name": "original_size", "val": ""}, {"name": "cropped_box_image", "val": ""}, {"name": "pred_iou_thresh", "val": " = 0.88"}, {"name": "stability_score_thresh", "val": " = 0.95"}, {"name": "mask_threshold", "val": " = 0"}, {"name": "stability_score_offset", "val": " = 1"}]- **masks** (`torch.Tensor`) --
  Input masks.
- **iou_scores** (`torch.Tensor`) --
  List of IoU scores.
- **original_size** (`tuple[int,int]`) --
  Size of the original image.
- **cropped_box_image** (`torch.Tensor`) --
  The cropped image.
- **pred_iou_thresh** (`float`, *optional*, defaults to 0.88) --
  The threshold for the iou scores.
- **stability_score_thresh** (`float`, *optional*, defaults to 0.95) --
  The threshold for the stability score.
- **mask_threshold** (`float`, *optional*, defaults to 0) --
  The threshold for the predicted masks.
- **stability_score_offset** (`float`, *optional*, defaults to 1) --
  The offset for the stability score used in the `_compute_stability_score` method.0

Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
bounding boxes and pad the predicted masks if necessary.

**Parameters:**

masks (`torch.Tensor`) : Input masks.

iou_scores (`torch.Tensor`) : List of IoU scores.

original_size (`tuple[int,int]`) : Size of the original image.

cropped_box_image (`torch.Tensor`) : The cropped image.

pred_iou_thresh (`float`, *optional*, defaults to 0.88) : The threshold for the iou scores.

stability_score_thresh (`float`, *optional*, defaults to 0.95) : The threshold for the stability score.

mask_threshold (`float`, *optional*, defaults to 0) : The threshold for the predicted masks.

stability_score_offset (`float`, *optional*, defaults to 1) : The offset for the stability score used in the `_compute_stability_score` method.
#### generate_crop_boxes[[transformers.SamImageProcessorFast.generate_crop_boxes]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam_fast.py#L276)

Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

**Parameters:**

image (`torch.Tensor`) : Input original image

target_size (`int`) : Target size of the resized image

crop_n_layers (`int`, *optional*, defaults to 0) : If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.

overlap_ratio (`float`, *optional*, defaults to 512/1500) : Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.

points_per_crop (`int`, *optional*, defaults to 32) : Number of points to sample from each crop.

crop_n_points_downscale_factor (`list[int]`, *optional*, defaults to 1) : The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.

device (`torch.device`, *optional*, defaults to None) : Device to use for the computation. If None, cpu will be used.

input_data_format (`str` or `ChannelDimension`, *optional*) : The channel dimension format of the input image. If not provided, it will be inferred.

return_tensors (`str`, *optional*, defaults to `pt`) : If `pt`, returns `torch.Tensor`.
#### post_process_for_mask_generation[[transformers.SamImageProcessorFast.post_process_for_mask_generation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam_fast.py#L461)

Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.

**Parameters:**

all_masks (`torch.Tensor`) : List of all predicted segmentation masks

all_scores (`torch.Tensor`) : List of all predicted iou scores

all_boxes (`torch.Tensor`) : List of all bounding boxes of the predicted masks

crops_nms_thresh (`float`) : Threshold for NMS (Non Maximum Suppression) algorithm.
#### post_process_masks[[transformers.SamImageProcessorFast.post_process_masks]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam_fast.py#L408)

Remove padding and upscale masks to the original image size.

**Parameters:**

masks (`Union[List[torch.Tensor], List[np.ndarray]]`) : Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.

original_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`) : The original sizes of each image before it was resized to the model's expected input shape, in (height, width) format.

reshaped_input_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`) : The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.

mask_threshold (`float`, *optional*, defaults to 0.0) : The threshold to use for binarizing the masks.

binarize (`bool`, *optional*, defaults to `True`) : Whether to binarize the masks.

pad_size (`int`, *optional*, defaults to `self.pad_size`) : The target size the images were padded to before being passed to the model. If None, the target size is assumed to be the processor's `pad_size`.

**Returns:**

`(`torch.Tensor`)`

Batched masks in batch_size, num_channels, height, width) format, where (height, width)
is given by original_size.
#### preprocess[[transformers.SamImageProcessorFast.preprocess]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam_fast.py#L162)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

segmentation_maps (`ImageInput`, *optional*) : The segmentation maps to preprocess.

do_convert_rgb (`bool`, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *optional*) : Whether to resize the image.

size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : Describes the maximum input dimensions to the model.

crop_size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[PILImageResampling, int, NoneType], None]`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *optional*) : Whether to rescale the image.

rescale_factor (`float`, *optional*) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *optional*) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...], NoneType]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...], NoneType]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *optional*) : Whether to center crop the image.

data_format (`Union[~image_utils.ChannelDimension, str, NoneType]`) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[~image_utils.ChannelDimension, str, NoneType]`) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`) : The device to process the images on. If unset, the device is inferred from the input images.

return_tensors (`Annotated[Union[str, ~utils.generic.TensorType, NoneType], None]`) : Returns stacked tensors if set to `pt, otherwise returns a list of tensors.

disable_grouping (`bool`, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

mask_size (`dict[str, int]`, *optional*) : The size `{"longest_edge": int}` to resize the segmentation maps to.

mask_pad_size (`dict[str, int]`, *optional*) : The size `{"height": int, "width": int}` to pad the segmentation maps to. Must be larger than any segmentation map size provided for preprocessing.

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.
#### resize[[transformers.SamImageProcessorFast.resize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam_fast.py#L83)

Resize an image to `(size["height"], size["width"])`.

**Parameters:**

image (`np.ndarray`) : Image to resize.

size (`dict[str, int]`) : Dictionary in the format `{"longest_edge": int}` specifying the size of the output image. The longest edge of the image will be resized to the specified size, while the other edge will be resized to maintain the aspect ratio.

interpolation : `F_t.InterpolationMode` filter to use when resizing the image e.g. `F_t.InterpolationMode.BICUBIC`.

**Returns:**

``torch.Tensor``

The resized image.

## SamVisionModel[[transformers.SamVisionModel]]

#### transformers.SamVisionModel[[transformers.SamVisionModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1078)

The vision model from Sam without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.SamVisionModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1090[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SamImageProcessor](/docs/transformers/main/en/model_doc/sam#transformers.SamImageProcessor). See `SamImageProcessor.__call__()` for details ([SamProcessor](/docs/transformers/main/en/model_doc/sam#transformers.SamProcessor) uses
  [SamImageProcessor](/docs/transformers/main/en/model_doc/sam#transformers.SamImageProcessor) for processing images).0`transformers.models.sam.modeling_sam.SamVisionEncoderOutput` or `tuple(torch.FloatTensor)`A `transformers.models.sam.modeling_sam.SamVisionEncoderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SamConfig](/docs/transformers/main/en/model_doc/sam#transformers.SamConfig)) and inputs.

- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) -- The image embeddings obtained by applying the projection layer to the pooler_output.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [SamVisionModel](/docs/transformers/main/en/model_doc/sam#transformers.SamVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([SamVisionConfig](/docs/transformers/main/en/model_doc/sam#transformers.SamVisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.sam.modeling_sam.SamVisionEncoderOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.sam.modeling_sam.SamVisionEncoderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SamConfig](/docs/transformers/main/en/model_doc/sam#transformers.SamConfig)) and inputs.

- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) -- The image embeddings obtained by applying the projection layer to the pooler_output.
- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## SamModel[[transformers.SamModel]]

#### transformers.SamModel[[transformers.SamModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1105)

Segment Anything Model (SAM) for generating segmentation masks, given an input image and
input points and labels, boxes, or masks.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.SamModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1185[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "input_points", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "input_labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "input_boxes", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "input_masks", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "image_embeddings", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "multimask_output", "val": ": bool = True"}, {"name": "attention_similarity", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "target_embedding", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SamImageProcessor](/docs/transformers/main/en/model_doc/sam#transformers.SamImageProcessor). See `SamImageProcessor.__call__()` for details ([SamProcessor](/docs/transformers/main/en/model_doc/sam#transformers.SamProcessor) uses
  [SamImageProcessor](/docs/transformers/main/en/model_doc/sam#transformers.SamImageProcessor) for processing images).
- **input_points** (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`) --
  Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
  better results. The points can be obtained by passing a list of list of list to the processor that will
  create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
  second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
  per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
  multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
  coordinates of the point. If a different number of points is passed either for each image, or for each
  mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
  computation of the embedding will be skipped for these points using the labels.
- **input_labels** (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`) --
  Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
  official implementation, there are 3 types of labels

  - `1`: the point is a point that contains the object of interest
  - `0`: the point is a point that does not contain the object of interest
  - `-1`: the point corresponds to the background

  We added the label:

  - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

  The padding labels should be automatically done by the processor.
- **input_boxes** (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`) --
  Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
  much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
  that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
  size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
  In the order (`x1`, `y1`, `x2`, `y2`):

  - `x1`: the x coordinate of the top left point of the input box
  - `y1`: the y coordinate of the top left point of the input box
  - `x2`: the x coordinate of the bottom right point of the input box
  - `y2`: the y coordinate of the bottom right point of the input box
- **input_masks** (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`) --
  SAM model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
  generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
  manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).
- **image_embeddings** (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`) --
  Image embeddings, this is used by the mask decder to generate masks and iou scores. For more memory
  efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
  method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
- **multimask_output** (`bool`, *optional*) --
  In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
  bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
  "best" mask, by specifying `multimask_output=False`.
- **attention_similarity** (`torch.FloatTensor`, *optional*) --
  Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
  model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).
- **target_embedding** (`torch.FloatTensor`, *optional*) --
  Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
  the model is used for personalization as introduced in [PerSAM](https://huggingface.co/papers/2305.03048).0`transformers.models.sam.modeling_sam.SamImageSegmentationOutput` or `tuple(torch.FloatTensor)`A `transformers.models.sam.modeling_sam.SamImageSegmentationOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SamConfig](/docs/transformers/main/en/model_doc/sam#transformers.SamConfig)) and inputs.

- **iou_scores** (`torch.FloatTensor` of shape `(batch_size, num_masks)`) -- The iou scores of the predicted masks.
- **pred_masks** (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`) -- The predicted low resolutions masks. Needs to be post-processed by the processor
- **vision_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
- **vision_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **mask_decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [SamModel](/docs/transformers/main/en/model_doc/sam#transformers.SamModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
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

**Parameters:**

config ([SamConfig](/docs/transformers/main/en/model_doc/sam#transformers.SamConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.sam.modeling_sam.SamImageSegmentationOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.sam.modeling_sam.SamImageSegmentationOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SamConfig](/docs/transformers/main/en/model_doc/sam#transformers.SamConfig)) and inputs.

- **iou_scores** (`torch.FloatTensor` of shape `(batch_size, num_masks)`) -- The iou scores of the predicted masks.
- **pred_masks** (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`) -- The predicted low resolutions masks. Needs to be post-processed by the processor
- **vision_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
- **vision_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **mask_decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
