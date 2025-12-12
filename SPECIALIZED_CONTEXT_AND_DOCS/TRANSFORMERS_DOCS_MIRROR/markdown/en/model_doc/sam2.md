![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat)

# SAM2

## Overview

SAM2 (Segment Anything Model 2) was proposed in [Segment Anything in Images and Videos](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/) by Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, Christoph Feichtenhofer.

The model can be used to predict segmentation masks of any object of interest given an input image or video, and input points or bounding boxes.

![example image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam2_header.gif)

The abstract from the paper is the following:

*We present Segment Anything Model 2 (SAM 2), a foundation model towards solving promptable visual segmentation in images and videos. We build a data engine, which improves model and data via user interaction, to collect the largest video segmentation dataset to date. Our model is a simple transformer architecture with streaming memory for real-time video processing. SAM 2 trained on our data provides strong performance across a wide range of tasks. In video segmentation, we observe better accuracy, using 3x fewer interactions than prior approaches. In image segmentation, our model is more accurate and 6x faster than the Segment Anything Model (SAM). We believe that our data, model, and insights will serve as a significant milestone for video segmentation and related perception tasks. We are releasing a version of our model, the dataset and an interactive demo.*

Tips:

* Batch & Video Support: SAM2 natively supports batch processing and seamless video segmentation, while original SAM is designed for static images and simpler one-image-at-a-time workflows.
* Accuracy & Generalization: SAM2 shows improved segmentation quality, robustness, and zero-shot generalization to new domains compared to the original SAM, especially with mixed prompts.

This model was contributed by [sangbumchoi](https://github.com/SangbumChoi) and [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/facebookresearch/sam2/tree/main).

## Usage example

### Automatic Mask Generation with Pipeline

SAM2 can be used for automatic mask generation to segment all objects in an image using the `mask-generation` pipeline:


```
>>> from transformers import pipeline

>>> generator = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device=0)
>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> outputs = generator(image_url, points_per_batch=64)

>>> len(outputs["masks"])  # Number of masks generated
39
```

### Basic Image Segmentation

#### Single Point Click

You can segment objects by providing a single point click on the object you want to segment:


```
>>> from transformers import Sam2Processor, Sam2Model, infer_device
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = infer_device()

>>> model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
>>> processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> input_points = [[[[500, 375]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
>>> input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

>>> # The model outputs multiple mask predictions ranked by quality score
>>> print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")
Generated 3 masks with shape torch.Size(1, 3, 1500, 2250)
```

#### Multiple Points for Refinement

You can provide multiple points to refine the segmentation:


```
>>> # Add both positive and negative points to refine the mask
>>> input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points for refinement
>>> input_labels = [[[1, 1]]]  # Both positive clicks

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
```

#### Bounding Box Input

SAM2 also supports bounding box inputs for segmentation:


```
>>> # Define bounding box as [x_min, y_min, x_max, y_max]
>>> input_boxes = [[[75, 275, 1725, 850]]]

>>> inputs = processor(images=raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
```

#### Multiple Objects Segmentation

You can segment multiple objects simultaneously:


```
>>> # Define points for two different objects
>>> input_points = [[[[500, 375]], [[650, 750]]]]  # Points for two objects in same image
>>> input_labels = [[[1], [1]]]  # Positive clicks for both objects

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Each object gets its own mask
>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
>>> print(f"Generated masks for {masks.shape[0]} objects")
Generated masks for 2 objects
```

### Batch Inference

#### Batched Images

Process multiple images simultaneously for improved efficiency:


```
>>> from transformers import Sam2Processor, Sam2Model, infer_device
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = infer_device()

>>> model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
>>> processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

>>> # Load multiple images
>>> image_urls = [
...     "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg",
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
... ]
>>> raw_images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in image_urls]

>>> # Single point per image
>>> input_points = [[[[500, 375]]], [[[770, 200]]]]  # One point for each image
>>> input_labels = [[[1]], [[1]]]  # Positive clicks for both images

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Post-process masks for each image
>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
>>> print(f"Processed {len(all_masks)} images, each with {all_masks[0].shape[0]} objects")
Processed 2 images, each with 1 objects
```

#### Batched Objects per Image

Segment multiple objects within each image using batch inference:


```
>>> # Multiple objects per image - different numbers of objects per image
>>> input_points = [
...     [[[500, 375]], [[650, 750]]],  # Truck image: 2 objects
...     [[[770, 200]]]  # Dog image: 1 object
... ]
>>> input_labels = [
...     [[1], [1]],  # Truck image: positive clicks for both objects
...     [[1]]  # Dog image: positive click for the object
... ]

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
```

#### Batched Images with Batched Objects and Multiple Points

Handle complex batch scenarios with multiple points per object:


```
>>> # Add groceries image for more complex example
>>> groceries_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/groceries.jpg"
>>> groceries_image = Image.open(requests.get(groceries_url, stream=True).raw).convert("RGB")
>>> raw_images = [raw_images[0], groceries_image]  # Use truck and groceries images

>>> # Complex batching: multiple images, multiple objects, multiple points per object
>>> input_points = [
...     [[[500, 375]], [[650, 750]]],  # Truck image: 2 objects with 1 point each
...     [[[400, 300]], [[630, 300], [550, 300]]]  # Groceries image: obj1 has 1 point, obj2 has 2 points
... ]
>>> input_labels = [
...     [[1], [1]],  # Truck image: positive clicks
...     [[1], [1, 1]]  # Groceries image: positive clicks for refinement
... ]

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
```

#### Batched Bounding Boxes

Process multiple images with bounding box inputs:


```
>>> # Multiple bounding boxes per image (using truck and groceries images)
>>> input_boxes = [
...     [[75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750]],  # Truck image: 4 boxes
...     [[450, 170, 520, 350], [350, 190, 450, 350], [500, 170, 580, 350], [580, 170, 640, 350]]  # Groceries image: 4 boxes
... ]

>>> # Update images for this example
>>> raw_images = [raw_images[0], groceries_image]  # truck and groceries

>>> inputs = processor(images=raw_images, input_boxes=input_boxes, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
>>> print(f"Processed {len(input_boxes)} images with {len(input_boxes[0])} and {len(input_boxes[1])} boxes respectively")
Processed 2 images with 4 and 4 boxes respectively
```

### Using Previous Masks as Input

SAM2 can use masks from previous predictions as input to refine segmentation:


```
>>> # Get initial segmentation
>>> input_points = [[[[500, 375]]]]
>>> input_labels = [[[1]]]
>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Use the best mask as input for refinement
>>> mask_input = outputs.pred_masks[:, :, torch.argmax(outputs.iou_scores.squeeze())]

>>> # Add additional points with the mask input
>>> new_input_points = [[[[500, 375], [450, 300]]]]
>>> new_input_labels = [[[1, 1]]]
>>> inputs = processor(
...     input_points=new_input_points,
...     input_labels=new_input_labels,
...     original_sizes=inputs["original_sizes"],
...     return_tensors="pt",
... ).to(device)

>>> with torch.no_grad():
...     refined_outputs = model(
...         **inputs,
...         input_masks=mask_input,
...         image_embeddings=outputs.image_embeddings,
...         multimask_output=False,
...     )
```

## Sam2Config

### class transformers.Sam2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/configuration_sam2.py#L363)

( vision\_config = None prompt\_encoder\_config = None mask\_decoder\_config = None initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **vision\_config** (Union[`dict`, `Sam2VisionConfig`], *optional*) —
  Dictionary of configuration options used to initialize [Sam2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2VisionConfig).
* **prompt\_encoder\_config** (Union[`dict`, `Sam2PromptEncoderConfig`], *optional*) —
  Dictionary of configuration options used to initialize [Sam2PromptEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2PromptEncoderConfig).
* **mask\_decoder\_config** (Union[`dict`, `Sam2MaskDecoderConfig`], *optional*) —
  Dictionary of configuration options used to initialize [Sam2MaskDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2MaskDecoderConfig).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  Standard deviation for parameter initialization.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[Sam2Config](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Config) is the configuration class to store the configuration of a [Sam2Model](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Model). It is used to instantiate a
SAM2 model according to the specified arguments, defining the memory attention, memory encoder, and image encoder
configs. Instantiating a configuration defaults will yield a similar configuration to that of the SAM 2.1 Hiera-tiny
[facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import (
...     Sam2VisionConfig,
...     Sam2PromptEncoderConfig,
...     Sam2MaskDecoderConfig,
...     Sam2Model,
... )

>>> # Initializing a Sam2Config with `"facebook/sam2.1_hiera_tiny"` style configuration
>>> configuration = Sam2config()

>>> # Initializing a Sam2Model (with random weights) from the `"facebook/sam2.1_hiera_tiny"` style configuration
>>> model = Sam2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a Sam2Config from a Sam2VisionConfig, Sam2PromptEncoderConfig, and Sam2MaskDecoderConfig

>>> # Initializing SAM2 vision encoder, memory attention, and memory encoder configurations
>>> vision_config = Sam2VisionConfig()
>>> prompt_encoder_config = Sam2PromptEncoderConfig()
>>> mask_decoder_config = Sam2MaskDecoderConfig()

>>> config = Sam2Config(vision_config, prompt_encoder_config, mask_decoder_config)
```

## Sam2HieraDetConfig

### class transformers.Sam2HieraDetConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/configuration_sam2.py#L25)

( hidden\_size = 96 num\_attention\_heads = 1 num\_channels = 3 image\_size = None patch\_kernel\_size = None patch\_stride = None patch\_padding = None query\_stride = None window\_positional\_embedding\_background\_size = None num\_query\_pool\_stages = 3 blocks\_per\_stage = None embed\_dim\_per\_stage = None num\_attention\_heads\_per\_stage = None window\_size\_per\_stage = None global\_attention\_blocks = None mlp\_ratio = 4.0 hidden\_act = 'gelu' layer\_norm\_eps = 1e-06 initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 96) —
  The hidden dimension of the image encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 1) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of channels in the image.
* **image\_size** (`list[int]`, *optional*, defaults to `[1024, 1024]`) —
  The size of the image.
* **patch\_kernel\_size** (`list[int]`, *optional*, defaults to `[7, 7]`) —
  The kernel size of the patch.
* **patch\_stride** (`list[int]`, *optional*, defaults to `[4, 4]`) —
  The stride of the patch.
* **patch\_padding** (`list[int]`, *optional*, defaults to `[3, 3]`) —
  The padding of the patch.
* **query\_stride** (`list[int]`, *optional*, defaults to `[2, 2]`) —
  The downsample stride between stages.
* **window\_positional\_embedding\_background\_size** (`list[int]`, *optional*, defaults to `[7, 7]`) —
  The window size per stage when not using global attention.
* **num\_query\_pool\_stages** (`int`, *optional*, defaults to 3) —
  The number of query pool stages.
* **blocks\_per\_stage** (`list[int]`, *optional*, defaults to `[1, 2, 7, 2]`) —
  The number of blocks per stage.
* **embed\_dim\_per\_stage** (`list[int]`, *optional*, defaults to `[96, 192, 384, 768]`) —
  The embedding dimension per stage.
* **num\_attention\_heads\_per\_stage** (`list[int]`, *optional*, defaults to `[1, 2, 4, 8]`) —
  The number of attention heads per stage.
* **window\_size\_per\_stage** (`list[int]`, *optional*, defaults to `[8, 4, 14, 7]`) —
  The window size per stage.
* **global\_attention\_blocks** (`list[int]`, *optional*, defaults to `[5, 7, 9]`) —
  The blocks where global attention is used.
* **mlp\_ratio** (`float`, *optional*, defaults to 4.0) —
  The ratio of the MLP hidden dimension to the embedding dimension.
* **hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function in the neck.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon for the layer normalization.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a [Sam2HieraDetModel](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2HieraDetModel). It is used to instantiate
a HieraDet model as defined in the original sam2 repo according to the specified arguments, defining the model architecture.
Instantiating a configuration defaults will yield a similar configuration to that of SAM 2.1 Hiera-tiny
[facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## Sam2VisionConfig

### class transformers.Sam2VisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/configuration_sam2.py#L144)

( backbone\_config = None backbone\_channel\_list = None backbone\_feature\_sizes = None fpn\_hidden\_size = 256 fpn\_kernel\_size = 1 fpn\_stride = 1 fpn\_padding = 0 fpn\_top\_down\_levels = None num\_feature\_levels = 3 hidden\_act = 'gelu' layer\_norm\_eps = 1e-06 initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **backbone\_config** (`Union[dict, "PretrainedConfig"]`, *optional*) —
  Configuration for the vision backbone. This is used to instantiate the backbone using
  `AutoModel.from_config`.
* **backbone\_channel\_list** (`List[int]`, *optional*, defaults to `[768, 384, 192, 96]`) —
  The list of channel dimensions for the backbone.
* **backbone\_feature\_sizes** (`List[List[int]]`, *optional*, defaults to `[[256, 256], [128, 128], [64, 64]]`) —
  The spatial sizes of the feature maps from the backbone.
* **fpn\_hidden\_size** (`int`, *optional*, defaults to 256) —
  The hidden dimension of the FPN.
* **fpn\_kernel\_size** (`int`, *optional*, defaults to 1) —
  The kernel size for the convolutions in the neck.
* **fpn\_stride** (`int`, *optional*, defaults to 1) —
  The stride for the convolutions in the neck.
* **fpn\_padding** (`int`, *optional*, defaults to 0) —
  The padding for the convolutions in the neck.
* **fpn\_top\_down\_levels** (`List[int]`, *optional*, defaults to `[2, 3]`) —
  The levels for the top-down FPN connections.
* **num\_feature\_levels** (`int`, *optional*, defaults to 3) —
  The number of feature levels from the FPN to use.
* **hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function in the neck.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon for the layer normalization.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a [Sam2VisionModel](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2VisionModel). It is used to instantiate a SAM
vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
defaults will yield a similar configuration to that of SAM 2.1 Hiera-tiny
[facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## Sam2MaskDecoderConfig

### class transformers.Sam2MaskDecoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/configuration_sam2.py#L290)

( hidden\_size = 256 hidden\_act = 'gelu' mlp\_dim = 2048 num\_hidden\_layers = 2 num\_attention\_heads = 8 attention\_downsample\_rate = 2 num\_multimask\_outputs = 3 iou\_head\_depth = 3 iou\_head\_hidden\_dim = 256 dynamic\_multimask\_via\_stability = True dynamic\_multimask\_stability\_delta = 0.05 dynamic\_multimask\_stability\_thresh = 0.98 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the hidden states.
* **hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function in the SAM2 mask decoder.
* **mlp\_dim** (`int`, *optional*, defaults to 2048) —
  The dimension of the MLP in the two-way transformer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 2) —
  The number of hidden layers in the two-way transformer.
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  The number of attention heads in the two-way transformer.
* **attention\_downsample\_rate** (`int`, *optional*, defaults to 2) —
  The downsample rate for the attention layers.
* **num\_multimask\_outputs** (`int`, *optional*, defaults to 3) —
  The number of multimask outputs.
* **iou\_head\_depth** (`int`, *optional*, defaults to 3) —
  The depth of the IoU head.
* **iou\_head\_hidden\_dim** (`int`, *optional*, defaults to 256) —
  The hidden dimension of the IoU head.
* **dynamic\_multimask\_via\_stability** (`bool`, *optional*, defaults to `True`) —
  Whether to use dynamic multimask via stability.
* **dynamic\_multimask\_stability\_delta** (`float`, *optional*, defaults to 0.05) —
  The stability delta for the dynamic multimask.
* **dynamic\_multimask\_stability\_thresh** (`float`, *optional*, defaults to 0.98) —
  The stability threshold for the dynamic multimask.

This is the configuration class to store the configuration of a `Sam2MaskDecoder`. It is used to instantiate a SAM2
memory encoder according to the specified arguments, defining the model architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## Sam2PromptEncoderConfig

### class transformers.Sam2PromptEncoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/configuration_sam2.py#L238)

( hidden\_size = 256 image\_size = 1024 patch\_size = 16 mask\_input\_channels = 16 num\_point\_embeddings = 4 hidden\_act = 'gelu' layer\_norm\_eps = 1e-06 scale = 1 \*\*kwargs  )

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
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **scale** (`float`, *optional*, defaults to 1) —
  The scale factor for the prompt encoder.

This is the configuration class to store the configuration of a `Sam2PromptEncoder`. The `Sam2PromptEncoder`
module is used to encode the input 2D points and bounding boxes.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## Sam2Processor

### class transformers.Sam2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/processing_sam2.py#L38)

( image\_processor target\_size: typing.Optional[int] = None point\_pad\_value: int = -10 \*\*kwargs  )

Parameters

* **image\_processor** (`Sam2ImageProcessorFast`) —
  An instance of [Sam2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2ImageProcessorFast).
* **target\_size** (`int`, *optional*) —
  The target size (target\_size, target\_size) to which the image will be resized.
* **point\_pad\_value** (`int`, *optional*, defaults to -10) —
  The value used for padding input points.

Constructs a SAM2 processor which wraps a SAM2 image processor and an 2D points & Bounding boxes processor into a
single processor.

[Sam2Processor](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Processor) offers all the functionalities of [Sam2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2ImageProcessorFast) and [Sam2VideoProcessor](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoProcessor). See the docstring of
`__call__()` and [**call**()](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoProcessor.__call__) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/processing_sam2.py#L63)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None input\_points: typing.Union[list[list[list[list[float]]]], torch.Tensor, NoneType] = None input\_labels: typing.Union[list[list[list[int]]], torch.Tensor, NoneType] = None input\_boxes: typing.Union[list[list[list[float]]], torch.Tensor, NoneType] = None original\_sizes: typing.Union[list[list[float]], torch.Tensor, NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None \*\*kwargs  ) → A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields

Parameters

* **images** (`ImageInput`, *optional*) —
  The image(s) to process.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  The segmentation maps to process.
* **input\_points** (`list[list[list[list[float]]]]`, `torch.Tensor`, *optional*) —
  The points to add to the frame.
* **input\_labels** (`list[list[list[int]]]`, `torch.Tensor`, *optional*) —
  The labels for the points.
* **input\_boxes** (`list[list[list[float]]]`, `torch.Tensor`, *optional*) —
  The bounding boxes to add to the frame.
* **original\_sizes** (`list[list[float]]`, `torch.Tensor`, *optional*) —
  The original sizes of the images.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return.
* \***\*kwargs** —
  Additional keyword arguments to pass to the image processor.

Returns

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields

* `pixel_values` (`torch.Tensor`): The processed image(s).
* `original_sizes` (`list[list[float]]`): The original sizes of the images.
* `reshaped_input_sizes` (`torch.Tensor`): The reshaped input sizes of the images.
* `labels` (`torch.Tensor`): The processed segmentation maps (if provided).
* `input_points` (`torch.Tensor`): The processed points.
* `input_labels` (`torch.Tensor`): The processed labels.
* `input_boxes` (`torch.Tensor`): The processed bounding boxes.

This method uses `Sam2ImageProcessorFast.__call__()` method to prepare image(s) for the model. It also prepares 2D
points and bounding boxes for the model if they are provided.

#### post\_process\_masks

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/processing_sam2.py#L479)

( masks original\_sizes mask\_threshold = 0.0 binarize = True max\_hole\_area = 0.0 max\_sprinkle\_area = 0.0 apply\_non\_overlapping\_constraints = False \*\*kwargs  ) → (`torch.Tensor`)

Parameters

* **masks** (`Union[List[torch.Tensor], List[np.ndarray]]`) —
  Batched masks from the mask\_decoder in (batch\_size, num\_channels, height, width) format.
* **original\_sizes** (`Union[torch.Tensor, List[Tuple[int,int]]]`) —
  The original sizes of each image before it was resized to the model’s expected input shape, in (height,
  width) format.
* **mask\_threshold** (`float`, *optional*, defaults to 0.0) —
  Threshold for binarization and post-processing operations.
* **binarize** (`bool`, *optional*, defaults to `True`) —
  Whether to binarize the masks.
* **max\_hole\_area** (`float`, *optional*, defaults to 0.0) —
  The maximum area of a hole to fill.
* **max\_sprinkle\_area** (`float`, *optional*, defaults to 0.0) —
  The maximum area of a sprinkle to fill.
* **apply\_non\_overlapping\_constraints** (`bool`, *optional*, defaults to `False`) —
  Whether to apply non-overlapping constraints to the masks.

Returns

(`torch.Tensor`)

Batched masks in batch\_size, num\_channels, height, width) format, where (height, width)
is given by original\_size.

Remove padding and upscale masks to the original image size.

## Sam2ImageProcessorFast

### class transformers.Sam2ImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/image_processing_sam2_fast.py#L380)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.sam2.image\_processing\_sam2\_fast.Sam2FastImageProcessorKwargs]  )

Constructs a fast Sam2 image processor.

#### filter\_masks

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/image_processing_sam2_fast.py#L568)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/image_processing_sam2_fast.py#L515)

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
  Number of points to sam2ple from each crop.
* **crop\_n\_points\_downscale\_factor** (`list[int]`, *optional*, defaults to 1) —
  The number of points-per-side sam2pled in layer n is scaled down by crop\_n\_points\_downscale\_factor\*\*n.
* **device** (`torch.device`, *optional*, defaults to None) —
  Device to use for the computation. If None, cpu will be used.
* **input\_data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format of the input image. If not provided, it will be inferred.
* **return\_tensors** (`str`, *optional*, defaults to `pt`) —
  If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.

Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

#### post\_process\_for\_mask\_generation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/image_processing_sam2_fast.py#L700)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/image_processing_sam2_fast.py#L647)

( masks original\_sizes mask\_threshold = 0.0 binarize = True max\_hole\_area = 0.0 max\_sprinkle\_area = 0.0 apply\_non\_overlapping\_constraints = False \*\*kwargs  ) → (`torch.Tensor`)

Parameters

* **masks** (`Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]`) —
  Batched masks from the mask\_decoder in (batch\_size, num\_channels, height, width) format.
* **original\_sizes** (`Union[torch.Tensor, List[Tuple[int,int]]]`) —
  The original sizes of each image before it was resized to the model’s expected input shape, in (height,
  width) format.
* **mask\_threshold** (`float`, *optional*, defaults to 0.0) —
  Threshold for binarization and post-processing operations.
* **binarize** (`bool`, *optional*, defaults to `True`) —
  Whether to binarize the masks.
* **max\_hole\_area** (`float`, *optional*, defaults to 0.0) —
  The maximum area of a hole to fill.
* **max\_sprinkle\_area** (`float`, *optional*, defaults to 0.0) —
  The maximum area of a sprinkle to fill.
* **apply\_non\_overlapping\_constraints** (`bool`, *optional*, defaults to `False`) —
  Whether to apply non-overlapping constraints to the masks.

Returns

(`torch.Tensor`)

Batched masks in batch\_size, num\_channels, height, width) format, where (height, width)
is given by original\_size.

Remove padding and upscale masks to the original image size.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/image_processing_sam2_fast.py#L445)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.sam2.image\_processing\_sam2\_fast.Sam2FastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
  The size `{"height": int, "width": int}` to resize the segmentation maps to.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## Sam2HieraDetModel

### class transformers.Sam2HieraDetModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/modeling_sam2.py#L583)

( config: Sam2HieraDetConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/modeling_sam2.py#L624)

( pixel\_values: typing.Optional[torch.FloatTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  )

## Sam2VisionModel

### class transformers.Sam2VisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/modeling_sam2.py#L654)

( config: Sam2VisionConfig  )

Parameters

* **config** ([Sam2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2VisionConfig)) —
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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/modeling_sam2.py#L676)

( pixel\_values: typing.Optional[torch.FloatTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  )

## Sam2Model

### class transformers.Sam2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/modeling_sam2.py#L1279)

( config: Sam2Config  )

Parameters

* **config** ([Sam2Config](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Segment Anything Model 2 (SAM 2) for generating segmentation masks, given an input image and
input points and labels, boxes, or masks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2/modeling_sam2.py#L1392)

( pixel\_values: typing.Optional[torch.FloatTensor] = None input\_points: typing.Optional[torch.FloatTensor] = None input\_labels: typing.Optional[torch.LongTensor] = None input\_boxes: typing.Optional[torch.FloatTensor] = None input\_masks: typing.Optional[torch.LongTensor] = None image\_embeddings: typing.Optional[torch.FloatTensor] = None multimask\_output: bool = True attention\_similarity: typing.Optional[torch.FloatTensor] = None target\_embedding: typing.Optional[torch.FloatTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.sam2.modeling_sam2.Sam2ImageSegmentationOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([Sam2Processor](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Processor) uses
  `image_processor_class` for processing images).
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
  Image embeddings, this is used by the mask decoder to generate masks and iou scores. For more memory
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

`transformers.models.sam2.modeling_sam2.Sam2ImageSegmentationOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.sam2.modeling_sam2.Sam2ImageSegmentationOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Sam2Config](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Config)) and inputs.

* **iou\_scores** (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_masks)`) — The Intersection over Union (IoU) scores of the predicted masks.
* **pred\_masks** (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_masks, height, width)`) — The predicted low-resolution masks. This is an alias for `low_res_masks`. These masks need to be post-processed
  by the processor to be brought to the original image size.
* **object\_score\_logits** (`torch.FloatTensor` of shape `(batch_size, point_batch_size, 1)`) — Logits for the object score, indicating if an object is present.
* **image\_embeddings** (`tuple(torch.FloatTensor)`) — The features from the FPN, which are used by the mask decoder. This is a tuple of `torch.FloatTensor` where each
  tensor has shape `(batch_size, channels, height, width)`.
* **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, height, width, hidden_size)`.
  Hidden-states of the vision model at the output of each stage.
* **vision\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
  Attentions weights of the vision model.
* **mask\_decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
  Attentions weights of the mask decoder.

The [Sam2Model](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoModel, AutoProcessor

>>> model = AutoModel.from_pretrained("danelcsb/sam2.1_hiera_tiny")
>>> processor = AutoProcessor.from_pretrained("danelcsb/sam2.1_hiera_tiny")

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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/sam2.md)
