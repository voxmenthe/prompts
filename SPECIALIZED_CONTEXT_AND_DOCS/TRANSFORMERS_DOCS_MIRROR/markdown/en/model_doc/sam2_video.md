![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat)

# SAM2 Video

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

### Video Segmentation and Tracking

SAM2’s key strength is its ability to track objects across video frames. Here’s how to use it for video segmentation:

#### Basic Video Tracking


```
>>> from transformers import Sam2VideoModel, Sam2VideoProcessor, infer_device
>>> import torch

>>> device = infer_device()
>>> model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to(device, dtype=torch.bfloat16)
>>> processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")

>>> # Load video frames (example assumes you have a list of PIL Images)
>>> # video_frames = [Image.open(f"frame_{i:05d}.jpg") for i in range(num_frames)]

>>> # For this example, we'll use the video loading utility
>>> from transformers.video_utils import load_video
>>> video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
>>> video_frames, _ = load_video(video_url)

>>> # Initialize video inference session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device=device,
...     dtype=torch.bfloat16,
... )

>>> # Add click on first frame to select object
>>> ann_frame_idx = 0
>>> ann_obj_id = 1
>>> points = [[[[210, 350]]]]
>>> labels = [[[1]]]

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=ann_obj_id,
...     input_points=points,
...     input_labels=labels,
... )

>>> # Segment the object on the first frame
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )
>>> video_res_masks = processor.post_process_masks(
...     [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
... )[0]
>>> print(f"Segmentation shape: {video_res_masks.shape}")
Segmentation shape: torch.Size([1, 1, 480, 854])

>>> # Propagate through the entire video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = video_res_masks

>>> print(f"Tracked object through {len(video_segments)} frames")
Tracked object through 180 frames
```

#### Multi-Object Video Tracking

Track multiple objects simultaneously across video frames:


```
>>> # Reset for new tracking session
>>> inference_session.reset_inference_session()

>>> # Add multiple objects on the first frame
>>> ann_frame_idx = 0
>>> obj_ids = [2, 3]
>>> input_points = [[[[200, 300]], [[400, 150]]]]  # Points for two objects (batched)
>>> input_labels = [[[1], [1]]]

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=obj_ids,
...     input_points=input_points,
...     input_labels=input_labels,
... )

>>> # Get masks for both objects on first frame
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )

>>> # Propagate both objects through video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = {
...         obj_id: video_res_masks[i]
...         for i, obj_id in enumerate(inference_session.obj_ids)
...     }

>>> print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")
Tracked 2 objects through 180 frames
```

#### Refining Video Segmentation

You can add additional clicks on any frame to refine the tracking:


```
>>> # Add refinement click on a later frame
>>> refine_frame_idx = 50
>>> ann_obj_id = 2  # Refining first object
>>> points = [[[[220, 280]]]]  # Additional point
>>> labels = [[[1]]]  # Positive click

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=refine_frame_idx,
...     obj_ids=ann_obj_id,
...     input_points=points,
...     input_labels=labels,
... )

>>> # Re-propagate with the additional information
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = video_res_masks
```

### Streaming Video Inference

For real-time applications, SAM2 supports processing video frames as they arrive:


```
>>> # Initialize session for streaming
>>> inference_session = processor.init_video_session(
...     inference_device=device,
...     dtype=torch.bfloat16,
... )

>>> # Process frames one by one
>>> for frame_idx, frame in enumerate(video_frames[:10]):  # Process first 10 frames
...     inputs = processor(images=frame, device=device, return_tensors="pt")
...
...     if frame_idx == 0:
...         # Add point input on first frame
...         processor.add_inputs_to_inference_session(
...             inference_session=inference_session,
...             frame_idx=0,
...             obj_ids=1,
...             input_points=[[[[210, 350], [250, 220]]]],
...             input_labels=[[[1, 1]]],
...             original_size=inputs.original_sizes[0], # need to be provided when using streaming video inference
...         )
...
...     # Process current frame
...     sam2_video_output = model(inference_session=inference_session, frame=inputs.pixel_values[0])
...
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=inputs.original_sizes, binarize=False
...     )[0]
...     print(f"Frame {frame_idx}: mask shape {video_res_masks.shape}")
```

#### Video Batch Processing for Multiple Objects

Track multiple objects simultaneously in video by adding them all at once:


```
>>> # Initialize video session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device=device,
...     dtype=torch.bfloat16,
... )

>>> # Add multiple objects on the first frame using batch processing
>>> ann_frame_idx = 0
>>> obj_ids = [2, 3]  # Track two different objects
>>> input_points = [
...     [[[200, 300], [230, 250], [275, 175]], [[400, 150]]]
... ]  # Object 2: 3 points (2 positive, 1 negative); Object 3: 1 point
>>> input_labels = [
...     [[1, 1, 0], [1]]
... ]  # Object 2: positive, positive, negative; Object 3: positive

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=obj_ids,
...     input_points=input_points,
...     input_labels=input_labels,
... )

>>> # Get masks for all objects on the first frame
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )
>>> video_res_masks = processor.post_process_masks(
...     [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
... )[0]
>>> print(f"Generated masks for {video_res_masks.shape[0]} objects")
Generated masks for 2 objects

>>> # Propagate all objects through the video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = {
...         obj_id: video_res_masks[i]
...         for i, obj_id in enumerate(inference_session.obj_ids)
...     }

>>> print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")
Tracked 2 objects through 180 frames
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SAM.

* [Demo notebook 🌎](https://colab.research.google.com/drive/1Z0NGLE7p8qnc9UpuI8KBETHd2xBbOEhv?usp=sharing) for using the model, contributed by [Sangbum Choi](https://github.com/SangbumChoi).

## Sam2VideoConfig

### class transformers.Sam2VideoConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/configuration_sam2_video.py#L150)

( vision\_config = None prompt\_encoder\_config = None mask\_decoder\_config = None initializer\_range = 0.02 num\_maskmem = 7 image\_size = 1024 sigmoid\_scale\_for\_mem\_enc = 20.0 sigmoid\_bias\_for\_mem\_enc = -10.0 enable\_occlusion\_spatial\_embedding = True multimask\_output\_in\_sam = True multimask\_min\_pt\_num = 0 multimask\_max\_pt\_num = 1 multimask\_output\_for\_tracking = True max\_object\_pointers\_in\_encoder = 16 enable\_temporal\_pos\_encoding\_for\_object\_pointers = True memory\_attention\_hidden\_size = 256 memory\_attention\_num\_layers = 4 memory\_attention\_num\_attention\_heads = 1 memory\_attention\_downsample\_rate = 1 memory\_attention\_feed\_forward\_hidden\_size = 2048 memory\_attention\_feed\_forward\_hidden\_act = 'relu' memory\_attention\_dropout = 0.1 memory\_attention\_rope\_theta = 10000 memory\_attention\_rope\_feat\_sizes = None memory\_attention\_rope\_dropout = 0.1 memory\_encoder\_hidden\_size = 256 memory\_encoder\_output\_channels = 64 mask\_downsampler\_embed\_dim = 256 mask\_downsampler\_kernel\_size = 3 mask\_downsampler\_stride = 2 mask\_downsampler\_padding = 1 mask\_downsampler\_total\_stride = 16 mask\_downsampler\_hidden\_act = 'gelu' memory\_fuser\_num\_layers = 2 memory\_fuser\_embed\_dim = 256 memory\_fuser\_intermediate\_dim = 1024 memory\_fuser\_kernel\_size = 7 memory\_fuser\_padding = 3 memory\_fuser\_layer\_scale\_init\_value = 1e-06 memory\_fuser\_hidden\_act = 'gelu' \*\*kwargs  )

Parameters

* **vision\_config** (Union[`dict`, `Sam2VisionConfig`], *optional*) —
  Dictionary of configuration options used to initialize [Sam2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2VisionConfig).
* **prompt\_encoder\_config** (Union[`dict`, `Sam2PromptEncoderConfig`], *optional*) —
  Dictionary of configuration options used to initialize [Sam2PromptEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2PromptEncoderConfig).
* **mask\_decoder\_config** (Union[`dict`, `Sam2MaskDecoderConfig`], *optional*) —
  Dictionary of configuration options used to initialize [Sam2MaskDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2MaskDecoderConfig).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  Standard deviation for parameter initialization.
* **num\_maskmem** (`int`, *optional*, defaults to 7) —
  The number of memory slots for the mask memory.
* **image\_size** (`int`, *optional*, defaults to 1024) —
  The size of the input images.
* **sigmoid\_scale\_for\_mem\_enc** (`float`, *optional*, defaults to 20.0) —
  Scale factor for the sigmoid function in the memory encoder.
* **sigmoid\_bias\_for\_mem\_enc** (`float`, *optional*, defaults to -10.0) —
  Bias for the sigmoid function in the memory encoder.
* **enable\_occlusion\_spatial\_embedding** (`bool`, *optional*, defaults to `True`) —
  Whether to enable spatial embedding for occlusions.
* **multimask\_output\_in\_sam** (`bool`, *optional*, defaults to `True`) —
  Whether to output multiple masks from the SAM head.
* **multimask\_min\_pt\_num** (`int`, *optional*, defaults to 0) —
  The minimum number of points to trigger multimask output.
* **multimask\_max\_pt\_num** (`int`, *optional*, defaults to 1) —
  The maximum number of points to trigger multimask output.
* **multimask\_output\_for\_tracking** (`bool`, *optional*, defaults to `True`) —
  Whether to use multimask output for tracking.
* **max\_object\_pointers\_in\_encoder** (`int`, *optional*, defaults to 16) —
  The maximum number of object pointers in the encoder.
* **enable\_temporal\_pos\_encoding\_for\_object\_pointers** (`bool`, *optional*, defaults to `True`) —
  Whether to enable temporal positional encoding for object pointers.
* **memory\_attention\_hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the memory attention hidden states.
* **memory\_attention\_num\_layers** (`int`, *optional*, defaults to 4) —
  The number of layers in the memory attention module.
* **memory\_attention\_num\_attention\_heads** (`int`, *optional*, defaults to 1) —
  Number of attention heads for each attention layer in the memory attention.
* **memory\_attention\_downsample\_rate** (`int`, *optional*, defaults to 1) —
  The downsample rate for the attention layers.
* **memory\_attention\_feed\_forward\_hidden\_size** (`int`, *optional*, defaults to 2048) —
  The dimension of the feedforward network in the memory attention module.
* **memory\_attention\_feed\_forward\_hidden\_act** (`str`, *optional*, defaults to `"relu"`) —
  The non-linear activation function in the feedforward network in the memory attention module.
* **memory\_attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout rate for the memory attention module.
* **memory\_attention\_rope\_theta** (`float`, *optional*, defaults to 10000) —
  The Rope theta parameter.
* **memory\_attention\_rope\_feat\_sizes** (`list[int]`, *optional*, defaults to `[64, 64]`) —
  The feature sizes for the Rope positional encoding.
* **memory\_attention\_rope\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout rate for the Rope positional encoding.
* **memory\_encoder\_hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the memory encoder hidden states.
* **memory\_encoder\_output\_channels** (`int`, *optional*, defaults to 64) —
  The number of output channels for the memory encoder.
* **mask\_downsampler\_embed\_dim** (`int`, *optional*, defaults to 256) —
  The dimension of the mask downsampler embedding.
* **mask\_downsampler\_kernel\_size** (`int`, *optional*, defaults to 3) —
  The kernel size for the mask downsampler.
* **mask\_downsampler\_stride** (`int`, *optional*, defaults to 2) —
  The stride for the mask downsampler.
* **mask\_downsampler\_padding** (`int`, *optional*, defaults to 1) —
  The padding for the mask downsampler.
* **mask\_downsampler\_total\_stride** (`int`, *optional*, defaults to 16) —
  The total stride for the mask downsampler.
* **mask\_downsampler\_hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function in the mask downsampler.
* **memory\_fuser\_num\_layers** (`int`, *optional*, defaults to 2) —
  The number of layers in the memory fuser.
* **memory\_fuser\_embed\_dim** (`int`, *optional*, defaults to 256) —
  The dimension of the embedding layer in the memory fuser.
* **memory\_fuser\_intermediate\_dim** (`int`, *optional*, defaults to 1024) —
  The dimension of the intermediate layer in the memory fuser.
* **memory\_fuser\_kernel\_size** (`int`, *optional*, defaults to 7) —
  The kernel size for the memory fuser.
* **memory\_fuser\_padding** (`int`, *optional*, defaults to 3) —
  The padding for the memory fuser.
* **memory\_fuser\_layer\_scale\_init\_value** (`float`, *optional*, defaults to 1e-06) —
  The initial value for the layer scale in the memory fuser.
* **memory\_fuser\_hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function in the memory fuser.
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

## Sam2VideoMaskDecoderConfig

### class transformers.Sam2VideoMaskDecoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/configuration_sam2_video.py#L77)

( hidden\_size = 256 hidden\_act = 'gelu' mlp\_dim = 2048 num\_hidden\_layers = 2 num\_attention\_heads = 8 attention\_downsample\_rate = 2 num\_multimask\_outputs = 3 iou\_head\_depth = 3 iou\_head\_hidden\_dim = 256 dynamic\_multimask\_via\_stability = True dynamic\_multimask\_stability\_delta = 0.05 dynamic\_multimask\_stability\_thresh = 0.98 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the hidden states.
* **hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function in the SAM2\_VIDEO mask decoder.
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

This is the configuration class to store the configuration of a `Sam2VideoMaskDecoder`. It is used to instantiate a SAM2\_VIDEO
memory encoder according to the specified arguments, defining the model architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## Sam2VideoPromptEncoderConfig

### class transformers.Sam2VideoPromptEncoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/configuration_sam2_video.py#L25)

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

This is the configuration class to store the configuration of a `Sam2VideoPromptEncoder`. The `Sam2VideoPromptEncoder`
module is used to encode the input 2D points and bounding boxes.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## Sam2VideoProcessor

### class transformers.Sam2VideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/processing_sam2_video.py#L37)

( image\_processor video\_processor target\_size: typing.Optional[int] = None point\_pad\_value: int = -10 \*\*kwargs  )

Parameters

* **image\_processor** (`Sam2ImageProcessorFast`) —
  An instance of [Sam2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2ImageProcessorFast).
* **video\_processor** (`Sam2VideoVideoProcessor`) —
  An instance of [Sam2VideoVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoVideoProcessor).
* **target\_size** (`int`, *optional*) —
  The target size (target\_size, target\_size) to which the image will be resized.
* **point\_pad\_value** (`int`, *optional*, defaults to -10) —
  The value used for padding input points.

Constructs a SAM2 processor which wraps a SAM2 image processor and an 2D points & Bounding boxes processor into a
single processor.

[Sam2VideoProcessor](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoProcessor) offers all the functionalities of [Sam2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/sam2#transformers.Sam2ImageProcessorFast) and [Sam2VideoProcessor](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoProcessor). See the docstring of
`__call__()` and [**call**()](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoProcessor.__call__) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/processing_sam2_video.py#L67)

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

This method uses `Sam2VideoImageProcessorFast.__call__` method to prepare image(s) for the model. It also prepares 2D
points and bounding boxes for the model if they are provided.

#### post\_process\_masks

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/processing_sam2_video.py#L483)

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

#### init\_video\_session

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/processing_sam2_video.py#L529)

( video: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]], NoneType] = None inference\_device: typing.Union[str, ForwardRef('torch.device')] = 'cpu' inference\_state\_device: typing.Union[str, ForwardRef('torch.device')] = None processing\_device: typing.Union[str, ForwardRef('torch.device')] = None video\_storage\_device: typing.Union[str, ForwardRef('torch.device')] = None max\_vision\_features\_cache\_size: int = 1 dtype: dtype = torch.float32  )

Parameters

* **video** (`VideoInput`, *optional*) —
  The video to process. No need to provide when streaming.
* **inference\_device** (`str` or `torch.device`, *optional*, defaults to “cpu”) —
  The device to use for inference.
* **inference\_state\_device** (`str` or `torch.device`, *optional*) —
  The device to store the inference state on.
* **processing\_device** (`str` or `torch.device`, *optional*) —
  The device to use for video processing.
* **video\_storage\_device** (`str` or `torch.device`, *optional*) —
  The device to store the processed video frames on.
* **max\_vision\_features\_cache\_size** (`int`, *optional*, defaults to 1) —
  The maximum number of vision features to cache.
* **dtype** (`torch.dtype`, *optional*, defaults to `torch.float32`) —
  The torch dtype to use for the whole session.

Initializes a video session for inference.
If a video is provided (async inference), the video will be processed and stored on the `video_storage_device`.

#### add\_inputs\_to\_inference\_session

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/processing_sam2_video.py#L582)

( inference\_session: Sam2VideoInferenceSession frame\_idx: int obj\_ids: typing.Union[list[int], int] input\_points: typing.Union[list[list[list[list[float]]]], torch.Tensor, NoneType] = None input\_labels: typing.Union[list[list[list[int]]], torch.Tensor, NoneType] = None input\_boxes: typing.Union[list[list[list[float]]], torch.Tensor, NoneType] = None input\_masks: typing.Union[numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], NoneType] = None original\_size: typing.Optional[tuple[int, int]] = None clear\_old\_inputs: bool = True  )

Parameters

* **inference\_session** (`Sam2VideoInferenceSession`) —
  The inference session for the video.
* **frame\_idx** (`int`) —
  The index of the frame to process.
* **obj\_ids** (`list[int]` or `int`) —
  The object ID(s) to associate with the points or box.
  These can be any integers and can be reused later on to specify an object.
* **input\_points** (`list[list[list[list[float]]]]`, `torch.Tensor`, *optional*) —
  The points to add to the frame.
* **input\_labels** (`list[list[list[int]]]`, `torch.Tensor`, *optional*) —
  The labels for the points.
* **input\_boxes** (`list[list[list[float]]]`, `torch.Tensor`, *optional*) —
  The bounding boxes to add to the frame.
* **input\_masks** (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, or `list[torch.Tensor]`, *optional*) —
  The mask(s) to add to the frame.
* **original\_size** (`tuple[int, int]`, *optional*) —
  The original size of the video. Provide when streaming.
* **clear\_old\_inputs** (`bool`, *optional*, defaults to `True`) —
  Whether to clear old inputs for the object.

Process new points, boxes, or masks for a video frame and add them to the inference session.

## Sam2VideoVideoProcessor

### class transformers.Sam2VideoVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/video_processing_sam2_video.py#L46)

( \*\*kwargs: typing\_extensions.Unpack[transformers.processing\_utils.VideosKwargs]  )

#### post\_process\_masks

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/video_processing_sam2_video.py#L76)

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

## Sam2VideoInferenceSession

### class transformers.Sam2VideoInferenceSession

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L102)

( video: FloatTensor = None video\_height: typing.Optional[int] = None video\_width: typing.Optional[int] = None inference\_device: typing.Union[torch.device, str] = 'cpu' inference\_state\_device: typing.Union[torch.device, str] = 'cpu' video\_storage\_device: typing.Union[torch.device, str] = 'cpu' dtype: typing.Union[torch.dtype, str] = 'float32' max\_vision\_features\_cache\_size: int = 1  )

Parameters

* **video** (`torch.FloatTensor`, *optional*) —
  The video to process. No need to provide when streaming.
* **video\_height** (`int`, *optional*) —
  The height of the video.
* **video\_width** (`int`, *optional*) —
  The width of the video.
* **inference\_device** (`torch.device`, *optional*, defaults to `"cpu"`) —
  The device to use for inference.
* **inference\_state\_device** (`torch.device`, *optional*, defaults to `"cpu"`) —
  The device to store the inference state on.
* **video\_storage\_device** (`torch.device`, *optional*, defaults to `"cpu"`) —
  The device to store the video on.
* **dtype** (`torch.dtype`, *optional*, defaults to `"float32"`) —
  The dtype to use for the video.
* **max\_vision\_features\_cache\_size** (`int`, *optional*, defaults to 1) —
  The maximum number of vision features to cache.

Manages video inference session parameters, state and cache.

#### add\_mask\_inputs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L220)

( obj\_idx: int frame\_idx: int inputs: Tensor  )

Add mask inputs with automatic device placement.

#### add\_new\_frame

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L295)

( pixel\_values: Tensor  )

Add new frame with automatic device placement.

#### add\_point\_inputs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L206)

( obj\_idx: int frame\_idx: int inputs: dict  )

Add point inputs with automatic device placement.

#### get\_frame

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L308)

( frame\_idx: int  )

Get frame from video.

#### get\_obj\_num

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L201)

( )

Get the total number of unique object ids received so far in this session.

#### get\_output

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L268)

( obj\_idx: int frame\_idx: int output\_key: str is\_conditioning\_frame: bool = True  )

Parameters

* **obj\_idx** (int) — The index of the object.
* **frame\_idx** (int) — The index of the frame.
* **output\_key** (str) — The key of the output.
* **is\_conditioning\_frame** (bool) — Whether the output is for a conditioning frame.

Get output with smart device management.

#### obj\_id\_to\_idx

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L175)

( obj\_id: int  )

Map object ID to index, creating new entry if needed.

#### obj\_idx\_to\_id

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L197)

( obj\_idx: int  )

Map model-side object index to client-side object id.

#### remove\_mask\_inputs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L226)

( obj\_idx: int frame\_idx: int  )

Remove mask inputs.

#### remove\_point\_inputs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L216)

( obj\_idx: int frame\_idx: int  )

Remove point inputs.

#### reset\_inference\_session

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L324)

( )

Reset tracking data and cache.

#### reset\_tracking\_data

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L312)

( )

Reset tracking data but keep cache.

#### store\_output

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L231)

( obj\_idx: int frame\_idx: int output\_key: typing.Optional[str] = None output\_value: typing.Union[torch.Tensor, dict, NoneType] = None is\_conditioning\_frame: bool = True  )

Parameters

* **obj\_idx** (int) — The index of the object.
* **frame\_idx** (int) — The index of the frame.
* **output\_key** (Optional[str]) — The key of the output. If None, the output is stored as a dictionary.
* **output\_value** (Optional[Union[torch.Tensor, dict]]) — The value of the output.
* **is\_conditioning\_frame** (bool) — Whether the output is for a conditioning frame.

Store output with smart device management.
If output\_key is None, the output is stored as a dictionary.

## Sam2VideoModel

### class transformers.Sam2VideoModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L1556)

( config: Sam2VideoConfig  )

Parameters

* **config** ([Sam2VideoConfig](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Sam2 Video Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L1694)

( inference\_session: Sam2VideoInferenceSession frame\_idx: typing.Optional[int] = None frame: typing.Optional[torch.Tensor] = None reverse: bool = False  ) → `transformers.models.sam2_video.modeling_sam2_video.Sam2VideoSegmentationOutput` or `tuple(torch.FloatTensor)`

Parameters

* **inference\_session** (`~models.sam2_video.modeling_sam2_video.Sam2VideoInferenceSession`) —
  The video inference session object.
* **frame\_idx** (`int`, *optional*) —
  The index of the frame on which to run inference. No need to provide when inferring
  on a new streamed frame.
* **frame** (`torch.Tensor`, *optional*) —
  The frame to process. Provide when streaming.
* **reverse** (`bool`, *optional*, defaults to `False`) —
  Whether to propagate in reverse.

Returns

`transformers.models.sam2_video.modeling_sam2_video.Sam2VideoSegmentationOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.sam2_video.modeling_sam2_video.Sam2VideoSegmentationOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Sam2VideoConfig](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoConfig)) and inputs.

* **pred\_masks** (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`) — The predicted masks stored at the model’s resolution.
* **frame\_idx** (`<class 'int'>.frame_idx`, defaults to `None`) — The frame index of the video.

Propagate the objects through a streamed video frame.

#### propagate\_in\_video\_iterator

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/sam2_video/modeling_sam2_video.py#L2491)

( inference\_session: Sam2VideoInferenceSession start\_frame\_idx: typing.Optional[int] = None max\_frame\_num\_to\_track: typing.Optional[int] = None reverse: bool = False  ) → `transformers.models.sam2_video.modeling_sam2_video.Sam2VideoSegmentationOutput` or `tuple(torch.FloatTensor)`

Parameters

* **inference\_session** (`~models.sam2_video.modeling_sam2_video.Sam2VideoInferenceSession`) —
  The video inference session object.
* **start\_frame\_idx** (`int`, *optional*) —
  The starting frame index for propagation.
  Need to be provided if `forward` hasn’t been called on new inputs yet.
  If not provided, the starting frame index will be the earliest frame with input points.
* **max\_frame\_num\_to\_track** (`int`, *optional*) —
  The maximum number of frames to track.
* **reverse** (`bool`, *optional*, defaults to `False`) —
  Whether to propagate in reverse.

Returns

`transformers.models.sam2_video.modeling_sam2_video.Sam2VideoSegmentationOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.sam2_video.modeling_sam2_video.Sam2VideoSegmentationOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Sam2VideoConfig](/docs/transformers/v4.56.2/en/model_doc/sam2_video#transformers.Sam2VideoConfig)) and inputs.

* **pred\_masks** (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`) — The predicted masks stored at the model’s resolution.
* **frame\_idx** (`<class 'int'>.frame_idx`, defaults to `None`) — The frame index of the video.

Propagate the objects through the video frames. Used when initializing an inference session with a whole video.
Yields Sam2VideoSegmentationOutput for each frame.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/sam2_video.md)
