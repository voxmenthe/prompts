# OWL-ViT

## Overview

The OWL-ViT (short for Vision Transformer for Open-World Localization) was proposed in [Simple Open-Vocabulary Object Detection with Vision Transformers](https://huggingface.co/papers/2205.06230) by Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. OWL-ViT is an open-vocabulary object detection network trained on a variety of (image, text) pairs. It can be used to query an image with one or multiple text queries to search for and detect target objects described in text.

The abstract from the paper is the following:

*Combining simple architectures with large-scale pre-training has led to massive improvements in image classification. For object detection, pre-training and scaling approaches are less well established, especially in the long-tailed and open-vocabulary setting, where training data is relatively scarce. In this paper, we propose a strong recipe for transferring image-text models to open-vocabulary object detection. We use a standard Vision Transformer architecture with minimal modifications, contrastive image-text pre-training, and end-to-end detection fine-tuning. Our analysis of the scaling properties of this setup shows that increasing image-level pre-training and model size yield consistent improvements on the downstream detection task. We provide the adaptation strategies and regularizations needed to attain very strong performance on zero-shot text-conditioned and one-shot image-conditioned object detection. Code and models are available on GitHub.*

 OWL-ViT architecture. Taken from the original paper. 

This model was contributed by [adirik](https://huggingface.co/adirik). The original code can be found [here](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit).

## Usage tips

OWL-ViT is a zero-shot text-conditioned object detection model. OWL-ViT uses [CLIP](clip) as its multi-modal backbone, with a ViT-like Transformer to get visual features and a causal language model to get the text features. To use CLIP for detection, OWL-ViT removes the final token pooling layer of the vision model and attaches a lightweight classification and box head to each transformer output token. Open-vocabulary classification is enabled by replacing the fixed classification layer weights with the class-name embeddings obtained from the text model. The authors first train CLIP from scratch and fine-tune it end-to-end with the classification and box heads on standard detection datasets using a bipartite matching loss. One or multiple text queries per image can be used to perform zero-shot text-conditioned object detection.

[OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor) can be used to resize (or rescale) and normalize images for the model and [CLIPTokenizer](/docs/transformers/main/en/model_doc/clip#transformers.CLIPTokenizer) is used to encode the text. [OwlViTProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTProcessor) wraps [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor) and [CLIPTokenizer](/docs/transformers/main/en/model_doc/clip#transformers.CLIPTokenizer) into a single instance to both encode the text and prepare the images. The following example shows how to perform object detection using [OwlViTProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTProcessor) and [OwlViTForObjectDetection](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTForObjectDetection).

```python
>>> import requests
>>> from PIL import Image
>>> import torch

>>> from transformers import OwlViTProcessor, OwlViTForObjectDetection

>>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
>>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text_labels = [["a photo of a cat", "a photo of a dog"]]
>>> inputs = processor(text=text_labels, images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
>>> target_sizes = torch.tensor([(image.height, image.width)])
>>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
>>> results = processor.post_process_grounded_object_detection(
...     outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
... )
>>> # Retrieve predictions for the first image for the corresponding text queries
>>> result = results[0]
>>> boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
>>> for box, score, text_label in zip(boxes, scores, text_labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
Detected a photo of a cat with confidence 0.707 at location [324.97, 20.44, 640.58, 373.29]
Detected a photo of a cat with confidence 0.717 at location [1.46, 55.26, 315.55, 472.17]
```

## Resources

A demo notebook on using OWL-ViT for zero- and one-shot (image-guided) object detection can be found [here](https://github.com/huggingface/notebooks/blob/main/examples/zeroshot_object_detection_with_owlvit.ipynb).

## OwlViTConfig[[transformers.OwlViTConfig]]

#### transformers.OwlViTConfig[[transformers.OwlViTConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/configuration_owlvit.py#L208)

[OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig) is the configuration class to store the configuration of an [OwlViTModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTModel). It is used to
instantiate an OWL-ViT model according to the specified arguments, defining the text model and vision model
configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the OWL-ViT
[google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

**Parameters:**

text_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [OwlViTTextConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextConfig).

vision_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [OwlViTVisionConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionConfig).

projection_dim (`int`, *optional*, defaults to 512) : Dimensionality of text and vision projection layers.

logit_scale_init_value (`float`, *optional*, defaults to 2.6592) : The initial value of the *logit_scale* parameter. Default is used as per the original OWL-ViT implementation.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not the model should return a dictionary. If `False`, returns a tuple.

kwargs (*optional*) : Dictionary of keyword arguments.

## OwlViTTextConfig[[transformers.OwlViTTextConfig]]

#### transformers.OwlViTTextConfig[[transformers.OwlViTTextConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/configuration_owlvit.py#L24)

This is the configuration class to store the configuration of an [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel). It is used to instantiate an
OwlViT text encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the OwlViT
[google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import OwlViTTextConfig, OwlViTTextModel

>>> # Initializing a OwlViTTextModel with google/owlvit-base-patch32 style configuration
>>> configuration = OwlViTTextConfig()

>>> # Initializing a OwlViTTextConfig from the google/owlvit-base-patch32 style configuration
>>> model = OwlViTTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 49408) : Vocabulary size of the OWL-ViT text model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).

hidden_size (`int`, *optional*, defaults to 512) : Dimensionality of the encoder layers and the pooler layer.

intermediate_size (`int`, *optional*, defaults to 2048) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 8) : Number of attention heads for each attention layer in the Transformer encoder.

max_position_embeddings (`int`, *optional*, defaults to 16) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

initializer_factor (`float`, *optional*, defaults to 1.0) : A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

pad_token_id (`int`, *optional*, defaults to 0) : The id of the padding token in the input sequences.

bos_token_id (`int`, *optional*, defaults to 49406) : The id of the beginning-of-sequence token in the input sequences.

eos_token_id (`int`, *optional*, defaults to 49407) : The id of the end-of-sequence token in the input sequences.

## OwlViTVisionConfig[[transformers.OwlViTVisionConfig]]

#### transformers.OwlViTVisionConfig[[transformers.OwlViTVisionConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/configuration_owlvit.py#L120)

This is the configuration class to store the configuration of an [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel). It is used to instantiate
an OWL-ViT image encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the OWL-ViT
[google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import OwlViTVisionConfig, OwlViTVisionModel

>>> # Initializing a OwlViTVisionModel with google/owlvit-base-patch32 style configuration
>>> configuration = OwlViTVisionConfig()

>>> # Initializing a OwlViTVisionModel model from the google/owlvit-base-patch32 style configuration
>>> model = OwlViTVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

num_channels (`int`, *optional*, defaults to 3) : Number of channels in the input images.

image_size (`int`, *optional*, defaults to 768) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 32) : The size (resolution) of each patch.

hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

initializer_factor (`float`, *optional*, defaults to 1.0) : A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

## OwlViTImageProcessor[[transformers.OwlViTImageProcessor]]

#### transformers.OwlViTImageProcessor[[transformers.OwlViTImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/image_processing_owlvit.py#L124)

Constructs an OWL-ViT image processor.

This image processor inherits from [ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

preprocesstransformers.OwlViTImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/image_processing_owlvit.py#L303[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "resample", "val": ": typing.Optional[PIL.Image.Resampling] = None"}, {"name": "do_center_crop", "val": ": typing.Optional[bool] = None"}, {"name": "crop_size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension] = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}]- **images** (`ImageInput`) --
  The image or batch of images to be prepared. Expects a single or batch of images with pixel values
  ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether or not to resize the input. If `True`, will resize the input to the size specified by `size`.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  The size to resize the input to. Only has an effect if `do_resize` is set to `True`.
- **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) --
  The resampling filter to use when resizing the input. Only has an effect if `do_resize` is set to
  `True`.
- **do_center_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) --
  Whether or not to center crop the input. If `True`, will center crop the input to the size specified by
  `crop_size`.
- **crop_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) --
  The size to center crop the input to. Only has an effect if `do_center_crop` is set to `True`.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether or not to rescale the input. If `True`, will rescale the input by dividing it by
  `rescale_factor`.
- **rescale_factor** (`float`, *optional*, defaults to `self.rescale_factor`) --
  The factor to rescale the input by. Only has an effect if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether or not to normalize the input. If `True`, will normalize the input by subtracting `image_mean`
  and dividing by `image_std`.
- **image_mean** (`Union[float, list[float]]`, *optional*, defaults to `self.image_mean`) --
  The mean to subtract from the input when normalizing. Only has an effect if `do_normalize` is set to
  `True`.
- **image_std** (`Union[float, list[float]]`, *optional*, defaults to `self.image_std`) --
  The standard deviation to divide the input by when normalizing. Only has an effect if `do_normalize` is
  set to `True`.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  The type of tensors to return. Can be one of:
  - Unset: Return a list of `np.ndarray`.
  - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
- **data_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) --
  The channel dimension format for the output image. Can be one of:
  - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - Unset: defaults to the channel dimension format of the input image.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.0

Prepares an image or batch of images for the model.

**Parameters:**

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the shorter edge of the input to a certain `size`.

size (`dict[str, int]`, *optional*, defaults to {"height" : 768, "width": 768}): The size to use for resizing the image. Only has an effect if `do_resize` is set to `True`. If `size` is a sequence like (h, w), output size will be matched to this. If `size` is an int, then image will be resized to (size, size).

resample (`int`, *optional*, defaults to `Resampling.BICUBIC`) : An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`, `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`, `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set to `True`.

do_center_crop (`bool`, *optional*, defaults to `False`) : Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the image is padded with 0's and then center cropped.

crop_size (`int`, *optional*, defaults to {"height" : 768, "width": 768}): The size to use for center cropping the image. Only has an effect if `do_center_crop` is set to `True`.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the input by a certain factor.

rescale_factor (`float`, *optional*, defaults to `1/255`) : The factor to use for rescaling the image. Only has an effect if `do_rescale` is set to `True`.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether or not to normalize the input with `image_mean` and `image_std`. Desired output size when applying center-cropping. Only has an effect if `do_center_crop` is set to `True`.

image_mean (`list[int]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`) : The sequence of means for each channel, to be used when normalizing images.

image_std (`list[int]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`) : The sequence of standard deviations for each channel, to be used when normalizing images.

## OwlViTImageProcessorFast[[transformers.OwlViTImageProcessorFast]]

#### transformers.OwlViTImageProcessorFast[[transformers.OwlViTImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/image_processing_owlvit_fast.py#L36)

Constructs a fast Owlvit image processor.

preprocesstransformers.OwlViTImageProcessorFast.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L839[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "*args", "val": ""}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.processing_utils.ImagesKwargs]"}]- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_convert_rgb** (`bool`, *optional*) --
  Whether to convert the image to RGB.
- **do_resize** (`bool`, *optional*) --
  Whether to resize the image.
- **size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  Describes the maximum input dimensions to the model.
- **crop_size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  Size of the output image after applying `center_crop`.
- **resample** (`Annotated[Union[PILImageResampling, int, NoneType], None]`) --
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
- **do_rescale** (`bool`, *optional*) --
  Whether to rescale the image.
- **rescale_factor** (`float`, *optional*) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*) --
  Whether to normalize the image.
- **image_mean** (`Union[float, list[float], tuple[float, ...], NoneType]`) --
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
- **image_std** (`Union[float, list[float], tuple[float, ...], NoneType]`) --
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
- **do_pad** (`bool`, *optional*) --
  Whether to pad the image. Padding is done either to the largest size in the batch
  or to a fixed square size per image. The exact padding strategy depends on the model.
- **pad_size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
  height and width in the batch. Applied only when `do_pad=True.`
- **do_center_crop** (`bool`, *optional*) --
  Whether to center crop the image.
- **data_format** (`Union[~image_utils.ChannelDimension, str, NoneType]`) --
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
- **input_data_format** (`Union[~image_utils.ChannelDimension, str, NoneType]`) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
- **device** (`Annotated[Union[str, torch.device, NoneType], None]`) --
  The device to process the images on. If unset, the device is inferred from the input images.
- **return_tensors** (`Annotated[Union[str, ~utils.generic.TensorType, NoneType], None]`) --
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
- **disable_grouping** (`bool`, *optional*) --
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157
- **image_seq_length** (`int`, *optional*) --
  The number of image tokens to be used for each image in the input.
  Added for backward compatibility but this should be set as a processor attribute in future models.0``- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

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

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.
#### post_process_object_detection[[transformers.OwlViTImageProcessorFast.post_process_object_detection]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/image_processing_owlvit_fast.py#L51)

Converts the raw output of [OwlViTForObjectDetection](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) into final bounding boxes in (top_left_x, top_left_y,
bottom_right_x, bottom_right_y) format.

**Parameters:**

outputs (`OwlViTObjectDetectionOutput`) : Raw outputs of the model.

threshold (`float`, *optional*, defaults to 0.1) : Score threshold to keep object detection predictions.

target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) : Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the batch. If unset, predictions will not be resized.

**Returns:**

``list[Dict]``

A list of dictionaries, each dictionary containing the following keys:
- "scores": The confidence scores for each predicted box on the image.
- "labels": Indexes of the classes predicted by the model on the image.
- "boxes": Image bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
#### post_process_image_guided_detection[[transformers.OwlViTImageProcessorFast.post_process_image_guided_detection]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/image_processing_owlvit_fast.py#L105)

Converts the output of [OwlViTForObjectDetection.image_guided_detection()](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTForObjectDetection.image_guided_detection) into the format expected by the COCO
api.

**Parameters:**

outputs (`OwlViTImageGuidedObjectDetectionOutput`) : Raw outputs of the model.

threshold (`float`, *optional*, defaults to 0.0) : Minimum confidence threshold to use to filter out predicted boxes.

nms_threshold (`float`, *optional*, defaults to 0.3) : IoU threshold for non-maximum suppression of overlapping boxes.

target_sizes (`torch.Tensor`, *optional*) : Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in the batch. If set, predicted normalized bounding boxes are rescaled to the target sizes. If left to None, predictions will not be unnormalized.

**Returns:**

``list[Dict]``

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
in the batch as predicted by the model. All labels are set to None as
`OwlViTForObjectDetection.image_guided_detection` perform one-shot object detection.

## OwlViTProcessor[[transformers.OwlViTProcessor]]

#### transformers.OwlViTProcessor[[transformers.OwlViTProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/processing_owlvit.py#L55)

Constructs an OWL-ViT processor which wraps [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor) and [CLIPTokenizer](/docs/transformers/main/en/model_doc/clip#transformers.CLIPTokenizer)/[CLIPTokenizerFast](/docs/transformers/main/en/model_doc/clip#transformers.CLIPTokenizer)
into a single processor that inherits both the image processor and tokenizer functionalities. See the
[__call__()](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTProcessor.__call__) and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

__call__transformers.OwlViTProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/processing_owlvit.py#L71[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "text", "val": ": typing.Union[str, list[str], list[list[str]]] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.owlvit.processing_owlvit.OwlViTProcessorKwargs]"}]- **images** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, --
- **`list[torch.Tensor]`)** --
  The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
  tensor. Both channels-first and channels-last formats are supported.
- **text** (`str`, `list[str]`, `list[list[str]]`) --
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **query_images** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`) --
  The query image to be prepared, one query image is expected per target image to be queried. Each image
  can be a PIL image, NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each image
  should be of shape (C, H, W), where C is a number of channels, H and W are image height and width.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors of a particular framework. Acceptable values are:
  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return NumPy `np.ndarray` objects.0[BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature)A [BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:
- **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
  `None`).
- **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
- **query_pixel_values** -- Pixel values of the query images to be fed to a model. Returned when `query_images` is not `None`.

Main method to prepare for the model one or several text(s) and image(s). This method forwards the `text` and
`kwargs` arguments to CLIPTokenizerFast's [__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) if `text` is not `None` to encode:
the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
CLIPImageProcessor's [__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) if `images` is not `None`. Please refer to the docstring
of the above two methods for more information.

**Parameters:**

image_processor ([OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor), *optional*) : The image processor is a required input.

tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`], *optional*) : The tokenizer is a required input.

**Returns:**

`[BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature)`

A [BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:
- **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
  `None`).
- **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
- **query_pixel_values** -- Pixel values of the query images to be fed to a model. Returned when `query_images` is not `None`.
#### post_process_grounded_object_detection[[transformers.OwlViTProcessor.post_process_grounded_object_detection]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/processing_owlvit.py#L177)

Converts the raw output of [OwlViTForObjectDetection](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) into final bounding boxes in (top_left_x, top_left_y,
bottom_right_x, bottom_right_y) format.

**Parameters:**

outputs (`OwlViTObjectDetectionOutput`) : Raw outputs of the model.

threshold (`float`, *optional*, defaults to 0.1) : Score threshold to keep object detection predictions.

target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) : Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the batch. If unset, predictions will not be resized.

text_labels (`list[list[str]]`, *optional*) : List of lists of text labels for each image in the batch. If unset, "text_labels" in output will be set to `None`.

**Returns:**

``list[Dict]``

A list of dictionaries, each dictionary containing the following keys:
- "scores": The confidence scores for each predicted box on the image.
- "labels": Indexes of the classes predicted by the model on the image.
- "boxes": Image bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
- "text_labels": The text labels for each predicted bounding box on the image.
#### post_process_image_guided_detection[[transformers.OwlViTProcessor.post_process_image_guided_detection]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/processing_owlvit.py#L225)

Converts the output of [OwlViTForObjectDetection.image_guided_detection()](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTForObjectDetection.image_guided_detection) into the format expected by the COCO
api.

**Parameters:**

outputs (`OwlViTImageGuidedObjectDetectionOutput`) : Raw outputs of the model.

threshold (`float`, *optional*, defaults to 0.0) : Minimum confidence threshold to use to filter out predicted boxes.

nms_threshold (`float`, *optional*, defaults to 0.3) : IoU threshold for non-maximum suppression of overlapping boxes.

target_sizes (`torch.Tensor`, *optional*) : Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in the batch. If set, predicted normalized bounding boxes are rescaled to the target sizes. If left to None, predictions will not be unnormalized.

**Returns:**

``list[Dict]``

A list of dictionaries, each dictionary containing the following keys:
- "scores": The confidence scores for each predicted box on the image.
- "boxes": Image bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
- "labels": Set to `None`.

## OwlViTModel[[transformers.OwlViTModel]]

#### transformers.OwlViTModel[[transformers.OwlViTModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L919)

The bare Owlvit Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.OwlViTModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L1026[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "return_loss", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "return_base_image_embeds", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor). See [OwlViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([OwlViTProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTProcessor) uses
  [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor) for processing images).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **return_loss** (`bool`, *optional*) --
  Whether or not to return the contrastive loss.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **return_base_image_embeds** (`bool`, *optional*) --
  Whether or not to return the base image embeddings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.owlvit.modeling_owlvit.OwlViTOutput` or `tuple(torch.FloatTensor)`A `transformers.models.owlvit.modeling_owlvit.OwlViTOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size * num_max_text_queries, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The image embeddings obtained by applying the projection layer to the pooled output of
  [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel).
The [OwlViTModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:
```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, OwlViTModel

>>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(text=[["a photo of a cat", "a photo of a dog"]], images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

**Parameters:**

config ([OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.owlvit.modeling_owlvit.OwlViTOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.owlvit.modeling_owlvit.OwlViTOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size * num_max_text_queries, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The image embeddings obtained by applying the projection layer to the pooled output of
  [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel).
#### get_text_features[[transformers.OwlViTModel.get_text_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L954)

Examples:
```python
>>> import torch
>>> from transformers import AutoProcessor, OwlViTModel

>>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
>>> inputs = processor(
...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
... )
>>> with torch.inference_mode():
...     text_features = model.get_text_features(**inputs)
```

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`) : Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

**Returns:**

`text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The text embeddings obtained by
applying the projection layer to the pooled output of [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).
#### get_image_features[[transformers.OwlViTModel.get_image_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L990)

Examples:
```python
>>> import torch
>>> from transformers.image_utils import load_image
>>> from transformers import AutoProcessor, OwlViTModel

>>> model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = load_image(url)

>>> inputs = processor(images=image, return_tensors="pt")
>>> with torch.inference_mode():
...     image_features = model.get_image_features(**inputs)
```

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor). See [OwlViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([OwlViTProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTProcessor) uses [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor) for processing images).

interpolate_pos_encoding (`bool`, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

**Returns:**

`image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The image embeddings obtained by
applying the projection layer to the pooled output of [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel).

## OwlViTTextModel[[transformers.OwlViTTextModel]]

#### transformers.OwlViTTextModel[[transformers.OwlViTTextModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L756)

forwardtransformers.OwlViTTextModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L772[{"name": "input_ids", "val": ": Tensor"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:
```python
>>> from transformers import AutoProcessor, OwlViTTextModel

>>> model = OwlViTTextModel.from_pretrained("google/owlvit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
>>> inputs = processor(
...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
... )
>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
```

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`) : Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## OwlViTVisionModel[[transformers.OwlViTVisionModel]]

#### transformers.OwlViTVisionModel[[transformers.OwlViTVisionModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L867)

forwardtransformers.OwlViTVisionModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L881[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor). See [OwlViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([OwlViTProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTProcessor) uses
  [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor) for processing images).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:
```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, OwlViTVisionModel

>>> model = OwlViTVisionModel.from_pretrained("google/owlvit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled CLS states
```

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor). See [OwlViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([OwlViTProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTProcessor) uses [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor) for processing images).

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

interpolate_pos_encoding (`bool`, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## OwlViTForObjectDetection[[transformers.OwlViTForObjectDetection]]

#### transformers.OwlViTForObjectDetection[[transformers.OwlViTForObjectDetection]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L1188)

forwardtransformers.OwlViTForObjectDetection.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L1539[{"name": "input_ids", "val": ": Tensor"}, {"name": "pixel_values", "val": ": FloatTensor"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids).
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor). See [OwlViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([OwlViTProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTProcessor) uses
  [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor) for processing images).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the last hidden state. See `text_model_last_hidden_state` and
  `vision_model_last_hidden_state` under returned tensors for more detail.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.owlvit.modeling_owlvit.OwlViTObjectDetectionOutput` or `tuple(torch.FloatTensor)`A `transformers.models.owlvit.modeling_owlvit.OwlViTObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) -- Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
- **loss_dict** (`Dict`, *optional*) -- A dictionary containing the individual losses. Useful for logging.
- **logits** (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`) -- Classification logits (including no-object) for all queries.
- **pred_boxes** (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`) -- Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use `post_process_object_detection()` to retrieve the
  unnormalized bounding boxes.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, num_max_text_queries, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`) -- Pooled output of [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel). OWL-ViT represents images as a set of image patches and computes
  image embeddings for each patch.
- **class_embeds** (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`) -- Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
  number of patches is (image_size / patch_size)**2.
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel).
The [OwlViTForObjectDetection](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:
```python
>>> import requests
>>> from PIL import Image
>>> import torch

>>> from transformers import OwlViTProcessor, OwlViTForObjectDetection

>>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
>>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text_labels = [["a photo of a cat", "a photo of a dog"]]
>>> inputs = processor(text=text_labels, images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
>>> target_sizes = torch.tensor([(image.height, image.width)])
>>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
>>> results = processor.post_process_grounded_object_detection(
...     outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
... )
>>> # Retrieve predictions for the first image for the corresponding text queries
>>> result = results[0]
>>> boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
>>> for box, score, text_label in zip(boxes, scores, text_labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
Detected a photo of a cat with confidence 0.707 at location [324.97, 20.44, 640.58, 373.29]
Detected a photo of a cat with confidence 0.717 at location [1.46, 55.26, 315.55, 472.17]
```

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input IDs?](../glossary#input-ids).

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor). See [OwlViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([OwlViTProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTProcessor) uses [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor) for processing images).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the last hidden state. See `text_model_last_hidden_state` and `vision_model_last_hidden_state` under returned tensors for more detail.

interpolate_pos_encoding (`bool`, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

``transformers.models.owlvit.modeling_owlvit.OwlViTObjectDetectionOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.owlvit.modeling_owlvit.OwlViTObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) -- Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
- **loss_dict** (`Dict`, *optional*) -- A dictionary containing the individual losses. Useful for logging.
- **logits** (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`) -- Classification logits (including no-object) for all queries.
- **pred_boxes** (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`) -- Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use `post_process_object_detection()` to retrieve the
  unnormalized bounding boxes.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, num_max_text_queries, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`) -- Pooled output of [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel). OWL-ViT represents images as a set of image patches and computes
  image embeddings for each patch.
- **class_embeds** (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`) -- Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
  number of patches is (image_size / patch_size)**2.
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel).
#### image_guided_detection[[transformers.OwlViTForObjectDetection.image_guided_detection]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L1436)

Examples:
```python
>>> import requests
>>> from PIL import Image
>>> import torch
>>> from transformers import AutoProcessor, OwlViTForObjectDetection

>>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch16")
>>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
>>> query_image = Image.open(requests.get(query_url, stream=True).raw)
>>> inputs = processor(images=image, query_images=query_image, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model.image_guided_detection(**inputs)
>>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
>>> target_sizes = torch.Tensor([image.size[::-1]])
>>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
>>> results = processor.post_process_image_guided_detection(
...     outputs=outputs, threshold=0.6, nms_threshold=0.3, target_sizes=target_sizes
... )
>>> i = 0  # Retrieve predictions for the first image
>>> boxes, scores = results[i]["boxes"], results[i]["scores"]
>>> for box, score in zip(boxes, scores):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")
Detected similar object with confidence 0.856 at location [10.94, 50.4, 315.8, 471.39]
Detected similar object with confidence 1.0 at location [334.84, 25.33, 636.16, 374.71]
```

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor). See [OwlViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([OwlViTProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTProcessor) uses [OwlViTImageProcessor](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTImageProcessor) for processing images).

query_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) : Pixel values of query image(s) to be detected. Pass in one query image per target image.

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

interpolate_pos_encoding (`bool`, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

``transformers.models.owlvit.modeling_owlvit.OwlViTImageGuidedObjectDetectionOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.owlvit.modeling_owlvit.OwlViTImageGuidedObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OwlViTConfig](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTConfig)) and inputs.

- **logits** (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`) -- Classification logits (including no-object) for all queries.
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`) -- Pooled output of [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel). OWL-ViT represents images as a set of image patches and computes
  image embeddings for each patch.
- **query_image_embeds** (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`) -- Pooled output of [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel). OWL-ViT represents images as a set of image patches and computes
  image embeddings for each patch.
- **target_pred_boxes** (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`) -- Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual target image in the batch
  (disregarding possible padding). You can use `post_process_object_detection()` to
  retrieve the unnormalized bounding boxes.
- **query_pred_boxes** (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`) -- Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual query image in the batch
  (disregarding possible padding). You can use `post_process_object_detection()` to
  retrieve the unnormalized bounding boxes.
- **class_embeds** (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`) -- Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
  number of patches is (image_size / patch_size)**2.
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [OwlViTTextModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [OwlViTVisionModel](/docs/transformers/main/en/model_doc/owlvit#transformers.OwlViTVisionModel).
