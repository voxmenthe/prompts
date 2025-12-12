# MobileViT

    
        
    

[MobileViT](https://huggingface.co/papers/2110.02178) is a lightweight vision transformer for mobile devices that merges CNNs's efficiency and inductive biases with transformers global context modeling. It treats transformers as convolutions, enabling global information processing without the heavy computational cost of standard ViTs.

   

You can find all the original MobileViT checkpoints under the [Apple](https://huggingface.co/apple/models?search=mobilevit) organization.

> [!TIP]
>
> - This model was contributed by [matthijs](https://huggingface.co/Matthijs).
>
> Click on the MobileViT models in the right sidebar for more examples of how to apply MobileViT to different vision tasks.

The example below demonstrates how to do [Image Classification] with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) and the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

```python

import torch
from transformers import pipeline

classifier = pipeline(
   task="image-classification",
   model="apple/mobilevit-small",
   dtype=torch.float16, device=0,
)

preds = classifier("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
print(f"Prediction: {preds}\n")
```

```python

import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, MobileViTForImageClassification

image_processor = AutoImageProcessor.from_pretrained(
   "apple/mobilevit-small",
   use_fast=True,
)
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small", device_map="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is:{predicted_class_label}")
```

## Notes

- Does **not** operate on sequential data, it's purely designed for image tasks.
- Feature maps are used directly instead of token embeddings.
- Use [MobileViTImageProcessor](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) to preprocess images.
- If using custom preprocessing, ensure that images are in **BGR** format (not RGB), as expected by the pretrained weights.
- The classification models are pretrained on [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).
- The segmentation models use a [DeepLabV3](https://huggingface.co/papers/1706.05587) head and are pretrained on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

## MobileViTConfig[[transformers.MobileViTConfig]]

#### transformers.MobileViTConfig[[transformers.MobileViTConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/configuration_mobilevit.py#L24)

This is the configuration class to store the configuration of a [MobileViTModel](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTModel). It is used to instantiate a
MobileViT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the MobileViT
[apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import MobileViTConfig, MobileViTModel

>>> # Initializing a mobilevit-small style configuration
>>> configuration = MobileViTConfig()

>>> # Initializing a model from the mobilevit-small style configuration
>>> model = MobileViTModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

image_size (`int`, *optional*, defaults to 256) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 2) : The size (resolution) of each patch.

hidden_sizes (`list[int]`, *optional*, defaults to `[144, 192, 240]`) : Dimensionality (hidden size) of the Transformer encoders at each stage.

neck_hidden_sizes (`list[int]`, *optional*, defaults to `[16, 32, 64, 96, 128, 160, 640]`) : The number of channels for the feature maps of the backbone.

num_attention_heads (`int`, *optional*, defaults to 4) : Number of attention heads for each attention layer in the Transformer encoder.

mlp_ratio (`float`, *optional*, defaults to 2.0) : The ratio of the number of channels in the output of the MLP to the number of channels in the input.

expand_ratio (`float`, *optional*, defaults to 4.0) : Expansion factor for the MobileNetv2 layers.

hidden_act (`str` or `function`, *optional*, defaults to `"silu"`) : The non-linear activation function (function or string) in the Transformer encoder and convolution layers.

conv_kernel_size (`int`, *optional*, defaults to 3) : The size of the convolutional kernel in the MobileViT layer.

output_stride (`int`, *optional*, defaults to 32) : The ratio of the spatial resolution of the output to the resolution of the input image.

hidden_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the Transformer encoder.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

classifier_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for attached classifiers.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries, keys and values.

aspp_out_channels (`int`, *optional*, defaults to 256) : Number of output channels used in the ASPP layer for semantic segmentation.

atrous_rates (`list[int]`, *optional*, defaults to `[6, 12, 18]`) : Dilation (atrous) factors used in the ASPP layer for semantic segmentation.

aspp_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for the ASPP layer for semantic segmentation.

semantic_loss_ignore_index (`int`, *optional*, defaults to 255) : The index that is ignored by the loss function of the semantic segmentation model.

## MobileViTImageProcessor[[transformers.MobileViTImageProcessor]]

#### transformers.MobileViTImageProcessor[[transformers.MobileViTImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/image_processing_mobilevit.py#L71)

Constructs a MobileViT image processor.

preprocesstransformers.MobileViTImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/image_processing_mobilevit.py#L341[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "segmentation_maps", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "resample", "val": ": typing.Optional[PIL.Image.Resampling] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_center_crop", "val": ": typing.Optional[bool] = None"}, {"name": "crop_size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "do_flip_channel_order", "val": ": typing.Optional[bool] = None"}, {"name": "do_reduce_labels", "val": ": typing.Optional[bool] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": ChannelDimension = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}]- **images** (`ImageInput`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **segmentation_maps** (`ImageInput`, *optional*) --
  Segmentation map to preprocess.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the image.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  Size of the image after resizing.
- **resample** (`int`, *optional*, defaults to `self.resample`) --
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the image by rescale factor.
- **rescale_factor** (`float`, *optional*, defaults to `self.rescale_factor`) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **do_center_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) --
  Whether to center crop the image.
- **crop_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) --
  Size of the center crop if `do_center_crop` is set to `True`.
- **do_flip_channel_order** (`bool`, *optional*, defaults to `self.do_flip_channel_order`) --
  Whether to flip the channel order of the image.
- **do_reduce_labels** (`bool`, *optional*, defaults to `self.do_reduce_labels`) --
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
  is used for background, and background itself is not included in all classes of a dataset (e.g.
  ADE20k). The background label will be replaced by 255.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  The type of tensors to return. Can be one of:
  - Unset: Return a list of `np.ndarray`.
  - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
- **data_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) --
  The channel dimension format for the output image. Can be one of:
  - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.0

Preprocess an image or batch of images.

**Parameters:**

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the `do_resize` parameter in the `preprocess` method.

size (`dict[str, int]` *optional*, defaults to `{"shortest_edge" : 224}`): Controls the size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess` method.

resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) : Defines the resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the `preprocess` method.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.

rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.

do_center_crop (`bool`, *optional*, defaults to `True`) : Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the image is padded with 0's and then center cropped. Can be overridden by the `do_center_crop` parameter in the `preprocess` method.

crop_size (`dict[str, int]`, *optional*, defaults to `{"height" : 256, "width": 256}`): Desired output size `(size["height"], size["width"])` when applying center-cropping. Can be overridden by the `crop_size` parameter in the `preprocess` method.

do_flip_channel_order (`bool`, *optional*, defaults to `True`) : Whether to flip the color channels from RGB to BGR. Can be overridden by the `do_flip_channel_order` parameter in the `preprocess` method.

do_reduce_labels (`bool`, *optional*, defaults to `False`) : Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255. Can be overridden by the `do_reduce_labels` parameter in the `preprocess` method.
#### post_process_semantic_segmentation[[transformers.MobileViTImageProcessor.post_process_semantic_segmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/image_processing_mobilevit.py#L484)

Converts the output of [MobileViTForSemanticSegmentation](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) into semantic segmentation maps.

**Parameters:**

outputs ([MobileViTForSemanticSegmentation](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation)) : Raw outputs of the model.

target_sizes (`list[Tuple]` of length `batch_size`, *optional*) : List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

**Returns:**

`semantic_segmentation`

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

## MobileViTImageProcessorFast[[transformers.MobileViTImageProcessorFast]]

#### transformers.MobileViTImageProcessorFast[[transformers.MobileViTImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/image_processing_mobilevit_fast.py#L44)

Constructs a fast Mobilevit image processor.

preprocesstransformers.MobileViTImageProcessorFast.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/image_processing_mobilevit_fast.py#L72[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "segmentation_maps", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.mobilevit.image_processing_mobilevit.MobileVitImageProcessorKwargs]"}]- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **segmentation_maps** (`ImageInput`, *optional*) --
  The segmentation maps to preprocess.
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
  Added for backward compatibility but this should be set as a processor attribute in future models.
- **do_flip_channel_order** (`bool`, *optional*, defaults to `self.do_flip_channel_order`) --
  Whether to flip the color channels from RGB to BGR or vice versa.
- **do_reduce_labels** (`bool`, *optional*, defaults to `self.do_reduce_labels`) --
  Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
  is used for background, and background itself is not included in all classes of a dataset (e.g.
  ADE20k). The background label will be replaced by 255.0``- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

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

do_flip_channel_order (`bool`, *optional*, defaults to `self.do_flip_channel_order`) : Whether to flip the color channels from RGB to BGR or vice versa.

do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`) : Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.
#### post_process_semantic_segmentation[[transformers.MobileViTImageProcessorFast.post_process_semantic_segmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/image_processing_mobilevit_fast.py#L189)

Converts the output of [MobileNetV2ForSemanticSegmentation](/docs/transformers/main/en/model_doc/mobilenet_v2#transformers.MobileNetV2ForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

**Parameters:**

outputs ([MobileNetV2ForSemanticSegmentation](/docs/transformers/main/en/model_doc/mobilenet_v2#transformers.MobileNetV2ForSemanticSegmentation)) : Raw outputs of the model.

target_sizes (`list[Tuple]` of length `batch_size`, *optional*) : List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

**Returns:**

`semantic_segmentation`

`list[torch.Tensor]` of length `batch_size`, where each item is a semantic
segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

## MobileViTModel[[transformers.MobileViTModel]]

#### transformers.MobileViTModel[[transformers.MobileViTModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/modeling_mobilevit.py#L624)

The bare Mobilevit Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.MobileViTModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/modeling_mobilevit.py#L656[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MobileViTImageProcessor](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTImageProcessor). See `MobileViTImageProcessor.__call__()` for details (`processor_class` uses
  [MobileViTImageProcessor](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) for processing images).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MobileViTConfig](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state after a pooling operation on the spatial dimensions.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
The [MobileViTModel](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
```

**Parameters:**

config ([MobileViTConfig](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

expand_output (`bool`, *optional*, defaults to `True`) : Whether to expand the output of the model using a 1x1 convolution. If `True`, the model will apply an additional 1x1 convolution to expand the output channels from `config.neck_hidden_sizes[5]` to `config.neck_hidden_sizes[6]`.

**Returns:**

``transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)``

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MobileViTConfig](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state after a pooling operation on the spatial dimensions.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

## MobileViTForImageClassification[[transformers.MobileViTForImageClassification]]

#### transformers.MobileViTForImageClassification[[transformers.MobileViTForImageClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/modeling_mobilevit.py#L706)

MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.MobileViTForImageClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/modeling_mobilevit.py#L722[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MobileViTImageProcessor](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTImageProcessor). See `MobileViTImageProcessor.__call__()` for details (`processor_class` uses
  [MobileViTImageProcessor](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) for processing images).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MobileViTConfig](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
The [MobileViTForImageClassification](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, MobileViTForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
>>> model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

**Parameters:**

config ([MobileViTConfig](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MobileViTConfig](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.

## MobileViTForSemanticSegmentation[[transformers.MobileViTForSemanticSegmentation]]

#### transformers.MobileViTForSemanticSegmentation[[transformers.MobileViTForSemanticSegmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/modeling_mobilevit.py#L876)

MobileViT model with a semantic segmentation head on top, e.g. for Pascal VOC.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.MobileViTForSemanticSegmentation.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilevit/modeling_mobilevit.py#L887[{"name": "pixel_values", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MobileViTImageProcessor](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTImageProcessor). See `MobileViTImageProcessor.__call__()` for details (`processor_class` uses
  [MobileViTImageProcessor](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) for processing images).
- **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) --
  Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.SemanticSegmenterOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.SemanticSegmenterOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MobileViTConfig](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`) -- Classification scores for each pixel.

  

  The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
  to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
  original image size as post-processing. You should always check your logits shape and resize as needed.

  

- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [MobileViTForSemanticSegmentation](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> import requests
>>> import torch
>>> from PIL import Image
>>> from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
>>> model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # logits are of shape (batch_size, num_labels, height, width)
>>> logits = outputs.logits
```

**Parameters:**

config ([MobileViTConfig](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.SemanticSegmenterOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.SemanticSegmenterOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MobileViTConfig](/docs/transformers/main/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`) -- Classification scores for each pixel.

  

  The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
  to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
  original image size as post-processing. You should always check your logits shape and resize as needed.

  

- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
