# PoolFormer

## Overview

The PoolFormer model was proposed in [MetaFormer is Actually What You Need for Vision](https://huggingface.co/papers/2111.11418)  by Sea AI Labs. Instead of designing complicated token mixer to achieve SOTA performance, the target of this work is to demonstrate the competence of transformer models largely stem from the general architecture MetaFormer.

The abstract from the paper is the following:

*Transformers have shown great potential in computer vision tasks. A common belief is their attention-based token mixer module contributes most to their competence. However, recent works show the attention-based module in transformers can be replaced by spatial MLPs and the resulted models still perform quite well. Based on this observation, we hypothesize that the general architecture of the transformers, instead of the specific token mixer module, is more essential to the model's performance. To verify this, we deliberately replace the attention module in transformers with an embarrassingly simple spatial pooling operator to conduct only the most basic token mixing. Surprisingly, we observe that the derived model, termed as PoolFormer, achieves competitive performance on multiple computer vision tasks. For example, on ImageNet-1K, PoolFormer achieves 82.1% top-1 accuracy, surpassing well-tuned vision transformer/MLP-like baselines DeiT-B/ResMLP-B24 by 0.3%/1.1% accuracy with 35%/52% fewer parameters and 48%/60% fewer MACs. The effectiveness of PoolFormer verifies our hypothesis and urges us to initiate the concept of "MetaFormer", a general architecture abstracted from transformers without specifying the token mixer. Based on the extensive experiments, we argue that MetaFormer is the key player in achieving superior results for recent transformer and MLP-like models on vision tasks. This work calls for more future research dedicated to improving MetaFormer instead of focusing on the token mixer modules. Additionally, our proposed PoolFormer could serve as a starting baseline for future MetaFormer architecture design.*

The figure below illustrates the architecture of PoolFormer. Taken from the [original paper](https://huggingface.co/papers/2111.11418).

This model was contributed by [heytanay](https://huggingface.co/heytanay). The original code can be found [here](https://github.com/sail-sg/poolformer).

## Usage tips

- PoolFormer has a hierarchical architecture, where instead of Attention, a simple Average Pooling layer is present. All checkpoints of the model can be found on the [hub](https://huggingface.co/models?other=poolformer).
- One can use [PoolFormerImageProcessor](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerImageProcessor) to prepare images for the model.
- As most models, PoolFormer comes in different sizes, the details of which can be found in the table below.

| **Model variant** | **Depths**    | **Hidden sizes**    | **Params (M)** | **ImageNet-1k Top 1** |
| :---------------: | ------------- | ------------------- | :------------: | :-------------------: |
| s12               | [2, 2, 6, 2]  | [64, 128, 320, 512] | 12             | 77.2                  |
| s24               | [4, 4, 12, 4] | [64, 128, 320, 512] | 21             | 80.3                  |
| s36               | [6, 6, 18, 6] | [64, 128, 320, 512] | 31             | 81.4                  |
| m36               | [6, 6, 18, 6] | [96, 192, 384, 768] | 56             | 82.1                  |
| m48               | [8, 8, 24, 8] | [96, 192, 384, 768] | 73             | 82.5                  |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with PoolFormer.

- [PoolFormerForImageClassification](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## PoolFormerConfig[[transformers.PoolFormerConfig]]

#### transformers.PoolFormerConfig[[transformers.PoolFormerConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/poolformer/configuration_poolformer.py#L24)

This is the configuration class to store the configuration of [PoolFormerModel](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerModel). It is used to instantiate a
PoolFormer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the PoolFormer
[sail/poolformer_s12](https://huggingface.co/sail/poolformer_s12) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import PoolFormerConfig, PoolFormerModel

>>> # Initializing a PoolFormer sail/poolformer_s12 style configuration
>>> configuration = PoolFormerConfig()

>>> # Initializing a model (with random weights) from the sail/poolformer_s12 style configuration
>>> model = PoolFormerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

num_channels (`int`, *optional*, defaults to 3) : The number of channels in the input image.

patch_size (`int`, *optional*, defaults to 16) : The size of the input patch.

stride (`int`, *optional*, defaults to 16) : The stride of the input patch.

pool_size (`int`, *optional*, defaults to 3) : The size of the pooling window.

mlp_ratio (`float`, *optional*, defaults to 4.0) : The ratio of the number of channels in the output of the MLP to the number of channels in the input.

depths (`list`, *optional*, defaults to `[2, 2, 6, 2]`) : The depth of each encoder block.

hidden_sizes (`list`, *optional*, defaults to `[64, 128, 320, 512]`) : The hidden sizes of each encoder block.

patch_sizes (`list`, *optional*, defaults to `[7, 3, 3, 3]`) : The size of the input patch for each encoder block.

strides (`list`, *optional*, defaults to `[4, 2, 2, 2]`) : The stride of the input patch for each encoder block.

padding (`list`, *optional*, defaults to `[2, 1, 1, 1]`) : The padding of the input patch for each encoder block.

num_encoder_blocks (`int`, *optional*, defaults to 4) : The number of encoder blocks.

drop_path_rate (`float`, *optional*, defaults to 0.0) : The dropout rate for the dropout layers.

hidden_act (`str`, *optional*, defaults to `"gelu"`) : The activation function for the hidden layers.

use_layer_scale (`bool`, *optional*, defaults to `True`) : Whether to use layer scale.

layer_scale_init_value (`float`, *optional*, defaults to 1e-05) : The initial value for the layer scale.

initializer_range (`float`, *optional*, defaults to 0.02) : The initializer range for the weights.

## PoolFormerImageProcessor[[transformers.PoolFormerImageProcessor]]

#### transformers.PoolFormerImageProcessor[[transformers.PoolFormerImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/poolformer/image_processing_poolformer.py#L60)

Constructs a PoolFormer image processor.

preprocesstransformers.PoolFormerImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/poolformer/image_processing_poolformer.py#L223[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "crop_pct", "val": ": typing.Optional[int] = None"}, {"name": "resample", "val": ": typing.Optional[PIL.Image.Resampling] = None"}, {"name": "do_center_crop", "val": ": typing.Optional[bool] = None"}, {"name": "crop_size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": ChannelDimension = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}]- **images** (`ImageInput`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the image.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  Size of the image after applying resize.
- **crop_pct** (`float`, *optional*, defaults to `self.crop_pct`) --
  Percentage of the image to crop. Only has an effect if `do_resize` is set to `True`.
- **resample** (`int`, *optional*, defaults to `self.resample`) --
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
  has an effect if `do_resize` is set to `True`.
- **do_center_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) --
  Whether to center crop the image.
- **crop_size** (`dict[str, int]`, *optional*, defaults to `self.crop_size`) --
  Size of the image after applying center crop.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the image values between [0 - 1].
- **rescale_factor** (`float`, *optional*, defaults to `self.rescale_factor`) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether to normalize the image.
- **image_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) --
  Image mean.
- **image_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) --
  Image standard deviation.
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

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by `do_resize` in the `preprocess` method.

size (`dict[str, int]` *optional*, defaults to `{"shortest_edge" : 224}`): Size of the image after resizing. Can be overridden by `size` in the `preprocess` method. If crop_pct is unset: - size is `{"height": h, "width": w}`: the image is resized to `(h, w)`. - size is `{"shortest_edge": s}`: the shortest edge of the image is resized to s whilst maintaining the aspect ratio.  If crop_pct is set: - size is `{"height": h, "width": w}`: the image is resized to `(int(floor(h/crop_pct)), int(floor(w/crop_pct)))` - size is `{"height": c, "width": c}`: the shortest edge of the image is resized to `int(floor(c/crop_pct)` whilst maintaining the aspect ratio. - size is `{"shortest_edge": c}`: the shortest edge of the image is resized to `int(floor(c/crop_pct)` whilst maintaining the aspect ratio.

crop_pct (`float`, *optional*, defaults to 0.9) : Percentage of the image to crop from the center. Can be overridden by `crop_pct` in the `preprocess` method.

resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) : Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.

do_center_crop (`bool`, *optional*, defaults to `True`) : Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image is padded with 0's and then center cropped. Can be overridden by `do_center_crop` in the `preprocess` method.

crop_size (`dict[str, int]`, *optional*, defaults to `{"height" : 224, "width": 224}`): Size of the image after applying center crop. Only has an effect if `do_center_crop` is set to `True`. Can be overridden by the `crop_size` parameter in the `preprocess` method.

rescale_factor (`int` or `float`, *optional*, defaults to `1/255`) : Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.

do_normalize (`bool`, *optional*, defaults to `True`) : Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method.

image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) : Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

## PoolFormerImageProcessorFast[[transformers.PoolFormerImageProcessorFast]]

#### transformers.PoolFormerImageProcessorFast[[transformers.PoolFormerImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/poolformer/image_processing_poolformer_fast.py#L47)

Constructs a fast Poolformer image processor.

preprocesstransformers.PoolFormerImageProcessorFast.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/poolformer/image_processing_poolformer_fast.py#L64[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.poolformer.image_processing_poolformer.PoolFormerImageProcessorKwargs]"}]- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) --
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
  Added for backward compatibility but this should be set as a processor attribute in future models.
- **crop_pct** (`float`, *optional*, defaults to `self.crop_pct`) --
  Percentage of the image to crop. Only has an effect if `do_resize` is set to `True`.0``- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
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

crop_pct (`float`, *optional*, defaults to `self.crop_pct`) : Percentage of the image to crop. Only has an effect if `do_resize` is set to `True`.

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## PoolFormerModel[[transformers.PoolFormerModel]]

#### transformers.PoolFormerModel[[transformers.PoolFormerModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/poolformer/modeling_poolformer.py#L260)

The bare Poolformer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.PoolFormerModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/poolformer/modeling_poolformer.py#L273[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PoolFormerImageProcessor](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerImageProcessor). See [PoolFormerImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [PoolFormerImageProcessor](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerImageProcessor) for processing images).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.modeling_outputs.BaseModelOutputWithNoAttention` or `tuple(torch.FloatTensor)`A `transformers.modeling_outputs.BaseModelOutputWithNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PoolFormerConfig](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
The [PoolFormerModel](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
```

**Parameters:**

config ([PoolFormerModel](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.modeling_outputs.BaseModelOutputWithNoAttention` or `tuple(torch.FloatTensor)``

A `transformers.modeling_outputs.BaseModelOutputWithNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PoolFormerConfig](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

## PoolFormerForImageClassification[[transformers.PoolFormerForImageClassification]]

#### transformers.PoolFormerForImageClassification[[transformers.PoolFormerForImageClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/poolformer/modeling_poolformer.py#L320)

PoolFormer Model transformer with an image classification head on top

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.PoolFormerForImageClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/poolformer/modeling_poolformer.py#L336[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [PoolFormerImageProcessor](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerImageProcessor). See [PoolFormerImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [PoolFormerImageProcessor](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerImageProcessor) for processing images).
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PoolFormerConfig](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
The [PoolFormerForImageClassification](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoImageProcessor, PoolFormerForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("sail/poolformer_s12")
>>> model = PoolFormerForImageClassification.from_pretrained("sail/poolformer_s12")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

**Parameters:**

config ([PoolFormerForImageClassification](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerForImageClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PoolFormerConfig](/docs/transformers/main/en/model_doc/poolformer#transformers.PoolFormerConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
  called feature maps) of the model at the output of each stage.
