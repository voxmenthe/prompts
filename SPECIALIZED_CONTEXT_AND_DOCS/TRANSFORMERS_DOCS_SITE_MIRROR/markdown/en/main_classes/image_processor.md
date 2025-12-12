# Image Processor

An image processor is in charge of loading images (optionally), preparing input features for vision models and post processing their outputs. This includes transformations such as resizing, normalization, and conversion to PyTorch and Numpy tensors. It may also include model specific post-processing such as converting logits to segmentation masks.
Fast image processors are available for a few models and more will be added in the future. They are based on the [torchvision](https://pytorch.org/vision/stable/index.html) library and provide a significant speed-up, especially when processing on GPU.
They have the same API as the base image processors and can be used as drop-in replacements.
To use a fast image processor, you need to install the `torchvision` library, and set the `use_fast` argument to `True` when instantiating the image processor:

```python
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)
```

Note that `use_fast` will be set to `True` by default in a future release.

When using a fast image processor, you can also set the `device` argument to specify the device on which the processing should be done. By default, the processing is done on the same device as the inputs if the inputs are tensors, or on the CPU otherwise.

```python
from torchvision.io import read_image
from transformers import DetrImageProcessorFast

images = read_image("image.jpg")
processor = DetrImageProcessorFast.from_pretrained("facebook/detr-resnet-50")
images_processed = processor(images, return_tensors="pt", device="cuda")
```

Here are some speed comparisons between the base and fast image processors for the `DETR` and `RT-DETR` models, and how they impact overall inference time:

  

  

  

  

These benchmarks were run on an [AWS EC2 g5.2xlarge instance](https://aws.amazon.com/ec2/instance-types/g5/), utilizing an NVIDIA A10G Tensor Core GPU.

## ImageProcessingMixin[[transformers.ImageProcessingMixin]]

#### transformers.ImageProcessingMixin[[transformers.ImageProcessingMixin]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L61)

This is an image processor mixin used to provide saving/loading functionality for sequential and image feature
extractors.

from_pretrainedtransformers.ImageProcessingMixin.from_pretrainedhttps://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L88[{"name": "pretrained_model_name_or_path", "val": ": typing.Union[str, os.PathLike]"}, {"name": "cache_dir", "val": ": typing.Union[str, os.PathLike, NoneType] = None"}, {"name": "force_download", "val": ": bool = False"}, {"name": "local_files_only", "val": ": bool = False"}, {"name": "token", "val": ": typing.Union[str, bool, NoneType] = None"}, {"name": "revision", "val": ": str = 'main'"}, {"name": "**kwargs", "val": ""}]- **pretrained_model_name_or_path** (`str` or `os.PathLike`) --
  This can be either:

  - a string, the *model id* of a pretrained image_processor hosted inside a model repo on
    huggingface.co.
  - a path to a *directory* containing a image processor file saved using the
    [save_pretrained()](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin.save_pretrained) method, e.g.,
    `./my_model_directory/`.
  - a path or url to a saved image processor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
- **cache_dir** (`str` or `os.PathLike`, *optional*) --
  Path to a directory in which a downloaded pretrained model image processor should be cached if the
  standard cache should not be used.
- **force_download** (`bool`, *optional*, defaults to `False`) --
  Whether or not to force to (re-)download the image processor files and override the cached versions if
  they exist.
- **proxies** (`dict[str, str]`, *optional*) --
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
  'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
- **token** (`str` or `bool`, *optional*) --
  The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
  the token generated when running `hf auth login` (stored in `~/.huggingface`).
- **revision** (`str`, *optional*, defaults to `"main"`) --
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.

  

  To test a pull request you made on the Hub, you can pass `revision="refs/pr/"`.

  

- **return_unused_kwargs** (`bool`, *optional*, defaults to `False`) --
  If `False`, then this function returns just the final image processor object. If `True`, then this
  functions returns a `Tuple(image_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
  consisting of the key/value pairs whose keys are not image processor attributes: i.e., the part of
  `kwargs` which has not been used to update `image_processor` and is otherwise ignored.
- **subfolder** (`str`, *optional*, defaults to `""`) --
  In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
  specify the folder name here.
- **kwargs** (`dict[str, Any]`, *optional*) --
  The values in kwargs of any keys which are image processor attributes will be used to override the
  loaded values. Behavior concerning key/value pairs whose keys are *not* image processor attributes is
  controlled by the `return_unused_kwargs` keyword parameter.0A image processor of type [ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin).

Instantiate a type of [ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin) from an image processor.

Examples:

```python
# We can't instantiate directly the base class *ImageProcessingMixin* so let's show the examples on a
# derived class: *CLIPImageProcessor*
image_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)  # Download image_processing_config from huggingface.co and cache.
image_processor = CLIPImageProcessor.from_pretrained(
    "./test/saved_model/"
)  # E.g. image processor (or model) was saved using *save_pretrained('./test/saved_model/')*
image_processor = CLIPImageProcessor.from_pretrained("./test/saved_model/preprocessor_config.json")
image_processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", do_normalize=False, foo=False
)
assert image_processor.do_normalize is False
image_processor, unused_kwargs = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", do_normalize=False, foo=False, return_unused_kwargs=True
)
assert image_processor.do_normalize is False
assert unused_kwargs == {"foo": False}
```

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : This can be either:  - a string, the *model id* of a pretrained image_processor hosted inside a model repo on huggingface.co. - a path to a *directory* containing a image processor file saved using the [save_pretrained()](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin.save_pretrained) method, e.g., `./my_model_directory/`. - a path or url to a saved image processor JSON *file*, e.g., `./my_model_directory/preprocessor_config.json`.

cache_dir (`str` or `os.PathLike`, *optional*) : Path to a directory in which a downloaded pretrained model image processor should be cached if the standard cache should not be used.

force_download (`bool`, *optional*, defaults to `False`) : Whether or not to force to (re-)download the image processor files and override the cached versions if they exist.

proxies (`dict[str, str]`, *optional*) : A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.

token (`str` or `bool`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*, defaults to `"main"`) : The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.     To test a pull request you made on the Hub, you can pass `revision="refs/pr/"`.   

return_unused_kwargs (`bool`, *optional*, defaults to `False`) : If `False`, then this function returns just the final image processor object. If `True`, then this functions returns a `Tuple(image_processor, unused_kwargs)` where *unused_kwargs* is a dictionary consisting of the key/value pairs whose keys are not image processor attributes: i.e., the part of `kwargs` which has not been used to update `image_processor` and is otherwise ignored.

subfolder (`str`, *optional*, defaults to `""`) : In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can specify the folder name here.

kwargs (`dict[str, Any]`, *optional*) : The values in kwargs of any keys which are image processor attributes will be used to override the loaded values. Behavior concerning key/value pairs whose keys are *not* image processor attributes is controlled by the `return_unused_kwargs` keyword parameter.

**Returns:**

A image processor of type [ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin).
#### save_pretrained[[transformers.ImageProcessingMixin.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L187)

Save an image processor object to the directory `save_directory`, so that it can be re-loaded using the
[from_pretrained()](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin.from_pretrained) class method.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory where the image processor JSON file will be saved (will be created if it does not exist).

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

## BatchFeature[[transformers.BatchFeature]]

#### transformers.BatchFeature[[transformers.BatchFeature]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/feature_extraction_utils.py#L57)

Holds the output of the [pad()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad) and feature extractor specific `__call__` methods.

This class is derived from a python dictionary and can be used as a dictionary.

convert_to_tensorstransformers.BatchFeature.convert_to_tensorshttps://github.com/huggingface/transformers/blob/main/src/transformers/feature_extraction_utils.py#L141[{"name": "tensor_type", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}]- **tensor_type** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  The type of tensors to use. If `str`, should be one of the values of the enum [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType). If
  `None`, no modification is done.0

Convert the inner content to tensors.

**Parameters:**

data (`dict`, *optional*) : Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask', etc.).

tensor_type (`Union[None, str, TensorType]`, *optional*) : You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at initialization.
#### to[[transformers.BatchFeature.to]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/feature_extraction_utils.py#L172)

Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
different `dtypes` and sending the `BatchFeature` to a different `device`.

**Parameters:**

args (`Tuple`) : Will be passed to the `to(...)` function of the tensors.

kwargs (`Dict`, *optional*) : Will be passed to the `to(...)` function of the tensors. To enable asynchronous data transfer, set the `non_blocking` flag in `kwargs` (defaults to `False`).

**Returns:**

`[BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature)`

The same instance after modification.

## BaseImageProcessor[[transformers.BaseImageProcessor]]

#### transformers.BaseImageProcessor[[transformers.BaseImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils.py#L39)

center_croptransformers.BaseImageProcessor.center_crophttps://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils.py#L128[{"name": "image", "val": ": ndarray"}, {"name": "size", "val": ": dict"}, {"name": "data_format", "val": ": typing.Union[transformers.image_utils.ChannelDimension, str, NoneType] = None"}, {"name": "input_data_format", "val": ": typing.Union[transformers.image_utils.ChannelDimension, str, NoneType] = None"}, {"name": "**kwargs", "val": ""}]- **image** (`np.ndarray`) --
  Image to center crop.
- **size** (`dict[str, int]`) --
  Size of the output image.
- **data_format** (`str` or `ChannelDimension`, *optional*) --
  The channel dimension format for the output image. If unset, the channel dimension format of the input
  image is used. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.0

Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
any edge, the image is padded with 0's and then center cropped.

**Parameters:**

image (`np.ndarray`) : Image to center crop.

size (`dict[str, int]`) : Size of the output image.

data_format (`str` or `ChannelDimension`, *optional*) : The channel dimension format for the output image. If unset, the channel dimension format of the input image is used. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

input_data_format (`ChannelDimension` or `str`, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
#### normalize[[transformers.BaseImageProcessor.normalize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils.py#L91)

Normalize an image. image = (image - image_mean) / image_std.

**Parameters:**

image (`np.ndarray`) : Image to normalize.

mean (`float` or `Iterable[float]`) : Image mean to use for normalization.

std (`float` or `Iterable[float]`) : Image standard deviation to use for normalization.

data_format (`str` or `ChannelDimension`, *optional*) : The channel dimension format for the output image. If unset, the channel dimension format of the input image is used. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

input_data_format (`ChannelDimension` or `str`, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

**Returns:**

``np.ndarray``

The normalized image.
#### rescale[[transformers.BaseImageProcessor.rescale]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils.py#L59)

Rescale an image by a scale factor. image = image * scale.

**Parameters:**

image (`np.ndarray`) : Image to rescale.

scale (`float`) : The scaling factor to rescale pixel values by.

data_format (`str` or `ChannelDimension`, *optional*) : The channel dimension format for the output image. If unset, the channel dimension format of the input image is used. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

input_data_format (`ChannelDimension` or `str`, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

**Returns:**

``np.ndarray``

The rescaled image.

## BaseImageProcessorFast[[transformers.BaseImageProcessorFast]]

#### transformers.BaseImageProcessorFast[[transformers.BaseImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L168)

center_croptransformers.BaseImageProcessorFast.center_crophttps://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L586[{"name": "image", "val": ": torch.Tensor"}, {"name": "size", "val": ": SizeDict"}, {"name": "**kwargs", "val": ""}]- **image** (`"torch.Tensor"`) --
  Image to center crop.
- **size** (`dict[str, int]`) --
  Size of the output image.0`torch.Tensor`The center cropped image.

Note: override torchvision's center_crop to have the same behavior as the slow processor.
Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
any edge, the image is padded with 0's and then center cropped.

**Parameters:**

image (`"torch.Tensor"`) : Image to center crop.

size (`dict[str, int]`) : Size of the output image.

**Returns:**

``torch.Tensor``

The center cropped image.
#### compile_friendly_resize[[transformers.BaseImageProcessorFast.compile_friendly_resize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L473)

A wrapper around `F.resize` so that it is compatible with torch.compile when the image is a uint8 tensor.
#### convert_to_rgb[[transformers.BaseImageProcessorFast.convert_to_rgb]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L627)

Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
as is.

**Parameters:**

image (ImageInput) : The image to convert.

**Returns:**

`ImageInput`

The converted image.
#### filter_out_unused_kwargs[[transformers.BaseImageProcessorFast.filter_out_unused_kwargs]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L643)

Filter out the unused kwargs from the kwargs dictionary.
#### normalize[[transformers.BaseImageProcessorFast.normalize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L518)

Normalize an image. image = (image - image_mean) / image_std.

**Parameters:**

image (`torch.Tensor`) : Image to normalize.

mean (`torch.Tensor`, `float` or `Iterable[float]`) : Image mean to use for normalization.

std (`torch.Tensor`, `float` or `Iterable[float]`) : Image standard deviation to use for normalization.

**Returns:**

``torch.Tensor``

The normalized image.
#### pad[[transformers.BaseImageProcessorFast.pad]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L346)

Pads images to `(pad_size["height"], pad_size["width"])` or to the largest size in the batch.

**Parameters:**

images (`list[torch.Tensor]`) : Images to pad.

pad_size (`SizeDict`, *optional*) : Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.

fill_value (`int`, *optional*, defaults to `0`) : The constant value used to fill the padded area.

padding_mode (`str`, *optional*, defaults to "constant") : The padding mode to use. Can be any of the modes supported by `torch.nn.functional.pad` (e.g. constant, reflection, replication).

return_mask (`bool`, *optional*, defaults to `False`) : Whether to return a pixel mask to denote padded regions.

disable_grouping (`bool`, *optional*, defaults to `False`) : Whether to disable grouping of images by size.

**Returns:**

``Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]``

The padded images and pixel masks if `return_mask` is `True`.
#### preprocess[[transformers.BaseImageProcessorFast.preprocess]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L839)

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
#### rescale[[transformers.BaseImageProcessorFast.rescale]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L498)

Rescale an image by a scale factor. image = image * scale.

**Parameters:**

image (`torch.Tensor`) : Image to rescale.

scale (`float`) : The scaling factor to rescale pixel values by.

**Returns:**

``torch.Tensor``

The rescaled image.
#### rescale_and_normalize[[transformers.BaseImageProcessorFast.rescale_and_normalize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L558)

Rescale and normalize images.
#### resize[[transformers.BaseImageProcessorFast.resize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils_fast.py#L417)

Resize an image to `(size["height"], size["width"])`.

**Parameters:**

image (`torch.Tensor`) : Image to resize.

size (`SizeDict`) : Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.

interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`) : `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

antialias (`bool`, *optional*, defaults to `True`) : Whether to use antialiasing.

**Returns:**

``torch.Tensor``

The resized image.
