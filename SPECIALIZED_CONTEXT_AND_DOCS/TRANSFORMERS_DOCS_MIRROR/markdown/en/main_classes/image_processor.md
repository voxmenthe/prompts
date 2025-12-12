# Image Processor

An image processor is in charge of loading images (optionally), preparing input features for vision models and post processing their outputs. This includes transformations such as resizing, normalization, and conversion to PyTorch and Numpy tensors. It may also include model specific post-processing such as converting logits to segmentation masks.
Fast image processors are available for a few models and more will be added in the future. They are based on the [torchvision](https://pytorch.org/vision/stable/index.html) library and provide a significant speed-up, especially when processing on GPU.
They have the same API as the base image processors and can be used as drop-in replacements.
To use a fast image processor, you need to install the `torchvision` library, and set the `use_fast` argument to `True` when instantiating the image processor:


```
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)
```

Note that `use_fast` will be set to `True` by default in a future release.

When using a fast image processor, you can also set the `device` argument to specify the device on which the processing should be done. By default, the processing is done on the same device as the inputs if the inputs are tensors, or on the CPU otherwise.


```
from torchvision.io import read_image
from transformers import DetrImageProcessorFast

images = read_image("image.jpg")
processor = DetrImageProcessorFast.from_pretrained("facebook/detr-resnet-50")
images_processed = processor(images, return_tensors="pt", device="cuda")
```

Here are some speed comparisons between the base and fast image processors for the `DETR` and `RT-DETR` models, and how they impact overall inference time:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_padded.png)

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_batched_compiled.png)

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_single.png)

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_batched.png)

These benchmarks were run on an [AWS EC2 g5.2xlarge instance](https://aws.amazon.com/ec2/instance-types/g5/), utilizing an NVIDIA A10G Tensor Core GPU.

## ImageProcessingMixin

### class transformers.ImageProcessingMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L64)

( \*\*kwargs  )

This is an image processor mixin used to provide saving/loading functionality for sequential and image feature
extractors.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L91)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[str, bool, NoneType] = None revision: str = 'main' \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained image\_processor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a image processor file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin.save_pretrained) method, e.g.,
    `./my_model_directory/`.
  + a path or url to a saved image processor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model image processor should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force to (re-)download the image processor files and override the cached versions if
  they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
* **token** (`str` or `bool`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
  the token generated when running `hf auth login` (stored in `~/.huggingface`).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.

Instantiate a type of [ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin) from an image processor.

Examples:


```
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

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L205)

( save\_directory: typing.Union[str, os.PathLike] push\_to\_hub: bool = False \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the image processor JSON file will be saved (will be created if it does not exist).
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Save an image processor object to the directory `save_directory`, so that it can be re-loaded using the
[from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin.from_pretrained) class method.

## BatchFeature

### class transformers.BatchFeature

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/feature_extraction_utils.py#L63)

( data: typing.Optional[dict[str, typing.Any]] = None tensor\_type: typing.Union[NoneType, str, transformers.utils.generic.TensorType] = None  )

Parameters

* **data** (`dict`, *optional*) —
  Dictionary of lists/arrays/tensors returned by the **call**/pad methods (‘input\_values’, ‘attention\_mask’,
  etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) —
  You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

Holds the output of the [pad()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad) and feature extractor specific `__call__` methods.

This class is derived from a python dictionary and can be used as a dictionary.

#### convert\_to\_tensors

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/feature_extraction_utils.py#L172)

( tensor\_type: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None  )

Parameters

* **tensor\_type** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  The type of tensors to use. If `str`, should be one of the values of the enum [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType). If
  `None`, no modification is done.

Convert the inner content to tensors.

#### to

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/feature_extraction_utils.py#L203)

( \*args \*\*kwargs  ) → [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

Parameters

* **args** (`Tuple`) —
  Will be passed to the `to(...)` function of the tensors.
* **kwargs** (`Dict`, *optional*) —
  Will be passed to the `to(...)` function of the tensors.
  To enable asynchronous data transfer, set the `non_blocking` flag in `kwargs` (defaults to `False`).

Returns

[BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

The same instance after modification.

Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
different `dtypes` and sending the `BatchFeature` to a different `device`.

## BaseImageProcessor

### class transformers.BaseImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L38)

( \*\*kwargs  )

#### center\_crop

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L125)

( image: ndarray size: dict data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None input\_data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None \*\*kwargs  )

Parameters

* **image** (`np.ndarray`) —
  Image to center crop.
* **size** (`dict[str, int]`) —
  Size of the output image.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the output image. If unset, the channel dimension format of the input
  image is used. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.

Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
any edge, the image is padded with 0’s and then center cropped.

#### normalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L88)

( image: ndarray mean: typing.Union[float, collections.abc.Iterable[float]] std: typing.Union[float, collections.abc.Iterable[float]] data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None input\_data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None \*\*kwargs  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  Image to normalize.
* **mean** (`float` or `Iterable[float]`) —
  Image mean to use for normalization.
* **std** (`float` or `Iterable[float]`) —
  Image standard deviation to use for normalization.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
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

The normalized image.

Normalize an image. image = (image - image\_mean) / image\_std.

#### rescale

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L56)

( image: ndarray scale: float data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None input\_data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None \*\*kwargs  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  Image to rescale.
* **scale** (`float`) —
  The scaling factor to rescale pixel values by.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
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

The rescaled image.

Rescale an image by a scale factor. image = image \* scale.

## BaseImageProcessorFast

### class transformers.BaseImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L193)

( \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  )

#### center\_crop

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L405)

( image: torch.Tensor size: dict \*\*kwargs  ) → `torch.Tensor`

Parameters

* **image** (`"torch.Tensor"`) —
  Image to center crop.
* **size** (`dict[str, int]`) —
  Size of the output image.

Returns

`torch.Tensor`

The center cropped image.

Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
any edge, the image is padded with 0’s and then center cropped.

#### compile\_friendly\_resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L296)

( image: torch.Tensor new\_size: tuple interpolation: typing.Optional[ForwardRef('F.InterpolationMode')] = None antialias: bool = True  )

A wrapper around `F.resize` so that it is compatible with torch.compile when the image is a uint8 tensor.

#### convert\_to\_rgb

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L428)

( image: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]  ) → ImageInput

Parameters

* **image** (ImageInput) —
  The image to convert.

Returns

ImageInput

The converted image.

Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
as is.

#### filter\_out\_unused\_kwargs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L444)

( kwargs: dict  )

Filter out the unused kwargs from the kwargs dictionary.

#### normalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L337)

( image: torch.Tensor mean: typing.Union[float, collections.abc.Iterable[float]] std: typing.Union[float, collections.abc.Iterable[float]] \*\*kwargs  ) → `torch.Tensor`

Parameters

* **image** (`torch.Tensor`) —
  Image to normalize.
* **mean** (`torch.Tensor`, `float` or `Iterable[float]`) —
  Image mean to use for normalization.
* **std** (`torch.Tensor`, `float` or `Iterable[float]`) —
  Image standard deviation to use for normalization.

Returns

`torch.Tensor`

The normalized image.

Normalize an image. image = (image - image\_mean) / image\_std.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L639)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*args \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
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

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

#### rescale

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L317)

( image: torch.Tensor scale: float \*\*kwargs  ) → `torch.Tensor`

Parameters

* **image** (`torch.Tensor`) —
  Image to rescale.
* **scale** (`float`) —
  The scaling factor to rescale pixel values by.

Returns

`torch.Tensor`

The rescaled image.

Rescale an image by a scale factor. image = image \* scale.

#### rescale\_and\_normalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L377)

( images: torch.Tensor do\_rescale: bool rescale\_factor: float do\_normalize: bool image\_mean: typing.Union[float, list[float]] image\_std: typing.Union[float, list[float]]  )

Rescale and normalize images.

#### resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L242)

( image: torch.Tensor size: SizeDict interpolation: F.InterpolationMode = None antialias: bool = True \*\*kwargs  ) → `torch.Tensor`

Parameters

* **image** (`torch.Tensor`) —
  Image to resize.
* **size** (`SizeDict`) —
  Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
* **interpolation** (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`) —
  `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

Returns

`torch.Tensor`

The resized image.

Resize an image to `(size["height"], size["width"])`.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/image_processor.md)
