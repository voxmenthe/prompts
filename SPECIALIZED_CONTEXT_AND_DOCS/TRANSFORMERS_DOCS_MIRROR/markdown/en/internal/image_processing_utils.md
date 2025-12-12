# Utilities for Image Processors

This page lists all the utility functions used by the image processors, mainly the functional
transformations used to process the images.

Most of those are only useful if you are studying the code of the image processors in the library.

## Image Transformations

#### transformers.image\_transforms.center\_crop

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_transforms.py#L455)

( image: ndarray size: tuple data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None input\_data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  The image to crop.
* **size** (`tuple[int, int]`) —
  The target size for the cropped image.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    If unset, will use the inferred format of the input image.
* **input\_data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    If unset, will use the inferred format of the input image.

Returns

`np.ndarray`

The cropped image.

Crops the `image` to the specified `size` using a center crop. Note that if the image is too small to be cropped to
the size given, it will be padded (so the returned result will always be of size `size`).

#### transformers.image\_transforms.center\_to\_corners\_format

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_transforms.py#L570)

( bboxes\_center: TensorType  )

Converts bounding boxes from center format to corners format.

center format: contains the coordinate for the center of the box and its width, height dimensions
(center\_x, center\_y, width, height)
corners format: contains the coordinates for the top-left and bottom-right corners of the box
(top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y)

#### transformers.image\_transforms.corners\_to\_center\_format

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_transforms.py#L630)

( bboxes\_corners: TensorType  )

Converts bounding boxes from corners format to center format.

corners format: contains the coordinates for the top-left and bottom-right corners of the box
(top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y)
center format: contains the coordinate for the center of the box and its the width, height dimensions
(center\_x, center\_y, width, height)

#### transformers.image\_transforms.id\_to\_rgb

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_transforms.py#L664)

( id\_map  )

Converts unique ID to RGB color.

#### transformers.image\_transforms.normalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_transforms.py#L394)

( image: ndarray mean: typing.Union[float, collections.abc.Collection[float]] std: typing.Union[float, collections.abc.Collection[float]] data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = None input\_data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None  )

Parameters

* **image** (`np.ndarray`) —
  The image to normalize.
* **mean** (`float` or `Collection[float]`) —
  The mean to use for normalization.
* **std** (`float` or `Collection[float]`) —
  The standard deviation to use for normalization.
* **data\_format** (`ChannelDimension`, *optional*) —
  The channel dimension format of the output image. If unset, will use the inferred format from the input.
* **input\_data\_format** (`ChannelDimension`, *optional*) —
  The channel dimension format of the input image. If unset, will use the inferred format from the input.

Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

image = (image - mean) / std

#### transformers.image\_transforms.pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_transforms.py#L694)

( image: ndarray padding: typing.Union[int, tuple[int, int], collections.abc.Iterable[tuple[int, int]]] mode: PaddingMode = <PaddingMode.CONSTANT: 'constant'> constant\_values: typing.Union[float, collections.abc.Iterable[float]] = 0.0 data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None input\_data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  The image to pad.
* **padding** (`int` or `tuple[int, int]` or `Iterable[tuple[int, int]]`) —
  Padding to apply to the edges of the height, width axes. Can be one of three formats:
  + `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
  + `((before, after),)` yields same before and after pad for height and width.
  + `(pad,)` or int is a shortcut for before = after = pad width for all axes.
* **mode** (`PaddingMode`) —
  The padding mode to use. Can be one of:
  + `"constant"`: pads with a constant value.
  + `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
    vector along each axis.
  + `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
  + `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
* **constant\_values** (`float` or `Iterable[float]`, *optional*) —
  The value to use for the padding if `mode` is `"constant"`.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    If unset, will use same as the input image.
* **input\_data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format for the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    If unset, will use the inferred format of the input image.

Returns

`np.ndarray`

The padded image.

Pads the `image` with the specified (height, width) `padding` and `mode`.

#### transformers.image\_transforms.rgb\_to\_id

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_transforms.py#L653)

( color  )

Converts RGB color to unique ID.

#### transformers.image\_transforms.rescale

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_transforms.py#L97)

( image: ndarray scale: float data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = None dtype: dtype = <class 'numpy.float32'> input\_data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  The image to rescale.
* **scale** (`float`) —
  The scale to use for rescaling the image.
* **data\_format** (`ChannelDimension`, *optional*) —
  The channel dimension format of the image. If not provided, it will be the same as the input image.
* **dtype** (`np.dtype`, *optional*, defaults to `np.float32`) —
  The dtype of the output image. Defaults to `np.float32`. Used for backwards compatibility with feature
  extractors.
* **input\_data\_format** (`ChannelDimension`, *optional*) —
  The channel dimension format of the input image. If not provided, it will be inferred from the input image.

Returns

`np.ndarray`

The rescaled image.

Rescales `image` by `scale`.

#### transformers.image\_transforms.resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_transforms.py#L323)

( image: ndarray size: tuple resample: PILImageResampling = None reducing\_gap: typing.Optional[int] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = None return\_numpy: bool = True input\_data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  The image to resize.
* **size** (`tuple[int, int]`) —
  The size to use for resizing the image.
* **resample** (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`) —
  The filter to user for resampling.
* **reducing\_gap** (`int`, *optional*) —
  Apply optimization by resizing the image in two steps. The bigger `reducing_gap`, the closer the result to
  the fair resampling. See corresponding Pillow documentation for more details.
* **data\_format** (`ChannelDimension`, *optional*) —
  The channel dimension format of the output image. If unset, will use the inferred format from the input.
* **return\_numpy** (`bool`, *optional*, defaults to `True`) —
  Whether or not to return the resized image as a numpy array. If False a `PIL.Image.Image` object is
  returned.
* **input\_data\_format** (`ChannelDimension`, *optional*) —
  The channel dimension format of the input image. If unset, will use the inferred format from the input.

Returns

`np.ndarray`

The resized image.

Resizes `image` to `(height, width)` specified by `size` using the PIL library.

#### transformers.image\_transforms.to\_pil\_image

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_transforms.py#L162)

( image: typing.Union[numpy.ndarray, ForwardRef('PIL.Image.Image'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor'), ForwardRef('jnp.ndarray')] do\_rescale: typing.Optional[bool] = None image\_mode: typing.Optional[str] = None input\_data\_format: typing.Union[transformers.image\_utils.ChannelDimension, str, NoneType] = None  ) → `PIL.Image.Image`

Parameters

* **image** (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor` or `tf.Tensor`) —
  The image to convert to the `PIL.Image` format.
* **do\_rescale** (`bool`, *optional*) —
  Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will default
  to `True` if the image type is a floating type and casting to `int` would result in a loss of precision,
  and `False` otherwise.
* **image\_mode** (`str`, *optional*) —
  The mode to use for the PIL image. If unset, will use the default mode for the input image type.
* **input\_data\_format** (`ChannelDimension`, *optional*) —
  The channel dimension format of the input image. If unset, will use the inferred format from the input.

Returns

`PIL.Image.Image`

The converted image.

Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
needed.

## ImageProcessingMixin

### class transformers.ImageProcessingMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L64)

( \*\*kwargs  )

This is an image processor mixin used to provide saving/loading functionality for sequential and image feature
extractors.

#### fetch\_images

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L522)

( image\_url\_or\_urls: typing.Union[str, list[str], list[list[str]]]  )

Convert a single or a list of urls into the corresponding `PIL.Image` objects.

If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
returned.

#### from\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L389)

( image\_processor\_dict: dict \*\*kwargs  ) → [ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)

Parameters

* **image\_processor\_dict** (`dict[str, Any]`) —
  Dictionary that will be used to instantiate the image processor object. Such a dictionary can be
  retrieved from a pretrained checkpoint by leveraging the
  [to\_dict()](/docs/transformers/v4.56.2/en/internal/image_processing_utils#transformers.ImageProcessingMixin.to_dict) method.
* **kwargs** (`dict[str, Any]`) —
  Additional parameters from which to initialize the image processor object.

Returns

[ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)

The image processor object instantiated from those
parameters.

Instantiates a type of [ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin) from a Python dictionary of parameters.

#### from\_json\_file

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L446)

( json\_file: typing.Union[str, os.PathLike]  ) → A image processor of type [ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)

Parameters

* **json\_file** (`str` or `os.PathLike`) —
  Path to the JSON file containing the parameters.

Returns

A image processor of type [ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)

The image\_processor object
instantiated from that JSON file.

Instantiates a image processor of type [ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin) from the path to a JSON
file of parameters.

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

#### get\_image\_processor\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L266)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] \*\*kwargs  ) → `tuple[Dict, Dict]`

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
* **subfolder** (`str`, *optional*, defaults to `""`) —
  In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
  specify the folder name here.
* **image\_processor\_filename** (`str`, *optional*, defaults to `"config.json"`) —
  The name of the file in the model directory to use for the image processor config.

Returns

`tuple[Dict, Dict]`

The dictionary(ies) that will be used to instantiate the image processor object.

From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
image processor of type `~image_processor_utils.ImageProcessingMixin` using `from_dict`.

#### push\_to\_hub

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/hub.py#L847)

( repo\_id: str use\_temp\_dir: typing.Optional[bool] = None commit\_message: typing.Optional[str] = None private: typing.Optional[bool] = None token: typing.Union[bool, str, NoneType] = None max\_shard\_size: typing.Union[str, int, NoneType] = '5GB' create\_pr: bool = False safe\_serialization: bool = True revision: typing.Optional[str] = None commit\_description: typing.Optional[str] = None tags: typing.Optional[list[str]] = None \*\*deprecated\_kwargs  )

Parameters

* **repo\_id** (`str`) —
  The name of the repository you want to push your image processor to. It should contain your organization name
  when pushing to a given organization.
* **use\_temp\_dir** (`bool`, *optional*) —
  Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
  Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
* **commit\_message** (`str`, *optional*) —
  Message to commit while pushing. Will default to `"Upload image processor"`.
* **private** (`bool`, *optional*) —
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
* **token** (`bool` or `str`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
  is not specified.
* **max\_shard\_size** (`int` or `str`, *optional*, defaults to `"5GB"`) —
  Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
  will then be each of size lower than this size. If expressed as a string, needs to be digits followed
  by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
  Google Colab instances without any CPU OOM issues.
* **create\_pr** (`bool`, *optional*, defaults to `False`) —
  Whether or not to create a PR with the uploaded files or directly commit.
* **safe\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether or not to convert the model weights in safetensors format for safer serialization.
* **revision** (`str`, *optional*) —
  Branch to push the uploaded files to.
* **commit\_description** (`str`, *optional*) —
  The description of the commit that will be created
* **tags** (`list[str]`, *optional*) —
  List of tags to push on the Hub.

Upload the image processor file to the 🤗 Model Hub.

Examples:


```
from transformers import AutoImageProcessor

image processor = AutoImageProcessor.from_pretrained("google-bert/bert-base-cased")

# Push the image processor to your namespace with the name "my-finetuned-bert".
image processor.push_to_hub("my-finetuned-bert")

# Push the image processor to an organization with the name "my-finetuned-bert".
image processor.push_to_hub("huggingface/my-finetuned-bert")
```

#### register\_for\_auto\_class

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L500)

( auto\_class = 'AutoImageProcessor'  )

Parameters

* **auto\_class** (`str` or `type`, *optional*, defaults to `"AutoImageProcessor "`) —
  The auto class to register this new image processor with.

Register this class with a given auto class. This should only be used for custom image processors as the ones
in the library are already mapped with `AutoImageProcessor` .

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

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L434)

( ) → `dict[str, Any]`

Returns

`dict[str, Any]`

Dictionary of all the attributes that make up this image processor instance.

Serializes this instance to a Python dictionary.

#### to\_json\_file

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L486)

( json\_file\_path: typing.Union[str, os.PathLike]  )

Parameters

* **json\_file\_path** (`str` or `os.PathLike`) —
  Path to the JSON file in which this image\_processor instance’s parameters will be saved.

Save this instance to a JSON file.

#### to\_json\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_base.py#L465)

( ) → `str`

Returns

`str`

String containing all the attributes that make up this feature\_extractor instance in JSON format.

Serializes this instance to a JSON string.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/image_processing_utils.md)
