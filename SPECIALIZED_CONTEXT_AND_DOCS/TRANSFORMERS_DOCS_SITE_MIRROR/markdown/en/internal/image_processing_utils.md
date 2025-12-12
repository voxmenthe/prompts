# Utilities for Image Processors

This page lists all the utility functions used by the image processors, mainly the functional
transformations used to process the images.

Most of those are only useful if you are studying the code of the image processors in the library.

## Image Transformations[[transformers.image_transforms.center_crop]]

#### transformers.image_transforms.center_crop[[transformers.image_transforms.center_crop]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L445)

Crops the `image` to the specified `size` using a center crop. Note that if the image is too small to be cropped to
the size given, it will be padded (so the returned result will always be of size `size`).

**Parameters:**

image (`np.ndarray`) : The image to crop.

size (`tuple[int, int]`) : The target size for the cropped image.

data_format (`str` or `ChannelDimension`, *optional*) : The channel dimension format for the output image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. If unset, will use the inferred format of the input image.

input_data_format (`str` or `ChannelDimension`, *optional*) : The channel dimension format for the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. If unset, will use the inferred format of the input image.

**Returns:**

``np.ndarray``

The cropped image.

#### transformers.image_transforms.center_to_corners_format[[transformers.image_transforms.center_to_corners_format]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L550)

Converts bounding boxes from center format to corners format.

center format: contains the coordinate for the center of the box and its width, height dimensions
(center_x, center_y, width, height)
corners format: contains the coordinates for the top-left and bottom-right corners of the box
(top_left_x, top_left_y, bottom_right_x, bottom_right_y)

#### transformers.image_transforms.corners_to_center_format[[transformers.image_transforms.corners_to_center_format]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L593)

Converts bounding boxes from corners format to center format.

corners format: contains the coordinates for the top-left and bottom-right corners of the box
(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
center format: contains the coordinate for the center of the box and its the width, height dimensions
(center_x, center_y, width, height)

#### transformers.image_transforms.id_to_rgb[[transformers.image_transforms.id_to_rgb]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L625)

Converts unique ID to RGB color.

#### transformers.image_transforms.normalize[[transformers.image_transforms.normalize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L384)

Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

image = (image - mean) / std

**Parameters:**

image (`np.ndarray`) : The image to normalize.

mean (`float` or `Collection[float]`) : The mean to use for normalization.

std (`float` or `Collection[float]`) : The standard deviation to use for normalization.

data_format (`ChannelDimension`, *optional*) : The channel dimension format of the output image. If unset, will use the inferred format from the input.

input_data_format (`ChannelDimension`, *optional*) : The channel dimension format of the input image. If unset, will use the inferred format from the input.

#### transformers.image_transforms.pad[[transformers.image_transforms.pad]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L655)

Pads the `image` with the specified (height, width) `padding` and `mode`.

**Parameters:**

image (`np.ndarray`) : The image to pad.

padding (`int` or `tuple[int, int]` or `Iterable[tuple[int, int]]`) : Padding to apply to the edges of the height, width axes. Can be one of three formats: - `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis. - `((before, after),)` yields same before and after pad for height and width. - `(pad,)` or int is a shortcut for before = after = pad width for all axes.

mode (`PaddingMode`) : The padding mode to use. Can be one of: - `"constant"`: pads with a constant value. - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the vector along each axis. - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis. - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.

constant_values (`float` or `Iterable[float]`, *optional*) : The value to use for the padding if `mode` is `"constant"`.

data_format (`str` or `ChannelDimension`, *optional*) : The channel dimension format for the output image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. If unset, will use same as the input image.

input_data_format (`str` or `ChannelDimension`, *optional*) : The channel dimension format for the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. If unset, will use the inferred format of the input image.

**Returns:**

``np.ndarray``

The padded image.

#### transformers.image_transforms.rgb_to_id[[transformers.image_transforms.rgb_to_id]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L614)

Converts RGB color to unique ID.

#### transformers.image_transforms.rescale[[transformers.image_transforms.rescale]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L89)

Rescales `image` by `scale`.

**Parameters:**

image (`np.ndarray`) : The image to rescale.

scale (`float`) : The scale to use for rescaling the image.

data_format (`ChannelDimension`, *optional*) : The channel dimension format of the image. If not provided, it will be the same as the input image.

dtype (`np.dtype`, *optional*, defaults to `np.float32`) : The dtype of the output image. Defaults to `np.float32`. Used for backwards compatibility with feature extractors.

input_data_format (`ChannelDimension`, *optional*) : The channel dimension format of the input image. If not provided, it will be inferred from the input image.

**Returns:**

``np.ndarray``

The rescaled image.

#### transformers.image_transforms.resize[[transformers.image_transforms.resize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L313)

Resizes `image` to `(height, width)` specified by `size` using the PIL library.

**Parameters:**

image (`np.ndarray`) : The image to resize.

size (`tuple[int, int]`) : The size to use for resizing the image.

resample (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`) : The filter to user for resampling.

reducing_gap (`int`, *optional*) : Apply optimization by resizing the image in two steps. The bigger `reducing_gap`, the closer the result to the fair resampling. See corresponding Pillow documentation for more details.

data_format (`ChannelDimension`, *optional*) : The channel dimension format of the output image. If unset, will use the inferred format from the input.

return_numpy (`bool`, *optional*, defaults to `True`) : Whether or not to return the resized image as a numpy array. If False a `PIL.Image.Image` object is returned.

input_data_format (`ChannelDimension`, *optional*) : The channel dimension format of the input image. If unset, will use the inferred format from the input.

**Returns:**

``np.ndarray``

The resized image.

#### transformers.image_transforms.to_pil_image[[transformers.image_transforms.to_pil_image]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L154)

Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
needed.

**Parameters:**

image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`) : The image to convert to the `PIL.Image` format.

do_rescale (`bool`, *optional*) : Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will default to `True` if the image type is a floating type and casting to `int` would result in a loss of precision, and `False` otherwise.

image_mode (`str`, *optional*) : The mode to use for the PIL image. If unset, will use the default mode for the input image type.

input_data_format (`ChannelDimension`, *optional*) : The channel dimension format of the input image. If unset, will use the inferred format from the input.

**Returns:**

``PIL.Image.Image``

The converted image.

## ImageProcessingMixin[[transformers.ImageProcessingMixin]]

#### transformers.ImageProcessingMixin[[transformers.ImageProcessingMixin]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L61)

This is an image processor mixin used to provide saving/loading functionality for sequential and image feature
extractors.

fetch_imagestransformers.ImageProcessingMixin.fetch_imageshttps://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L475[{"name": "image_url_or_urls", "val": ": typing.Union[str, list[str], list[list[str]]]"}]

Convert a single or a list of urls into the corresponding `PIL.Image` objects.

If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
returned.
#### from_dict[[transformers.ImageProcessingMixin.from_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L354)

Instantiates a type of [ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin) from a Python dictionary of parameters.

**Parameters:**

image_processor_dict (`dict[str, Any]`) : Dictionary that will be used to instantiate the image processor object. Such a dictionary can be retrieved from a pretrained checkpoint by leveraging the [to_dict()](/docs/transformers/main/en/internal/image_processing_utils#transformers.ImageProcessingMixin.to_dict) method.

kwargs (`dict[str, Any]`) : Additional parameters from which to initialize the image processor object.

**Returns:**

`[ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin)`

The image processor object instantiated from those
parameters.
#### from_json_file[[transformers.ImageProcessingMixin.from_json_file]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L399)

Instantiates a image processor of type [ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin) from the path to a JSON
file of parameters.

**Parameters:**

json_file (`str` or `os.PathLike`) : Path to the JSON file containing the parameters.

**Returns:**

`A image processor of type [ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin)`

The image_processor object
instantiated from that JSON file.
#### from_pretrained[[transformers.ImageProcessingMixin.from_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L88)

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
#### get_image_processor_dict[[transformers.ImageProcessingMixin.get_image_processor_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L235)

From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
image processor of type `~image_processor_utils.ImageProcessingMixin` using `from_dict`.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

subfolder (`str`, *optional*, defaults to `""`) : In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can specify the folder name here.

image_processor_filename (`str`, *optional*, defaults to `"config.json"`) : The name of the file in the model directory to use for the image processor config.

**Returns:**

``tuple[Dict, Dict]``

The dictionary(ies) that will be used to instantiate the image processor object.
#### push_to_hub[[transformers.ImageProcessingMixin.push_to_hub]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L711)

Upload the image processor file to the 🤗 Model Hub.

Examples:

```python
from transformers import AutoImageProcessor

image processor = AutoImageProcessor.from_pretrained("google-bert/bert-base-cased")

# Push the image processor to your namespace with the name "my-finetuned-bert".
image processor.push_to_hub("my-finetuned-bert")

# Push the image processor to an organization with the name "my-finetuned-bert".
image processor.push_to_hub("huggingface/my-finetuned-bert")
```

**Parameters:**

repo_id (`str`) : The name of the repository you want to push your image processor to. It should contain your organization name when pushing to a given organization.

commit_message (`str`, *optional*) : Message to commit while pushing. Will default to `"Upload image processor"`.

commit_description (`str`, *optional*) : The description of the commit that will be created

private (`bool`, *optional*) : Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.

token (`bool` or `str`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True` (default), will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*) : Branch to push the uploaded files to.

create_pr (`bool`, *optional*, defaults to `False`) : Whether or not to create a PR with the uploaded files or directly commit.

max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`) : Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

tags (`list[str]`, *optional*) : List of tags to push on the Hub.
#### register_for_auto_class[[transformers.ImageProcessingMixin.register_for_auto_class]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L453)

Register this class with a given auto class. This should only be used for custom image processors as the ones
in the library are already mapped with `AutoImageProcessor `.

**Parameters:**

auto_class (`str` or `type`, *optional*, defaults to `"AutoImageProcessor "`) : The auto class to register this new image processor with.
#### save_pretrained[[transformers.ImageProcessingMixin.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L187)

Save an image processor object to the directory `save_directory`, so that it can be re-loaded using the
[from_pretrained()](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin.from_pretrained) class method.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory where the image processor JSON file will be saved (will be created if it does not exist).

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### to_dict[[transformers.ImageProcessingMixin.to_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L387)

Serializes this instance to a Python dictionary.

**Returns:**

``dict[str, Any]``

Dictionary of all the attributes that make up this image processor instance.
#### to_json_file[[transformers.ImageProcessingMixin.to_json_file]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L439)

Save this instance to a JSON file.

**Parameters:**

json_file_path (`str` or `os.PathLike`) : Path to the JSON file in which this image_processor instance's parameters will be saved.
#### to_json_string[[transformers.ImageProcessingMixin.to_json_string]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_base.py#L418)

Serializes this instance to a JSON string.

**Returns:**

``str``

String containing all the attributes that make up this feature_extractor instance in JSON format.
