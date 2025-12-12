# Feature Extractor

A feature extractor is in charge of preparing input features for audio or vision models. This includes feature extraction from sequences, e.g., pre-processing audio files to generate Log-Mel Spectrogram features, feature extraction from images, e.g., cropping image files, but also padding, normalization, and conversion to NumPy and PyTorch tensors.

## FeatureExtractionMixin

### class transformers.FeatureExtractionMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/feature_extraction_utils.py#L251)

( \*\*kwargs  )

This is a feature extraction mixin used to provide saving/loading functionality for sequential and image feature
extractors.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/feature_extraction_utils.py#L275)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained feature\_extractor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a feature extractor file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g.,
    `./my_model_directory/`.
  + a path or url to a saved feature extractor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force to (re-)download the feature extractor files and override the cached versions
  if they exist.
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

Instantiate a type of [FeatureExtractionMixin](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin) from a feature extractor, *e.g.* a
derived class of [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor).

Examples:


```
# We can't instantiate directly the base class *FeatureExtractionMixin* nor *SequenceFeatureExtractor* so let's show the examples on a
# derived class: *Wav2Vec2FeatureExtractor*
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)  # Download feature_extraction_config from huggingface.co and cache.
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "./test/saved_model/"
)  # E.g. feature_extractor (or model) was saved using *save_pretrained('./test/saved_model/')*
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./test/saved_model/preprocessor_config.json")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False
)
assert feature_extractor.return_attention_mask is False
feature_extractor, unused_kwargs = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False, return_unused_kwargs=True
)
assert feature_extractor.return_attention_mask is False
assert unused_kwargs == {"foo": False}
```

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/feature_extraction_utils.py#L387)

( save\_directory: typing.Union[str, os.PathLike] push\_to\_hub: bool = False \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the feature extractor JSON file will be saved (will be created if it does not exist).
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Save a feature\_extractor object to the directory `save_directory`, so that it can be re-loaded using the
[from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) class method.

## SequenceFeatureExtractor

### class transformers.SequenceFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/feature_extraction_sequence_utils.py#L29)

( feature\_size: int sampling\_rate: int padding\_value: float \*\*kwargs  )

Parameters

* **feature\_size** (`int`) —
  The feature dimension of the extracted features.
* **sampling\_rate** (`int`) —
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
* **padding\_value** (`float`) —
  The value that is used to fill the padding values / vectors.

This is a general feature extraction class for speech recognition.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/feature_extraction_sequence_utils.py#L52)

( processed\_features: typing.Union[transformers.feature\_extraction\_utils.BatchFeature, list[transformers.feature\_extraction\_utils.BatchFeature], dict[str, transformers.feature\_extraction\_utils.BatchFeature], dict[str, list[transformers.feature\_extraction\_utils.BatchFeature]], list[dict[str, transformers.feature\_extraction\_utils.BatchFeature]]] padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True max\_length: typing.Optional[int] = None truncation: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None return\_attention\_mask: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None  )

Parameters

* **processed\_features** ([BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature), list of [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature), `dict[str, list[float]]`, `dict[str, list[list[float]]` or `list[dict[str, list[float]]]`) —
  Processed inputs. Can represent one input ([BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature) or `dict[str, list[float]]`) or a batch of
  input values / vectors (list of [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature), *dict[str, list[list[float]]]* or *list[dict[str,
  list[float]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
  collate function.

  Instead of `list[float]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
  see the note above for the return type.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) —
  Select a strategy to pad the returned sequences (according to the model’s padding side and padding
  index) among:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **max\_length** (`int`, *optional*) —
  Maximum length of the returned list and optionally padding length (see above).
* **truncation** (`bool`) —
  Activates truncation to cut input sequences longer than `max_length` to `max_length`.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific feature\_extractor’s default.

  [What are attention masks?](../glossary#attention-mask)
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.

Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the
max sequence length in the batch.

Padding side (left/right) padding values are defined at the feature extractor level (with `self.padding_side`,
`self.padding_value`)

If the `processed_features` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
PyTorch tensors, you will lose the specific device of your tensors however.

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

## ImageFeatureExtractionMixin

### class transformers.ImageFeatureExtractionMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L573)

( )

Mixin that contain utilities for preparing image features.

#### center\_crop

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L807)

( image size  ) → new\_image

Parameters

* **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape (n\_channels, height, width) or (height, width, n\_channels)) —
  The image to resize.
* **size** (`int` or `tuple[int, int]`) —
  The size to which crop the image.

Returns

new\_image

A center cropped `PIL.Image.Image` or `np.ndarray` or `torch.Tensor` of shape: (n\_channels,
height, width).

Crops `image` to the given size using a center crop. Note that if the image is too small to be cropped to the
size given, it will be padded (so the returned result has the size asked).

#### convert\_rgb

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L615)

( image  )

Parameters

* **image** (`PIL.Image.Image`) —
  The image to convert.

Converts `PIL.Image.Image` to RGB format.

#### expand\_dims

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L668)

( image  )

Parameters

* **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) —
  The image to expand.

Expands 2-dimensional `image` to 3 dimensions.

#### flip\_channel\_order

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L882)

( image  )

Parameters

* **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) —
  The image whose color channels to flip. If `np.ndarray` or `torch.Tensor`, the channel dimension should
  be first.

Flips the channel order of `image` from RGB to BGR, or vice versa. Note that this will trigger a conversion of
`image` to a NumPy array if it’s a PIL Image.

#### normalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L688)

( image mean std rescale = False  )

Parameters

* **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) —
  The image to normalize.
* **mean** (`list[float]` or `np.ndarray` or `torch.Tensor`) —
  The mean (per channel) to use for normalization.
* **std** (`list[float]` or `np.ndarray` or `torch.Tensor`) —
  The standard deviation (per channel) to use for normalization.
* **rescale** (`bool`, *optional*, defaults to `False`) —
  Whether or not to rescale the image to be between 0 and 1. If a PIL image is provided, scaling will
  happen automatically.

Normalizes `image` with `mean` and `std`. Note that this will trigger a conversion of `image` to a NumPy array
if it’s a PIL Image.

#### rescale

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L629)

( image: ndarray scale: typing.Union[float, int]  )

Rescale a numpy image by scale amount

#### resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L740)

( image size resample = None default\_to\_square = True max\_size = None  ) → image

Parameters

* **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) —
  The image to resize.
* **size** (`int` or `tuple[int, int]`) —
  The size to use for resizing the image. If `size` is a sequence like (h, w), output size will be
  matched to this.

  If `size` is an int and `default_to_square` is `True`, then image will be resized to (size, size). If
  `size` is an int and `default_to_square` is `False`, then smaller edge of the image will be matched to
  this number. i.e, if height > width, then image will be rescaled to (size \* height / width, size).
* **resample** (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`) —
  The filter to user for resampling.
* **default\_to\_square** (`bool`, *optional*, defaults to `True`) —
  How to convert `size` when it is a single int. If set to `True`, the `size` will be converted to a
  square (`size`,`size`). If set to `False`, will replicate
  [`torchvision.transforms.Resize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
  with support for resizing only the smallest edge and providing an optional `max_size`.
* **max\_size** (`int`, *optional*, defaults to `None`) —
  The maximum allowed for the longer edge of the resized image: if the longer edge of the image is
  greater than `max_size` after being resized according to `size`, then the image is resized again so
  that the longer edge is equal to `max_size`. As a result, `size` might be overruled, i.e the smaller
  edge may be shorter than `size`. Only used if `default_to_square` is `False`.

Returns

image

A resized `PIL.Image.Image`.

Resizes `image`. Enforces conversion of input to PIL.Image.

#### rotate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L899)

( image angle resample = None expand = 0 center = None translate = None fillcolor = None  ) → image

Parameters

* **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) —
  The image to rotate. If `np.ndarray` or `torch.Tensor`, will be converted to `PIL.Image.Image` before
  rotating.

Returns

image

A rotated `PIL.Image.Image`.

Returns a rotated copy of `image`. This method returns a copy of `image`, rotated the given number of degrees
counter clockwise around its centre.

#### to\_numpy\_array

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L636)

( image rescale = None channel\_first = True  )

Parameters

* **image** (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`) —
  The image to convert to a NumPy array.
* **rescale** (`bool`, *optional*) —
  Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
  default to `True` if the image is a PIL Image or an array/tensor of integers, `False` otherwise.
* **channel\_first** (`bool`, *optional*, defaults to `True`) —
  Whether or not to permute the dimensions of the image to put the channel dimension first.

Converts `image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
dimension.

#### to\_pil\_image

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_utils.py#L585)

( image rescale = None  )

Parameters

* **image** (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`) —
  The image to convert to the PIL Image format.
* **rescale** (`bool`, *optional*) —
  Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
  default to `True` if the image type is a floating type, `False` otherwise.

Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
needed.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/feature_extractor.md)
