# Video Processor

A **Video Processor** is a utility responsible for preparing input features for video models, as well as handling the post-processing of their outputs. It provides transformations such as resizing, normalization, and conversion into PyTorch. Along ith transformations the `VideoProcessor` class handles video decoding from local paths or URLs (requires [`torchcodec`](https://pypi.org/project/torchcodec/)) and frame sampling according to model-specific strategies.

The video processor extends the functionality of image processors by allowing Vision Large Language Models (VLMs) to handle videos with a distinct set of arguments compared to images. It serves as the bridge between raw video data and the model, ensuring that input features are optimized for the VLM.

When adding a new VLM or updating an existing one to enable distinct video preprocessing, saving and reloading the processor configuration will store the video related arguments in a dedicated file named `video_preprocessing_config.json`. Don’t worry if you haven’t updated your VLM, the processor will try to load video related configurations from a file named `preprocessing_config.json`.

### Usage Example

Here’s an example of how to load a video processor with [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) model:


```
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
```

Currently, if using base image processor for videos, it processes video data by treating each frame as an individual image and applying transformations frame-by-frame. While functional, this approach is not highly efficient. Using `AutoVideoProcessor` allows us to take advantage of **fast video processors**, leveraging the [torchvision](https://pytorch.org/vision/stable/index.html) library. Fast processors handle the whole batch of videos at once, without iterating over each video or frame. These updates introduce GPU acceleration and significantly enhance processing speed, especially for tasks requiring high throughput.

Fast video processors are available for all models and are loaded by default when an `AutoVideoProcessor` is initialized. When using a fast video processor, you can also set the `device` argument to specify the device on which the processing should be done. By default, the processing is done on the same device as the inputs if the inputs are tensors, or on the CPU otherwise. For even more speed improvement, we can compile the processor when using ‘cuda’ as device.


```
import torch
from transformers.video_utils import load_video
from transformers import AutoVideoProcessor

video = load_video("video.mp4")
processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
processor = torch.compile(processor)
processed_video = processor(video, return_tensors="pt")
```

#### Sampling behavior

The video processor can also sample video frames using the technique best suited for the given model. Sampling behavior is controlled with the `do_sample_frames` argument and can be configured through model-specific parameters such as `num_frames` or `fps` (the rate at which the video will be sampled). If the input video is given as a local path or URL (`str`), the processor will decode it automatically. To obtain metadata about the decoded video, such as sampled frame indices, original dimensions, duration, and fps, pass `return_metadata=True` to the processor.

* Specifying `num_frames` does not guarantee the output will contain exactly that number of frames. Depending on the model, the sampler may enforce minimum or maximum frame limits.
* The default decoder is [`torchcodec`](https://pypi.org/project/torchcodec/), which must be installed.


```
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
processed_video_inputs = processor(videos=["video_path.mp4"], return_metadata=True, do_sample_frames=True, return_tensors="pt")
video_metadata = processed_video_inputs["video_metadata"]

# See how many frames the original video had and what was the original FPS
print(video_metadata.total_num_frames, video_metadata.fps)
```

If you pass an already decoded video array but still want to enable model-specific frame sampling, it is strongly recommended to provide video\_metadata. This allows the sampler to know the original video’s duration and FPS. You can pass metadata as a `VideoMetadata` object or as a plain dict.


```
from transformers import AutoVideoProcessor
from transformers.video_utils import VideoMetadata

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
my_decodec_video = torch.randint(0, 255, size=(100, 3, 1280, 1280)) # short video of 100 frames
video_metadata = VideoMetadata(
    total_num_frames=100,
    fps=24,
    duration=4.1, # in seconds
)
processed_video_inputs = processor(videos=["video_path.mp4"], video_metadata=video_metadata, do_sample_frames=True, num_frames=10, return_tensors="pt")
print(processed_video_inputs.pixel_values_videos.shape)
>>> [10, 3, 384, 384]
```

## BaseVideoProcessor

### class transformers.BaseVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L155)

( \*\*kwargs: typing\_extensions.Unpack[transformers.processing\_utils.VideosKwargs]  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the video’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `self.size`) —
  Size of the output video after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **size\_divisor** (`int`, *optional*, defaults to `self.size_divisor`) —
  The size by which to make sure both the height and width can be divided.
* **default\_to\_square** (`bool`, *optional*, defaults to `self.default_to_square`) —
  Whether to default to a square video when resizing, if size is an int.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the video to the specified `crop_size`. Can be overridden by `do_center_crop` in the
  `preprocess` method.
* **do\_pad** (`bool`, *optional*) —
  Whether to pad the video to the `(max_height, max_width)` of the videos in the batch.
* **crop\_size** (`dict[str, int]` *optional*, defaults to `self.crop_size`) —
  Size of the output video after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
  method.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `self.rescale_factor`) —
  Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Mean to use if normalizing the video. This is a float or list of floats the length of the number of
  channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
  number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.image_std`) —
  Whether to convert the video to RGB.
* **video\_metadata** (`VideoMetadata`, *optional*) —
  Metadata of the video containing information about total duration, fps and total number of frames.
* **do\_sample\_frames** (`int`, *optional*, defaults to `self.do_sample_frames`) —
  Whether to sample frames from the video before processing or to process the whole video.
* **num\_frames** (`int`, *optional*, defaults to `self.num_frames`) —
  Maximum number of frames to sample when `do_sample_frames=True`.
* **fps** (`int` or `float`, *optional*, defaults to `self.fps`) —
  Target frames to sample per second when `do_sample_frames=True`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input video.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input video. If unset, the channel dimension format is inferred
  from the input video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the videos on. If unset, the device is inferred from the input videos.
* **return\_metadata** (`bool`, *optional*) —
  Whether to return video metadata or not.

Constructs a base VideoProcessor.

#### convert\_to\_rgb

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L214)

( video: torch.Tensor  ) → `torch.Tensor`

Parameters

* **video** (`"torch.Tensor"`) —
  The video to convert.

Returns

`torch.Tensor`

The converted video.

Converts a video to RGB format.

#### fetch\_videos

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L880)

( video\_url\_or\_urls: typing.Union[str, list[str], list[list[str]]] sample\_indices\_fn = None  )

Convert a single or a list of urls into the corresponding `np.array` objects.

If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
returned.

#### from\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L741)

( video\_processor\_dict: dict \*\*kwargs  ) → `~video_processing_utils.VideoProcessorBase`

Parameters

* **video\_processor\_dict** (`dict[str, Any]`) —
  Dictionary that will be used to instantiate the video processor object. Such a dictionary can be
  retrieved from a pretrained checkpoint by leveraging the
  `~video_processing_utils.VideoProcessorBase.to_dict` method.
* **kwargs** (`dict[str, Any]`) —
  Additional parameters from which to initialize the video processor object.

Returns

`~video_processing_utils.VideoProcessorBase`

The video processor object instantiated from those
parameters.

Instantiates a type of `~video_processing_utils.VideoProcessorBase` from a Python dictionary of parameters.

#### from\_json\_file

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L835)

( json\_file: typing.Union[str, os.PathLike]  ) → A video processor of type `~video_processing_utils.VideoProcessorBase`

Parameters

* **json\_file** (`str` or `os.PathLike`) —
  Path to the JSON file containing the parameters.

Returns

A video processor of type `~video_processing_utils.VideoProcessorBase`

The video\_processor object
instantiated from that JSON file.

Instantiates a video processor of type `~video_processing_utils.VideoProcessorBase` from the path to a JSON
file of parameters.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L448)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained video hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a video processor file saved using the
    `~video_processing_utils.VideoProcessorBase.save_pretrained` method, e.g.,
    `./my_model_directory/`.
  + a path or url to a saved video processor JSON *file*, e.g.,
    `./my_model_directory/video_preprocessor_config.json`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model video processor should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force to (re-)download the video processor files and override the cached versions if
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

Instantiate a type of `~video_processing_utils.VideoProcessorBase` from an video processor.

Examples:


```
# We can't instantiate directly the base class *VideoProcessorBase* so let's show the examples on a
# derived class: *LlavaOnevisionVideoProcessor*
video_processor = LlavaOnevisionVideoProcessor.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
)  # Download video_processing_config from huggingface.co and cache.
video_processor = LlavaOnevisionVideoProcessor.from_pretrained(
    "./test/saved_model/"
)  # E.g. video processor (or model) was saved using *save_pretrained('./test/saved_model/')*
video_processor = LlavaOnevisionVideoProcessor.from_pretrained("./test/saved_model/video_preprocessor_config.json")
video_processor = LlavaOnevisionVideoProcessor.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", do_normalize=False, foo=False
)
assert video_processor.do_normalize is False
video_processor, unused_kwargs = LlavaOnevisionVideoProcessor.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", do_normalize=False, foo=False, return_unused_kwargs=True
)
assert video_processor.do_normalize is False
assert unused_kwargs == {"foo": False}
```

#### get\_video\_processor\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L623)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] \*\*kwargs  ) → `tuple[Dict, Dict]`

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
* **subfolder** (`str`, *optional*, defaults to `""`) —
  In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
  specify the folder name here.

Returns

`tuple[Dict, Dict]`

The dictionary(ies) that will be used to instantiate the video processor object.

From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
video processor of type `~video_processing_utils.VideoProcessorBase` using `from_dict`.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L355)

( videos: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]]] \*\*kwargs: typing\_extensions.Unpack[transformers.processing\_utils.VideosKwargs]  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the video’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `self.size`) —
  Size of the output video after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **size\_divisor** (`int`, *optional*, defaults to `self.size_divisor`) —
  The size by which to make sure both the height and width can be divided.
* **default\_to\_square** (`bool`, *optional*, defaults to `self.default_to_square`) —
  Whether to default to a square video when resizing, if size is an int.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the video to the specified `crop_size`. Can be overridden by `do_center_crop` in the
  `preprocess` method.
* **do\_pad** (`bool`, *optional*) —
  Whether to pad the video to the `(max_height, max_width)` of the videos in the batch.
* **crop\_size** (`dict[str, int]` *optional*, defaults to `self.crop_size`) —
  Size of the output video after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
  method.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `self.rescale_factor`) —
  Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Mean to use if normalizing the video. This is a float or list of floats the length of the number of
  channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
  number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.image_std`) —
  Whether to convert the video to RGB.
* **video\_metadata** (`VideoMetadata`, *optional*) —
  Metadata of the video containing information about total duration, fps and total number of frames.
* **do\_sample\_frames** (`int`, *optional*, defaults to `self.do_sample_frames`) —
  Whether to sample frames from the video before processing or to process the whole video.
* **num\_frames** (`int`, *optional*, defaults to `self.num_frames`) —
  Maximum number of frames to sample when `do_sample_frames=True`.
* **fps** (`int` or `float`, *optional*, defaults to `self.fps`) —
  Target frames to sample per second when `do_sample_frames=True`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input video.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input video. If unset, the channel dimension format is inferred
  from the input video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the videos on. If unset, the device is inferred from the input videos.
* **return\_metadata** (`bool`, *optional*) —
  Whether to return video metadata or not.

#### register\_for\_auto\_class

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L854)

( auto\_class = 'AutoVideoProcessor'  )

Parameters

* **auto\_class** (`str` or `type`, *optional*, defaults to `"AutoVideoProcessor "`) —
  The auto class to register this new video processor with.

Register this class with a given auto class. This should only be used for custom video processors as the ones
in the library are already mapped with `AutoVideoProcessor` .

This API is experimental and may have some slight breaking changes in the next releases.

#### sample\_frames

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L239)

( metadata: VideoMetadata num\_frames: typing.Optional[int] = None fps: typing.Union[int, float, NoneType] = None \*\*kwargs  ) → np.ndarray

Parameters

* **metadata** (`VideoMetadata`) —
  Metadata of the video containing information about total duration, fps and total number of frames.
* **num\_frames** (`int`, *optional*) —
  Maximum number of frames to sample. Defaults to `self.num_frames`.
* **fps** (`int` or `float`, *optional*) —
  Target frames to sample per second. Defaults to `self.fps`.

Returns

np.ndarray

Indices to sample video frames.

Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
and `fps` are mutually exclusive.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L562)

( save\_directory: typing.Union[str, os.PathLike] push\_to\_hub: bool = False \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the video processor JSON file will be saved (will be created if it does not exist).
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Save an video processor object to the directory `save_directory`, so that it can be re-loaded using the
`~video_processing_utils.VideoProcessorBase.from_pretrained` class method.

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L786)

( ) → `dict[str, Any]`

Returns

`dict[str, Any]`

Dictionary of all the attributes that make up this video processor instance.

Serializes this instance to a Python dictionary.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/video_processor.md)
