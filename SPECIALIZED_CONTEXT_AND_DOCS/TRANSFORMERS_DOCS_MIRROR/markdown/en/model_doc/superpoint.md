*This model was released on 2017-12-20 and added to Hugging Face Transformers on 2024-03-19.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# SuperPoint

[SuperPoint](https://huggingface.co/papers/1712.07629) is the result of self-supervised training of a fully-convolutional network for interest point detection and description. The model is able to detect interest points that are repeatable under homographic transformations and provide a descriptor for each point. Usage on it’s own is limited, but it can be used as a feature extractor for other tasks such as homography estimation and image matching.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/superpoint_architecture.png)

You can find all the original SuperPoint checkpoints under the [Magic Leap Community](https://huggingface.co/magic-leap-community) organization.

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).

Click on the SuperPoint models in the right sidebar for more examples of how to apply SuperPoint to different computer vision tasks.

The example below demonstrates how to detect interest points in an image with the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

AutoModel


```
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Post-process to get keypoints, scores, and descriptors
image_size = (image.height, image.width)
processed_outputs = processor.post_process_keypoint_detection(outputs, [image_size])
```

## Notes

* SuperPoint outputs a dynamic number of keypoints per image, which makes it suitable for tasks requiring variable-length feature representations.


  ```
  from transformers import AutoImageProcessor, SuperPointForKeypointDetection
  import torch
  from PIL import Image
  import requests
  processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
  model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
  url_image_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
  image_1 = Image.open(requests.get(url_image_1, stream=True).raw)
  url_image_2 = "http://images.cocodataset.org/test-stuff2017/000000000568.jpg"
  image_2 = Image.open(requests.get(url_image_2, stream=True).raw)
  images = [image_1, image_2]
  inputs = processor(images, return_tensors="pt")
  # Example of handling dynamic keypoint output
  outputs = model(**inputs)
  keypoints = outputs.keypoints  # Shape varies per image
  scores = outputs.scores        # Confidence scores for each keypoint
  descriptors = outputs.descriptors  # 256-dimensional descriptors
  mask = outputs.mask # Value of 1 corresponds to a keypoint detection
  ```
* The model provides both keypoint coordinates and their corresponding descriptors (256-dimensional vectors) in a single forward pass.
* For batch processing with multiple images, you need to use the mask attribute to retrieve the respective information for each image. You can use the `post_process_keypoint_detection` from the `SuperPointImageProcessor` to retrieve the each image information.


  ```
  # Batch processing example
  images = [image1, image2, image3]
  inputs = processor(images, return_tensors="pt")
  outputs = model(**inputs)
  image_sizes = [(img.height, img.width) for img in images]
  processed_outputs = processor.post_process_keypoint_detection(outputs, image_sizes)
  ```
* You can then print the keypoints on the image of your choice to visualize the result:


  ```
  import matplotlib.pyplot as plt
  plt.axis("off")
  plt.imshow(image_1)
  plt.scatter(
      outputs[0]["keypoints"][:, 0],
      outputs[0]["keypoints"][:, 1],
      c=outputs[0]["scores"] * 100,
      s=outputs[0]["scores"] * 50,
      alpha=0.8
  )
  plt.savefig(f"output_image.png")
  ```

![](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/ZtFmphEhx8tcbEQqOolyE.png)

## Resources

* Refer to this [notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SuperPoint/Inference_with_SuperPoint_to_detect_interest_points_in_an_image.ipynb) for an inference and visualization example.

## SuperPointConfig

### class transformers.SuperPointConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superpoint/configuration_superpoint.py#L22)

( encoder\_hidden\_sizes: list = [64, 64, 128, 128] decoder\_hidden\_size: int = 256 keypoint\_decoder\_dim: int = 65 descriptor\_decoder\_dim: int = 256 keypoint\_threshold: float = 0.005 max\_keypoints: int = -1 nms\_radius: int = 4 border\_removal\_distance: int = 4 initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **encoder\_hidden\_sizes** (`List`, *optional*, defaults to `[64, 64, 128, 128]`) —
  The number of channels in each convolutional layer in the encoder.
* **decoder\_hidden\_size** (`int`, *optional*, defaults to 256) — The hidden size of the decoder.
* **keypoint\_decoder\_dim** (`int`, *optional*, defaults to 65) — The output dimension of the keypoint decoder.
* **descriptor\_decoder\_dim** (`int`, *optional*, defaults to 256) — The output dimension of the descriptor decoder.
* **keypoint\_threshold** (`float`, *optional*, defaults to 0.005) —
  The threshold to use for extracting keypoints.
* **max\_keypoints** (`int`, *optional*, defaults to -1) —
  The maximum number of keypoints to extract. If `-1`, will extract all keypoints.
* **nms\_radius** (`int`, *optional*, defaults to 4) —
  The radius for non-maximum suppression.
* **border\_removal\_distance** (`int`, *optional*, defaults to 4) —
  The distance from the border to remove keypoints.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a [SuperPointForKeypointDetection](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointForKeypointDetection). It is used to instantiate a
SuperPoint model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the SuperPoint
[magic-leap-community/superpoint](https://huggingface.co/magic-leap-community/superpoint) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import SuperPointConfig, SuperPointForKeypointDetection

>>> # Initializing a SuperPoint superpoint style configuration
>>> configuration = SuperPointConfig()
>>> # Initializing a model from the superpoint style configuration
>>> model = SuperPointForKeypointDetection(configuration)
>>> # Accessing the model configuration
>>> configuration = model.config
```

## SuperPointImageProcessor

### class transformers.SuperPointImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superpoint/image_processing_superpoint.py#L100)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: float = 0.00392156862745098 do\_grayscale: bool = False \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Controls whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden
  by `do_resize` in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 480, "width": 640}`):
  Resolution of the output image after `resize` is applied. Only has an effect if `do_resize` is set to
  `True`. Can be overridden by `size` in the `preprocess` method.
* **resample** (`Resampling`, *optional*, defaults to `2`) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.
* **do\_grayscale** (`bool`, *optional*, defaults to `False`) —
  Whether to convert the image to grayscale. Can be overridden by `do_grayscale` in the `preprocess` method.

Constructs a SuperPoint image processor.

#### post\_process\_keypoint\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superpoint/image_processing_superpoint.py#L302)

( outputs: SuperPointKeypointDescriptionOutput target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]]  ) → `list[Dict]`

Parameters

* **outputs** (`SuperPointKeypointDescriptionOutput`) —
  Raw outputs of the model containing keypoints in a relative (x, y) format, with scores and descriptors.
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  `(height, width)` of each image in the batch. This must be the original
  image size (before any processing).

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the keypoints in absolute format according
to target\_sizes, scores and descriptors for an image in the batch as predicted by the model.

Converts the raw output of [SuperPointForKeypointDetection](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointForKeypointDetection) into lists of keypoints, scores and descriptors
with coordinates absolute to the original image sizes.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superpoint/image_processing_superpoint.py#L185)

( images do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_grayscale: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the output image after `resize` has been applied. If `size["shortest_edge"]` >= 384, the image
  is resized to `(size["shortest_edge"], size["shortest_edge"])`. Otherwise, the smaller edge of the
  image will be matched to `int(size["shortest_edge"]/ crop_pct)`, after which the image is cropped to
  `(size["shortest_edge"], size["shortest_edge"])`. Only has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_grayscale** (`bool`, *optional*, defaults to `self.do_grayscale`) —
  Whether to convert the image to grayscale.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

#### resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superpoint/image_processing_superpoint.py#L146)

( image: ndarray size: dict data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  )

Parameters

* **image** (`np.ndarray`) —
  Image to resize.
* **size** (`dict[str, int]`) —
  Dictionary of the form `{"height": int, "width": int}`, specifying the size of the output image.
* **data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format of the output image. If not provided, it will be inferred from the input
  image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Resize an image.

* preprocess

## SuperPointImageProcessorFast

### class transformers.SuperPointImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superpoint/image_processing_superpoint_fast.py#L92)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.superpoint.image\_processing\_superpoint\_fast.SuperPointFastImageProcessorKwargs]  )

Constructs a fast Superpoint image processor.

#### post\_process\_keypoint\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superpoint/image_processing_superpoint_fast.py#L132)

( outputs: SuperPointKeypointDescriptionOutput target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]]  ) → `List[Dict]`

Parameters

* **outputs** (`SuperPointKeypointDescriptionOutput`) —
  Raw outputs of the model containing keypoints in a relative (x, y) format, with scores and descriptors.
* **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
  `(height, width)` of each image in the batch. This must be the original
  image size (before any processing).

Returns

`List[Dict]`

A list of dictionaries, each dictionary containing the keypoints in absolute format according
to target\_sizes, scores and descriptors for an image in the batch as predicted by the model.

Converts the raw output of [SuperPointForKeypointDetection](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointForKeypointDetection) into lists of keypoints, scores and descriptors
with coordinates absolute to the original image sizes.

* preprocess
* post\_process\_keypoint\_detection

## SuperPointForKeypointDetection

### class transformers.SuperPointForKeypointDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superpoint/modeling_superpoint.py#L364)

( config: SuperPointConfig  )

Parameters

* **config** ([SuperPointConfig](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SuperPoint model outputting keypoints and descriptors.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superpoint/modeling_superpoint.py#L385)

( pixel\_values: FloatTensor labels: typing.Optional[torch.LongTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.superpoint.modeling_superpoint.SuperPointKeypointDescriptionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SuperPointImageProcessor](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointImageProcessor). See [SuperPointImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [SuperPointImageProcessor](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.superpoint.modeling_superpoint.SuperPointKeypointDescriptionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.superpoint.modeling_superpoint.SuperPointKeypointDescriptionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SuperPointConfig](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*) — Loss computed during training.
* **keypoints** (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`) — Relative (x, y) coordinates of predicted keypoints in a given image.
* **scores** (`torch.FloatTensor` of shape `(batch_size, num_keypoints)`) — Scores of predicted keypoints.
* **descriptors** (`torch.FloatTensor` of shape `(batch_size, num_keypoints, descriptor_size)`) — Descriptors of predicted keypoints.
* **mask** (`torch.BoolTensor` of shape `(batch_size, num_keypoints)`) — Mask indicating which values in keypoints, scores and descriptors are keypoint information.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
* **when** `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.

The [SuperPointForKeypointDetection](/docs/transformers/v4.56.2/en/model_doc/superpoint#transformers.SuperPointForKeypointDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, SuperPointForKeypointDetection
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
>>> model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

>>> inputs = processor(image, return_tensors="pt")
>>> outputs = model(**inputs)
```

* forward

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/superpoint.md)
