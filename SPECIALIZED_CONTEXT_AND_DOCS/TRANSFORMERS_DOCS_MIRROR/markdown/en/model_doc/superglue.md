*This model was released on 2019-11-26 and added to Hugging Face Transformers on 2025-01-20.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# SuperGlue

[SuperGlue](https://huggingface.co/papers/1911.11763) is a neural network that matches two sets of local features by jointly finding correspondences and rejecting non-matchable points. Assignments are estimated by solving a differentiable optimal transport problem, whose costs are predicted by a graph neural network. SuperGlue introduces a flexible context aggregation mechanism based on attention, enabling it to reason about the underlying 3D scene and feature assignments jointly. Paired with the [SuperPoint model](https://huggingface.co/magic-leap-community/superpoint), it can be used to match two images and estimate the pose between them. This model is useful for tasks such as image matching, homography estimation, etc.

You can find all the original SuperGlue checkpoints under the [Magic Leap Community](https://huggingface.co/magic-leap-community) organization.

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).

Click on the SuperGlue models in the right sidebar for more examples of how to apply SuperGlue to different computer vision tasks.

The example below demonstrates how to match keypoints between two images with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
from transformers import pipeline

keypoint_matcher = pipeline(task="keypoint-matching", model="magic-leap-community/superglue_outdoor")

url_0 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
url_1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"

results = keypoint_matcher([url_0, url_1], threshold=0.9)
print(results[0])
# {'keypoint_image_0': {'x': ..., 'y': ...}, 'keypoint_image_1': {'x': ..., 'y': ...}, 'score': ...}
```

## Notes

* SuperGlue performs feature matching between two images simultaneously, requiring pairs of images as input.


  ```
  from transformers import AutoImageProcessor, AutoModel
  import torch
  from PIL import Image
  import requests

  processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
  model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")

  # SuperGlue requires pairs of images
  images = [image1, image2]
  inputs = processor(images, return_tensors="pt")
  with torch.inference_mode():
      outputs = model(**inputs)

  # Extract matching information
  keypoints0 = outputs.keypoints0  # Keypoints in first image
  keypoints1 = outputs.keypoints1  # Keypoints in second image
  matches = outputs.matches        # Matching indices
  matching_scores = outputs.matching_scores  # Confidence scores
  ```
* The model outputs matching indices, keypoints, and confidence scores for each match.
* For better visualization and analysis, use the [SuperGlueImageProcessor.post\_process\_keypoint\_matching()](/docs/transformers/v4.56.2/en/model_doc/superglue#transformers.SuperGlueImageProcessor.post_process_keypoint_matching) method to get matches in a more readable format.


  ```
  # Process outputs for visualization
  image_sizes = [[(image.height, image.width) for image in images]]
  processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

  for i, output in enumerate(processed_outputs):
      print(f"For the image pair {i}")
      for keypoint0, keypoint1, matching_score in zip(
              output["keypoints0"], output["keypoints1"], output["matching_scores"]
      ):
          print(f"Keypoint at {keypoint0.numpy()} matches with keypoint at {keypoint1.numpy()} with score {matching_score}")
  ```
* Visualize the matches between the images using the built-in plotting functionality.


  ```
  # Easy visualization using the built-in plotting method
  processor.visualize_keypoint_matching(images, processed_outputs)
  ```

![](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/01ZYaLB1NL5XdA8u7yCo4.png)

## Resources

* Refer to the [original SuperGlue repository](https://github.com/magicleap/SuperGluePretrainedNetwork) for more examples and implementation details.

## SuperGlueConfig

### class transformers.SuperGlueConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superglue/configuration_superglue.py#L27)

( keypoint\_detector\_config: SuperPointConfig = None hidden\_size: int = 256 keypoint\_encoder\_sizes: typing.Optional[list[int]] = None gnn\_layers\_types: typing.Optional[list[str]] = None num\_attention\_heads: int = 4 sinkhorn\_iterations: int = 100 matching\_threshold: float = 0.0 initializer\_range: float = 0.02 \*\*kwargs  )

Parameters

* **keypoint\_detector\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `SuperPointConfig`) —
  The config object or dictionary of the keypoint detector.
* **hidden\_size** (`int`, *optional*, defaults to 256) —
  The dimension of the descriptors.
* **keypoint\_encoder\_sizes** (`list[int]`, *optional*, defaults to `[32, 64, 128, 256]`) —
  The sizes of the keypoint encoder layers.
* **gnn\_layers\_types** (`list[str]`, *optional*, defaults to `['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross']`) —
  The types of the GNN layers. Must be either ‘self’ or ‘cross’.
* **num\_attention\_heads** (`int`, *optional*, defaults to 4) —
  The number of heads in the GNN layers.
* **sinkhorn\_iterations** (`int`, *optional*, defaults to 100) —
  The number of Sinkhorn iterations.
* **matching\_threshold** (`float`, *optional*, defaults to 0.0) —
  The matching threshold.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a `SuperGlueModel`. It is used to instantiate a
SuperGlue model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the SuperGlue
[magic-leap-community/superglue\_indoor](https://huggingface.co/magic-leap-community/superglue_indoor) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import SuperGlueConfig, SuperGlueModel

>>> # Initializing a SuperGlue superglue style configuration
>>> configuration = SuperGlueConfig()

>>> # Initializing a model from the superglue style configuration
>>> model = SuperGlueModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## SuperGlueImageProcessor

### class transformers.SuperGlueImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superglue/image_processing_superglue.py#L138)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: float = 0.00392156862745098 do\_grayscale: bool = True \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Controls whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden
  by `do_resize` in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 480, "width": 640}`):
  Resolution of the output image after `resize` is applied. Only has an effect if `do_resize` is set to
  `True`. Can be overridden by `size` in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.
* **do\_grayscale** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to grayscale. Can be overridden by `do_grayscale` in the `preprocess` method.

Constructs a SuperGlue image processor.

#### post\_process\_keypoint\_matching

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superglue/image_processing_superglue.py#L342)

( outputs: KeypointMatchingOutput target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] threshold: float = 0.0  ) → `list[Dict]`

Parameters

* **outputs** (`KeypointMatchingOutput`) —
  Raw outputs of the model.
* **target\_sizes** (`torch.Tensor` or `list[tuple[tuple[int, int]]]`, *optional*) —
  Tensor of shape `(batch_size, 2, 2)` or list of tuples of tuples (`tuple[int, int]`) containing the
  target size `(height, width)` of each image in the batch. This must be the original image size (before
  any processing).
* **threshold** (`float`, *optional*, defaults to 0.0) —
  Threshold to filter out the matches with low scores.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the keypoints in the first and second image
of the pair, the matching scores and the matching indices.

Converts the raw output of `KeypointMatchingOutput` into lists of keypoints, scores and descriptors
with coordinates absolute to the original image sizes.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superglue/image_processing_superglue.py#L224)

( images do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_grayscale: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  )

Parameters

* **images** (`ImageInput`) —
  Image pairs to preprocess. Expects either a list of 2 images or a list of list of 2 images list with
  pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set
  `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the output image after `resize` has been applied. If `size["shortest_edge"]` >= 384, the image
  is resized to `(size["shortest_edge"], size["shortest_edge"])`. Otherwise, the smaller edge of the
  image will be matched to `int(size["shortest_edge"]/ crop_pct)`, after which the image is cropped to
  `(size["shortest_edge"], size["shortest_edge"])`. Only has an effect if `do_resize` is set to `True`.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of `PILImageResampling`, filters. Only
  has an effect if `do_resize` is set to `True`.
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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superglue/image_processing_superglue.py#L185)

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

#### visualize\_keypoint\_matching

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superglue/image_processing_superglue.py#L411)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] keypoint\_matching\_output: list  ) → `List[PIL.Image.Image]`

Parameters

* **images** (`ImageInput`) —
  Image pairs to plot. Same as `SuperGlueImageProcessor.preprocess`. Expects either a list of 2
  images or a list of list of 2 images list with pixel values ranging from 0 to 255.
* **keypoint\_matching\_output** (List[Dict[str, torch.Tensor]]]) —
  A post processed keypoint matching output

Returns

`List[PIL.Image.Image]`

A list of PIL images, each containing the image pairs side by side with the detected
keypoints as well as the matching between them.

Plots the image pairs side by side with the detected keypoints as well as the matching between them.

* preprocess
* post\_process\_keypoint\_matching
* visualize\_keypoint\_matching

Pytorch

Hide Pytorch content

## SuperGlueForKeypointMatching

### class transformers.SuperGlueForKeypointMatching

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superglue/modeling_superglue.py#L545)

( config: SuperGlueConfig  )

Parameters

* **config** ([SuperGlueConfig](/docs/transformers/v4.56.2/en/model_doc/superglue#transformers.SuperGlueConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SuperGlue model taking images as inputs and outputting the matching of them.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/superglue/modeling_superglue.py#L724)

( pixel\_values: FloatTensor labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.superglue.modeling_superglue.KeypointMatchingOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SuperGlueImageProcessor](/docs/transformers/v4.56.2/en/model_doc/superglue#transformers.SuperGlueImageProcessor). See [SuperGlueImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [SuperGlueImageProcessor](/docs/transformers/v4.56.2/en/model_doc/superglue#transformers.SuperGlueImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.superglue.modeling_superglue.KeypointMatchingOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.superglue.modeling_superglue.KeypointMatchingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SuperGlueConfig](/docs/transformers/v4.56.2/en/model_doc/superglue#transformers.SuperGlueConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*) — Loss computed during training.
* **matches** (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`) — Index of keypoint matched in the other image.
* **matching\_scores** (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`) — Scores of predicted matches.
* **keypoints** (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`) — Absolute (x, y) coordinates of predicted keypoints in a given image.
* **mask** (`torch.IntTensor` of shape `(batch_size, num_keypoints)`) — Mask indicating which values in matches and matching\_scores are keypoint matching information.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*) — Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, 2, num_channels, num_keypoints)`, returned when `output_hidden_states=True` is passed or when
  `config.output_hidden_states=True`)
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, 2, num_heads, num_keypoints, num_keypoints)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`)

The [SuperGlueForKeypointMatching](/docs/transformers/v4.56.2/en/model_doc/superglue#transformers.SuperGlueForKeypointMatching) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, AutoModel
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_78916675_4568141288.jpg?raw=true"
>>> image1 = Image.open(requests.get(url, stream=True).raw)
>>> url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg?raw=true"
>>> image2 = Image.open(requests.get(url, stream=True).raw)
>>> images = [image1, image2]

>>> processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
>>> model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")

>>> with torch.no_grad():
>>>     inputs = processor(images, return_tensors="pt")
>>>     outputs = model(**inputs)
```

* forward

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/superglue.md)
