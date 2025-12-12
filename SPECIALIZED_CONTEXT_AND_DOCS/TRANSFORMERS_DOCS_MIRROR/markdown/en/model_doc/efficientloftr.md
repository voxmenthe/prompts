*This model was released on 2024-03-07 and added to Hugging Face Transformers on 2025-07-22.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# EfficientLoFTR

[EfficientLoFTR](https://huggingface.co/papers/2403.04765) is an efficient detector-free local feature matching method that produces semi-dense matches across images with sparse-like speed. It builds upon the original [LoFTR](https://huggingface.co/papers/2104.00680) architecture but introduces significant improvements for both efficiency and accuracy. The key innovation is an aggregated attention mechanism with adaptive token selection that makes the model ~2.5× faster than LoFTR while achieving higher accuracy. EfficientLoFTR can even surpass state-of-the-art efficient sparse matching pipelines like [SuperPoint](./superpoint) + [LightGlue](./lightglue) in terms of speed, making it suitable for large-scale or latency-sensitive applications such as image retrieval and 3D reconstruction.

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).

Click on the EfficientLoFTR models in the right sidebar for more examples of how to apply EfficientLoFTR to different computer vision tasks.

The example below demonstrates how to match keypoints between two images with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
from transformers import pipeline

keypoint_matcher = pipeline(task="keypoint-matching", model="zju-community/efficientloftr")

url_0 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
url_1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"

results = keypoint_matcher([url_0, url_1], threshold=0.9)
print(results[0])
# {'keypoint_image_0': {'x': ..., 'y': ...}, 'keypoint_image_1': {'x': ..., 'y': ...}, 'score': ...}
```

## Notes

* EfficientLoFTR is designed for efficiency while maintaining high accuracy. It uses an aggregated attention mechanism with adaptive token selection to reduce computational overhead compared to the original LoFTR.


  ```
  from transformers import AutoImageProcessor, AutoModelForKeypointMatching
  import torch
  from PIL import Image
  import requests

  processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
  model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")

  # EfficientLoFTR requires pairs of images
  images = [image1, image2]
  inputs = processor(images, return_tensors="pt")
  with torch.inference_mode():
      outputs = model(**inputs)

  # Extract matching information
  keypoints = outputs.keypoints        # Keypoints in both images
  matches = outputs.matches            # Matching indices 
  matching_scores = outputs.matching_scores  # Confidence scores
  ```
* The model produces semi-dense matches, offering a good balance between the density of matches and computational efficiency. It excels in handling large viewpoint changes and texture-poor scenarios.
* For better visualization and analysis, use the [post\_process\_keypoint\_matching()](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRImageProcessor.post_process_keypoint_matching) method to get matches in a more readable format.


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
  visualized_images = processor.visualize_keypoint_matching(images, processed_outputs)
  ```
* EfficientLoFTR uses a novel two-stage correlation layer that achieves accurate subpixel correspondences, improving upon the original LoFTR’s fine correlation module.

![](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/2nJZQlFToCYp_iLurvcZ4.png)

## Resources

* Refer to the [original EfficientLoFTR repository](https://github.com/zju3dv/EfficientLoFTR) for more examples and implementation details.
* [EfficientLoFTR project page](https://zju3dv.github.io/efficientloftr/) with interactive demos and additional information.

## EfficientLoFTRConfig

### class transformers.EfficientLoFTRConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientloftr/configuration_efficientloftr.py#L20)

( stage\_num\_blocks: typing.Optional[list[int]] = None out\_features: typing.Optional[list[int]] = None stage\_stride: typing.Optional[list[int]] = None hidden\_size: int = 256 activation\_function: str = 'relu' q\_aggregation\_kernel\_size: int = 4 kv\_aggregation\_kernel\_size: int = 4 q\_aggregation\_stride: int = 4 kv\_aggregation\_stride: int = 4 num\_attention\_layers: int = 4 num\_attention\_heads: int = 8 attention\_dropout: float = 0.0 attention\_bias: bool = False mlp\_activation\_function: str = 'leaky\_relu' coarse\_matching\_skip\_softmax: bool = False coarse\_matching\_threshold: float = 0.2 coarse\_matching\_temperature: float = 0.1 coarse\_matching\_border\_removal: int = 2 fine\_kernel\_size: int = 8 batch\_norm\_eps: float = 1e-05 rope\_theta: float = 10000.0 partial\_rotary\_factor: float = 4.0 rope\_scaling: typing.Optional[dict] = None fine\_matching\_slice\_dim: int = 8 fine\_matching\_regress\_temperature: float = 10.0 initializer\_range: float = 0.02 \*\*kwargs  )

Parameters

* **stage\_num\_blocks** (`List`, *optional*, defaults to [1, 2, 4, 14]) —
  The number of blocks in each stages
* **out\_features** (`List`, *optional*, defaults to [64, 64, 128, 256]) —
  The number of channels in each stage
* **stage\_stride** (`List`, *optional*, defaults to [2, 1, 2, 2]) —
  The stride used in each stage
* **hidden\_size** (`int`, *optional*, defaults to 256) —
  The dimension of the descriptors.
* **activation\_function** (`str`, *optional*, defaults to `"relu"`) —
  The activation function used in the backbone
* **q\_aggregation\_kernel\_size** (`int`, *optional*, defaults to 4) —
  The kernel size of the aggregation of query states in the fusion network
* **kv\_aggregation\_kernel\_size** (`int`, *optional*, defaults to 4) —
  The kernel size of the aggregation of key and value states in the fusion network
* **q\_aggregation\_stride** (`int`, *optional*, defaults to 4) —
  The stride of the aggregation of query states in the fusion network
* **kv\_aggregation\_stride** (`int`, *optional*, defaults to 4) —
  The stride of the aggregation of key and value states in the fusion network
* **num\_attention\_layers** (`int`, *optional*, defaults to 4) —
  Number of attention layers in the LocalFeatureTransformer
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  The number of heads in the GNN layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **attention\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value and output projection layers during attention.
* **mlp\_activation\_function** (`str`, *optional*, defaults to `"leaky_relu"`) —
  Activation function used in the attention mlp layer.
* **coarse\_matching\_skip\_softmax** (`bool`, *optional*, defaults to `False`) —
  Whether to skip softmax or not at the coarse matching step.
* **coarse\_matching\_threshold** (`float`, *optional*, defaults to 0.2) —
  The threshold for the minimum score required for a match.
* **coarse\_matching\_temperature** (`float`, *optional*, defaults to 0.1) —
  The temperature to apply to the coarse similarity matrix
* **coarse\_matching\_border\_removal** (`int`, *optional*, defaults to 2) —
  The size of the border to remove during coarse matching
* **fine\_kernel\_size** (`int`, *optional*, defaults to 8) —
  Kernel size used for the fine feature matching
* **batch\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the batch normalization layers.
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **partial\_rotary\_factor** (`float`, *optional*, defaults to 4.0) —
  Dim factor for the RoPE embeddings, in EfficientLoFTR, frequencies should be generated for
  the whole hidden\_size, so this factor is used to compensate.
* **rope\_scaling** (`Dict`, *optional*) —
  Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
  and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
  accordingly.
  Expected contents:
  `rope_type` (`str`):
  The sub-variant of RoPE to use. Can be one of [‘default’, ‘linear’, ‘dynamic’, ‘yarn’, ‘longrope’,
  ‘llama3’, ‘2d’], with ‘default’ being the original RoPE implementation.
  `dim` (`int`): The dimension of the RoPE embeddings.
* **fine\_matching\_slice\_dim** (`int`, *optional*, defaults to 8) —
  The size of the slice used to divide the fine features for the first and second fine matching stages.
* **fine\_matching\_regress\_temperature** (`float`, *optional*, defaults to 10.0) —
  The temperature to apply to the fine similarity matrix
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a `EffientLoFTRFromKeypointMatching`.
It is used to instantiate a EfficientLoFTR model according to the specified arguments, defining the model
architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
EfficientLoFTR [zju-community/efficientloftr](https://huggingface.co/zju-community/efficientloftr) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import EfficientLoFTRConfig, EfficientLoFTRForKeypointMatching

>>> # Initializing a EfficientLoFTR configuration
>>> configuration = EfficientLoFTRConfig()

>>> # Initializing a model from the EfficientLoFTR configuration
>>> model = EfficientLoFTRForKeypointMatching(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## EfficientLoFTRImageProcessor

### class transformers.EfficientLoFTRImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientloftr/image_processing_efficientloftr.py#L135)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: float = 0.00392156862745098 do\_grayscale: bool = True \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Controls whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden
  by `do_resize` in the `preprocess` method.
* **size** (`Dict[str, int]` *optional*, defaults to `{"height" -- 480, "width": 640}`):
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

Constructs a EfficientLoFTR image processor.

#### post\_process\_keypoint\_matching

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientloftr/image_processing_efficientloftr.py#L340)

( outputs: KeypointMatchingOutput target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] threshold: float = 0.0  ) → `List[Dict]`

Parameters

* **outputs** (`KeypointMatchingOutput`) —
  Raw outputs of the model.
* **target\_sizes** (`torch.Tensor` or `List[Tuple[Tuple[int, int]]]`, *optional*) —
  Tensor of shape `(batch_size, 2, 2)` or list of tuples of tuples (`Tuple[int, int]`) containing the
  target size `(height, width)` of each image in the batch. This must be the original image size (before
  any processing).
* **threshold** (`float`, *optional*, defaults to 0.0) —
  Threshold to filter out the matches with low scores.

Returns

`List[Dict]`

A list of dictionaries, each dictionary containing the keypoints in the first and second image
of the pair, the matching scores and the matching indices.

Converts the raw output of `KeypointMatchingOutput` into lists of keypoints, scores and descriptors
with coordinates absolute to the original image sizes.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientloftr/image_processing_efficientloftr.py#L222)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientloftr/image_processing_efficientloftr.py#L182)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientloftr/image_processing_efficientloftr.py#L399)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] keypoint\_matching\_output: list  ) → `List[PIL.Image.Image]`

Parameters

* **images** (`ImageInput`) —
  Image pairs to plot. Same as `EfficientLoFTRImageProcessor.preprocess`. Expects either a list of 2
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

## EfficientLoFTRModel

### class transformers.EfficientLoFTRModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientloftr/modeling_efficientloftr.py#L672)

( config: EfficientLoFTRConfig  )

Parameters

* **config** ([EfficientLoFTRConfig](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

EfficientLoFTR model taking images as inputs and outputting the features of the images.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientloftr/modeling_efficientloftr.py#L683)

( pixel\_values: FloatTensor labels: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.modeling_outputs.BackboneOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EfficientLoFTRImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRImageProcessor). See [EfficientLoFTRImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [EfficientLoFTRImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

Returns

`transformers.modeling_outputs.BackboneOutput` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BackboneOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([EfficientLoFTRConfig](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRConfig)) and inputs.

* **feature\_maps** (`tuple(torch.FloatTensor)` of shape `(batch_size, num_channels, height, width)`) — Feature maps of the stages.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, num_channels, height, width)`,
  depending on the backbone.

  Hidden-states of the model at the output of each stage plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Only applicable if the backbone uses attention.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [EfficientLoFTRModel](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRModel) forward method, overrides the `__call__` special method.

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

>>> processor = AutoImageProcessor.from_pretrained("zju-community/efficient_loftr")
>>> model = AutoModel.from_pretrained("zju-community/efficient_loftr")

>>> with torch.no_grad():
>>>     inputs = processor(images, return_tensors="pt")
>>>     outputs = model(**inputs)
```

* forward

## EfficientLoFTRForKeypointMatching

### class transformers.EfficientLoFTRForKeypointMatching

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientloftr/modeling_efficientloftr.py#L890)

( config: EfficientLoFTRConfig  )

Parameters

* **config** ([EfficientLoFTRConfig](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

EfficientLoFTR model taking images as inputs and outputting the matching of them.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/efficientloftr/modeling_efficientloftr.py#L1251)

( pixel\_values: FloatTensor labels: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.efficientloftr.modeling_efficientloftr.KeypointMatchingOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

Returns

`transformers.models.efficientloftr.modeling_efficientloftr.KeypointMatchingOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.efficientloftr.modeling_efficientloftr.KeypointMatchingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **matches** (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`) — Index of keypoint matched in the other image.
* **matching\_scores** (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`) — Scores of predicted matches.
* **keypoints** (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`) — Absolute (x, y) coordinates of predicted keypoints in a given image.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*) — Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, 2, num_channels, num_keypoints)`, returned when `output_hidden_states=True` is passed or when
  `config.output_hidden_states=True`)
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, 2, num_heads, num_keypoints, num_keypoints)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`)

The [EfficientLoFTRForKeypointMatching](/docs/transformers/v4.56.2/en/model_doc/efficientloftr#transformers.EfficientLoFTRForKeypointMatching) forward method, overrides the `__call__` special method.

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

>>> processor = AutoImageProcessor.from_pretrained("zju-community/efficient_loftr")
>>> model = AutoModel.from_pretrained("zju-community/efficient_loftr")

>>> with torch.no_grad():
>>>     inputs = processor(images, return_tensors="pt")
>>>     outputs = model(**inputs)
```

* forward

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/efficientloftr.md)
