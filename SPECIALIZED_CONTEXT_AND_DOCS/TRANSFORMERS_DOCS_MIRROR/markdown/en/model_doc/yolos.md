*This model was released on 2021-06-01 and added to Hugging Face Transformers on 2022-05-02.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# YOLOS

[YOLOS](https://huggingface.co/papers/2106.00666) uses a [Vision Transformer (ViT)](./vit) for object detection with minimal modifications and region priors. It can achieve performance comparable to specialized object detection models and frameworks with knowledge about 2D spatial structures.

You can find all the original YOLOS checkpoints under the [HUST Vision Lab](https://huggingface.co/hustvl/models?search=yolos) organization.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/yolos_architecture.png) YOLOS architecture. Taken from the [original paper](https://huggingface.co/papers/2106.00666).

This model wasa contributed by [nielsr](https://huggingface.co/nielsr).
Click on the YOLOS models in the right sidebar for more examples of how to apply YOLOS to different object detection tasks.

The example below demonstrates how to detect objects with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

Automodel


```
import torch
from transformers import pipeline

detector = pipeline(
    task="object-detection",
    model="hustvl/yolos-base",
    dtype=torch.float16,
    device=0
)
detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
```

## Notes

* Use [YolosImageProcessor](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosImageProcessor) for preparing images (and optional targets) for the model. Contrary to [DETR](./detr), YOLOS doesn’t require a `pixel_mask`.

## Resources

* Refer to these [notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/YOLOS) for inference and fine-tuning with [YolosForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosForObjectDetection) on a custom dataset.

## YolosConfig

### class transformers.YolosConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/configuration_yolos.py#L30)

( hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 image\_size = [512, 864] patch\_size = 16 num\_channels = 3 qkv\_bias = True num\_detection\_tokens = 100 use\_mid\_position\_embeddings = True auxiliary\_loss = False class\_cost = 1 bbox\_cost = 5 giou\_cost = 2 bbox\_loss\_coefficient = 5 giou\_loss\_coefficient = 2 eos\_coefficient = 0.1 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **image\_size** (`list[int]`, *optional*, defaults to `[512, 864]`) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.
* **num\_detection\_tokens** (`int`, *optional*, defaults to 100) —
  The number of detection tokens.
* **use\_mid\_position\_embeddings** (`bool`, *optional*, defaults to `True`) —
  Whether to use the mid-layer position encodings.
* **auxiliary\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
* **class\_cost** (`float`, *optional*, defaults to 1) —
  Relative weight of the classification error in the Hungarian matching cost.
* **bbox\_cost** (`float`, *optional*, defaults to 5) —
  Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
* **giou\_cost** (`float`, *optional*, defaults to 2) —
  Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
* **bbox\_loss\_coefficient** (`float`, *optional*, defaults to 5) —
  Relative weight of the L1 bounding box loss in the object detection loss.
* **giou\_loss\_coefficient** (`float`, *optional*, defaults to 2) —
  Relative weight of the generalized IoU loss in the object detection loss.
* **eos\_coefficient** (`float`, *optional*, defaults to 0.1) —
  Relative classification weight of the ‘no-object’ class in the object detection loss.

This is the configuration class to store the configuration of a [YolosModel](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosModel). It is used to instantiate a YOLOS
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the YOLOS
[hustvl/yolos-base](https://huggingface.co/hustvl/yolos-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import YolosConfig, YolosModel

>>> # Initializing a YOLOS hustvl/yolos-base style configuration
>>> configuration = YolosConfig()

>>> # Initializing a model (with random weights) from the hustvl/yolos-base style configuration
>>> model = YolosModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## YolosImageProcessor

### class transformers.YolosImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/image_processing_yolos.py#L726)

( format: typing.Union[str, transformers.image\_utils.AnnotationFormat] = <AnnotationFormat.COCO\_DETECTION: 'coco\_detection'> do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_annotations: typing.Optional[bool] = None do\_pad: bool = True pad\_size: typing.Optional[dict[str, int]] = None \*\*kwargs  )

Parameters

* **format** (`str`, *optional*, defaults to `"coco_detection"`) —
  Data format of the annotations. One of “coco\_detection” or “coco\_panoptic”.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Controls whether to resize the image’s (height, width) dimensions to the specified `size`. Can be
  overridden by the `do_resize` parameter in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"shortest_edge" -- 800, "longest_edge": 1333}`):
  Size of the image’s `(height, width)` dimensions after resizing. Can be overridden by the `size` parameter
  in the `preprocess` method. Available options are:
  + `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
    Do NOT keep the aspect ratio.
  + `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
    the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
    less or equal to `longest_edge`.
  + `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
    aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
    `max_width`.
* **resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`) —
  Resampling filter to use if resizing the image.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
  `preprocess` method.
* **do\_normalize** —
  Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
  `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`) —
  Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
  channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`) —
  Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
  for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
  method. If `True`, padding will be applied to the bottom and right of the image with zeros.
  If `pad_size` is provided, the image will be padded to the specified dimensions.
  Otherwise, the image will be padded to the maximum height and width of the batch.
* **pad\_size** (`dict[str, int]`, *optional*) —
  The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
  height and width in the batch.

Constructs a Detr image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/image_processing_yolos.py#L1179)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] annotations: typing.Union[dict[str, typing.Union[int, str, list[dict]]], list[dict[str, typing.Union[int, str, list[dict]]]], NoneType] = None return\_segmentation\_masks: typing.Optional[bool] = None masks\_path: typing.Union[str, pathlib.Path, NoneType] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Union[int, float, NoneType] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_annotations: typing.Optional[bool] = None do\_pad: typing.Optional[bool] = None format: typing.Union[str, transformers.image\_utils.AnnotationFormat, NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None pad\_size: typing.Optional[dict[str, int]] = None \*\*kwargs  )

Parameters

* **images** (`ImageInput`) —
  Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging
  from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **annotations** (`AnnotationType` or `list[AnnotationType]`, *optional*) —
  List of annotations associated with the image or batch of images. If annotation is for object
  detection, the annotations should be a dictionary with the following keys:
  + “image\_id” (`int`): The image id.
  + “annotations” (`list[Dict]`): List of annotations for an image. Each annotation should be a
    dictionary. An image can have no annotations, in which case the list should be empty.
    If annotation is for segmentation, the annotations should be a dictionary with the following keys:
  + “image\_id” (`int`): The image id.
  + “segments\_info” (`list[Dict]`): List of segments for an image. Each segment should be a dictionary.
    An image can have no segments, in which case the list should be empty.
  + “file\_name” (`str`): The file name of the image.
* **return\_segmentation\_masks** (`bool`, *optional*, defaults to self.return\_segmentation\_masks) —
  Whether to return segmentation masks.
* **masks\_path** (`str` or `pathlib.Path`, *optional*) —
  Path to the directory containing the segmentation masks.
* **do\_resize** (`bool`, *optional*, defaults to self.do\_resize) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to self.size) —
  Size of the image’s `(height, width)` dimensions after resizing. Available options are:
  + `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
    Do NOT keep the aspect ratio.
  + `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
    the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
    less or equal to `longest_edge`.
  + `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
    aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
    `max_width`.
* **resample** (`PILImageResampling`, *optional*, defaults to self.resample) —
  Resampling filter to use when resizing the image.
* **do\_rescale** (`bool`, *optional*, defaults to self.do\_rescale) —
  Whether to rescale the image.
* **rescale\_factor** (`float`, *optional*, defaults to self.rescale\_factor) —
  Rescale factor to use when rescaling the image.
* **do\_normalize** (`bool`, *optional*, defaults to self.do\_normalize) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to self.image\_mean) —
  Mean to use when normalizing the image.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to self.image\_std) —
  Standard deviation to use when normalizing the image.
* **do\_convert\_annotations** (`bool`, *optional*, defaults to self.do\_convert\_annotations) —
  Whether to convert the annotations to the format expected by the model. Converts the bounding
  boxes from the format `(top_left_x, top_left_y, width, height)` to `(center_x, center_y, width, height)`
  and in relative coordinates.
* **do\_pad** (`bool`, *optional*, defaults to self.do\_pad) —
  Whether to pad the image. If `True`, padding will be applied to the bottom and right of
  the image with zeros. If `pad_size` is provided, the image will be padded to the specified
  dimensions. Otherwise, the image will be padded to the maximum height and width of the batch.
* **format** (`str` or `AnnotationFormat`, *optional*, defaults to self.format) —
  Format of the annotations.
* **return\_tensors** (`str` or `TensorType`, *optional*, defaults to self.return\_tensors) —
  Type of tensors to return. If `None`, will return the list of images.
* **data\_format** (`str` or `ChannelDimension`, *optional*, defaults to self.data\_format) —
  The channel dimension format of the image. If not provided, it will be the same as the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **pad\_size** (`dict[str, int]`, *optional*) —
  The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
  height and width in the batch.

Preprocess an image or a batch of images so that it can be used by the model.

## YolosImageProcessorFast

### class transformers.YolosImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/image_processing_yolos_fast.py#L338)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.yolos.image\_processing\_yolos\_fast.YolosFastImageProcessorKwargs]  )

Constructs a fast Yolos image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/image_processing_yolos_fast.py#L616)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] annotations: typing.Union[dict[str, typing.Union[int, str, list[dict]]], list[dict[str, typing.Union[int, str, list[dict]]]], NoneType] = None masks\_path: typing.Union[str, pathlib.Path, NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.yolos.image\_processing\_yolos\_fast.YolosFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **annotations** (`AnnotationType` or `list[AnnotationType]`, *optional*) —
  List of annotations associated with the image or batch of images. If annotation is for object
  detection, the annotations should be a dictionary with the following keys:
  + “image\_id” (`int`): The image id.
  + “annotations” (`list[Dict]`): List of annotations for an image. Each annotation should be a
    dictionary. An image can have no annotations, in which case the list should be empty.
    If annotation is for segmentation, the annotations should be a dictionary with the following keys:
  + “image\_id” (`int`): The image id.
  + “segments\_info” (`list[Dict]`): List of segments for an image. Each segment should be a dictionary.
    An image can have no segments, in which case the list should be empty.
  + “file\_name” (`str`): The file name of the image.
* **masks\_path** (`str` or `pathlib.Path`, *optional*) —
  Path to the directory containing the segmentation masks.
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
* **format** (`str`, *optional*, defaults to `AnnotationFormat.COCO_DETECTION`) —
  Data format of the annotations. One of “coco\_detection” or “coco\_panoptic”.
* **do\_convert\_annotations** (`bool`, *optional*, defaults to `True`) —
  Controls whether to convert the annotations to the format expected by the YOLOS model. Converts the
  bounding boxes to the format `(center_x, center_y, width, height)` and in the range `[0, 1]`.
  Can be overridden by the `do_convert_annotations` parameter in the `preprocess` method.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
  method. If `True`, padding will be applied to the bottom and right of the image with zeros.
  If `pad_size` is provided, the image will be padded to the specified dimensions.
  Otherwise, the image will be padded to the maximum height and width of the batch.
* **pad\_size** (`dict[str, int]`, *optional*) —
  The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
  height and width in the batch.
* **return\_segmentation\_masks** (`bool`, *optional*, defaults to `False`) —
  Whether to return segmentation masks.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/image_processing_yolos_fast.py#L586)

( image: Tensor padded\_size: tuple annotation: typing.Optional[dict[str, typing.Any]] = None update\_bboxes: bool = True fill: int = 0  )

#### post\_process\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/image_processing_yolos_fast.py#L816)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None top\_k: int = 100  ) → `list[Dict]`

Parameters

* **outputs** (`YolosObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*) —
  Score threshold to keep object detection predictions.
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  (height, width) of each image in the batch. If left to None, predictions will not be resized.
* **top\_k** (`int`, *optional*, defaults to 100) —
  Keep only top k bounding boxes before filtering by thresholding.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
in the batch as predicted by the model.

Converts the raw output of [YolosForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosForObjectDetection) into final bounding boxes in (top\_left\_x,
top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## YolosFeatureExtractor

### class transformers.YolosFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/feature_extraction_yolos.py#L38)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/image_processing_yolos.py#L1093)

( images: list annotations: typing.Optional[list[dict[str, typing.Any]]] = None constant\_values: typing.Union[float, collections.abc.Iterable[float]] = 0 return\_pixel\_mask: bool = False return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None update\_bboxes: bool = True pad\_size: typing.Optional[dict[str, int]] = None  )

Parameters

* **image** (`np.ndarray`) —
  Image to pad.
* **annotations** (`list[dict[str, any]]`, *optional*) —
  Annotations to pad along with the images. If provided, the bounding boxes will be updated to match the
  padded images.
* **constant\_values** (`float` or `Iterable[float]`, *optional*) —
  The value to use for the padding if `mode` is `"constant"`.
* **return\_pixel\_mask** (`bool`, *optional*, defaults to `True`) —
  Whether to return a pixel mask.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`str` or `ChannelDimension`, *optional*) —
  The channel dimension format of the image. If not provided, it will be the same as the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format of the input image. If not provided, it will be inferred.
* **update\_bboxes** (`bool`, *optional*, defaults to `True`) —
  Whether to update the bounding boxes in the annotations to match the padded images. If the
  bounding boxes have not been converted to relative coordinates and `(centre_x, centre_y, width, height)`
  format, the bounding boxes will not be updated.
* **pad\_size** (`dict[str, int]`, *optional*) —
  The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
  height and width in the batch.

Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width
in the batch and optionally returns their corresponding pixel mask.

#### post\_process\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/image_processing_yolos.py#L1483)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None  ) → `list[Dict]`

Parameters

* **outputs** (`YolosObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*) —
  Score threshold to keep object detection predictions.
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  `(height, width)` of each image in the batch. If unset, predictions will not be resized.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
in the batch as predicted by the model.

Converts the raw output of [YolosForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## YolosModel

### class transformers.YolosModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/modeling_yolos.py#L486)

( config: YolosConfig add\_pooling\_layer: bool = True  )

Parameters

* **config** ([YolosConfig](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The bare Yolos Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/modeling_yolos.py#L519)

( pixel\_values: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [YolosImageProcessor](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosImageProcessor). See [YolosImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [YolosImageProcessor](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosImageProcessor) for processing images).
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([YolosConfig](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [YolosModel](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


## YolosForObjectDetection

### class transformers.YolosForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/modeling_yolos.py#L592)

( config: YolosConfig  )

Parameters

* **config** ([YolosConfig](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

YOLOS Model (consisting of a ViT encoder) with object detection heads on top, for tasks such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yolos/modeling_yolos.py#L619)

( pixel\_values: FloatTensor labels: typing.Optional[list[dict]] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.yolos.modeling_yolos.YolosObjectDetectionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [YolosImageProcessor](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosImageProcessor). See [YolosImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [YolosImageProcessor](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosImageProcessor) for processing images).
* **labels** (`list[Dict]` of len `(batch_size,)`, *optional*) —
  Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
  following 2 keys: `'class_labels'` and `'boxes'` (the class labels and bounding boxes of an image in the
  batch respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

Returns

`transformers.models.yolos.modeling_yolos.YolosObjectDetectionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.yolos.modeling_yolos.YolosObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([YolosConfig](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) — A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`) — Classification logits (including no-object) for all queries.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use `post_process()` to retrieve the unnormalized bounding
  boxes.
* **auxiliary\_outputs** (`list[Dict]`, *optional*) — Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
  and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
  `pred_boxes`) for each decoder layer.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [YolosForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/yolos#transformers.YolosForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
>>> model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
>>> target_sizes = torch.tensor([image.size[::-1]])
>>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
...     0
... ]

>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     print(
...         f"Detected {model.config.id2label[label.item()]} with confidence "
...         f"{round(score.item(), 3)} at location {box}"
...     )
Detected remote with confidence 0.991 at location [46.48, 72.78, 178.98, 119.3]
Detected remote with confidence 0.908 at location [336.48, 79.27, 368.23, 192.36]
Detected cat with confidence 0.934 at location [337.18, 18.06, 638.14, 373.09]
Detected cat with confidence 0.979 at location [10.93, 53.74, 313.41, 470.67]
Detected remote with confidence 0.974 at location [41.63, 72.23, 178.09, 119.99]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/yolos.md)
