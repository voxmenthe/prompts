*This model was released on 2020-10-08 and added to Hugging Face Transformers on 2022-09-14.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# Deformable DETR

[Deformable DETR](https://huggingface.co/papers/2010.04159) improves on the original [DETR](./detr) by using a deformable attention module. This mechanism selectively attends to a small set of key sampling points around a reference. It improves training speed and improves accuracy.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deformable_detr_architecture.png) Deformable DETR architecture. Taken from the [original paper](https://huggingface.co/papers/2010.04159).

You can find all the available Deformable DETR checkpoints under the [SenseTime](https://huggingface.co/SenseTime) organization.

This model was contributed by [nielsr](https://huggingface.co/nielsr).

Click on the Deformable DETR models in the right sidebar for more examples of how to apply Deformable DETR to different object detection and segmentation tasks.

The example below demonstrates how to perform object detection with the [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) and the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
from transformers import pipeline
import torch

pipeline = pipeline(
    "object-detection", 
    model="SenseTime/deformable-detr",
    dtype=torch.float16,
    device_map=0
)

pipeline("http://images.cocodataset.org/val2017/000000039769.jpg")
```

## Resources

* Refer to this set of [notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Deformable-DETR) for inference and fine-tuning [DeformableDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) on a custom dataset.

## DeformableDetrImageProcessor

### class transformers.DeformableDetrImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/image_processing_deformable_detr.py#L805)

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
* **do\_convert\_annotations** (`bool`, *optional*, defaults to `True`) —
  Controls whether to convert the annotations to the format expected by the DETR model. Converts the
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

Constructs a Deformable DETR image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/image_processing_deformable_detr.py#L1264)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] annotations: typing.Union[dict[str, typing.Union[int, str, list[dict]]], list[dict[str, typing.Union[int, str, list[dict]]]], NoneType] = None return\_segmentation\_masks: typing.Optional[bool] = None masks\_path: typing.Union[str, pathlib.Path, NoneType] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Union[int, float, NoneType] = None do\_normalize: typing.Optional[bool] = None do\_convert\_annotations: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: typing.Optional[bool] = None format: typing.Union[str, transformers.image\_utils.AnnotationFormat, NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None pad\_size: typing.Optional[dict[str, int]] = None \*\*kwargs  )

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
* **do\_convert\_annotations** (`bool`, *optional*, defaults to self.do\_convert\_annotations) —
  Whether to convert the annotations to the format expected by the model. Converts the bounding
  boxes from the format `(top_left_x, top_left_y, width, height)` to `(center_x, center_y, width, height)`
  and in relative coordinates.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to self.image\_mean) —
  Mean to use when normalizing the image.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to self.image\_std) —
  Standard deviation to use when normalizing the image.
* **do\_pad** (`bool`, *optional*, defaults to self.do\_pad) —
  Whether to pad the image. If `True`, padding will be applied to the bottom and right of
  the image with zeros. If `pad_size` is provided, the image will be padded to the specified
  dimensions. Otherwise, the image will be padded to the maximum height and width of the batch.
* **format** (`str` or `AnnotationFormat`, *optional*, defaults to self.format) —
  Format of the annotations.
* **return\_tensors** (`str` or `TensorType`, *optional*, defaults to self.return\_tensors) —
  Type of tensors to return. If `None`, will return the list of images.
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
* **pad\_size** (`dict[str, int]`, *optional*) —
  The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
  height and width in the batch.

Preprocess an image or a batch of images so that it can be used by the model.

#### post\_process\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/image_processing_deformable_detr.py#L1574)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None top\_k: int = 100  ) → `list[Dict]`

Parameters

* **outputs** (`DetrObjectDetectionOutput`) —
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

Converts the raw output of [DeformableDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) into final bounding boxes in (top\_left\_x,
top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## DeformableDetrImageProcessorFast

### class transformers.DeformableDetrImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/image_processing_deformable_detr_fast.py#L290)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.deformable\_detr.image\_processing\_deformable\_detr\_fast.DeformableDetrFastImageProcessorKwargs]  )

Constructs a fast Deformable Detr image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/image_processing_deformable_detr_fast.py#L568)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] annotations: typing.Union[dict[str, typing.Union[int, str, list[dict]]], list[dict[str, typing.Union[int, str, list[dict]]]], NoneType] = None masks\_path: typing.Union[str, pathlib.Path, NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.deformable\_detr.image\_processing\_deformable\_detr\_fast.DeformableDetrFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
  Controls whether to convert the annotations to the format expected by the DEFORMABLE\_DETR model. Converts the
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

#### post\_process\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/image_processing_deformable_detr_fast.py#L768)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None top\_k: int = 100  ) → `list[Dict]`

Parameters

* **outputs** (`DetrObjectDetectionOutput`) —
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

Converts the raw output of [DeformableDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) into final bounding boxes in (top\_left\_x,
top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## DeformableDetrFeatureExtractor

### class transformers.DeformableDetrFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/feature_extraction_deformable_detr.py#L38)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

#### post\_process\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/image_processing_deformable_detr.py#L1574)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None top\_k: int = 100  ) → `list[Dict]`

Parameters

* **outputs** (`DetrObjectDetectionOutput`) —
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

Converts the raw output of [DeformableDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) into final bounding boxes in (top\_left\_x,
top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## DeformableDetrConfig

### class transformers.DeformableDetrConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/configuration_deformable_detr.py#L26)

( use\_timm\_backbone = True backbone\_config = None num\_channels = 3 num\_queries = 300 max\_position\_embeddings = 1024 encoder\_layers = 6 encoder\_ffn\_dim = 1024 encoder\_attention\_heads = 8 decoder\_layers = 6 decoder\_ffn\_dim = 1024 decoder\_attention\_heads = 8 encoder\_layerdrop = 0.0 is\_encoder\_decoder = True activation\_function = 'relu' d\_model = 256 dropout = 0.1 attention\_dropout = 0.0 activation\_dropout = 0.0 init\_std = 0.02 init\_xavier\_std = 1.0 return\_intermediate = True auxiliary\_loss = False position\_embedding\_type = 'sine' backbone = 'resnet50' use\_pretrained\_backbone = True backbone\_kwargs = None dilation = False num\_feature\_levels = 4 encoder\_n\_points = 4 decoder\_n\_points = 4 two\_stage = False two\_stage\_num\_proposals = 300 with\_box\_refine = False class\_cost = 1 bbox\_cost = 5 giou\_cost = 2 mask\_loss\_coefficient = 1 dice\_loss\_coefficient = 1 bbox\_loss\_coefficient = 5 giou\_loss\_coefficient = 2 eos\_coefficient = 0.1 focal\_alpha = 0.25 disable\_custom\_kernels = False \*\*kwargs  )

Parameters

* **use\_timm\_backbone** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone)
  API.
* **backbone\_config** (`PretrainedConfig` or `dict`, *optional*) —
  The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
  case it will default to `ResNetConfig()`.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **num\_queries** (`int`, *optional*, defaults to 300) —
  Number of object queries, i.e. detection slots. This is the maximal number of objects
  [DeformableDetrModel](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrModel) can detect in a single image. In case `two_stage` is set to `True`, we use
  `two_stage_num_proposals` instead.
* **d\_model** (`int`, *optional*, defaults to 256) —
  Dimension of the layers.
* **encoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of encoder layers.
* **decoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of decoder layers.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **decoder\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 1024) —
  Dimension of the “intermediate” (often named feed-forward) layer in decoder.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 1024) —
  Dimension of the “intermediate” (often named feed-forward) layer in decoder.
* **activation\_function** (`str` or `function`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **activation\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for activations inside the fully connected layer.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **init\_xavier\_std** (`float`, *optional*, defaults to 1) —
  The scaling factor used for the Xavier initialization gain in the HM Attention map module.
* **encoder\_layerdrop** (`float`, *optional*, defaults to 0.0) —
  The LayerDrop probability for the encoder. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **auxiliary\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"sine"`) —
  Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
* **backbone** (`str`, *optional*, defaults to `"resnet50"`) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `True`) —
  Whether to use pretrained weights for the backbone.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **dilation** (`bool`, *optional*, defaults to `False`) —
  Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when
  `use_timm_backbone` = `True`.
* **class\_cost** (`float`, *optional*, defaults to 1) —
  Relative weight of the classification error in the Hungarian matching cost.
* **bbox\_cost** (`float`, *optional*, defaults to 5) —
  Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
* **giou\_cost** (`float`, *optional*, defaults to 2) —
  Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
* **mask\_loss\_coefficient** (`float`, *optional*, defaults to 1) —
  Relative weight of the Focal loss in the panoptic segmentation loss.
* **dice\_loss\_coefficient** (`float`, *optional*, defaults to 1) —
  Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
* **bbox\_loss\_coefficient** (`float`, *optional*, defaults to 5) —
  Relative weight of the L1 bounding box loss in the object detection loss.
* **giou\_loss\_coefficient** (`float`, *optional*, defaults to 2) —
  Relative weight of the generalized IoU loss in the object detection loss.
* **eos\_coefficient** (`float`, *optional*, defaults to 0.1) —
  Relative classification weight of the ‘no-object’ class in the object detection loss.
* **num\_feature\_levels** (`int`, *optional*, defaults to 4) —
  The number of input feature levels.
* **encoder\_n\_points** (`int`, *optional*, defaults to 4) —
  The number of sampled keys in each feature level for each attention head in the encoder.
* **decoder\_n\_points** (`int`, *optional*, defaults to 4) —
  The number of sampled keys in each feature level for each attention head in the decoder.
* **two\_stage** (`bool`, *optional*, defaults to `False`) —
  Whether to apply a two-stage deformable DETR, where the region proposals are also generated by a variant of
  Deformable DETR, which are further fed into the decoder for iterative bounding box refinement.
* **two\_stage\_num\_proposals** (`int`, *optional*, defaults to 300) —
  The number of region proposals to be generated, in case `two_stage` is set to `True`.
* **with\_box\_refine** (`bool`, *optional*, defaults to `False`) —
  Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
  based on the predictions from the previous layer.
* **focal\_alpha** (`float`, *optional*, defaults to 0.25) —
  Alpha parameter in the focal loss.
* **disable\_custom\_kernels** (`bool`, *optional*, defaults to `False`) —
  Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
  kernels are not supported by PyTorch ONNX export.

This is the configuration class to store the configuration of a [DeformableDetrModel](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrModel). It is used to instantiate
a Deformable DETR model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Deformable DETR
[SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import DeformableDetrConfig, DeformableDetrModel

>>> # Initializing a Deformable DETR SenseTime/deformable-detr style configuration
>>> configuration = DeformableDetrConfig()

>>> # Initializing a model (with random weights) from the SenseTime/deformable-detr style configuration
>>> model = DeformableDetrModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## DeformableDetrModel

### class transformers.DeformableDetrModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L1297)

( config: DeformableDetrConfig  )

Parameters

* **config** ([DeformableDetrConfig](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L1465)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DeformableDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrImageProcessor). See [DeformableDetrImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DeformableDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrImageProcessor) for processing images).
* **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*) —
  Not used by default. Can be used to mask object queries.
* **encoder\_outputs** (`torch.FloatTensor`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
  can choose to directly pass a flattened representation of an image.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*) —
  Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
  embedded representation.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DeformableDetrConfig](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrConfig)) and inputs.

* **init\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Initial reference points sent through the Transformer decoder.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **intermediate\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`) — Stacked intermediate hidden states (output of each layer of the decoder).
* **intermediate\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`) — Stacked intermediate reference points (reference points of each layer of the decoder).
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **enc\_outputs\_class** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
  picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
  foreground and background).
* **enc\_outputs\_coord\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Logits of predicted bounding boxes coordinates in the first stage.

The [DeformableDetrModel](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, DeformableDetrModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
>>> model = DeformableDetrModel.from_pretrained("SenseTime/deformable-detr")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 300, 256]
```

## DeformableDetrForObjectDetection

### class transformers.DeformableDetrForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L1705)

( config: DeformableDetrConfig  )

Parameters

* **config** ([DeformableDetrConfig](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Deformable DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
top, for tasks such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L1743)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[list[dict]] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrObjectDetectionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DeformableDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrImageProcessor). See [DeformableDetrImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DeformableDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrImageProcessor) for processing images).
* **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*) —
  Not used by default. Can be used to mask object queries.
* **encoder\_outputs** (`torch.FloatTensor`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
  can choose to directly pass a flattened representation of an image.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*) —
  Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
  embedded representation.
* **labels** (`list[Dict]` of len `(batch_size,)`, *optional*) —
  Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
  following 2 keys: ‘class\_labels’ and ‘boxes’ (the class labels and bounding boxes of an image in the batch
  respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrObjectDetectionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DeformableDetrConfig](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) — A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`) — Classification logits (including no-object) for all queries.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use `~DeformableDetrProcessor.post_process_object_detection` to retrieve the
  unnormalized bounding boxes.
* **auxiliary\_outputs** (`list[Dict]`, *optional*) — Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
  and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
  `pred_boxes`) for each decoder layer.
* **init\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Initial reference points sent through the Transformer decoder.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **intermediate\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`) — Stacked intermediate hidden states (output of each layer of the decoder).
* **intermediate\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`) — Stacked intermediate reference points (reference points of each layer of the decoder).
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **enc\_outputs\_class** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
  picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
  foreground and background).
* **enc\_outputs\_coord\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Logits of predicted bounding boxes coordinates in the first stage.

The [DeformableDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deformable_detr#transformers.DeformableDetrForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
>>> model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
>>> target_sizes = torch.tensor([image.size[::-1]])
>>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
...     0
... ]
>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     print(
...         f"Detected {model.config.id2label[label.item()]} with confidence "
...         f"{round(score.item(), 3)} at location {box}"
...     )
Detected cat with confidence 0.8 at location [16.5, 52.84, 318.25, 470.78]
Detected cat with confidence 0.789 at location [342.19, 24.3, 640.02, 372.25]
Detected remote with confidence 0.633 at location [40.79, 72.78, 176.76, 117.25]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deformable_detr.md)
