*This model was released on 2022-12-12 and added to Hugging Face Transformers on 2023-06-20.*

# DETA

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

This model is in maintenance mode only, we don’t accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

## Overview

The DETA model was proposed in [NMS Strikes Back](https://huggingface.co/papers/2212.06137) by Jeffrey Ouyang-Zhang, Jang Hyun Cho, Xingyi Zhou, Philipp Krähenbühl.
DETA (short for Detection Transformers with Assignment) improves [Deformable DETR](deformable_detr) by replacing the one-to-one bipartite Hungarian matching loss
with one-to-many label assignments used in traditional detectors with non-maximum suppression (NMS). This leads to significant gains of up to 2.5 mAP.

The abstract from the paper is the following:

*Detection Transformer (DETR) directly transforms queries to unique objects by using one-to-one bipartite matching during training and enables end-to-end object detection. Recently, these models have surpassed traditional detectors on COCO with undeniable elegance. However, they differ from traditional detectors in multiple designs, including model architecture and training schedules, and thus the effectiveness of one-to-one matching is not fully understood. In this work, we conduct a strict comparison between the one-to-one Hungarian matching in DETRs and the one-to-many label assignments in traditional detectors with non-maximum supervision (NMS). Surprisingly, we observe one-to-many assignments with NMS consistently outperform standard one-to-one matching under the same setting, with a significant gain of up to 2.5 mAP. Our detector that trains Deformable-DETR with traditional IoU-based label assignment achieved 50.2 COCO mAP within 12 epochs (1x schedule) with ResNet50 backbone, outperforming all existing traditional or transformer-based detectors in this setting. On multiple datasets, schedules, and architectures, we consistently show bipartite matching is unnecessary for performant detection transformers. Furthermore, we attribute the success of detection transformers to their expressive transformer architecture.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/deta_architecture.jpg) DETA overview. Taken from the [original paper](https://huggingface.co/papers/2212.06137).

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/jozhang97/DETA).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DETA.

* Demo notebooks for DETA can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETA).
* Scripts for finetuning [DetaForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaForObjectDetection) with [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
* See also: [Object detection task guide](../tasks/object_detection).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DetaConfig

### class transformers.DetaConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/deta/configuration_deta.py#L25)

( backbone\_config = None backbone = None use\_pretrained\_backbone = False use\_timm\_backbone = False backbone\_kwargs = None num\_queries = 900 max\_position\_embeddings = 2048 encoder\_layers = 6 encoder\_ffn\_dim = 2048 encoder\_attention\_heads = 8 decoder\_layers = 6 decoder\_ffn\_dim = 1024 decoder\_attention\_heads = 8 encoder\_layerdrop = 0.0 is\_encoder\_decoder = True activation\_function = 'relu' d\_model = 256 dropout = 0.1 attention\_dropout = 0.0 activation\_dropout = 0.0 init\_std = 0.02 init\_xavier\_std = 1.0 return\_intermediate = True auxiliary\_loss = False position\_embedding\_type = 'sine' num\_feature\_levels = 5 encoder\_n\_points = 4 decoder\_n\_points = 4 two\_stage = True two\_stage\_num\_proposals = 300 with\_box\_refine = True assign\_first\_stage = True assign\_second\_stage = True class\_cost = 1 bbox\_cost = 5 giou\_cost = 2 mask\_loss\_coefficient = 1 dice\_loss\_coefficient = 1 bbox\_loss\_coefficient = 5 giou\_loss\_coefficient = 2 eos\_coefficient = 0.1 focal\_alpha = 0.25 disable\_custom\_kernels = True \*\*kwargs  )

Parameters

* **backbone\_config** (`PretrainedConfig` or `dict`, *optional*, defaults to `ResNetConfig()`) —
  The configuration of the backbone model.
* **backbone** (`str`, *optional*) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, `False`) —
  Whether to use pretrained weights for the backbone.
* **use\_timm\_backbone** (`bool`, *optional*, `False`) —
  Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
  library.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **num\_queries** (`int`, *optional*, defaults to 900) —
  Number of object queries, i.e. detection slots. This is the maximal number of objects [DetaModel](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaModel) can
  detect in a single image. In case `two_stage` is set to `True`, we use `two_stage_num_proposals` instead.
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
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 2048) —
  Dimension of the “intermediate” (often named feed-forward) layer in decoder.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 2048) —
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
* **num\_feature\_levels** (`int`, *optional*, defaults to 5) —
  The number of input feature levels.
* **encoder\_n\_points** (`int`, *optional*, defaults to 4) —
  The number of sampled keys in each feature level for each attention head in the encoder.
* **decoder\_n\_points** (`int`, *optional*, defaults to 4) —
  The number of sampled keys in each feature level for each attention head in the decoder.
* **two\_stage** (`bool`, *optional*, defaults to `True`) —
  Whether to apply a two-stage deformable DETR, where the region proposals are also generated by a variant of
  DETA, which are further fed into the decoder for iterative bounding box refinement.
* **two\_stage\_num\_proposals** (`int`, *optional*, defaults to 300) —
  The number of region proposals to be generated, in case `two_stage` is set to `True`.
* **with\_box\_refine** (`bool`, *optional*, defaults to `True`) —
  Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
  based on the predictions from the previous layer.
* **focal\_alpha** (`float`, *optional*, defaults to 0.25) —
  Alpha parameter in the focal loss.
* **assign\_first\_stage** (`bool`, *optional*, defaults to `True`) —
  Whether to assign each prediction i to the highest overlapping ground truth object if the overlap is larger than a threshold 0.7.
* **assign\_second\_stage** (`bool`, *optional*, defaults to `True`) —
  Whether to assign second assignment procedure in the second stage closely follows the first stage assignment procedure.
* **disable\_custom\_kernels** (`bool`, *optional*, defaults to `True`) —
  Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
  kernels are not supported by PyTorch ONNX export.

This is the configuration class to store the configuration of a [DetaModel](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaModel). It is used to instantiate a DETA
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the DETA
[SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import DetaConfig, DetaModel

>>> # Initializing a DETA SenseTime/deformable-detr style configuration
>>> configuration = DetaConfig()

>>> # Initializing a model (with random weights) from the SenseTime/deformable-detr style configuration
>>> model = DetaModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## DetaImageProcessor

### class transformers.DetaImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/deta/image_processing_deta.py#L498)

( format: typing.Union[str, transformers.image\_utils.AnnotationFormat] = <AnnotationFormat.COCO\_DETECTION: 'coco\_detection'> do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_annotations: bool = True do\_pad: bool = True pad\_size: typing.Optional[dict[str, int]] = None \*\*kwargs  )

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/deta/image_processing_deta.py#L889)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] annotations: typing.Union[list[dict], list[list[dict]], NoneType] = None return\_segmentation\_masks: typing.Optional[bool] = None masks\_path: typing.Union[str, pathlib.Path, NoneType] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Union[int, float, NoneType] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_annotations: typing.Optional[bool] = None do\_pad: typing.Optional[bool] = None format: typing.Union[str, transformers.image\_utils.AnnotationFormat, NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None pad\_size: typing.Optional[dict[str, int]] = None \*\*kwargs  )

Parameters

* **images** (`ImageInput`) —
  Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging
  from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **annotations** (`list[Dict]` or `list[list[Dict]]`, *optional*) —
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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/deta/image_processing_deta.py#L1144)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None nms\_threshold: float = 0.7  ) → `list[Dict]`

Parameters

* **outputs** (`DetrObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.5) —
  Score threshold to keep object detection predictions.
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  (height, width) of each image in the batch. If left to None, predictions will not be resized.
* **nms\_threshold** (`float`, *optional*, defaults to 0.7) —
  NMS threshold.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
in the batch as predicted by the model.

Converts the output of [DetaForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## DetaModel

### class transformers.DetaModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/deta/modeling_deta.py#L1425)

( config: DetaConfig  )

Parameters

* **config** ([DetaConfig](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DETA Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/deta/modeling_deta.py#L1581)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deprecated.deta.modeling_deta.DetaModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) —
  Pixel values. Padding will be ignored by default should you provide it.

  Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See `AutoImageProcessor.__call__()` for details.
* **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*) —
  Not used by default. Can be used to mask object queries.
* **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) —
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

`transformers.models.deprecated.deta.modeling_deta.DetaModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deprecated.deta.modeling_deta.DetaModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DetaConfig](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaConfig)) and inputs.

* **init\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Initial reference points sent through the Transformer decoder.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **intermediate\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`) — Stacked intermediate hidden states (output of each layer of the decoder).
* **intermediate\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`) — Stacked intermediate reference points (reference points of each layer of the decoder).
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
  plus the initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries, num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
  layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **enc\_outputs\_class** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
  picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
  foreground and background).
* **enc\_outputs\_coord\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Logits of predicted bounding boxes coordinates in the first stage.
* **output\_proposals** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.two_stage=True`) — Logits of proposal bounding boxes coordinates in the gen\_encoder\_output\_proposals.

The [DetaModel](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, DetaModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-swin-large-o365")
>>> model = DetaModel.from_pretrained("jozhang97/deta-swin-large-o365", two_stage=False)

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 900, 256]
```

## DetaForObjectDetection

### class transformers.DetaForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/deta/modeling_deta.py#L1840)

( config: DetaConfig  )

Parameters

* **config** ([DetaConfig](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DETA Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks
such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/deta/modeling_deta.py#L1897)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[list[dict]] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.deprecated.deta.modeling_deta.DetaObjectDetectionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) —
  Pixel values. Padding will be ignored by default should you provide it.

  Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See `AutoImageProcessor.__call__()` for details.
* **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*) —
  Not used by default. Can be used to mask object queries.
* **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, *optional*) —
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
* **labels** (`list[Dict]` of len `(batch_size,)`, *optional*) —
  Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
  following 2 keys: ‘class\_labels’ and ‘boxes’ (the class labels and bounding boxes of an image in the batch
  respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

Returns

`transformers.models.deprecated.deta.modeling_deta.DetaObjectDetectionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.deprecated.deta.modeling_deta.DetaObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DetaConfig](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) — A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`) — Classification logits (including no-object) for all queries.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use `~DetaProcessor.post_process_object_detection` to retrieve the
  unnormalized bounding boxes.
* **auxiliary\_outputs** (`list[Dict]`, *optional*) — Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
  and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
  `pred_boxes`) for each decoder layer.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
  plus the initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries, num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
  average in the self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
  layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_heads, 4, 4)`. Attentions weights of the encoder, after the attention softmax, used to compute the weighted average
  in the self-attention heads.
* **intermediate\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`) — Stacked intermediate hidden states (output of each layer of the decoder).
* **intermediate\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`) — Stacked intermediate reference points (reference points of each layer of the decoder).
* **init\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Initial reference points sent through the Transformer decoder.
* **enc\_outputs\_class** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
  picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
  foreground and background).
* **enc\_outputs\_coord\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Logits of predicted bounding boxes coordinates in the first stage.
* **output\_proposals** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.two_stage=True`) — Logits of proposal bounding boxes coordinates in the gen\_encoder\_output\_proposals.

The [DetaForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/deta#transformers.DetaForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, DetaForObjectDetection
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-swin-large")
>>> model = DetaForObjectDetection.from_pretrained("jozhang97/deta-swin-large")

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
Detected cat with confidence 0.802 at location [9.87, 54.36, 316.93, 473.44]
Detected cat with confidence 0.795 at location [346.62, 24.35, 639.62, 373.2]
Detected remote with confidence 0.725 at location [40.41, 73.36, 175.77, 117.29]
Detected remote with confidence 0.638 at location [333.34, 76.81, 370.22, 187.94]
Detected couch with confidence 0.584 at location [0.03, 0.99, 640.02, 474.93]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deta.md)
