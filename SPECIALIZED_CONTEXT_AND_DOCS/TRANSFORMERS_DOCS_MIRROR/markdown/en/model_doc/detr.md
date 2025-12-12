*This model was released on 2020-05-26 and added to Hugging Face Transformers on 2021-06-09.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# DETR

[DETR](https://huggingface.co/papers/2005.12872) consists of a convolutional backbone followed by an encoder-decoder Transformer which can be trained end-to-end for object detection. It greatly simplifies a lot of the complexity of models like Faster-R-CNN and Mask-R-CNN, which use things like region proposals, non-maximum suppression procedure and anchor generation. Moreover, DETR can also be naturally extended to perform panoptic segmentation, by simply adding a mask head on top of the decoder outputs.

You can find all the original DETR checkpoints under the [AI at Meta](https://huggingface.co/facebook/models?search=detr) organization.

This model was contributed by [nielsr](https://huggingface.co/nielsr).

Click on the DETR models in the right sidebar for more examples of how to apply DETR to different object detection and segmentation tasks.

The example below demonstrates how to perform object detection with the [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
from transformers import pipeline
import torch

pipeline = pipeline(
    "object-detection", 
    model="facebook/detr-resnet-50",
    dtype=torch.float16,
    device_map=0
)

pipeline("http://images.cocodataset.org/val2017/000000039769.jpg")
```

How DETR works

Here’s a TLDR explaining how [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) works:

First, an image is sent through a pre-trained convolutional backbone (in the paper, the authors use ResNet-50/ResNet-101). Let’s assume we also add a batch dimension. This means that the input to the backbone is a tensor of shape `(batch_size, 3, height, width)`, assuming the image has 3 color channels (RGB). The CNN backbone outputs a new lower-resolution feature map, typically of shape `(batch_size, 2048, height/32, width/32)`. This is then projected to match the hidden dimension of the Transformer of DETR, which is `256` by default, using a `nn.Conv2D` layer. So now, we have a tensor of shape `(batch_size, 256, height/32, width/32).` Next, the feature map is flattened and transposed to obtain a tensor of shape `(batch_size, seq_len, d_model)` = `(batch_size, width/32*height/32, 256)`. So a difference with NLP models is that the sequence length is actually longer than usual, but with a smaller `d_model` (which in NLP is typically 768 or higher).

Next, this is sent through the encoder, outputting `encoder_hidden_states` of the same shape (you can consider these as image features). Next, so-called **object queries** are sent through the decoder. This is a tensor of shape `(batch_size, num_queries, d_model)`, with `num_queries` typically set to 100 and initialized with zeros. These input embeddings are learnt positional encodings that the authors refer to as object queries, and similarly to the encoder, they are added to the input of each attention layer. Each object query will look for a particular object in the image. The decoder updates these embeddings through multiple self-attention and encoder-decoder attention layers to output `decoder_hidden_states` of the same shape: `(batch_size, num_queries, d_model)`. Next, two heads are added on top for object detection: a linear layer for classifying each object query into one of the objects or “no object”, and a MLP to predict bounding boxes for each query.

The model is trained using a **bipartite matching loss**: so what we actually do is compare the predicted classes + bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N (so if an image only contains 4 objects, 96 annotations will just have a “no object” as class and “no bounding box” as bounding box). The [Hungarian matching algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) is used to find an optimal one-to-one mapping of each of the N queries to each of the N annotations. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and [generalized IoU loss](https://giou.stanford.edu/) (for the bounding boxes) are used to optimize the parameters of the model.

DETR can be naturally extended to perform panoptic segmentation (which unifies semantic segmentation and instance segmentation). [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) adds a segmentation mask head on top of [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection). The mask head can be trained either jointly, or in a two steps process, where one first trains a [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) model to detect bounding boxes around both “things” (instances) and “stuff” (background things like trees, roads, sky), then freeze all the weights and train only the mask head for 25 epochs. Experimentally, these two approaches give similar results. Note that predicting boxes is required for the training to be possible, since the Hungarian matching is computed using distances between boxes.

## Notes

* DETR uses so-called **object queries** to detect objects in an image. The number of queries determines the maximum number of objects that can be detected in a single image, and is set to 100 by default (see parameter `num_queries` of [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig)). Note that it’s good to have some slack (in COCO, the authors used 100, while the maximum number of objects in a COCO image is ~70).
* The decoder of DETR updates the query embeddings in parallel. This is different from language models like GPT-2, which use autoregressive decoding instead of parallel. Hence, no causal attention mask is used.
* DETR adds position embeddings to the hidden states at each self-attention and cross-attention layer before projecting to queries and keys. For the position embeddings of the image, one can choose between fixed sinusoidal or learned absolute position embeddings. By default, the parameter `position_embedding_type` of [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig) is set to `"sine"`.
* During training, the authors of DETR did find it helpful to use auxiliary losses in the decoder, especially to help the model output the correct number of objects of each class. If you set the parameter `auxiliary_loss` of [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig) to `True`, then prediction feedforward neural networks and Hungarian losses are added after each decoder layer (with the FFNs sharing parameters).
* If you want to train the model in a distributed environment across multiple nodes, then one should update the *num\_boxes* variable in the *DetrLoss* class of *modeling\_detr.py*. When training on multiple nodes, this should be set to the average number of target boxes across all nodes, as can be seen in the original implementation [here](https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/detr.py#L227-L232).
* [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) and [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) can be initialized with any convolutional backbone available in the [timm library](https://github.com/rwightman/pytorch-image-models). Initializing with a MobileNet backbone for example can be done by setting the `backbone` attribute of [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig) to `"tf_mobilenetv3_small_075"`, and then initializing the model with that config.
* DETR resizes the input images such that the shortest side is at least a certain amount of pixels while the longest is at most 1333 pixels. At training time, scale augmentation is used such that the shortest side is randomly set to at least 480 and at most 800 pixels. At inference time, the shortest side is set to 800. One can use [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor) to prepare images (and optional annotations in COCO format) for the model. Due to this resizing, images in a batch can have different sizes. DETR solves this by padding images up to the largest size in a batch, and by creating a pixel mask that indicates which pixels are real/which are padding. Alternatively, one can also define a custom `collate_fn` in order to batch images together, using `~transformers.DetrImageProcessor.pad_and_create_pixel_mask`.
* The size of the images will determine the amount of memory being used, and will thus determine the `batch_size`. It is advised to use a batch size of 2 per GPU. See [this Github thread](https://github.com/facebookresearch/detr/issues/150) for more info.

There are three other ways to instantiate a DETR model (depending on what you prefer):

* Option 1: Instantiate DETR with pre-trained weights for entire model


```
from transformers import DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
```

* Option 2: Instantiate DETR with randomly initialized weights for Transformer, but pre-trained weights for backbone


```
from transformers import DetrConfig, DetrForObjectDetection

config = DetrConfig()
model = DetrForObjectDetection(config)
```

* Option 3: Instantiate DETR with randomly initialized weights for backbone + Transformer


```
config = DetrConfig(use_pretrained_backbone=False)
model = DetrForObjectDetection(config)
```

As a summary, consider the following table:

| Task | Object detection | Instance segmentation | Panoptic segmentation |
| --- | --- | --- | --- |
| **Description** | Predicting bounding boxes and class labels around objects in an image | Predicting masks around objects (i.e. instances) in an image | Predicting masks around both objects (i.e. instances) as well as “stuff” (i.e. background things like trees and roads) in an image |
| **Model** | [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) | [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) | [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) |
| **Example dataset** | COCO detection | COCO detection, COCO panoptic | COCO panoptic |
| **Format of annotations to provide to** [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor) | {‘image\_id’: `int`, ‘annotations’: `list[Dict]`} each Dict being a COCO object annotation | {‘image\_id’: `int`, ‘annotations’: `list[Dict]`} (in case of COCO detection) or {‘file\_name’: `str`, ‘image\_id’: `int`, ‘segments\_info’: `list[Dict]`} (in case of COCO panoptic) | {‘file\_name’: `str`, ‘image\_id’: `int`, ‘segments\_info’: `list[Dict]`} and masks\_path (path to directory containing PNG files of the masks) |
| **Postprocessing** (i.e. converting the output of the model to Pascal VOC format) | `post_process()` | `post_process_segmentation()` | `post_process_segmentation()`, `post_process_panoptic()` |
| **evaluators** | `CocoEvaluator` with `iou_types="bbox"` | `CocoEvaluator` with `iou_types="bbox"` or `"segm"` | `CocoEvaluator` with `iou_tupes="bbox"` or `"segm"`, `PanopticEvaluator` |

* In short, one should prepare the data either in COCO detection or COCO panoptic format, then use [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor) to create `pixel_values`, `pixel_mask` and optional `labels`, which can then be used to train (or fine-tune) a model.
* For evaluation, one should first convert the outputs of the model using one of the postprocessing methods of [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor). These can be provided to either `CocoEvaluator` or `PanopticEvaluator`, which allow you to calculate metrics like mean Average Precision (mAP) and Panoptic Quality (PQ). The latter objects are implemented in the [original repository](https://github.com/facebookresearch/detr). See the [example notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR) for more info regarding evaluation.

## Resources

* Refer to these [notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR) for examples of fine-tuning [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) and [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) on a custom dataset.

## DetrConfig

### class transformers.DetrConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/configuration_detr.py#L32)

( use\_timm\_backbone = True backbone\_config = None num\_channels = 3 num\_queries = 100 encoder\_layers = 6 encoder\_ffn\_dim = 2048 encoder\_attention\_heads = 8 decoder\_layers = 6 decoder\_ffn\_dim = 2048 decoder\_attention\_heads = 8 encoder\_layerdrop = 0.0 decoder\_layerdrop = 0.0 is\_encoder\_decoder = True activation\_function = 'relu' d\_model = 256 dropout = 0.1 attention\_dropout = 0.0 activation\_dropout = 0.0 init\_std = 0.02 init\_xavier\_std = 1.0 auxiliary\_loss = False position\_embedding\_type = 'sine' backbone = 'resnet50' use\_pretrained\_backbone = True backbone\_kwargs = None dilation = False class\_cost = 1 bbox\_cost = 5 giou\_cost = 2 mask\_loss\_coefficient = 1 dice\_loss\_coefficient = 1 bbox\_loss\_coefficient = 5 giou\_loss\_coefficient = 2 eos\_coefficient = 0.1 \*\*kwargs  )

Parameters

* **use\_timm\_backbone** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone)
  API.
* **backbone\_config** (`PretrainedConfig` or `dict`, *optional*) —
  The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
  case it will default to `ResNetConfig()`.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **num\_queries** (`int`, *optional*, defaults to 100) —
  Number of object queries, i.e. detection slots. This is the maximal number of objects [DetrModel](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrModel) can
  detect in a single image. For COCO, we recommend 100 queries.
* **d\_model** (`int`, *optional*, defaults to 256) —
  This parameter is a general dimension parameter, defining dimensions for components such as the encoder layer and projection parameters in the decoder layer, among others.
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
* **decoder\_layerdrop** (`float`, *optional*, defaults to 0.0) —
  The LayerDrop probability for the decoder. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **auxiliary\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"sine"`) —
  Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
* **backbone** (`str`, *optional*, defaults to `"resnet50"`) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, `True`) —
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

This is the configuration class to store the configuration of a [DetrModel](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrModel). It is used to instantiate a DETR
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the DETR
[facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import DetrConfig, DetrModel

>>> # Initializing a DETR facebook/detr-resnet-50 style configuration
>>> configuration = DetrConfig()

>>> # Initializing a model (with random weights) from the facebook/detr-resnet-50 style configuration
>>> model = DetrModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

#### from\_backbone\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/configuration_detr.py#L263)

( backbone\_config: PretrainedConfig \*\*kwargs  ) → [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig)

Parameters

* **backbone\_config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The backbone configuration.

Returns

[DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig)

An instance of a configuration object

Instantiate a [DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig) (or a derived class) from a pre-trained backbone model configuration.

## DetrImageProcessor

### class transformers.DetrImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr.py#L790)

( format: typing.Union[str, transformers.image\_utils.AnnotationFormat] = <AnnotationFormat.COCO\_DETECTION: 'coco\_detection'> do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_annotations: typing.Optional[bool] = None do\_pad: bool = True pad\_size: typing.Optional[dict[str, int]] = None \*\*kwargs  )

Parameters

* **format** (`str`, *optional*, defaults to `"coco_detection"`) —
  Data format of the annotations. One of “coco\_detection” or “coco\_panoptic”.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Controls whether to resize the image’s `(height, width)` dimensions to the specified `size`. Can be
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
* **do\_normalize** (`bool`, *optional*, defaults to True) —
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

Constructs a Detr image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr.py#L1239)

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr.py#L1775)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None  ) → `list[Dict]`

Parameters

* **outputs** (`DetrObjectDetectionOutput`) —
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

Converts the raw output of [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr.py#L1828)

( outputs target\_sizes: typing.Optional[list[tuple[int, int]]] = None  ) → `list[torch.Tensor]`

Parameters

* **outputs** ([DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[tuple[int, int]]`, *optional*) —
  A list of tuples (`tuple[int, int]`) containing the target size (height, width) of each image in the
  batch. If unset, predictions will not be resized.

Returns

`list[torch.Tensor]`

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each
`torch.Tensor` correspond to a semantic class id.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) into semantic segmentation maps. Only supports PyTorch.

#### post\_process\_instance\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr.py#L1876)

( outputs threshold: float = 0.5 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 target\_sizes: typing.Optional[list[tuple[int, int]]] = None return\_coco\_annotation: typing.Optional[bool] = False  ) → `list[Dict]`

Parameters

* **outputs** ([DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation)) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The probability score threshold to keep predicted instance masks.
* **mask\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to use when turning the predicted masks into binary values.
* **overlap\_mask\_area\_threshold** (`float`, *optional*, defaults to 0.8) —
  The overlap mask area threshold to merge or discard small disconnected parts within each binary
  instance mask.
* **target\_sizes** (`list[Tuple]`, *optional*) —
  List of length (batch\_size), where each list item (`tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction. If unset, predictions will not be resized.
* **return\_coco\_annotation** (`bool`, *optional*) —
  Defaults to `False`. If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE)
  format.

Returns

`list[Dict]`

A list of dictionaries, one per image, each dictionary containing two keys:

* **segmentation** — A tensor of shape `(height, width)` where each pixel represents a `segment_id` or
  `list[List]` run-length encoding (RLE) of the segmentation map if return\_coco\_annotation is set to
  `True`. Set to `None` if no mask if found above `threshold`.
* **segments\_info** — A dictionary that contains additional information on each segment.
  + **id** — An integer representing the `segment_id`.
  + **label\_id** — An integer representing the label / semantic class id corresponding to `segment_id`.
  + **score** — Prediction score of segment with `segment_id`.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) into instance segmentation predictions. Only supports PyTorch.

#### post\_process\_panoptic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr.py#L1960)

( outputs threshold: float = 0.5 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 label\_ids\_to\_fuse: typing.Optional[set[int]] = None target\_sizes: typing.Optional[list[tuple[int, int]]] = None  ) → `list[Dict]`

Parameters

* **outputs** ([DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation)) —
  The outputs from [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation).
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The probability score threshold to keep predicted instance masks.
* **mask\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to use when turning the predicted masks into binary values.
* **overlap\_mask\_area\_threshold** (`float`, *optional*, defaults to 0.8) —
  The overlap mask area threshold to merge or discard small disconnected parts within each binary
  instance mask.
* **label\_ids\_to\_fuse** (`Set[int]`, *optional*) —
  The labels in this state will have all their instances be fused together. For instance we could say
  there can only be one sky in an image, but several persons, so the label ID for sky would be in that
  set, but not the one for person.
* **target\_sizes** (`list[Tuple]`, *optional*) —
  List of length (batch\_size), where each list item (`tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction in batch. If unset, predictions will not be resized.

Returns

`list[Dict]`

A list of dictionaries, one per image, each dictionary containing two keys:

* **segmentation** — a tensor of shape `(height, width)` where each pixel represents a `segment_id` or
  `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized to
  the corresponding `target_sizes` entry.
* **segments\_info** — A dictionary that contains additional information on each segment.
  + **id** — an integer representing the `segment_id`.
  + **label\_id** — An integer representing the label / semantic class id corresponding to `segment_id`.
  + **was\_fused** — a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
    Multiple instances of the same class / label were fused and assigned a single `segment_id`.
  + **score** — Prediction score of segment with `segment_id`.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) into image panoptic segmentation predictions. Only supports
PyTorch.

## DetrImageProcessorFast

### class transformers.DetrImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr_fast.py#L311)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.detr.image\_processing\_detr\_fast.DetrFastImageProcessorKwargs]  )

Constructs a fast Detr image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr_fast.py#L589)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] annotations: typing.Union[dict[str, typing.Union[int, str, list[dict]]], list[dict[str, typing.Union[int, str, list[dict]]]], NoneType] = None masks\_path: typing.Union[str, pathlib.Path, NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.detr.image\_processing\_detr\_fast.DetrFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **return\_segmentation\_masks** (`bool`, *optional*, defaults to `False`) —
  Whether to return segmentation masks.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

#### post\_process\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr_fast.py#L1016)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None  ) → `list[Dict]`

Parameters

* **outputs** (`DetrObjectDetectionOutput`) —
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

Converts the raw output of [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr_fast.py#L1070)

( outputs target\_sizes: typing.Optional[list[tuple[int, int]]] = None  ) → `list[torch.Tensor]`

Parameters

* **outputs** ([DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[tuple[int, int]]`, *optional*) —
  A list of tuples (`tuple[int, int]`) containing the target size (height, width) of each image in the
  batch. If unset, predictions will not be resized.

Returns

`list[torch.Tensor]`

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each
`torch.Tensor` correspond to a semantic class id.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) into semantic segmentation maps. Only supports PyTorch.

#### post\_process\_instance\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr_fast.py#L1118)

( outputs threshold: float = 0.5 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 target\_sizes: typing.Optional[list[tuple[int, int]]] = None return\_coco\_annotation: typing.Optional[bool] = False  ) → `list[Dict]`

Parameters

* **outputs** ([DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation)) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The probability score threshold to keep predicted instance masks.
* **mask\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to use when turning the predicted masks into binary values.
* **overlap\_mask\_area\_threshold** (`float`, *optional*, defaults to 0.8) —
  The overlap mask area threshold to merge or discard small disconnected parts within each binary
  instance mask.
* **target\_sizes** (`list[Tuple]`, *optional*) —
  List of length (batch\_size), where each list item (`tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction. If unset, predictions will not be resized.
* **return\_coco\_annotation** (`bool`, *optional*) —
  Defaults to `False`. If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE)
  format.

Returns

`list[Dict]`

A list of dictionaries, one per image, each dictionary containing two keys:

* **segmentation** — A tensor of shape `(height, width)` where each pixel represents a `segment_id` or
  `list[List]` run-length encoding (RLE) of the segmentation map if return\_coco\_annotation is set to
  `True`. Set to `None` if no mask if found above `threshold`.
* **segments\_info** — A dictionary that contains additional information on each segment.
  + **id** — An integer representing the `segment_id`.
  + **label\_id** — An integer representing the label / semantic class id corresponding to `segment_id`.
  + **score** — Prediction score of segment with `segment_id`.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) into instance segmentation predictions. Only supports PyTorch.

#### post\_process\_panoptic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr_fast.py#L1202)

( outputs threshold: float = 0.5 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 label\_ids\_to\_fuse: typing.Optional[set[int]] = None target\_sizes: typing.Optional[list[tuple[int, int]]] = None  ) → `list[Dict]`

Parameters

* **outputs** ([DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation)) —
  The outputs from [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation).
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The probability score threshold to keep predicted instance masks.
* **mask\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to use when turning the predicted masks into binary values.
* **overlap\_mask\_area\_threshold** (`float`, *optional*, defaults to 0.8) —
  The overlap mask area threshold to merge or discard small disconnected parts within each binary
  instance mask.
* **label\_ids\_to\_fuse** (`Set[int]`, *optional*) —
  The labels in this state will have all their instances be fused together. For instance we could say
  there can only be one sky in an image, but several persons, so the label ID for sky would be in that
  set, but not the one for person.
* **target\_sizes** (`list[Tuple]`, *optional*) —
  List of length (batch\_size), where each list item (`tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction in batch. If unset, predictions will not be resized.

Returns

`list[Dict]`

A list of dictionaries, one per image, each dictionary containing two keys:

* **segmentation** — a tensor of shape `(height, width)` where each pixel represents a `segment_id` or
  `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized to
  the corresponding `target_sizes` entry.
* **segments\_info** — A dictionary that contains additional information on each segment.
  + **id** — an integer representing the `segment_id`.
  + **label\_id** — An integer representing the label / semantic class id corresponding to `segment_id`.
  + **was\_fused** — a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
    Multiple instances of the same class / label were fused and assigned a single `segment_id`.
  + **score** — Prediction score of segment with `segment_id`.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) into image panoptic segmentation predictions. Only supports
PyTorch.

## DetrFeatureExtractor

### class transformers.DetrFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/feature_extraction_detr.py#L38)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

#### post\_process\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr.py#L1775)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None  ) → `list[Dict]`

Parameters

* **outputs** (`DetrObjectDetectionOutput`) —
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

Converts the raw output of [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr.py#L1828)

( outputs target\_sizes: typing.Optional[list[tuple[int, int]]] = None  ) → `list[torch.Tensor]`

Parameters

* **outputs** ([DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[tuple[int, int]]`, *optional*) —
  A list of tuples (`tuple[int, int]`) containing the target size (height, width) of each image in the
  batch. If unset, predictions will not be resized.

Returns

`list[torch.Tensor]`

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each
`torch.Tensor` correspond to a semantic class id.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) into semantic segmentation maps. Only supports PyTorch.

#### post\_process\_instance\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr.py#L1876)

( outputs threshold: float = 0.5 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 target\_sizes: typing.Optional[list[tuple[int, int]]] = None return\_coco\_annotation: typing.Optional[bool] = False  ) → `list[Dict]`

Parameters

* **outputs** ([DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation)) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The probability score threshold to keep predicted instance masks.
* **mask\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to use when turning the predicted masks into binary values.
* **overlap\_mask\_area\_threshold** (`float`, *optional*, defaults to 0.8) —
  The overlap mask area threshold to merge or discard small disconnected parts within each binary
  instance mask.
* **target\_sizes** (`list[Tuple]`, *optional*) —
  List of length (batch\_size), where each list item (`tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction. If unset, predictions will not be resized.
* **return\_coco\_annotation** (`bool`, *optional*) —
  Defaults to `False`. If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE)
  format.

Returns

`list[Dict]`

A list of dictionaries, one per image, each dictionary containing two keys:

* **segmentation** — A tensor of shape `(height, width)` where each pixel represents a `segment_id` or
  `list[List]` run-length encoding (RLE) of the segmentation map if return\_coco\_annotation is set to
  `True`. Set to `None` if no mask if found above `threshold`.
* **segments\_info** — A dictionary that contains additional information on each segment.
  + **id** — An integer representing the `segment_id`.
  + **label\_id** — An integer representing the label / semantic class id corresponding to `segment_id`.
  + **score** — Prediction score of segment with `segment_id`.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) into instance segmentation predictions. Only supports PyTorch.

#### post\_process\_panoptic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/image_processing_detr.py#L1960)

( outputs threshold: float = 0.5 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 label\_ids\_to\_fuse: typing.Optional[set[int]] = None target\_sizes: typing.Optional[list[tuple[int, int]]] = None  ) → `list[Dict]`

Parameters

* **outputs** ([DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation)) —
  The outputs from [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation).
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The probability score threshold to keep predicted instance masks.
* **mask\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to use when turning the predicted masks into binary values.
* **overlap\_mask\_area\_threshold** (`float`, *optional*, defaults to 0.8) —
  The overlap mask area threshold to merge or discard small disconnected parts within each binary
  instance mask.
* **label\_ids\_to\_fuse** (`Set[int]`, *optional*) —
  The labels in this state will have all their instances be fused together. For instance we could say
  there can only be one sky in an image, but several persons, so the label ID for sky would be in that
  set, but not the one for person.
* **target\_sizes** (`list[Tuple]`, *optional*) —
  List of length (batch\_size), where each list item (`tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction in batch. If unset, predictions will not be resized.

Returns

`list[Dict]`

A list of dictionaries, one per image, each dictionary containing two keys:

* **segmentation** — a tensor of shape `(height, width)` where each pixel represents a `segment_id` or
  `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized to
  the corresponding `target_sizes` entry.
* **segments\_info** — A dictionary that contains additional information on each segment.
  + **id** — an integer representing the `segment_id`.
  + **label\_id** — An integer representing the label / semantic class id corresponding to `segment_id`.
  + **was\_fused** — a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
    Multiple instances of the same class / label were fused and assigned a single `segment_id`.
  + **score** — Prediction score of segment with `segment_id`.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) into image panoptic segmentation predictions. Only supports
PyTorch.

## DETR specific outputs

### class transformers.models.detr.modeling\_detr.DetrModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/modeling_detr.py#L77)

( last\_hidden\_state: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.EncoderDecoderCache] = None decoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None decoder\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None encoder\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None encoder\_attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None intermediate\_hidden\_states: typing.Optional[torch.FloatTensor] = None  )

Parameters

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) —
  Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **past\_key\_values** (`~cache_utils.EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) —
  It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **intermediate\_hidden\_states** (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`) —
  Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
  layernorm.

Base class for outputs of the DETR encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput,
namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

### class transformers.models.detr.modeling\_detr.DetrObjectDetectionOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/modeling_detr.py#L95)

( loss: typing.Optional[torch.FloatTensor] = None loss\_dict: typing.Optional[dict] = None logits: typing.Optional[torch.FloatTensor] = None pred\_boxes: typing.Optional[torch.FloatTensor] = None auxiliary\_outputs: typing.Optional[list[dict]] = None last\_hidden\_state: typing.Optional[torch.FloatTensor] = None decoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) —
  Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) —
  A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`) —
  Classification logits (including no-object) for all queries.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) —
  Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use [post\_process\_object\_detection()](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor.post_process_object_detection) to retrieve the
  unnormalized bounding boxes.
* **auxiliary\_outputs** (`list[Dict]`, *optional*) —
  Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
  and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
  `pred_boxes`) for each decoder layer.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) —
  Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

Output type of [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection).

### class transformers.models.detr.modeling\_detr.DetrSegmentationOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/modeling_detr.py#L138)

( loss: typing.Optional[torch.FloatTensor] = None loss\_dict: typing.Optional[dict] = None logits: typing.Optional[torch.FloatTensor] = None pred\_boxes: typing.Optional[torch.FloatTensor] = None pred\_masks: typing.Optional[torch.FloatTensor] = None auxiliary\_outputs: typing.Optional[list[dict]] = None last\_hidden\_state: typing.Optional[torch.FloatTensor] = None decoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None decoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None cross\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None encoder\_attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) —
  Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) —
  A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`) —
  Classification logits (including no-object) for all queries.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) —
  Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use [post\_process\_object\_detection()](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor.post_process_object_detection) to retrieve the
  unnormalized bounding boxes.
* **pred\_masks** (`torch.FloatTensor` of shape `(batch_size, num_queries, height/4, width/4)`) —
  Segmentation masks logits for all queries. See also
  [post\_process\_semantic\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor.post_process_semantic_segmentation) or
  [post\_process\_instance\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor.post_process_instance_segmentation)
  [post\_process\_panoptic\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor.post_process_panoptic_segmentation) to evaluate semantic, instance and panoptic
  segmentation masks respectively.
* **auxiliary\_outputs** (`list[Dict]`, *optional*) —
  Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
  and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
  `pred_boxes`) for each decoder layer.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) —
  Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

Output type of [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation).

## DetrModel

### class transformers.DetrModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/modeling_detr.py#L1039)

( config: DetrConfig  )

Parameters

* **config** ([DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/modeling_detr.py#L1070)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.detr.modeling\_detr.DetrModelOutput](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor). See [DetrImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor) for processing images).
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

[transformers.models.detr.modeling\_detr.DetrModelOutput](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.detr.modeling\_detr.DetrModelOutput](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **past\_key\_values** (`~cache_utils.EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **intermediate\_hidden\_states** (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`) — Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
  layernorm.

The [DetrModel](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, DetrModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
>>> model = DetrModel.from_pretrained("facebook/detr-resnet-50")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**inputs)

>>> # the last hidden states are the final query embeddings of the Transformer decoder
>>> # these are of shape (batch_size, num_queries, hidden_size)
>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 100, 256]
```

## DetrForObjectDetection

### class transformers.DetrForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/modeling_detr.py#L1231)

( config: DetrConfig  )

Parameters

* **config** ([DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks
such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/modeling_detr.py#L1249)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[list[dict]] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.detr.modeling\_detr.DetrObjectDetectionOutput](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrObjectDetectionOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor). See [DetrImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor) for processing images).
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

[transformers.models.detr.modeling\_detr.DetrObjectDetectionOutput](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrObjectDetectionOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.detr.modeling\_detr.DetrObjectDetectionOutput](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrObjectDetectionOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) — A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`) — Classification logits (including no-object) for all queries.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use [post\_process\_object\_detection()](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor.post_process_object_detection) to retrieve the
  unnormalized bounding boxes.
* **auxiliary\_outputs** (`list[Dict]`, *optional*) — Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
  and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
  `pred_boxes`) for each decoder layer.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
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

The [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, DetrForObjectDetection
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
>>> model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

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
Detected remote with confidence 0.998 at location [40.16, 70.81, 175.55, 117.98]
Detected remote with confidence 0.996 at location [333.24, 72.55, 368.33, 187.66]
Detected couch with confidence 0.995 at location [-0.02, 1.15, 639.73, 473.76]
Detected cat with confidence 0.999 at location [13.24, 52.05, 314.02, 470.93]
Detected cat with confidence 0.999 at location [345.4, 23.85, 640.37, 368.72]
```

## DetrForSegmentation

### class transformers.DetrForSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/modeling_detr.py#L1374)

( config: DetrConfig  )

Parameters

* **config** ([DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top, for tasks
such as COCO panoptic.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/detr/modeling_detr.py#L1395)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[list[dict]] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.detr.modeling\_detr.DetrSegmentationOutput](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrSegmentationOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor). See [DetrImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor) for processing images).
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
  Labels for computing the bipartite matching loss, DICE/F-1 loss and Focal loss. List of dicts, each
  dictionary containing at least the following 3 keys: ‘class\_labels’, ‘boxes’ and ‘masks’ (the class labels,
  bounding boxes and segmentation masks of an image in the batch respectively). The class labels themselves
  should be a `torch.LongTensor` of len `(number of bounding boxes in the image,)`, the boxes a
  `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)` and the masks a
  `torch.FloatTensor` of shape `(number of bounding boxes in the image, height, width)`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.detr.modeling\_detr.DetrSegmentationOutput](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrSegmentationOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.detr.modeling\_detr.DetrSegmentationOutput](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrSegmentationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DetrConfig](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) — A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`) — Classification logits (including no-object) for all queries.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use [post\_process\_object\_detection()](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor.post_process_object_detection) to retrieve the
  unnormalized bounding boxes.
* **pred\_masks** (`torch.FloatTensor` of shape `(batch_size, num_queries, height/4, width/4)`) — Segmentation masks logits for all queries. See also
  [post\_process\_semantic\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor.post_process_semantic_segmentation) or
  [post\_process\_instance\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor.post_process_instance_segmentation)
  [post\_process\_panoptic\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor.post_process_panoptic_segmentation) to evaluate semantic, instance and panoptic
  segmentation masks respectively.
* **auxiliary\_outputs** (`list[Dict]`, *optional*) — Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
  and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
  `pred_boxes`) for each decoder layer.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
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

The [DetrForSegmentation](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import io
>>> import requests
>>> from PIL import Image
>>> import torch
>>> import numpy

>>> from transformers import AutoImageProcessor, DetrForSegmentation
>>> from transformers.image_transforms import rgb_to_id

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
>>> model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**inputs)

>>> # Use the `post_process_panoptic_segmentation` method of the `image_processor` to retrieve post-processed panoptic segmentation maps
>>> # Segmentation results are returned as a list of dictionaries
>>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(300, 500)])

>>> # A tensor of shape (height, width) where each value denotes a segment id, filled with -1 if no segment is found
>>> panoptic_seg = result[0]["segmentation"]
>>> # Get prediction score and segment_id to class_id mapping of each segment
>>> panoptic_segments_info = result[0]["segments_info"]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/detr.md)
