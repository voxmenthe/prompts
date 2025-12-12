*This model was released on 2023-04-17 and added to Hugging Face Transformers on 2024-06-22.*

# RT-DETR

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The RT-DETR model was proposed in [DETRs Beat YOLOs on Real-time Object Detection](https://huggingface.co/papers/2304.08069) by Wenyu Lv, Yian Zhao, Shangliang Xu, Jinman Wei, Guanzhong Wang, Cheng Cui, Yuning Du, Qingqing Dang, Yi Liu.

RT-DETR is an object detection model that stands for “Real-Time DEtection Transformer.” This model is designed to perform object detection tasks with a focus on achieving real-time performance while maintaining high accuracy. Leveraging the transformer architecture, which has gained significant popularity in various fields of deep learning, RT-DETR processes images to identify and locate multiple objects within them.

The abstract from the paper is the following:

*Recently, end-to-end transformer-based detectors (DETRs) have achieved remarkable performance. However, the issue of the high computational cost of DETRs has not been effectively addressed, limiting their practical application and preventing them from fully exploiting the benefits of no post-processing, such as non-maximum suppression (NMS). In this paper, we first analyze the influence of NMS in modern real-time object detectors on inference speed, and establish an end-to-end speed benchmark. To avoid the inference delay caused by NMS, we propose a Real-Time DEtection TRansformer (RT-DETR), the first real-time end-to-end object detector to our best knowledge. Specifically, we design an efficient hybrid encoder to efficiently process multi-scale features by decoupling the intra-scale interaction and cross-scale fusion, and propose IoU-aware query selection to improve the initialization of object queries. In addition, our proposed detector supports flexibly adjustment of the inference speed by using different decoder layers without the need for retraining, which facilitates the practical application of real-time object detectors. Our RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, our RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/rt_detr_overview.png) RT-DETR performance relative to YOLO models. Taken from the [original paper.](https://huggingface.co/papers/2304.08069)

The model version was contributed by [rafaelpadilla](https://huggingface.co/rafaelpadilla) and [sangbumchoi](https://github.com/SangbumChoi). The original code can be found [here](https://github.com/lyuwenyu/RT-DETR/).

## Usage tips

Initially, an image is processed using a pre-trained convolutional neural network, specifically a Resnet-D variant as referenced in the original code. This network extracts features from the final three layers of the architecture. Following this, a hybrid encoder is employed to convert the multi-scale features into a sequential array of image features. Then, a decoder, equipped with auxiliary prediction heads is used to refine the object queries. This process facilitates the direct generation of bounding boxes, eliminating the need for any additional post-processing to acquire the logits and coordinates for the bounding boxes.


```
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
>>> model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3)

>>> for result in results:
...     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
...         score, label = score.item(), label_id.item()
...         box = [round(i, 2) for i in box.tolist()]
...         print(f"{model.config.id2label[label]}: {score:.2f} {box}")
sofa: 0.97 [0.14, 0.38, 640.13, 476.21]
cat: 0.96 [343.38, 24.28, 640.14, 371.5]
cat: 0.96 [13.23, 54.18, 318.98, 472.22]
remote: 0.95 [40.11, 73.44, 175.96, 118.48]
remote: 0.92 [333.73, 76.58, 369.97, 186.99]
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with RT-DETR.

Object Detection

* Scripts for finetuning [RTDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrForObjectDetection) with [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
* See also: [Object detection task guide](../tasks/object_detection).
* Notebooks regarding inference and fine-tuning RT-DETR on a custom dataset can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/RT-DETR). 🌎

## RTDetrConfig

### class transformers.RTDetrConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/configuration_rt_detr.py#L27)

( initializer\_range = 0.01 initializer\_bias\_prior\_prob = None layer\_norm\_eps = 1e-05 batch\_norm\_eps = 1e-05 backbone\_config = None backbone = None use\_pretrained\_backbone = False use\_timm\_backbone = False freeze\_backbone\_batch\_norms = True backbone\_kwargs = None encoder\_hidden\_dim = 256 encoder\_in\_channels = [512, 1024, 2048] feat\_strides = [8, 16, 32] encoder\_layers = 1 encoder\_ffn\_dim = 1024 encoder\_attention\_heads = 8 dropout = 0.0 activation\_dropout = 0.0 encode\_proj\_layers = [2] positional\_encoding\_temperature = 10000 encoder\_activation\_function = 'gelu' activation\_function = 'silu' eval\_size = None normalize\_before = False hidden\_expansion = 1.0 d\_model = 256 num\_queries = 300 decoder\_in\_channels = [256, 256, 256] decoder\_ffn\_dim = 1024 num\_feature\_levels = 3 decoder\_n\_points = 4 decoder\_layers = 6 decoder\_attention\_heads = 8 decoder\_activation\_function = 'relu' attention\_dropout = 0.0 num\_denoising = 100 label\_noise\_ratio = 0.5 box\_noise\_scale = 1.0 learn\_initial\_query = False anchor\_image\_size = None disable\_custom\_kernels = True with\_box\_refine = True is\_encoder\_decoder = True matcher\_alpha = 0.25 matcher\_gamma = 2.0 matcher\_class\_cost = 2.0 matcher\_bbox\_cost = 5.0 matcher\_giou\_cost = 2.0 use\_focal\_loss = True auxiliary\_loss = True focal\_loss\_alpha = 0.75 focal\_loss\_gamma = 2.0 weight\_loss\_vfl = 1.0 weight\_loss\_bbox = 5.0 weight\_loss\_giou = 2.0 eos\_coefficient = 0.0001 \*\*kwargs  )

Parameters

* **initializer\_range** (`float`, *optional*, defaults to 0.01) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **initializer\_bias\_prior\_prob** (`float`, *optional*) —
  The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
  If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **batch\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the batch normalization layers.
* **backbone\_config** (`Dict`, *optional*, defaults to `RTDetrResNetConfig()`) —
  The configuration of the backbone model.
* **backbone** (`str`, *optional*) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to use pretrained weights for the backbone.
* **use\_timm\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
  library.
* **freeze\_backbone\_batch\_norms** (`bool`, *optional*, defaults to `True`) —
  Whether to freeze the batch normalization layers in the backbone.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **encoder\_hidden\_dim** (`int`, *optional*, defaults to 256) —
  Dimension of the layers in hybrid encoder.
* **encoder\_in\_channels** (`list`, *optional*, defaults to `[512, 1024, 2048]`) —
  Multi level features input for encoder.
* **feat\_strides** (`list[int]`, *optional*, defaults to `[8, 16, 32]`) —
  Strides used in each feature map.
* **encoder\_layers** (`int`, *optional*, defaults to 1) —
  Total of layers to be used by the encoder.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 1024) —
  Dimension of the “intermediate” (often named feed-forward) layer in decoder.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **dropout** (`float`, *optional*, defaults to 0.0) —
  The ratio for all dropout layers.
* **activation\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for activations inside the fully connected layer.
* **encode\_proj\_layers** (`list[int]`, *optional*, defaults to `[2]`) —
  Indexes of the projected layers to be used in the encoder.
* **positional\_encoding\_temperature** (`int`, *optional*, defaults to 10000) —
  The temperature parameter used to create the positional encodings.
* **encoder\_activation\_function** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **activation\_function** (`str`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the general layer. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **eval\_size** (`tuple[int, int]`, *optional*) —
  Height and width used to computes the effective height and width of the position embeddings after taking
  into account the stride.
* **normalize\_before** (`bool`, *optional*, defaults to `False`) —
  Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
  feed-forward modules.
* **hidden\_expansion** (`float`, *optional*, defaults to 1.0) —
  Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.
* **d\_model** (`int`, *optional*, defaults to 256) —
  Dimension of the layers exclude hybrid encoder.
* **num\_queries** (`int`, *optional*, defaults to 300) —
  Number of object queries.
* **decoder\_in\_channels** (`list`, *optional*, defaults to `[256, 256, 256]`) —
  Multi level features dimension for decoder
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 1024) —
  Dimension of the “intermediate” (often named feed-forward) layer in decoder.
* **num\_feature\_levels** (`int`, *optional*, defaults to 3) —
  The number of input feature levels.
* **decoder\_n\_points** (`int`, *optional*, defaults to 4) —
  The number of sampled keys in each feature level for each attention head in the decoder.
* **decoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of decoder layers.
* **decoder\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **decoder\_activation\_function** (`str`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the decoder. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **num\_denoising** (`int`, *optional*, defaults to 100) —
  The total number of denoising tasks or queries to be used for contrastive denoising.
* **label\_noise\_ratio** (`float`, *optional*, defaults to 0.5) —
  The fraction of denoising labels to which random noise should be added.
* **box\_noise\_scale** (`float`, *optional*, defaults to 1.0) —
  Scale or magnitude of noise to be added to the bounding boxes.
* **learn\_initial\_query** (`bool`, *optional*, defaults to `False`) —
  Indicates whether the initial query embeddings for the decoder should be learned during training
* **anchor\_image\_size** (`tuple[int, int]`, *optional*) —
  Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.
* **disable\_custom\_kernels** (`bool`, *optional*, defaults to `True`) —
  Whether to disable custom kernels.
* **with\_box\_refine** (`bool`, *optional*, defaults to `True`) —
  Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
  based on the predictions from the previous layer.
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Whether the architecture has an encoder decoder structure.
* **matcher\_alpha** (`float`, *optional*, defaults to 0.25) —
  Parameter alpha used by the Hungarian Matcher.
* **matcher\_gamma** (`float`, *optional*, defaults to 2.0) —
  Parameter gamma used by the Hungarian Matcher.
* **matcher\_class\_cost** (`float`, *optional*, defaults to 2.0) —
  The relative weight of the class loss used by the Hungarian Matcher.
* **matcher\_bbox\_cost** (`float`, *optional*, defaults to 5.0) —
  The relative weight of the bounding box loss used by the Hungarian Matcher.
* **matcher\_giou\_cost** (`float`, *optional*, defaults to 2.0) —
  The relative weight of the giou loss of used by the Hungarian Matcher.
* **use\_focal\_loss** (`bool`, *optional*, defaults to `True`) —
  Parameter informing if focal focal should be used.
* **auxiliary\_loss** (`bool`, *optional*, defaults to `True`) —
  Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
* **focal\_loss\_alpha** (`float`, *optional*, defaults to 0.75) —
  Parameter alpha used to compute the focal loss.
* **focal\_loss\_gamma** (`float`, *optional*, defaults to 2.0) —
  Parameter gamma used to compute the focal loss.
* **weight\_loss\_vfl** (`float`, *optional*, defaults to 1.0) —
  Relative weight of the varifocal loss in the object detection loss.
* **weight\_loss\_bbox** (`float`, *optional*, defaults to 5.0) —
  Relative weight of the L1 bounding box loss in the object detection loss.
* **weight\_loss\_giou** (`float`, *optional*, defaults to 2.0) —
  Relative weight of the generalized IoU loss in the object detection loss.
* **eos\_coefficient** (`float`, *optional*, defaults to 0.0001) —
  Relative classification weight of the ‘no-object’ class in the object detection loss.

This is the configuration class to store the configuration of a [RTDetrModel](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrModel). It is used to instantiate a
RT-DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the RT-DETR
[PekingU/rtdetr\_r50vd](https://huggingface.co/PekingU/rtdetr_r50vd) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import RTDetrConfig, RTDetrModel

>>> # Initializing a RT-DETR configuration
>>> configuration = RTDetrConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = RTDetrModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

#### from\_backbone\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/configuration_rt_detr.py#L354)

( backbone\_config: PretrainedConfig \*\*kwargs  ) → [RTDetrConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrConfig)

Parameters

* **backbone\_config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The backbone configuration.

Returns

[RTDetrConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrConfig)

An instance of a configuration object

Instantiate a [RTDetrConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrConfig) (or a derived class) from a pre-trained backbone model configuration and DETR model
configuration.

## RTDetrResNetConfig

### class transformers.RTDetrResNetConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/configuration_rt_detr_resnet.py#L25)

( num\_channels = 3 embedding\_size = 64 hidden\_sizes = [256, 512, 1024, 2048] depths = [3, 4, 6, 3] layer\_type = 'bottleneck' hidden\_act = 'relu' downsample\_in\_first\_stage = False downsample\_in\_bottleneck = False out\_features = None out\_indices = None \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **embedding\_size** (`int`, *optional*, defaults to 64) —
  Dimensionality (hidden size) for the embedding layer.
* **hidden\_sizes** (`list[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`) —
  Dimensionality (hidden size) at each stage.
* **depths** (`list[int]`, *optional*, defaults to `[3, 4, 6, 3]`) —
  Depth (number of layers) for each stage.
* **layer\_type** (`str`, *optional*, defaults to `"bottleneck"`) —
  The layer to use, it can be either `"basic"` (used for smaller models, like resnet-18 or resnet-34) or
  `"bottleneck"` (used for larger models like resnet-50 and above).
* **hidden\_act** (`str`, *optional*, defaults to `"relu"`) —
  The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
  are supported.
* **downsample\_in\_first\_stage** (`bool`, *optional*, defaults to `False`) —
  If `True`, the first stage will downsample the inputs using a `stride` of 2.
* **downsample\_in\_bottleneck** (`bool`, *optional*, defaults to `False`) —
  If `True`, the first conv 1x1 in ResNetBottleNeckLayer will downsample the inputs using a `stride` of 2.
* **out\_features** (`list[str]`, *optional*) —
  If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
  (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
  corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
  same order as defined in the `stage_names` attribute.
* **out\_indices** (`list[int]`, *optional*) —
  If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
  many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
  If unset and `out_features` is unset, will default to the last stage. Must be in the
  same order as defined in the `stage_names` attribute.

This is the configuration class to store the configuration of a `RTDetrResnetBackbone`. It is used to instantiate an
ResNet model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the ResNet
[microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import RTDetrResNetConfig, RTDetrResnetBackbone

>>> # Initializing a ResNet resnet-50 style configuration
>>> configuration = RTDetrResNetConfig()

>>> # Initializing a model (with random weights) from the resnet-50 style configuration
>>> model = RTDetrResnetBackbone(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## RTDetrImageProcessor

### class transformers.RTDetrImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/image_processing_rt_detr.py#L384)

( format: typing.Union[str, transformers.image\_utils.AnnotationFormat] = <AnnotationFormat.COCO\_DETECTION: 'coco\_detection'> do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = False image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_annotations: bool = True do\_pad: bool = False pad\_size: typing.Optional[dict[str, int]] = None \*\*kwargs  )

Parameters

* **format** (`str`, *optional*, defaults to `AnnotationFormat.COCO_DETECTION`) —
  Data format of the annotations. One of “coco\_detection” or “coco\_panoptic”.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Controls whether to resize the image’s (height, width) dimensions to the specified `size`. Can be
  overridden by the `do_resize` parameter in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 640, "width": 640}`):
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
  Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
  `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `False`) —
  Whether to normalize the image.
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
* **do\_pad** (`bool`, *optional*, defaults to `False`) —
  Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
  method. If `True`, padding will be applied to the bottom and right of the image with zeros.
  If `pad_size` is provided, the image will be padded to the specified dimensions.
  Otherwise, the image will be padded to the maximum height and width of the batch.
* **pad\_size** (`dict[str, int]`, *optional*) —
  The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
  height and width in the batch.

Constructs a RT-DETR image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/image_processing_rt_detr.py#L783)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] annotations: typing.Union[dict[str, typing.Union[int, str, list[dict]]], list[dict[str, typing.Union[int, str, list[dict]]]], NoneType] = None return\_segmentation\_masks: typing.Optional[bool] = None masks\_path: typing.Union[str, pathlib.Path, NoneType] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Union[float, int, NoneType] = None do\_normalize: typing.Optional[bool] = None do\_convert\_annotations: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: typing.Optional[bool] = None format: typing.Union[str, transformers.image\_utils.AnnotationFormat, NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None pad\_size: typing.Optional[dict[str, int]] = None  )

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/image_processing_rt_detr.py#L1030)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None use\_focal\_loss: bool = True  ) → `list[Dict]`

Parameters

* **outputs** (`DetrObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.5) —
  Score threshold to keep object detection predictions.
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  `(height, width)` of each image in the batch. If unset, predictions will not be resized.
* **use\_focal\_loss** (`bool` defaults to `True`) —
  Variable informing if the focal loss was used to predict the outputs. If `True`, a sigmoid is applied
  to compute the scores of each detection, otherwise, a softmax function is used.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
in the batch as predicted by the model.

Converts the raw output of [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## RTDetrImageProcessorFast

### class transformers.RTDetrImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/image_processing_rt_detr_fast.py#L146)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.rt\_detr.image\_processing\_rt\_detr\_fast.RTDetrFastImageProcessorKwargs]  )

Constructs a fast Rt Detr image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/image_processing_rt_detr_fast.py#L386)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] annotations: typing.Union[dict[str, typing.Union[int, str, list[dict]]], list[dict[str, typing.Union[int, str, list[dict]]]], NoneType] = None masks\_path: typing.Union[str, pathlib.Path, NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.rt\_detr.image\_processing\_rt\_detr\_fast.RTDetrFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
  Controls whether to convert the annotations to the format expected by the RT\_DETR model. Converts the
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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/image_processing_rt_detr_fast.py#L517)

( outputs threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None use\_focal\_loss: bool = True  ) → `list[Dict]`

Parameters

* **outputs** (`DetrObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.5) —
  Score threshold to keep object detection predictions.
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  `(height, width)` of each image in the batch. If unset, predictions will not be resized.
* **use\_focal\_loss** (`bool` defaults to `True`) —
  Variable informing if the focal loss was used to predict the outputs. If `True`, a sigmoid is applied
  to compute the scores of each detection, otherwise, a softmax function is used.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
in the batch as predicted by the model.

Converts the raw output of [DetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

## RTDetrModel

### class transformers.RTDetrModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/modeling_rt_detr.py#L1470)

( config: RTDetrConfig  )

Parameters

* **config** ([RTDetrConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

RT-DETR Model (consisting of a backbone and encoder-decoder) outputting raw hidden states without any head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/modeling_rt_detr.py#L1584)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[list[dict]] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.rt_detr.modeling_rt_detr.RTDetrModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [RTDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrImageProcessor). See [RTDetrImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [RTDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrImageProcessor) for processing images).
* **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
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

`transformers.models.rt_detr.modeling_rt_detr.RTDetrModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.rt_detr.modeling_rt_detr.RTDetrModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RTDetrConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **intermediate\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`) — Stacked intermediate hidden states (output of each layer of the decoder).
* **intermediate\_logits** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, config.num_labels)`) — Stacked intermediate logits (logits of each layer of the decoder).
* **intermediate\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`) — Stacked intermediate reference points (reference points of each layer of the decoder).
* **intermediate\_predicted\_corners** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`) — Stacked intermediate predicted corners (predicted corners of each layer of the decoder).
* **initial\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Initial reference points used for the first decoder layer.
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
* **init\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Initial reference points sent through the Transformer decoder.
* **enc\_topk\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
  picked as region proposals in the encoder stage. Output of bounding box binary classification (i.e.
  foreground and background).
* **enc\_topk\_bboxes** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`) — Logits of predicted bounding boxes coordinates in the encoder stage.
* **enc\_outputs\_class** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
  picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
  foreground and background).
* **enc\_outputs\_coord\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Logits of predicted bounding boxes coordinates in the first stage.
* **denoising\_meta\_values** (`dict`, *optional*, defaults to `None`) — Extra dictionary for the denoising related values.

The [RTDetrModel](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, RTDetrModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
>>> model = RTDetrModel.from_pretrained("PekingU/rtdetr_r50vd")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 300, 256]
```

## RTDetrForObjectDetection

### class transformers.RTDetrForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/modeling_rt_detr.py#L1813)

( config: RTDetrConfig  )

Parameters

* **config** ([RTDetrConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

RT-DETR Model (consisting of a backbone and encoder-decoder) outputting bounding boxes and logits to be further
decoded into scores and classes.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/modeling_rt_detr.py#L1852)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[list[dict]] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → `transformers.models.rt_detr.modeling_rt_detr.RTDetrObjectDetectionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [RTDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrImageProcessor). See [RTDetrImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details (`processor_class` uses
  [RTDetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrImageProcessor) for processing images).
* **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
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

`transformers.models.rt_detr.modeling_rt_detr.RTDetrObjectDetectionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.rt_detr.modeling_rt_detr.RTDetrObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RTDetrConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) — A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`) — Classification logits (including no-object) for all queries.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use [post\_process\_object\_detection()](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrImageProcessor.post_process_object_detection) to retrieve the
  unnormalized (absolute) bounding boxes.
* **auxiliary\_outputs** (`list[Dict]`, *optional*) — Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
  and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
  `pred_boxes`) for each decoder layer.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
* **intermediate\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`) — Stacked intermediate hidden states (output of each layer of the decoder).
* **intermediate\_logits** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, config.num_labels)`) — Stacked intermediate logits (logits of each layer of the decoder).
* **intermediate\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`) — Stacked intermediate reference points (reference points of each layer of the decoder).
* **intermediate\_predicted\_corners** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`) — Stacked intermediate predicted corners (predicted corners of each layer of the decoder).
* **initial\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`) — Stacked initial reference points (initial reference points of each layer of the decoder).
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
* **init\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Initial reference points sent through the Transformer decoder.
* **enc\_topk\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Logits of predicted bounding boxes coordinates in the encoder.
* **enc\_topk\_bboxes** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Logits of predicted bounding boxes coordinates in the encoder.
* **enc\_outputs\_class** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
  picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
  foreground and background).
* **enc\_outputs\_coord\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`) — Logits of predicted bounding boxes coordinates in the first stage.
* **denoising\_meta\_values** (`dict`, *optional*, defaults to `None`) — Extra dictionary for the denoising related values

The [RTDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
>>> from PIL import Image
>>> import requests
>>> import torch

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
>>> model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> list(logits.shape)
[1, 300, 80]

>>> boxes = outputs.pred_boxes
>>> list(boxes.shape)
[1, 300, 4]

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
Detected sofa with confidence 0.97 at location [0.14, 0.38, 640.13, 476.21]
Detected cat with confidence 0.96 at location [343.38, 24.28, 640.14, 371.5]
Detected cat with confidence 0.958 at location [13.23, 54.18, 318.98, 472.22]
Detected remote with confidence 0.951 at location [40.11, 73.44, 175.96, 118.48]
Detected remote with confidence 0.924 at location [333.73, 76.58, 369.97, 186.99]
```

## RTDetrResNetBackbone

### class transformers.RTDetrResNetBackbone

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/modeling_rt_detr_resnet.py#L325)

( config  )

Parameters

* **config** ([RTDetrResNetBackbone](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrResNetBackbone)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ResNet backbone, to be used with frameworks like RTDETR.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr/modeling_rt_detr_resnet.py#L339)

( pixel\_values: Tensor output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  )

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

The [RTDetrResNetBackbone](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrResNetBackbone) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

<returns>

A `transformers.modeling_outputs.BackboneOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RTDetrResNetConfig](/docs/transformers/v4.56.2/en/model_doc/rt_detr#transformers.RTDetrResNetConfig)) and inputs.

* **feature\_maps** (`tuple(torch.FloatTensor)` of shape `(batch_size, num_channels, height, width)`) — Feature maps of the stages.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, num_channels, height, width)`,
  depending on the backbone.

  Hidden-states of the model at the output of each stage plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Only applicable if the backbone uses attention.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Examples:


```
>>> from transformers import RTDetrResNetConfig, RTDetrResNetBackbone
>>> import torch


`transformers.modeling_outputs.BackboneOutput` or `tuple(torch.FloatTensor)`</returntype>

precation import deprecate_kwarg
precation import deprecate_kwarg
precation import deprecate_kwarg
utils.deprecation import deprecate_kwarg
utils.deprecation import deprecate_kwarg

>>> config = RTDetrResNetConfig()
>>> model = RTDetrResNetBackbone(config)

>>> pixel_values = torch.randn(1, 3, 224, 224)

>>> with torch.no_grad():
...     outputs = model(pixel_values)

>>> feature_maps = outputs.feature_maps
>>> list(feature_maps[-1].shape)
[1, 2048, 7, 7]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/rt_detr.md)
