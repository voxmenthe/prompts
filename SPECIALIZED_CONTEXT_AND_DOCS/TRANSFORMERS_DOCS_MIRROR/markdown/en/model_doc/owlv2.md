*This model was released on 2023-06-16 and added to Hugging Face Transformers on 2023-10-13.*

# OWLv2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

OWLv2 was proposed in [Scaling Open-Vocabulary Object Detection](https://huggingface.co/papers/2306.09683) by Matthias Minderer, Alexey Gritsenko, Neil Houlsby. OWLv2 scales up [OWL-ViT](owlvit) using self-training, which uses an existing detector to generate pseudo-box annotations on image-text pairs. This results in large gains over the previous state-of-the-art for zero-shot object detection.

The abstract from the paper is the following:

*Open-vocabulary object detection has benefited greatly from pretrained vision-language models, but is still limited by the amount of available detection training data. While detection training data can be expanded by using Web image-text pairs as weak supervision, this has not been done at scales comparable to image-level pretraining. Here, we scale up detection data with self-training, which uses an existing detector to generate pseudo-box annotations on image-text pairs. Major challenges in scaling self-training are the choice of label space, pseudo-annotation filtering, and training efficiency. We present the OWLv2 model and OWL-ST self-training recipe, which address these challenges. OWLv2 surpasses the performance of previous state-of-the-art open-vocabulary detectors already at comparable training scales (~10M examples). However, with OWL-ST, we can scale to over 1B examples, yielding further large improvement: With an L/14 architecture, OWL-ST improves AP on LVIS rare classes, for which the model has seen no human box annotations, from 31.2% to 44.6% (43% relative improvement). OWL-ST unlocks Web-scale training for open-world localization, similar to what has been seen for image classification and language modelling.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/owlv2_overview.png) OWLv2 high-level overview. Taken from the [original paper](https://huggingface.co/papers/2306.09683).

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit).

## Usage example

OWLv2 is, just like its predecessor [OWL-ViT](owlvit), a zero-shot text-conditioned object detection model. OWL-ViT uses [CLIP](clip) as its multi-modal backbone, with a ViT-like Transformer to get visual features and a causal language model to get the text features. To use CLIP for detection, OWL-ViT removes the final token pooling layer of the vision model and attaches a lightweight classification and box head to each transformer output token. Open-vocabulary classification is enabled by replacing the fixed classification layer weights with the class-name embeddings obtained from the text model. The authors first train CLIP from scratch and fine-tune it end-to-end with the classification and box heads on standard detection datasets using a bipartite matching loss. One or multiple text queries per image can be used to perform zero-shot text-conditioned object detection.

[Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor) can be used to resize (or rescale) and normalize images for the model and [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) is used to encode the text. [Owlv2Processor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Processor) wraps [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor) and [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) into a single instance to both encode the text and prepare the images. The following example shows how to perform object detection using [Owlv2Processor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Processor) and [Owlv2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ForObjectDetection).


```
>>> import requests
>>> from PIL import Image
>>> import torch

>>> from transformers import Owlv2Processor, Owlv2ForObjectDetection

>>> processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
>>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text_labels = [["a photo of a cat", "a photo of a dog"]]
>>> inputs = processor(text=text_labels, images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
>>> target_sizes = torch.tensor([(image.height, image.width)])
>>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
>>> results = processor.post_process_grounded_object_detection(
...     outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
... )
>>> # Retrieve predictions for the first image for the corresponding text queries
>>> result = results[0]
>>> boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
>>> for box, score, text_label in zip(boxes, scores, text_labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
Detected a photo of a cat with confidence 0.614 at location [341.67, 23.39, 642.32, 371.35]
Detected a photo of a cat with confidence 0.665 at location [6.75, 51.96, 326.62, 473.13]
```

## Resources

* A demo notebook on using OWLv2 for zero- and one-shot (image-guided) object detection can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/OWLv2).
* [Zero-shot object detection task guide](../tasks/zero_shot_object_detection)

The architecture of OWLv2 is identical to [OWL-ViT](owlvit), however the object detection head now also includes an objectness classifier, which predicts the (query-agnostic) likelihood that a predicted box contains an object (as opposed to background). The objectness score can be used to rank or filter predictions independently of text queries.
Usage of OWLv2 is identical to [OWL-ViT](owlvit) with a new, updated image processor ([Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor)).

## Owlv2Config

### class transformers.Owlv2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/configuration_owlv2.py#L211)

( text\_config = None vision\_config = None projection\_dim = 512 logit\_scale\_init\_value = 2.6592 return\_dict = True \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Owlv2TextConfig](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2TextConfig).
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Owlv2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionConfig).
* **projection\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of text and vision projection layers.
* **logit\_scale\_init\_value** (`float`, *optional*, defaults to 2.6592) —
  The initial value of the *logit\_scale* parameter. Default is used as per the original OWLv2
  implementation.
* **return\_dict** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return a dictionary. If `False`, returns a tuple.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config) is the configuration class to store the configuration of an [Owlv2Model](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Model). It is used to
instantiate an OWLv2 model according to the specified arguments, defining the text model and vision model
configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the OWLv2
[google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

#### from\_text\_vision\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/configuration_owlv2.py#L267)

( text\_config: dict vision\_config: dict \*\*kwargs  ) → [Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config)

Returns

[Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config)

An instance of a configuration object

Instantiate a [Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config) (or a derived class) from owlv2 text model configuration and owlv2 vision
model configuration.

## Owlv2TextConfig

### class transformers.Owlv2TextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/configuration_owlv2.py#L25)

( vocab\_size = 49408 hidden\_size = 512 intermediate\_size = 2048 num\_hidden\_layers = 12 num\_attention\_heads = 8 max\_position\_embeddings = 16 hidden\_act = 'quick\_gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 pad\_token\_id = 0 bos\_token\_id = 49406 eos\_token\_id = 49407 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 49408) —
  Vocabulary size of the OWLv2 text model. Defines the number of different tokens that can be represented
  by the `inputs_ids` passed when calling [Owlv2TextModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2TextModel).
* **hidden\_size** (`int`, *optional*, defaults to 512) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 16) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"quick_gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the padding token in the input sequences.
* **bos\_token\_id** (`int`, *optional*, defaults to 49406) —
  The id of the beginning-of-sequence token in the input sequences.
* **eos\_token\_id** (`int`, *optional*, defaults to 49407) —
  The id of the end-of-sequence token in the input sequences.

This is the configuration class to store the configuration of an [Owlv2TextModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2TextModel). It is used to instantiate an
Owlv2 text encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Owlv2
[google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Owlv2TextConfig, Owlv2TextModel

>>> # Initializing a Owlv2TextModel with google/owlv2-base-patch16 style configuration
>>> configuration = Owlv2TextConfig()

>>> # Initializing a Owlv2TextConfig from the google/owlv2-base-patch16 style configuration
>>> model = Owlv2TextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Owlv2VisionConfig

### class transformers.Owlv2VisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/configuration_owlv2.py#L122)

( hidden\_size = 768 intermediate\_size = 3072 num\_hidden\_layers = 12 num\_attention\_heads = 12 num\_channels = 3 image\_size = 768 patch\_size = 16 hidden\_act = 'quick\_gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  Number of channels in the input images.
* **image\_size** (`int`, *optional*, defaults to 768) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"quick_gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).

This is the configuration class to store the configuration of an [Owlv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionModel). It is used to instantiate
an OWLv2 image encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the OWLv2
[google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Owlv2VisionConfig, Owlv2VisionModel

>>> # Initializing a Owlv2VisionModel with google/owlv2-base-patch16 style configuration
>>> configuration = Owlv2VisionConfig()

>>> # Initializing a Owlv2VisionModel model from the google/owlv2-base-patch16 style configuration
>>> model = Owlv2VisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Owlv2ImageProcessor

### class transformers.Owlv2ImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/image_processing_owlv2.py#L211)

( do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_pad: bool = True do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None \*\*kwargs  )

Parameters

* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
  the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
  method.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the image to a square with gray pixels on the bottom and the right. Can be overridden by
  `do_pad` in the `preprocess` method.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Controls whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden
  by `do_resize` in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 960, "width": 960}`):
  Size to resize the image to. Can be overridden by `size` in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling method to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `OPENAI_CLIP_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs an OWLv2 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/image_processing_owlv2.py#L368)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_pad: typing.Optional[bool] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether to pad the image to a square with gray pixels on the bottom and the right.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size to resize the image to.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation.
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

#### post\_process\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/image_processing_owlv2.py#L503)

( outputs: Owlv2ObjectDetectionOutput threshold: float = 0.1 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple], NoneType] = None  ) → `list[Dict]`

Parameters

* **outputs** (`Owlv2ObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.1) —
  Score threshold to keep object detection predictions.
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  `(height, width)` of each image in the batch. If unset, predictions will not be resized.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the following keys:

* “scores”: The confidence scores for each predicted box on the image.
* “labels”: Indexes of the classes predicted by the model on the image.
* “boxes”: Image bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format.

Converts the raw output of [Owlv2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format.

#### post\_process\_image\_guided\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/image_processing_owlv2.py#L557)

( outputs threshold = 0.0 nms\_threshold = 0.3 target\_sizes = None  ) → `list[Dict]`

Parameters

* **outputs** (`OwlViTImageGuidedObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.0) —
  Minimum confidence threshold to use to filter out predicted boxes.
* **nms\_threshold** (`float`, *optional*, defaults to 0.3) —
  IoU threshold for non-maximum suppression of overlapping boxes.
* **target\_sizes** (`torch.Tensor`, *optional*) —
  Tensor of shape (batch\_size, 2) where each entry is the (height, width) of the corresponding image in
  the batch. If set, predicted normalized bounding boxes are rescaled to the target sizes. If left to
  None, predictions will not be unnormalized.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
in the batch as predicted by the model. All labels are set to None as
`OwlViTForObjectDetection.image_guided_detection` perform one-shot object detection.

Converts the output of [OwlViTForObjectDetection.image\_guided\_detection()](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTForObjectDetection.image_guided_detection) into the format expected by the COCO
api.

## Owlv2ImageProcessorFast

### class transformers.Owlv2ImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/image_processing_owlv2_fast.py#L74)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.owlv2.image\_processing\_owlv2\_fast.Owlv2FastImageProcessorKwargs]  )

Constructs a fast Owlv2 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/image_processing_owlv2_fast.py#L270)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.owlv2.image\_processing\_owlv2\_fast.Owlv2FastImageProcessorKwargs]  )

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
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
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
  method. If `True`, padding will be applied to the bottom and right of the image with grey pixels.

#### post\_process\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/image_processing_owlv2_fast.py#L137)

( outputs: Owlv2ObjectDetectionOutput threshold: float = 0.1 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple], NoneType] = None  ) → `list[Dict]`

Parameters

* **outputs** (`Owlv2ObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.1) —
  Score threshold to keep object detection predictions.
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  `(height, width)` of each image in the batch. If unset, predictions will not be resized.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the following keys:

* “scores”: The confidence scores for each predicted box on the image.
* “labels”: Indexes of the classes predicted by the model on the image.
* “boxes”: Image bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format.

Converts the raw output of [Owlv2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format.

#### post\_process\_image\_guided\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/image_processing_owlv2_fast.py#L190)

( outputs threshold = 0.0 nms\_threshold = 0.3 target\_sizes = None  ) → `list[Dict]`

Parameters

* **outputs** (`Owlv2ImageGuidedObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.0) —
  Minimum confidence threshold to use to filter out predicted boxes.
* **nms\_threshold** (`float`, *optional*, defaults to 0.3) —
  IoU threshold for non-maximum suppression of overlapping boxes.
* **target\_sizes** (`torch.Tensor`, *optional*) —
  Tensor of shape (batch\_size, 2) where each entry is the (height, width) of the corresponding image in
  the batch. If set, predicted normalized bounding boxes are rescaled to the target sizes. If left to
  None, predictions will not be unnormalized.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
in the batch as predicted by the model. All labels are set to None as
`Owlv2ForObjectDetection.image_guided_detection` perform one-shot object detection.

Converts the output of [Owlv2ForObjectDetection.image\_guided\_detection()](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ForObjectDetection.image_guided_detection) into the format expected by the COCO
api.

## Owlv2Processor

### class transformers.Owlv2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/processing_owlv2.py#L57)

( image\_processor tokenizer \*\*kwargs  )

Parameters

* **image\_processor** ([`Owlv2ImageProcessor`, `Owlv2ImageProcessorFast`]) —
  The image processor is a required input.
* **tokenizer** ([`CLIPTokenizer`, `CLIPTokenizerFast`]) —
  The tokenizer is a required input.

Constructs an Owlv2 processor which wraps [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor)/[Owlv2ImageProcessorFast](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessorFast) and [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer)/[CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) into
a single processor that inherits both the image processor and tokenizer functionalities. See the
[**call**()](/docs/transformers/v4.56.2/en/model_doc/owlvit#transformers.OwlViTProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/processing_owlv2.py#L78)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None text: typing.Union[str, list[str], list[list[str]]] = None audio = None videos = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.owlv2.processing\_owlv2.Owlv2ProcessorKwargs]  ) → [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

Parameters

* **images** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, —
* **`list[torch.Tensor]`)** —
  The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
  tensor. Both channels-first and channels-last formats are supported.
* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **query\_images** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`) —
  The query image to be prepared, one query image is expected per target image to be queried. Each image
  can be a PIL image, NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each image
  should be of shape (C, H, W), where C is a number of channels, H and W are image height and width.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors of a particular framework. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return NumPy `np.ndarray` objects.
  + `'jax'`: Return JAX `jnp.ndarray` objects.

Returns

[BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

A [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

* **input\_ids** — List of token ids to be fed to a model. Returned when `text` is not `None`.
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names` and if `text` is not
  `None`).
* **pixel\_values** — Pixel values to be fed to a model. Returned when `images` is not `None`.
* **query\_pixel\_values** — Pixel values of the query images to be fed to a model. Returned when `query_images` is not `None`.

Main method to prepare for the model one or several text(s) and image(s). This method forwards the `text` and
`kwargs` arguments to CLIPTokenizerFast’s [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) if `text` is not `None` to encode:
the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
CLIPImageProcessor’s [**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) if `images` is not `None`. Please refer to the docstring
of the above two methods for more information.

#### post\_process\_grounded\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/processing_owlv2.py#L209)

( outputs: Owlv2ObjectDetectionOutput threshold: float = 0.1 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple], NoneType] = None text\_labels: typing.Optional[list[list[str]]] = None  ) → `list[Dict]`

Parameters

* **outputs** (`Owlv2ObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.1) —
  Score threshold to keep object detection predictions.
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  `(height, width)` of each image in the batch. If unset, predictions will not be resized.
* **text\_labels** (`list[list[str]]`, *optional*) —
  List of lists of text labels for each image in the batch. If unset, “text\_labels” in output will be
  set to `None`.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the following keys:

* “scores”: The confidence scores for each predicted box on the image.
* “labels”: Indexes of the classes predicted by the model on the image.
* “boxes”: Image bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format.
* “text\_labels”: The text labels for each predicted bounding box on the image.

Converts the raw output of [Owlv2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format.

#### post\_process\_image\_guided\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/processing_owlv2.py#L258)

( outputs: Owlv2ImageGuidedObjectDetectionOutput threshold: float = 0.0 nms\_threshold: float = 0.3 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple], NoneType] = None  ) → `list[Dict]`

Parameters

* **outputs** (`Owlv2ImageGuidedObjectDetectionOutput`) —
  Raw outputs of the model.
* **threshold** (`float`, *optional*, defaults to 0.0) —
  Minimum confidence threshold to use to filter out predicted boxes.
* **nms\_threshold** (`float`, *optional*, defaults to 0.3) —
  IoU threshold for non-maximum suppression of overlapping boxes.
* **target\_sizes** (`torch.Tensor`, *optional*) —
  Tensor of shape (batch\_size, 2) where each entry is the (height, width) of the corresponding image in
  the batch. If set, predicted normalized bounding boxes are rescaled to the target sizes. If left to
  None, predictions will not be unnormalized.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the following keys:

* “scores”: The confidence scores for each predicted box on the image.
* “boxes”: Image bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format.
* “labels”: Set to `None`.

Converts the output of [Owlv2ForObjectDetection.image\_guided\_detection()](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ForObjectDetection.image_guided_detection) into the format expected by the COCO
api.

## Owlv2Model

### class transformers.Owlv2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L925)

( config: Owlv2Config  )

Parameters

* **config** ([Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Owlv2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L1047)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None return\_loss: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_base\_image\_embeds: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.owlv2.modeling_owlv2.Owlv2Output` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor). See [Owlv2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Owlv2Processor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Processor) uses
  [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **return\_loss** (`bool`, *optional*) —
  Whether or not to return the contrastive loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_base\_image\_embeds** (`bool`, *optional*) —
  Whether or not to return the base image embeddings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.owlv2.modeling_owlv2.Owlv2Output` or `tuple(torch.FloatTensor)`

A `transformers.models.owlv2.modeling_owlv2.Owlv2Output` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
* **logits\_per\_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size * num_max_text_queries, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [Owlv2TextModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2TextModel).
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of
  [Owlv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionModel).
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [Owlv2TextModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2TextModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output`, defaults to `None`) — The output of the [Owlv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionModel).

The [Owlv2Model](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Owlv2Model

>>> model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble")
>>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(text=[["a photo of a cat", "a photo of a dog"]], images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

#### get\_text\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L960)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by
applying the projection layer to the pooled output of [Owlv2TextModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2TextModel).

Examples:


```
>>> from transformers import AutoProcessor, Owlv2Model

>>> model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble")
>>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
>>> inputs = processor(
...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
... )
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L1000)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor). See [Owlv2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Owlv2Processor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Processor) uses
  [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by
applying the projection layer to the pooled output of [Owlv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionModel).

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Owlv2Model

>>> model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble")
>>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt")
>>> image_features = model.get_image_features(**inputs)
```

## Owlv2TextModel

### class transformers.Owlv2TextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L763)

( config: Owlv2TextConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L778)

( input\_ids: Tensor attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config)) and inputs.

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

The [Owlv2TextModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoProcessor, Owlv2TextModel

>>> model = Owlv2TextModel.from_pretrained("google/owlv2-base-patch16")
>>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
>>> inputs = processor(
...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
... )
>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
```

## Owlv2VisionModel

### class transformers.Owlv2VisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L874)

( config: Owlv2VisionConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L887)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor). See [Owlv2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Owlv2Processor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Processor) uses
  [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config)) and inputs.

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

The [Owlv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Owlv2VisionModel

>>> model = Owlv2VisionModel.from_pretrained("google/owlv2-base-patch16")
>>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled CLS states
```

## Owlv2ForObjectDetection

### class transformers.Owlv2ForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L1210)

( config: Owlv2Config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L1599)

( input\_ids: Tensor pixel\_values: FloatTensor attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → `transformers.models.owlv2.modeling_owlv2.Owlv2ObjectDetectionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size * num_max_text_queries, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are input
  IDs?](../glossary#input-ids).
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor). See [Owlv2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Owlv2Processor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Processor) uses
  [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the last hidden state. See `text_model_last_hidden_state` and
  `vision_model_last_hidden_state` under returned tensors for more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.owlv2.modeling_owlv2.Owlv2ObjectDetectionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.owlv2.modeling_owlv2.Owlv2ObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) — A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`) — Classification logits (including no-object) for all queries.
* **objectness\_logits** (`torch.FloatTensor` of shape `(batch_size, num_patches, 1)`) — The objectness logits of all image patches. OWL-ViT represents images as a set of image patches where the
  total number of patches is (image\_size / patch\_size)\*\*2.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use [post\_process\_object\_detection()](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor.post_process_object_detection) to retrieve the
  unnormalized bounding boxes.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_max_text_queries, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [Owlv2TextModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2TextModel).
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`) — Pooled output of [Owlv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionModel). OWLv2 represents images as a set of image patches and computes image
  embeddings for each patch.
* **class\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`) — Class embeddings of all image patches. OWLv2 represents images as a set of image patches where the total
  number of patches is (image\_size / patch\_size)\*\*2.
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [Owlv2TextModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2TextModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output`, defaults to `None`) — The output of the [Owlv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionModel).

The [Owlv2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import requests
>>> from PIL import Image
>>> import torch

>>> from transformers import Owlv2Processor, Owlv2ForObjectDetection

>>> processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
>>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text_labels = [["a photo of a cat", "a photo of a dog"]]
>>> inputs = processor(text=text_labels, images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
>>> target_sizes = torch.tensor([(image.height, image.width)])
>>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
>>> results = processor.post_process_grounded_object_detection(
...     outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
... )
>>> # Retrieve predictions for the first image for the corresponding text queries
>>> result = results[0]
>>> boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
>>> for box, score, text_label in zip(boxes, scores, text_labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
Detected a photo of a cat with confidence 0.614 at location [341.67, 23.39, 642.32, 371.35]
Detected a photo of a cat with confidence 0.665 at location [6.75, 51.96, 326.62, 473.13]
```

#### image\_guided\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/owlv2/modeling_owlv2.py#L1481)

( pixel\_values: FloatTensor query\_pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → `transformers.models.owlv2.modeling_owlv2.Owlv2ImageGuidedObjectDetectionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor). See [Owlv2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Owlv2Processor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Processor) uses
  [Owlv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor) for processing images).
* **query\_pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) —
  Pixel values of query image(s) to be detected. Pass in one query image per target image.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.owlv2.modeling_owlv2.Owlv2ImageGuidedObjectDetectionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.owlv2.modeling_owlv2.Owlv2ImageGuidedObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Owlv2Config](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2Config)) and inputs.

* **logits** (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`) — Classification logits (including no-object) for all queries.
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`) — Pooled output of [Owlv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionModel). OWLv2 represents images as a set of image patches and computes
  image embeddings for each patch.
* **query\_image\_embeds** (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`) — Pooled output of [Owlv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionModel). OWLv2 represents images as a set of image patches and computes
  image embeddings for each patch.
* **target\_pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual target image in the batch
  (disregarding possible padding). You can use [post\_process\_object\_detection()](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor.post_process_object_detection) to
  retrieve the unnormalized bounding boxes.
* **query\_pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual query image in the batch
  (disregarding possible padding). You can use [post\_process\_object\_detection()](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2ImageProcessor.post_process_object_detection) to
  retrieve the unnormalized bounding boxes.
* **class\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`) — Class embeddings of all image patches. OWLv2 represents images as a set of image patches where the total
  number of patches is (image\_size / patch\_size)\*\*2.
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [Owlv2TextModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2TextModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output`, defaults to `None`) — The output of the [Owlv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/owlv2#transformers.Owlv2VisionModel).

Examples:


```
>>> import requests
>>> from PIL import Image
>>> import torch
>>> from transformers import AutoProcessor, Owlv2ForObjectDetection

>>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
>>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
>>> query_image = Image.open(requests.get(query_url, stream=True).raw)
>>> inputs = processor(images=image, query_images=query_image, return_tensors="pt")

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model.image_guided_detection(**inputs)

>>> target_sizes = torch.Tensor([image.size[::-1]])

>>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
>>> results = processor.post_process_image_guided_detection(
...     outputs=outputs, threshold=0.9, nms_threshold=0.3, target_sizes=target_sizes
... )
>>> i = 0  # Retrieve predictions for the first image
>>> boxes, scores = results[i]["boxes"], results[i]["scores"]
>>> for box, score in zip(boxes, scores):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")
Detected similar object with confidence 0.938 at location [327.31, 54.94, 547.39, 268.06]
Detected similar object with confidence 0.959 at location [5.78, 360.65, 619.12, 366.39]
Detected similar object with confidence 0.902 at location [2.85, 360.01, 627.63, 380.8]
Detected similar object with confidence 0.985 at location [176.98, -29.45, 672.69, 182.83]
Detected similar object with confidence 1.0 at location [6.53, 14.35, 624.87, 470.82]
Detected similar object with confidence 0.998 at location [579.98, 29.14, 615.49, 489.05]
Detected similar object with confidence 0.985 at location [206.15, 10.53, 247.74, 466.01]
Detected similar object with confidence 0.947 at location [18.62, 429.72, 646.5, 457.72]
Detected similar object with confidence 0.996 at location [523.88, 20.69, 586.84, 483.18]
Detected similar object with confidence 0.998 at location [3.39, 360.59, 617.29, 499.21]
Detected similar object with confidence 0.969 at location [4.47, 449.05, 614.5, 474.76]
Detected similar object with confidence 0.966 at location [31.44, 463.65, 654.66, 471.07]
Detected similar object with confidence 0.924 at location [30.93, 468.07, 635.35, 475.39]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/owlv2.md)
