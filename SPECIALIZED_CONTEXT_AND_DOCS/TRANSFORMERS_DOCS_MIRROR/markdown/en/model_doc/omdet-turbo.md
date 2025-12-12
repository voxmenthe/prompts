*This model was released on 2024-03-11 and added to Hugging Face Transformers on 2024-09-25.*

# OmDet-Turbo

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The OmDet-Turbo model was proposed in [Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head](https://huggingface.co/papers/2403.06892) by Tiancheng Zhao, Peng Liu, Xuan He, Lu Zhang, Kyusong Lee. OmDet-Turbo incorporates components from RT-DETR and introduces a swift multimodal fusion module to achieve real-time open-vocabulary object detection capabilities while maintaining high accuracy. The base model achieves performance of up to 100.2 FPS and 53.4 AP on COCO zero-shot.

The abstract from the paper is the following:

*End-to-end transformer-based detectors (DETRs) have shown exceptional performance in both closed-set and open-vocabulary object detection (OVD) tasks through the integration of language modalities. However, their demanding computational requirements have hindered their practical application in real-time object detection (OD) scenarios. In this paper, we scrutinize the limitations of two leading models in the OVDEval benchmark, OmDet and Grounding-DINO, and introduce OmDet-Turbo. This novel transformer-based real-time OVD model features an innovative Efficient Fusion Head (EFH) module designed to alleviate the bottlenecks observed in OmDet and Grounding-DINO. Notably, OmDet-Turbo-Base achieves a 100.2 frames per second (FPS) with TensorRT and language cache techniques applied. Notably, in zero-shot scenarios on COCO and LVIS datasets, OmDet-Turbo achieves performance levels nearly on par with current state-of-the-art supervised models. Furthermore, it establishes new state-of-the-art benchmarks on ODinW and OVDEval, boasting an AP of 30.1 and an NMS-AP of 26.86, respectively. The practicality of OmDet-Turbo in industrial applications is underscored by its exceptional performance on benchmark datasets and superior inference speed, positioning it as a compelling choice for real-time object detection tasks.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/omdet_turbo_architecture.jpeg) OmDet-Turbo architecture overview. Taken from the [original paper](https://huggingface.co/papers/2403.06892).

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/om-ai-lab/OmDet).

## Usage tips

One unique property of OmDet-Turbo compared to other zero-shot object detection models, such as [Grounding DINO](grounding-dino), is the decoupled classes and prompt embedding structure that allows caching of text embeddings. This means that the model needs both classes and task as inputs, where classes is a list of objects we want to detect and task is the grounded text used to guide open-vocabulary detection. This approach limits the scope of the open-vocabulary detection and makes the decoding process faster.

[OmDetTurboProcessor](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboProcessor) is used to prepare the classes, task and image triplet. The task input is optional, and when not provided, it will default to `"Detect [class1], [class2], [class3], ..."`. To process the results from the model, one can use `post_process_grounded_object_detection` from [OmDetTurboProcessor](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboProcessor). Notably, this function takes in the input classes, as unlike other zero-shot object detection models, the decoupling of classes and task embeddings means that no decoding of the predicted class embeddings is needed in the post-processing step, and the predicted classes can be matched to the inputted ones directly.

## Usage example

### Single image inference

Here’s how to load the model and prepare the inputs to perform zero-shot object detection on a single image:


```
>>> import torch
>>> import requests
>>> from PIL import Image

>>> from transformers import AutoProcessor, OmDetTurboForObjectDetection

>>> processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
>>> model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text_labels = ["cat", "remote"]
>>> inputs = processor(image, text=text_labels, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # convert outputs (bounding boxes and class logits)
>>> results = processor.post_process_grounded_object_detection(
...     outputs,
...     target_sizes=[(image.height, image.width)],
...     text_labels=text_labels,
...     threshold=0.3,
...     nms_threshold=0.3,
... )
>>> result = results[0]
>>> boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]
>>> for box, score, text_label in zip(boxes, scores, text_labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
Detected remote with confidence 0.768 at location [39.89, 70.35, 176.74, 118.04]
Detected cat with confidence 0.72 at location [11.6, 54.19, 314.8, 473.95]
Detected remote with confidence 0.563 at location [333.38, 75.77, 370.7, 187.03]
Detected cat with confidence 0.552 at location [345.15, 23.95, 639.75, 371.67]
```

### Multi image inference

OmDet-Turbo can perform batched multi-image inference, with support for different text prompts and classes in the same batch:


```
>>> import torch
>>> import requests
>>> from io import BytesIO
>>> from PIL import Image
>>> from transformers import AutoProcessor, OmDetTurboForObjectDetection

>>> processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
>>> model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")

>>> url1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image1 = Image.open(BytesIO(requests.get(url1).content)).convert("RGB")
>>> text_labels1 = ["cat", "remote"]
>>> task1 = "Detect {}.".format(", ".join(text_labels1))

>>> url2 = "http://images.cocodataset.org/train2017/000000257813.jpg"
>>> image2 = Image.open(BytesIO(requests.get(url2).content)).convert("RGB")
>>> text_labels2 = ["boat"]
>>> task2 = "Detect everything that looks like a boat."

>>> url3 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
>>> image3 = Image.open(BytesIO(requests.get(url3).content)).convert("RGB")
>>> text_labels3 = ["statue", "trees"]
>>> task3 = "Focus on the foreground, detect statue and trees."

>>> inputs = processor(
...     images=[image1, image2, image3],
...     text=[text_labels1, text_labels2, text_labels3],
...     task=[task1, task2, task3],
...     return_tensors="pt",
... )

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # convert outputs (bounding boxes and class logits)
>>> results = processor.post_process_grounded_object_detection(
...     outputs,
...     text_labels=[text_labels1, text_labels2, text_labels3],
...     target_sizes=[(image.height, image.width) for image in [image1, image2, image3]],
...     threshold=0.2,
...     nms_threshold=0.3,
... )

>>> for i, result in enumerate(results):
...     for score, text_label, box in zip(
...         result["scores"], result["text_labels"], result["boxes"]
...     ):
...         box = [round(i, 1) for i in box.tolist()]
...         print(
...             f"Detected {text_label} with confidence "
...             f"{round(score.item(), 2)} at location {box} in image {i}"
...         )
Detected remote with confidence 0.77 at location [39.9, 70.4, 176.7, 118.0] in image 0
Detected cat with confidence 0.72 at location [11.6, 54.2, 314.8, 474.0] in image 0
Detected remote with confidence 0.56 at location [333.4, 75.8, 370.7, 187.0] in image 0
Detected cat with confidence 0.55 at location [345.2, 24.0, 639.8, 371.7] in image 0
Detected boat with confidence 0.32 at location [146.9, 219.8, 209.6, 250.7] in image 1
Detected boat with confidence 0.3 at location [319.1, 223.2, 403.2, 238.4] in image 1
Detected boat with confidence 0.27 at location [37.7, 220.3, 84.0, 235.9] in image 1
Detected boat with confidence 0.22 at location [407.9, 207.0, 441.7, 220.2] in image 1
Detected statue with confidence 0.73 at location [544.7, 210.2, 651.9, 502.8] in image 2
Detected trees with confidence 0.25 at location [3.9, 584.3, 391.4, 785.6] in image 2
Detected trees with confidence 0.25 at location [1.4, 621.2, 118.2, 787.8] in image 2
Detected statue with confidence 0.2 at location [428.1, 205.5, 767.3, 759.5] in image 2
```

## OmDetTurboConfig

### class transformers.OmDetTurboConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/omdet_turbo/configuration_omdet_turbo.py#L26)

( text\_config = None backbone\_config = None use\_timm\_backbone = True backbone = 'swin\_tiny\_patch4\_window7\_224' backbone\_kwargs = None use\_pretrained\_backbone = False apply\_layernorm\_after\_vision\_backbone = True image\_size = 640 disable\_custom\_kernels = False layer\_norm\_eps = 1e-05 batch\_norm\_eps = 1e-05 init\_std = 0.02 text\_projection\_in\_dim = 512 text\_projection\_out\_dim = 512 task\_encoder\_hidden\_dim = 1024 class\_embed\_dim = 512 class\_distance\_type = 'cosine' num\_queries = 900 csp\_activation = 'silu' conv\_norm\_activation = 'gelu' encoder\_feedforward\_activation = 'relu' encoder\_feedforward\_dropout = 0.0 encoder\_dropout = 0.0 hidden\_expansion = 1 vision\_features\_channels = [256, 256, 256] encoder\_hidden\_dim = 256 encoder\_in\_channels = [192, 384, 768] encoder\_projection\_indices = [2] encoder\_attention\_heads = 8 encoder\_dim\_feedforward = 2048 encoder\_layers = 1 positional\_encoding\_temperature = 10000 num\_feature\_levels = 3 decoder\_hidden\_dim = 256 decoder\_num\_heads = 8 decoder\_num\_layers = 6 decoder\_activation = 'relu' decoder\_dim\_feedforward = 2048 decoder\_num\_points = 4 decoder\_dropout = 0.0 eval\_size = None learn\_initial\_query = False cache\_size = 100 is\_encoder\_decoder = True \*\*kwargs  )

Parameters

* **text\_config** (`PretrainedConfig`, *optional*) —
  The configuration of the text backbone.
* **backbone\_config** (`PretrainedConfig`, *optional*) —
  The configuration of the vision backbone.
* **use\_timm\_backbone** (`bool`, *optional*, defaults to `True`) —
  Whether to use the timm for the vision backbone.
* **backbone** (`str`, *optional*, defaults to `"swin_tiny_patch4_window7_224"`) —
  The name of the pretrained vision backbone to use. If `use_pretrained_backbone=False` a randomly initialized
  backbone with the same architecture `backbone` is used.
* **backbone\_kwargs** (`dict`, *optional*) —
  Additional kwargs for the vision backbone.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to use a pretrained vision backbone.
* **apply\_layernorm\_after\_vision\_backbone** (`bool`, *optional*, defaults to `True`) —
  Whether to apply layer normalization on the feature maps of the vision backbone output.
* **image\_size** (`int`, *optional*, defaults to 640) —
  The size (resolution) of each image.
* **disable\_custom\_kernels** (`bool`, *optional*, defaults to `False`) —
  Whether to disable custom kernels.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon value for layer normalization.
* **batch\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon value for batch normalization.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **text\_projection\_in\_dim** (`int`, *optional*, defaults to 512) —
  The input dimension for the text projection.
* **text\_projection\_out\_dim** (`int`, *optional*, defaults to 512) —
  The output dimension for the text projection.
* **task\_encoder\_hidden\_dim** (`int`, *optional*, defaults to 1024) —
  The feedforward dimension for the task encoder.
* **class\_embed\_dim** (`int`, *optional*, defaults to 512) —
  The dimension of the classes embeddings.
* **class\_distance\_type** (`str`, *optional*, defaults to `"cosine"`) —
  The type of of distance to compare predicted classes to projected classes embeddings.
  Can be `"cosine"` or `"dot"`.
* **num\_queries** (`int`, *optional*, defaults to 900) —
  The number of queries.
* **csp\_activation** (`str`, *optional*, defaults to `"silu"`) —
  The activation function of the Cross Stage Partial (CSP) networks of the encoder.
* **conv\_norm\_activation** (`str`, *optional*, defaults to `"gelu"`) —
  The activation function of the ConvNormLayer layers of the encoder.
* **encoder\_feedforward\_activation** (`str`, *optional*, defaults to `"relu"`) —
  The activation function for the feedforward network of the encoder.
* **encoder\_feedforward\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout rate following the activation of the encoder feedforward network.
* **encoder\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout rate of the encoder multi-head attention module.
* **hidden\_expansion** (`int`, *optional*, defaults to 1) —
  The hidden expansion of the CSP networks in the encoder.
* **vision\_features\_channels** (`tuple(int)`, *optional*, defaults to `[256, 256, 256]`) —
  The projected vision features channels used as inputs for the decoder.
* **encoder\_hidden\_dim** (`int`, *optional*, defaults to 256) —
  The hidden dimension of the encoder.
* **encoder\_in\_channels** (`List(int)`, *optional*, defaults to `[192, 384, 768]`) —
  The input channels for the encoder.
* **encoder\_projection\_indices** (`List(int)`, *optional*, defaults to `[2]`) —
  The indices of the input features projected by each layers.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 8) —
  The number of attention heads for the encoder.
* **encoder\_dim\_feedforward** (`int`, *optional*, defaults to 2048) —
  The feedforward dimension for the encoder.
* **encoder\_layers** (`int`, *optional*, defaults to 1) —
  The number of layers in the encoder.
* **positional\_encoding\_temperature** (`int`, *optional*, defaults to 10000) —
  The positional encoding temperature in the encoder.
* **num\_feature\_levels** (`int`, *optional*, defaults to 3) —
  The number of feature levels for the multi-scale deformable attention module of the decoder.
* **decoder\_hidden\_dim** (`int`, *optional*, defaults to 256) —
  The hidden dimension of the decoder.
* **decoder\_num\_heads** (`int`, *optional*, defaults to 8) —
  The number of heads for the decoder.
* **decoder\_num\_layers** (`int`, *optional*, defaults to 6) —
  The number of layers for the decoder.
* **decoder\_activation** (`str`, *optional*, defaults to `"relu"`) —
  The activation function for the decoder.
* **decoder\_dim\_feedforward** (`int`, *optional*, defaults to 2048) —
  The feedforward dimension for the decoder.
* **decoder\_num\_points** (`int`, *optional*, defaults to 4) —
  The number of points sampled in the decoder multi-scale deformable attention module.
* **decoder\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout rate for the decoder.
* **eval\_size** (`tuple[int, int]`, *optional*) —
  Height and width used to computes the effective height and width of the position embeddings after taking
  into account the stride (see RTDetr).
* **learn\_initial\_query** (`bool`, *optional*, defaults to `False`) —
  Whether to learn the initial query.
* **cache\_size** (`int`, *optional*, defaults to 100) —
  The cache size for the classes and prompts caches.
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Whether the model is used as an encoder-decoder model or not.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional parameters from the architecture. The values in kwargs will be saved as part of the configuration
  and can be used to control the model outputs.

This is the configuration class to store the configuration of a [OmDetTurboForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboForObjectDetection).
It is used to instantiate a OmDet-Turbo model according to the specified arguments, defining the model architecture
Instantiating a configuration with the defaults will yield a similar configuration to that of the OmDet-Turbo
[omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import OmDetTurboConfig, OmDetTurboForObjectDetection

>>> # Initializing a OmDet-Turbo omlab/omdet-turbo-swin-tiny-hf style configuration
>>> configuration = OmDetTurboConfig()

>>> # Initializing a model (with random weights) from the omlab/omdet-turbo-swin-tiny-hf style configuration
>>> model = OmDetTurboForObjectDetection(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## OmDetTurboProcessor

### class transformers.OmDetTurboProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/omdet_turbo/processing_omdet_turbo.py#L203)

( image\_processor tokenizer  )

Parameters

* **image\_processor** (`DetrImageProcessor`) —
  An instance of [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor). The image processor is a required input.
* **tokenizer** (`AutoTokenizer`) —
  An instance of [‘PreTrainedTokenizer`]. The tokenizer is a required input.

Constructs a OmDet-Turbo processor which wraps a Deformable DETR image processor and an AutoTokenizer into a
single processor.

[OmDetTurboProcessor](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboProcessor) offers all the functionalities of [DetrImageProcessor](/docs/transformers/v4.56.2/en/model_doc/detr#transformers.DetrImageProcessor) and
[AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See the docstring of `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode)
for more information.

#### post\_process\_grounded\_object\_detection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/omdet_turbo/processing_omdet_turbo.py#L317)

( outputs: OmDetTurboObjectDetectionOutput text\_labels: typing.Union[list[str], list[list[str]], NoneType] = None threshold: float = 0.3 nms\_threshold: float = 0.5 target\_sizes: typing.Union[transformers.utils.generic.TensorType, list[tuple], NoneType] = None max\_num\_det: typing.Optional[int] = None  ) → `list[Dict]`

Parameters

* **outputs** (`OmDetTurboObjectDetectionOutput`) —
  Raw outputs of the model.
* **text\_labels** (Union[list[str], list[list[str]]], *optional*) —
  The input classes names. If not provided, `text_labels` will be set to `None` in `outputs`.
* **threshold** (float, defaults to 0.3) —
  Only return detections with a confidence score exceeding this threshold.
* **nms\_threshold** (float, defaults to 0.5) —
  The threshold to use for box non-maximum suppression. Value in [0, 1].
* **target\_sizes** (`torch.Tensor` or `list[tuple[int, int]]`, *optional*) —
  Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
  `(height, width)` of each image in the batch. If unset, predictions will not be resized.
* **max\_num\_det** (`int`, *optional*) —
  The maximum number of detections to return.

Returns

`list[Dict]`

A list of dictionaries, each dictionary containing the scores, classes and boxes for an image
in the batch as predicted by the model.

Converts the raw output of [OmDetTurboForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y,
bottom\_right\_x, bottom\_right\_y) format and get the associated text class.

## OmDetTurboForObjectDetection

### class transformers.OmDetTurboForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/omdet_turbo/modeling_omdet_turbo.py#L1462)

( config: OmDetTurboConfig  )

Parameters

* **config** ([OmDetTurboConfig](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OmDetTurbo Model (consisting of a vision and a text backbone, and encoder-decoder architecture) outputting
bounding boxes and classes scores for tasks such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/omdet_turbo/modeling_omdet_turbo.py#L1492)

( pixel\_values: FloatTensor classes\_input\_ids: LongTensor classes\_attention\_mask: LongTensor tasks\_input\_ids: LongTensor tasks\_attention\_mask: LongTensor classes\_structure: LongTensor labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.omdet_turbo.modeling_omdet_turbo.OmDetTurboObjectDetectionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
* **classes\_input\_ids** (`torch.LongTensor` of shape `(total_classes (>= batch_size), sequence_length)`) —
  Indices of input classes sequence tokens in the vocabulary of the language model.
  Several classes can be provided for each tasks, thus the tokenized classes are flattened
  and the structure of the classes is provided in the `classes_structure` argument.

  Indices can be obtained using [OmDetTurboProcessor](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboProcessor). See `OmDetTurboProcessor.__call__()` for
  details.

  [What are input IDs?](../glossary#input-ids)
* **classes\_attention\_mask** (`torch.BoolTensor` of shape `(total_classes (>= batch_size), num_classes, sequence_length)`) —
  Attention mask for the classes. This is a binary mask that indicates which tokens should be attended to,
  and which should not.
* **tasks\_input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input tasks sequence tokens in the vocabulary of the language model.

  Indices can be obtained using [OmDetTurboProcessor](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboProcessor). See `OmDetTurboProcessor.__call__()` for
  details.

  [What are input IDs?](../glossary#input-ids)
* **tasks\_attention\_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`) —
  Attention mask for the tasks. This is a binary mask that indicates which tokens should be attended to,
  and which should not.
* **classes\_structure** (`torch.LongTensor` of shape `(batch_size)`) —
  Structure of the classes. This tensor indicates the number of classes for each task.
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

`transformers.models.omdet_turbo.modeling_omdet_turbo.OmDetTurboObjectDetectionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.omdet_turbo.modeling_omdet_turbo.OmDetTurboObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OmDetTurboConfig](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboConfig)) and inputs.

* **loss** (`torch.FloatTensor`, *optional*, defaults to `None`) — The loss value.
* **decoder\_coord\_logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — The predicted coordinates logits of the objects.
* **decoder\_class\_logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes)`) — The predicted class of the objects.
* **init\_reference\_points** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — The initial reference points.
* **intermediate\_reference\_points** (`tuple[tuple[torch.FloatTensor]]`, *optional*, defaults to `None`) — The intermediate reference points.
* **encoder\_coord\_logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — The predicted coordinates of the objects from the encoder.
* **encoder\_class\_logits** (`tuple[torch.FloatTensor].encoder_class_logits`, defaults to `None`) — The predicted class of the objects from the encoder.
* **encoder\_extracted\_states** (`torch.FloatTensor`, *optional*, defaults to `None`) — The extracted states from the Feature Pyramid Network (FPN) and Path Aggregation Network (PAN) of the encoder.
* **decoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape
  `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
  plus the initial embedding outputs.
* **decoder\_attentions** (`tuple[tuple[torch.FloatTensor]]`, *optional*) — Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
  weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
* **encoder\_hidden\_states** (`tuple[torch.FloatTensor]`, *optional*) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape
  `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
  plus the initial embedding outputs.
* **encoder\_attentions** (`tuple[tuple[torch.FloatTensor]]`, *optional*) — Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
  weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
* **classes\_structure** (`torch.LongTensor`, *optional*) — The number of queried classes for each image.

The [OmDetTurboForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/omdet-turbo#transformers.OmDetTurboForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import requests
>>> from PIL import Image

>>> from transformers import AutoProcessor, OmDetTurboForObjectDetection

>>> processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
>>> model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> classes = ["cat", "remote"]
>>> task = "Detect {}.".format(", ".join(classes))
>>> inputs = processor(image, text=classes, task=task, return_tensors="pt")

>>> outputs = model(**inputs)

>>> # convert outputs (bounding boxes and class logits)
>>> results = processor.post_process_grounded_object_detection(
...     outputs,
...     classes=classes,
...     target_sizes=[image.size[::-1]],
...     score_threshold=0.3,
...     nms_threshold=0.3,
>>> )[0]
>>> for score, class_name, box in zip(results["scores"], results["classes"], results["boxes"]):
...     box = [round(i, 1) for i in box.tolist()]
...     print(
...         f"Detected {class_name} with confidence "
...         f"{round(score.item(), 2)} at location {box}"
...     )
Detected remote with confidence 0.76 at location [39.9, 71.3, 176.5, 117.9]
Detected cat with confidence 0.72 at location [345.1, 22.5, 639.7, 371.9]
Detected cat with confidence 0.65 at location [12.7, 53.8, 315.5, 475.3]
Detected remote with confidence 0.57 at location [333.4, 75.6, 370.7, 187.0]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/omdet-turbo.md)
