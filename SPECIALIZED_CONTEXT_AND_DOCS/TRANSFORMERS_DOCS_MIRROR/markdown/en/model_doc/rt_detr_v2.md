*This model was released on 2024-07-24 and added to Hugging Face Transformers on 2025-02-06.*

# RT-DETRv2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The RT-DETRv2 model was proposed in [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://huggingface.co/papers/2407.17140) by Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu.

RT-DETRv2 refines RT-DETR by introducing selective multi-scale feature extraction, a discrete sampling operator for broader deployment compatibility, and improved training strategies like dynamic data augmentation and scale-adaptive hyperparameters. These changes enhance flexibility and practicality while maintaining real-time performance.

The abstract from the paper is the following:

*In this report, we present RT-DETRv2, an improved Real-Time DEtection TRansformer (RT-DETR). RT-DETRv2 builds upon the previous state-of-the-art real-time detector, RT-DETR, and opens up a set of bag-of-freebies for flexibility and practicality, as well as optimizing the training strategy to achieve enhanced performance. To improve the flexibility, we suggest setting a distinct number of sampling points for features at different scales in the deformable attention to achieve selective multi-scale feature extraction by the decoder. To enhance practicality, we propose an optional discrete sampling operator to replace the grid\_sample operator that is specific to RT-DETR compared to YOLOs. This removes the deployment constraints typically associated with DETRs. For the training strategy, we propose dynamic data augmentation and scale-adaptive hyperparameters customization to improve performance without loss of speed.*

This model was contributed by [jadechoghari](https://huggingface.co/jadechoghari).
The original code can be found [here](https://github.com/lyuwenyu/RT-DETR).

## Usage tips

This second version of RT-DETR improves how the decoder finds objects in an image.

* **better sampling** – adjusts offsets so the model looks at the right areas
* **flexible attention** – can use smooth (bilinear) or fixed (discrete) sampling
* **optimized processing** – improves how attention weights mix information


```
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
>>> model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

>>> for result in results:
...     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
...         score, label = score.item(), label_id.item()
...         box = [round(i, 2) for i in box.tolist()]
...         print(f"{model.config.id2label[label]}: {score:.2f} {box}")
cat: 0.97 [341.14, 25.11, 639.98, 372.89]
cat: 0.96 [12.78, 56.35, 317.67, 471.34]
remote: 0.95 [39.96, 73.12, 175.65, 117.44]
sofa: 0.86 [-0.11, 2.97, 639.89, 473.62]
sofa: 0.82 [-0.12, 1.78, 639.87, 473.52]
remote: 0.79 [333.65, 76.38, 370.69, 187.48]
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with RT-DETRv2.

Object Detection

* Scripts for finetuning [RTDetrV2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2ForObjectDetection) with [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
* See also: [Object detection task guide](../tasks/object_detection).
* Notebooks for [inference](https://github.com/qubvel/transformers-notebooks/blob/main/notebooks/RT_DETR_v2_inference.ipynb) and [fine-tuning](https://github.com/qubvel/transformers-notebooks/blob/main/notebooks/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb) RT-DETRv2 on a custom dataset (🌎).

## RTDetrV2Config

### class transformers.RTDetrV2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/configuration_rt_detr_v2.py#L31)

( initializer\_range = 0.01 initializer\_bias\_prior\_prob = None layer\_norm\_eps = 1e-05 batch\_norm\_eps = 1e-05 backbone\_config = None backbone = None use\_pretrained\_backbone = False use\_timm\_backbone = False freeze\_backbone\_batch\_norms = True backbone\_kwargs = None encoder\_hidden\_dim = 256 encoder\_in\_channels = [512, 1024, 2048] feat\_strides = [8, 16, 32] encoder\_layers = 1 encoder\_ffn\_dim = 1024 encoder\_attention\_heads = 8 dropout = 0.0 activation\_dropout = 0.0 encode\_proj\_layers = [2] positional\_encoding\_temperature = 10000 encoder\_activation\_function = 'gelu' activation\_function = 'silu' eval\_size = None normalize\_before = False hidden\_expansion = 1.0 d\_model = 256 num\_queries = 300 decoder\_in\_channels = [256, 256, 256] decoder\_ffn\_dim = 1024 num\_feature\_levels = 3 decoder\_n\_points = 4 decoder\_layers = 6 decoder\_attention\_heads = 8 decoder\_activation\_function = 'relu' attention\_dropout = 0.0 num\_denoising = 100 label\_noise\_ratio = 0.5 box\_noise\_scale = 1.0 learn\_initial\_query = False anchor\_image\_size = None with\_box\_refine = True is\_encoder\_decoder = True matcher\_alpha = 0.25 matcher\_gamma = 2.0 matcher\_class\_cost = 2.0 matcher\_bbox\_cost = 5.0 matcher\_giou\_cost = 2.0 use\_focal\_loss = True auxiliary\_loss = True focal\_loss\_alpha = 0.75 focal\_loss\_gamma = 2.0 weight\_loss\_vfl = 1.0 weight\_loss\_bbox = 5.0 weight\_loss\_giou = 2.0 eos\_coefficient = 0.0001 decoder\_n\_levels = 3 decoder\_offset\_scale = 0.5 decoder\_method = 'default' \*\*kwargs  )

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
* **backbone\_config** (`Dict`, *optional*, defaults to `RTDetrV2ResNetConfig()`) —
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
  Height and width used to compute the effective height and width of the position embeddings after taking
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
  Parameter informing if focal loss should be used.
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
* **decoder\_n\_levels** (`int`, *optional*, defaults to 3) —
  The number of feature levels used by the decoder.
* **decoder\_offset\_scale** (`float`, *optional*, defaults to 0.5) —
  Scaling factor applied to the attention offsets in the decoder.
* **decoder\_method** (`str`, *optional*, defaults to `"default"`) —
  The method to use for the decoder: `"default"` or `"discrete"`.

This is the configuration class to store the configuration of a [RTDetrV2Model](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Model). It is used to instantiate a
RT-DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the RT-DETR architecture.

e.g. [PekingU/rtdetr\_r18vd](https://huggingface.co/PekingU/rtdetr_r18vd)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import RTDetrV2Config, RTDetrV2Model

>>> # Initializing a RT-DETR configuration
>>> configuration = RTDetrV2Config()

>>> # Initializing a model (with random weights) from the configuration
>>> model = RTDetrV2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

#### from\_backbone\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/configuration_rt_detr_v2.py#L369)

( backbone\_config: PretrainedConfig \*\*kwargs  ) → [RTDetrV2Config](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config)

Parameters

* **backbone\_config** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) —
  The backbone configuration.

Returns

[RTDetrV2Config](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config)

An instance of a configuration object

Instantiate a [RTDetrV2Config](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config) (or a derived class) from a pre-trained backbone model configuration and DETR model
configuration.

## RTDetrV2Model

### class transformers.RTDetrV2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py#L1367)

( config: RTDetrV2Config  )

Parameters

* **config** ([RTDetrV2Config](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config)) —
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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py#L1480)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[list[dict]] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
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

`transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RTDetrV2Config](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config)) and inputs.

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

The [RTDetrV2Model](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, RTDetrV2Model
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("PekingU/RTDetrV2_r50vd")
>>> model = RTDetrV2Model.from_pretrained("PekingU/RTDetrV2_r50vd")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 300, 256]
```

## RTDetrV2ForObjectDetection

### class transformers.RTDetrV2ForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py#L1810)

( config: RTDetrV2Config  )

Parameters

* **config** ([RTDetrV2Config](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config)) —
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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py#L1841)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[list[dict]] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → `transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ObjectDetectionOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
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

`transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ObjectDetectionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([RTDetrV2Config](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) — A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`) — Classification logits (including no-object) for all queries.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use `~RTDetrV2ImageProcessor.post_process_object_detection` to retrieve the
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

The [RTDetrV2ForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2ForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import RTDetrV2ImageProcessor, RTDetrV2ForObjectDetection
>>> from PIL import Image
>>> import requests
>>> import torch

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = RTDetrV2ImageProcessor.from_pretrained("PekingU/RTDetrV2_r50vd")
>>> model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/RTDetrV2_r50vd")

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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/rt_detr_v2.md)
