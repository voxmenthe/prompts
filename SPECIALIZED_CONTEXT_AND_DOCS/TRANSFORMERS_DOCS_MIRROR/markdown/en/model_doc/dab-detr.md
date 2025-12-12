*This model was released on 2022-01-28 and added to Hugging Face Transformers on 2025-02-04.*

# DAB-DETR

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The DAB-DETR model was proposed in [DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR](https://huggingface.co/papers/2201.12329) by Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, Lei Zhang.
DAB-DETR is an enhanced variant of Conditional DETR. It utilizes dynamically updated anchor boxes to provide both a reference query point (x, y) and a reference anchor size (w, h), improving cross-attention computation. This new approach achieves 45.7% AP when trained for 50 epochs with a single ResNet-50 model as the backbone.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dab_detr_convergence_plot.png)

The abstract from the paper is the following:

*We present in this paper a novel query formulation using dynamic anchor boxes
for DETR (DEtection TRansformer) and offer a deeper understanding of the role
of queries in DETR. This new formulation directly uses box coordinates as queries
in Transformer decoders and dynamically updates them layer-by-layer. Using box
coordinates not only helps using explicit positional priors to improve the query-to-feature similarity and eliminate the slow training convergence issue in DETR,
but also allows us to modulate the positional attention map using the box width
and height information. Such a design makes it clear that queries in DETR can be
implemented as performing soft ROI pooling layer-by-layer in a cascade manner.
As a result, it leads to the best performance on MS-COCO benchmark among
the DETR-like detection models under the same setting, e.g., AP 45.7% using
ResNet50-DC5 as backbone trained in 50 epochs. We also conducted extensive
experiments to confirm our analysis and verify the effectiveness of our methods.*

This model was contributed by [davidhajdu](https://huggingface.co/davidhajdu).
The original code can be found [here](https://github.com/IDEA-Research/DAB-DETR).

## How to Get Started with the Model

Use the code below to get started with the model.


```
import torch
import requests

from PIL import Image
from transformers import AutoModelForObjectDetection, AutoImageProcessor

url = 'http://images.cocodataset.org/val2017/000000039769.jpg' 
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("IDEA-Research/dab-detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("IDEA-Research/dab-detr-resnet-50")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
```

This should output


```
cat: 0.87 [14.7, 49.39, 320.52, 469.28]
remote: 0.86 [41.08, 72.37, 173.39, 117.2]
cat: 0.86 [344.45, 19.43, 639.85, 367.86]
remote: 0.61 [334.27, 75.93, 367.92, 188.81]
couch: 0.59 [-0.04, 1.34, 639.9, 477.09]
```

There are three other ways to instantiate a DAB-DETR model (depending on what you prefer):

Option 1: Instantiate DAB-DETR with pre-trained weights for entire model


```
>>> from transformers import DabDetrForObjectDetection

>>> model = DabDetrForObjectDetection.from_pretrained("IDEA-Research/dab-detr-resnet-50")
```

Option 2: Instantiate DAB-DETR with randomly initialized weights for Transformer, but pre-trained weights for backbone


```
>>> from transformers import DabDetrConfig, DabDetrForObjectDetection

>>> config = DabDetrConfig()
>>> model = DabDetrForObjectDetection(config)
```

Option 3: Instantiate DAB-DETR with randomly initialized weights for backbone + Transformer


```
>>> config = DabDetrConfig(use_pretrained_backbone=False)
>>> model = DabDetrForObjectDetection(config)
```

## DabDetrConfig

### class transformers.DabDetrConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dab_detr/configuration_dab_detr.py#L26)

( use\_timm\_backbone = True backbone\_config = None backbone = 'resnet50' use\_pretrained\_backbone = True backbone\_kwargs = None num\_queries = 300 encoder\_layers = 6 encoder\_ffn\_dim = 2048 encoder\_attention\_heads = 8 decoder\_layers = 6 decoder\_ffn\_dim = 2048 decoder\_attention\_heads = 8 is\_encoder\_decoder = True activation\_function = 'prelu' hidden\_size = 256 dropout = 0.1 attention\_dropout = 0.0 activation\_dropout = 0.0 init\_std = 0.02 init\_xavier\_std = 1.0 auxiliary\_loss = False dilation = False class\_cost = 2 bbox\_cost = 5 giou\_cost = 2 cls\_loss\_coefficient = 2 bbox\_loss\_coefficient = 5 giou\_loss\_coefficient = 2 focal\_alpha = 0.25 temperature\_height = 20 temperature\_width = 20 query\_dim = 4 random\_refpoints\_xy = False keep\_query\_pos = False num\_patterns = 0 normalize\_before = False sine\_position\_embedding\_scale = None initializer\_bias\_prior\_prob = None \*\*kwargs  )

Parameters

* **use\_timm\_backbone** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [AutoBackbone](/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone)
  API.
* **backbone\_config** (`PretrainedConfig` or `dict`, *optional*) —
  The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
  case it will default to `ResNetConfig()`.
* **backbone** (`str`, *optional*, defaults to `"resnet50"`) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `True`) —
  Whether to use pretrained weights for the backbone.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **num\_queries** (`int`, *optional*, defaults to 300) —
  Number of object queries, i.e. detection slots. This is the maximal number of objects
  [DabDetrModel](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrModel) can detect in a single image. For COCO, we recommend 100 queries.
* **encoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of encoder layers.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 2048) —
  Dimension of the “intermediate” (often named feed-forward) layer in encoder.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **decoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of decoder layers.
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 2048) —
  Dimension of the “intermediate” (often named feed-forward) layer in decoder.
* **decoder\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Indicates whether the transformer model architecture is an encoder-decoder or not.
* **activation\_function** (`str` or `function`, *optional*, defaults to `"prelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **hidden\_size** (`int`, *optional*, defaults to 256) —
  This parameter is a general dimension parameter, defining dimensions for components such as the encoder layer and projection parameters in the decoder layer, among others.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **activation\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for activations inside the fully connected layer.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **init\_xavier\_std** (`float`, *optional*, defaults to 1.0) —
  The scaling factor used for the Xavier initialization gain in the HM Attention map module.
* **auxiliary\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
* **dilation** (`bool`, *optional*, defaults to `False`) —
  Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when `use_timm_backbone` = `True`.
* **class\_cost** (`float`, *optional*, defaults to 2) —
  Relative weight of the classification error in the Hungarian matching cost.
* **bbox\_cost** (`float`, *optional*, defaults to 5) —
  Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
* **giou\_cost** (`float`, *optional*, defaults to 2) —
  Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
* **cls\_loss\_coefficient** (`float`, *optional*, defaults to 2) —
  Relative weight of the classification loss in the object detection loss function.
* **bbox\_loss\_coefficient** (`float`, *optional*, defaults to 5) —
  Relative weight of the L1 bounding box loss in the object detection loss.
* **giou\_loss\_coefficient** (`float`, *optional*, defaults to 2) —
  Relative weight of the generalized IoU loss in the object detection loss.
* **focal\_alpha** (`float`, *optional*, defaults to 0.25) —
  Alpha parameter in the focal loss.
* **temperature\_height** (`int`, *optional*, defaults to 20) —
  Temperature parameter to tune the flatness of positional attention (HEIGHT)
* **temperature\_width** (`int`, *optional*, defaults to 20) —
  Temperature parameter to tune the flatness of positional attention (WIDTH)
* **query\_dim** (`int`, *optional*, defaults to 4) —
  Query dimension parameter represents the size of the output vector.
* **random\_refpoints\_xy** (`bool`, *optional*, defaults to `False`) —
  Whether to fix the x and y coordinates of the anchor boxes with random initialization.
* **keep\_query\_pos** (`bool`, *optional*, defaults to `False`) —
  Whether to concatenate the projected positional embedding from the object query into the original query (key) in every decoder layer.
* **num\_patterns** (`int`, *optional*, defaults to 0) —
  Number of pattern embeddings.
* **normalize\_before** (`bool`, *optional*, defaults to `False`) —
  Whether we use a normalization layer in the Encoder or not.
* **sine\_position\_embedding\_scale** (`float`, *optional*, defaults to ‘None’) —
  Scaling factor applied to the normalized positional encodings.
* **initializer\_bias\_prior\_prob** (`float`, *optional*) —
  The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
  If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.

This is the configuration class to store the configuration of a [DabDetrModel](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrModel). It is used to instantiate
a DAB-DETR model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the DAB-DETR
[IDEA-Research/dab\_detr-base](https://huggingface.co/IDEA-Research/dab_detr-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import DabDetrConfig, DabDetrModel

>>> # Initializing a DAB-DETR IDEA-Research/dab_detr-base style configuration
>>> configuration = DabDetrConfig()

>>> # Initializing a model (with random weights) from the IDEA-Research/dab_detr-base style configuration
>>> model = DabDetrModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## DabDetrModel

### class transformers.DabDetrModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dab_detr/modeling_dab_detr.py#L1159)

( config: DabDetrConfig  )

Parameters

* **config** ([DabDetrConfig](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DAB-DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
hidden-states, intermediate hidden states, reference points, output coordinates without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dab_detr/modeling_dab_detr.py#L1214)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.dab_detr.modeling_dab_detr.DabDetrModelOutput` or `tuple(torch.FloatTensor)`

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

`transformers.models.dab_detr.modeling_dab_detr.DabDetrModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.dab_detr.modeling_dab_detr.DabDetrModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DabDetrConfig](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrConfig)) and inputs.

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
* **reference\_points** (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, 2 (anchor points))`) — Reference points (reference points of each layer of the decoder).

The [DabDetrModel](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, AutoModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("IDEA-Research/dab_detr-base")
>>> model = AutoModel.from_pretrained("IDEA-Research/dab_detr-base")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**inputs)

>>> # the last hidden states are the final query embeddings of the Transformer decoder
>>> # these are of shape (batch_size, num_queries, hidden_size)
>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 300, 256]
```

## DabDetrForObjectDetection

### class transformers.DabDetrForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dab_detr/modeling_dab_detr.py#L1431)

( config: DabDetrConfig  )

Parameters

* **config** ([DabDetrConfig](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DAB\_DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
top, for tasks such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dab_detr/modeling_dab_detr.py#L1468)

( pixel\_values: FloatTensor pixel\_mask: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[list[dict]] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.dab_detr.modeling_dab_detr.DabDetrObjectDetectionOutput` or `tuple(torch.FloatTensor)`

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

`transformers.models.dab_detr.modeling_dab_detr.DabDetrObjectDetectionOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.dab_detr.modeling_dab_detr.DabDetrObjectDetectionOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DabDetrConfig](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
  bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
  scale-invariant IoU loss.
* **loss\_dict** (`Dict`, *optional*) — A dictionary containing the individual losses. Useful for logging.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`) — Classification logits (including no-object) for all queries.
* **pred\_boxes** (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`) — Normalized boxes coordinates for all queries, represented as (center\_x, center\_y, width, height). These
  values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
  possible padding). You can use `~DabDetrImageProcessor.post_process_object_detection` to retrieve the
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

The [DabDetrForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/dab-detr#transformers.DabDetrForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("IDEA-Research/dab-detr-resnet-50")
>>> model = AutoModelForObjectDetection.from_pretrained("IDEA-Research/dab-detr-resnet-50")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
>>>     outputs = model(**inputs)

>>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
>>> target_sizes = torch.tensor([(image.height, image.width)])
>>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     print(
...         f"Detected {model.config.id2label[label.item()]} with confidence "
...         f"{round(score.item(), 3)} at location {box}"
...     )
Detected remote with confidence 0.833 at location [38.31, 72.1, 177.63, 118.45]
Detected cat with confidence 0.831 at location [9.2, 51.38, 321.13, 469.0]
Detected cat with confidence 0.804 at location [340.3, 16.85, 642.93, 370.95]
Detected remote with confidence 0.683 at location [334.48, 73.49, 366.37, 190.01]
Detected couch with confidence 0.535 at location [0.52, 1.19, 640.35, 475.1]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dab-detr.md)
