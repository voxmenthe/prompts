*This model was released on 2024-01-04 and added to Hugging Face Transformers on 2025-08-01.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# MM Grounding DINO

[MM Grounding DINO](https://huggingface.co/papers/2401.02361) model was proposed in [An Open and Comprehensive Pipeline for Unified Object Grounding and Detection](https://huggingface.co/papers/2401.02361) by Xiangyu Zhao, Yicheng Chen, Shilin Xu, Xiangtai Li, Xinjiang Wang, Yining Li, Haian Huang>.

MM Grounding DINO improves upon the [Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino) by improving the contrastive class head and removing the parameter sharing in the decoder, improving zero-shot detection performance on both COCO (50.6(+2.2) AP) and LVIS (31.9(+11.8) val AP and 41.4(+12.6) minival AP).

You can find all the original MM Grounding DINO checkpoints under the [MM Grounding DINO](https://huggingface.co/collections/openmmlab-community/mm-grounding-dino-688cbde05b814c4e2832f9df) collection. This model also supports LLMDet inference. You can find LLMDet checkpoints under the [LLMDet](https://huggingface.co/collections/iSEE-Laboratory/llmdet-688475906dc235d5f1dc678e) collection.

Click on the MM Grounding DINO models in the right sidebar for more examples of how to apply MM Grounding DINO to different MM Grounding DINO tasks.

The example below demonstrates how to generate text based on an image with the [AutoModelForZeroShotObjectDetection](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModelForZeroShotObjectDetection) class.

AutoModel


```
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, infer_device
from transformers.image_utils import load_image


# Prepare processor and model
model_id = "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"
device = infer_device()
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Prepare inputs
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(image_url)
text_labels = [["a cat", "a remote control"]]
inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Postprocess outputs
results = processor.post_process_grounded_object_detection(
    outputs,
    threshold=0.4,
    target_sizes=[(image.height, image.width)]
)

# Retrieve the first image result
result = results[0]
for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
```

## Notes

* Here’s a table of models and their object detection performance results on COCO (results from [official repo](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)):

  | Model | Backbone | Pre-Train Data | Style | COCO mAP |
  | --- | --- | --- | --- | --- |
  | [mm\_grounding\_dino\_tiny\_o365v1\_goldg](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg) | Swin-T | O365,GoldG | Zero-shot | 50.4(+2.3) |
  | [mm\_grounding\_dino\_tiny\_o365v1\_goldg\_grit](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit) | Swin-T | O365,GoldG,GRIT | Zero-shot | 50.5(+2.1) |
  | [mm\_grounding\_dino\_tiny\_o365v1\_goldg\_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det) | Swin-T | O365,GoldG,V3Det | Zero-shot | 50.6(+2.2) |
  | [mm\_grounding\_dino\_tiny\_o365v1\_goldg\_grit\_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det) | Swin-T | O365,GoldG,GRIT,V3Det | Zero-shot | 50.4(+2.0) |
  | [mm\_grounding\_dino\_base\_o365v1\_goldg\_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det) | Swin-B | O365,GoldG,V3Det | Zero-shot | 52.5 |
  | [mm\_grounding\_dino\_base\_all](https://huggingface.co/openmmlab-community/mm_grounding_dino_base_all) | Swin-B | O365,ALL | - | 59.5 |
  | [mm\_grounding\_dino\_large\_o365v2\_oiv6\_goldg](https://huggingface.co/openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg) | Swin-L | O365V2,OpenImageV6,GoldG | Zero-shot | 53.0 |
  | [mm\_grounding\_dino\_large\_all](https://huggingface.co/openmmlab-community/mm_grounding_dino_large_all) | Swin-L | O365V2,OpenImageV6,ALL | - | 60.3 |
* Here’s a table of MM Grounding DINO tiny models and their object detection performance on LVIS (results from [official repo](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)):

  | Model | Pre-Train Data | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP | Val1.0 APr | Val1.0 APc | Val1.0 APf | Val1.0 AP |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | [mm\_grounding\_dino\_tiny\_o365v1\_goldg](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg) | O365,GoldG | 28.1 | 30.2 | 42.0 | 35.7(+6.9) | 17.1 | 22.4 | 36.5 | 27.0(+6.9) |
  | [mm\_grounding\_dino\_tiny\_o365v1\_goldg\_grit](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit) | O365,GoldG,GRIT | 26.6 | 32.4 | 41.8 | 36.5(+7.7) | 17.3 | 22.6 | 36.4 | 27.1(+7.0) |
  | [mm\_grounding\_dino\_tiny\_o365v1\_goldg\_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det) | O365,GoldG,V3Det | 33.0 | 36.0 | 45.9 | 40.5(+11.7) | 21.5 | 25.5 | 40.2 | 30.6(+10.5) |
  | [mm\_grounding\_dino\_tiny\_o365v1\_goldg\_grit\_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det) | O365,GoldG,GRIT,V3Det | 34.2 | 37.4 | 46.2 | 41.4(+12.6) | 23.6 | 27.6 | 40.5 | 31.9(+11.8) |

* This implementation also supports inference for [LLMDet](https://github.com/iSEE-Laboratory/LLMDet). Here’s a table of LLMDet models and their performance on LVIS (results from [official repo](https://github.com/iSEE-Laboratory/LLMDet)):

  | Model | Pre-Train Data | MiniVal APr | MiniVal APc | MiniVal APf | MiniVal AP | Val1.0 APr | Val1.0 APc | Val1.0 APf | Val1.0 AP |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | [llmdet\_tiny](https://huggingface.co/iSEE-Laboratory/llmdet_tiny) | (O365,GoldG,GRIT,V3Det) + GroundingCap-1M | 44.7 | 37.3 | 39.5 | 50.7 | 34.9 | 26.0 | 30.1 | 44.3 |
  | [llmdet\_base](https://huggingface.co/iSEE-Laboratory/llmdet_base) | (O365,GoldG,V3Det) + GroundingCap-1M | 48.3 | 40.8 | 43.1 | 54.3 | 38.5 | 28.2 | 34.3 | 47.8 |
  | [llmdet\_large](https://huggingface.co/iSEE-Laboratory/llmdet_large) | (O365V2,OpenImageV6,GoldG) + GroundingCap-1M | 51.1 | 45.1 | 46.1 | 56.6 | 42.0 | 31.6 | 38.8 | 50.2 |

## MMGroundingDinoConfig

### class transformers.MMGroundingDinoConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mm_grounding_dino/configuration_mm_grounding_dino.py#L31)

( backbone\_config = None backbone = None use\_pretrained\_backbone = False use\_timm\_backbone = False backbone\_kwargs = None text\_config = None num\_queries = 900 encoder\_layers = 6 encoder\_ffn\_dim = 2048 encoder\_attention\_heads = 8 decoder\_layers = 6 decoder\_ffn\_dim = 2048 decoder\_attention\_heads = 8 is\_encoder\_decoder = True activation\_function = 'relu' d\_model = 256 dropout = 0.1 attention\_dropout = 0.0 activation\_dropout = 0.0 auxiliary\_loss = False position\_embedding\_type = 'sine' num\_feature\_levels = 4 encoder\_n\_points = 4 decoder\_n\_points = 4 two\_stage = True class\_cost = 1.0 bbox\_cost = 5.0 giou\_cost = 2.0 bbox\_loss\_coefficient = 5.0 giou\_loss\_coefficient = 2.0 focal\_alpha = 0.25 disable\_custom\_kernels = False max\_text\_len = 256 text\_enhancer\_dropout = 0.0 fusion\_droppath = 0.1 fusion\_dropout = 0.0 embedding\_init\_target = True query\_dim = 4 positional\_embedding\_temperature = 20 init\_std = 0.02 layer\_norm\_eps = 1e-05 \*\*kwargs  )

Parameters

* **backbone\_config** (`PretrainedConfig` or `dict`, *optional*, defaults to `ResNetConfig()`) —
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
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `BertConfig`) —
  The config object or dictionary of the text backbone.
* **num\_queries** (`int`, *optional*, defaults to 900) —
  Number of object queries, i.e. detection slots. This is the maximal number of objects
  [MMGroundingDinoModel](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoModel) can detect in a single image.
* **encoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of encoder layers.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 2048) —
  Dimension of the “intermediate” (often named feed-forward) layer in decoder.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **decoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of decoder layers.
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 2048) —
  Dimension of the “intermediate” (often named feed-forward) layer in decoder.
* **decoder\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Whether the model is used as an encoder/decoder or not.
* **activation\_function** (`str` or `function`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **d\_model** (`int`, *optional*, defaults to 256) —
  Dimension of the layers.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **activation\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for activations inside the fully connected layer.
* **auxiliary\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"sine"`) —
  Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
* **num\_feature\_levels** (`int`, *optional*, defaults to 4) —
  The number of input feature levels.
* **encoder\_n\_points** (`int`, *optional*, defaults to 4) —
  The number of sampled keys in each feature level for each attention head in the encoder.
* **decoder\_n\_points** (`int`, *optional*, defaults to 4) —
  The number of sampled keys in each feature level for each attention head in the decoder.
* **two\_stage** (`bool`, *optional*, defaults to `True`) —
  Whether to apply a two-stage deformable DETR, where the region proposals are also generated by a variant of
  Grounding DINO, which are further fed into the decoder for iterative bounding box refinement.
* **class\_cost** (`float`, *optional*, defaults to 1.0) —
  Relative weight of the classification error in the Hungarian matching cost.
* **bbox\_cost** (`float`, *optional*, defaults to 5.0) —
  Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
* **giou\_cost** (`float`, *optional*, defaults to 2.0) —
  Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
* **bbox\_loss\_coefficient** (`float`, *optional*, defaults to 5.0) —
  Relative weight of the L1 bounding box loss in the object detection loss.
* **giou\_loss\_coefficient** (`float`, *optional*, defaults to 2.0) —
  Relative weight of the generalized IoU loss in the object detection loss.
* **focal\_alpha** (`float`, *optional*, defaults to 0.25) —
  Alpha parameter in the focal loss.
* **disable\_custom\_kernels** (`bool`, *optional*, defaults to `False`) —
  Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
  kernels are not supported by PyTorch ONNX export.
* **max\_text\_len** (`int`, *optional*, defaults to 256) —
  The maximum length of the text input.
* **text\_enhancer\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the text enhancer.
* **fusion\_droppath** (`float`, *optional*, defaults to 0.1) —
  The droppath ratio for the fusion module.
* **fusion\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the fusion module.
* **embedding\_init\_target** (`bool`, *optional*, defaults to `True`) —
  Whether to initialize the target with Embedding weights.
* **query\_dim** (`int`, *optional*, defaults to 4) —
  The dimension of the query vector.
* **positional\_embedding\_temperature** (`float`, *optional*, defaults to 20) —
  The temperature for Sine Positional Embedding that is used together with vision backbone.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.

This is the configuration class to store the configuration of a [MMGroundingDinoModel](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoModel). It is used to instantiate a
MM Grounding DINO model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the MM Grounding DINO tiny architecture
[openmmlab-community/mm\_grounding\_dino\_tiny\_o365v1\_goldg\_v3det](https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import MMGroundingDinoConfig, MMGroundingDinoModel

>>> # Initializing a MM Grounding DINO configuration
>>> configuration = MMGroundingDinoConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = MMGroundingDinoModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MMGroundingDinoModel

### class transformers.MMGroundingDinoModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mm_grounding_dino/modeling_mm_grounding_dino.py#L1824)

( config: MMGroundingDinoConfig  )

Parameters

* **config** ([MMGroundingDinoConfig](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Grounding DINO Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mm_grounding_dino/modeling_mm_grounding_dino.py#L1945)

( pixel\_values: Tensor input\_ids: Tensor token\_type\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None pixel\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs = None output\_attentions = None output\_hidden\_states = None return\_dict = None  )

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [GroundingDinoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoImageProcessor). See [GroundingDinoImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([GroundingDinoProcessor](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoProcessor) uses
  [GroundingDinoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoImageProcessor) for processing images).
* **input\_ids** (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [BertTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`: 0 corresponds to a `sentence A` token, 1 corresponds to a `sentence B` token

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **pixel\_mask** (`torch.Tensor` of shape `(batch_size, height, width)`, *optional*) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
* **encoder\_outputs** (``) -- Tuple consists of (`last\_hidden\_state`, *optional*:` hidden\_states`, *optional*:` attentions`)` last\_hidden\_state`of shape`(batch\_size, sequence\_length, hidden\_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **output\_attentions** (``) -- Whether or not to return the attentions tensors of all attention layers. See` attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (``) -- Whether or not to return the hidden states of all layers. See` hidden\_states` under returned tensors for
  more detail.
* **return\_dict** (“) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

The [MMGroundingDinoModel](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoProcessor, AutoModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "a cat."

>>> processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
>>> model = AutoModel.from_pretrained("IDEA-Research/grounding-dino-tiny")

>>> inputs = processor(images=image, text=text, return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 900, 256]
```

## MMGroundingDinoForObjectDetection

### class transformers.MMGroundingDinoForObjectDetection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mm_grounding_dino/modeling_mm_grounding_dino.py#L2389)

( config: MMGroundingDinoConfig  )

Parameters

* **config** ([MMGroundingDinoConfig](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Grounding DINO Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top,
for tasks such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mm_grounding_dino/modeling_mm_grounding_dino.py#L2423)

( pixel\_values: FloatTensor input\_ids: LongTensor token\_type\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None pixel\_mask: typing.Optional[torch.BoolTensor] = None encoder\_outputs: typing.Union[transformers.models.mm\_grounding\_dino.modeling\_mm\_grounding\_dino.MMGroundingDinoEncoderOutput, tuple, NoneType] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[list[dict[str, typing.Union[torch.LongTensor, torch.FloatTensor]]]] = None  )

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [GroundingDinoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoImageProcessor). See [GroundingDinoImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([GroundingDinoProcessor](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoProcessor) uses
  [GroundingDinoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/grounding-dino#transformers.GroundingDinoImageProcessor) for processing images).
* **input\_ids** (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [BertTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`: 0 corresponds to a `sentence A` token, 1 corresponds to a `sentence B` token

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **pixel\_mask** (`torch.BoolTensor` of shape `(batch_size, height, width)`, *optional*) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
* **encoder\_outputs** (`Union[~models.mm_grounding_dino.modeling_mm_grounding_dino.MMGroundingDinoEncoderOutput, tuple, NoneType]`) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
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

The [MMGroundingDinoForObjectDetection](/docs/transformers/v4.56.2/en/model_doc/mm-grounding-dino#transformers.MMGroundingDinoForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import requests

>>> import torch
>>> from PIL import Image
>>> from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

>>> model_id = "IDEA-Research/grounding-dino-tiny"
>>> device = "cuda"

>>> processor = AutoProcessor.from_pretrained(model_id)
>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

>>> image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(image_url, stream=True).raw)
>>> # Check for cats and remote controls
>>> text_labels = [["a cat", "a remote control"]]

>>> inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = processor.post_process_grounded_object_detection(
...     outputs,
...     threshold=0.4,
...     text_threshold=0.3,
...     target_sizes=[(image.height, image.width)]
... )
>>> # Retrieve the first image result
>>> result = results[0]
>>> for box, score, text_label in zip(result["boxes"], result["scores"], result["text_labels"]):
...     box = [round(x, 2) for x in box.tolist()]
...     print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")
Detected a cat with confidence 0.479 at location [344.7, 23.11, 637.18, 374.28]
Detected a cat with confidence 0.438 at location [12.27, 51.91, 316.86, 472.44]
Detected a remote control with confidence 0.478 at location [38.57, 70.0, 176.78, 118.18]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mm-grounding-dino.md)
