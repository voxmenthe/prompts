*This model was released on 2022-09-08 and added to Hugging Face Transformers on 2023-03-13.*

# MGP-STR

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The MGP-STR model was proposed in [Multi-Granularity Prediction for Scene Text Recognition](https://huggingface.co/papers/2209.03592) by Peng Wang, Cheng Da, and Cong Yao. MGP-STR is a conceptually **simple** yet **powerful** vision Scene Text Recognition (STR) model, which is built upon the [Vision Transformer (ViT)](vit). To integrate linguistic knowledge, Multi-Granularity Prediction (MGP) strategy is proposed to inject information from the language modality into the model in an implicit way.

The abstract from the paper is the following:

*Scene text recognition (STR) has been an active research topic in computer vision for years. To tackle this challenging problem, numerous innovative methods have been successively proposed and incorporating linguistic knowledge into STR models has recently become a prominent trend. In this work, we first draw inspiration from the recent progress in Vision Transformer (ViT) to construct a conceptually simple yet powerful vision STR model, which is built upon ViT and outperforms previous state-of-the-art models for scene text recognition, including both pure vision models and language-augmented methods. To integrate linguistic knowledge, we further propose a Multi-Granularity Prediction strategy to inject information from the language modality into the model in an implicit way, i.e. , subword representations (BPE and WordPiece) widely-used in NLP are introduced into the output space, in addition to the conventional character level representation, while no independent language model (LM) is adopted. The resultant algorithm (termed MGP-STR) is able to push the performance envelop of STR to an even higher level. Specifically, it achieves an average recognition accuracy of 93.35% on standard benchmarks.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mgp_str_architecture.png) MGP-STR architecture. Taken from the [original paper](https://huggingface.co/papers/2209.03592).

MGP-STR is trained on two synthetic datasets [MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/) (MJ) and [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) (ST) without fine-tuning on other datasets. It achieves state-of-the-art results on six standard Latin scene text benchmarks, including 3 regular text datasets (IC13, SVT, IIIT) and 3 irregular ones (IC15, SVTP, CUTE).
This model was contributed by [yuekun](https://huggingface.co/yuekun). The original code can be found [here](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR).

## Inference example

[MgpstrModel](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrModel) accepts images as input and generates three types of predictions, which represent textual information at different granularities.
The three types of predictions are fused to give the final prediction result.

The [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) class is responsible for preprocessing the input image and
[MgpstrTokenizer](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrTokenizer) decodes the generated character tokens to the target string. The
[MgpstrProcessor](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrProcessor) wraps [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) and [MgpstrTokenizer](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrTokenizer)
into a single instance to both extract the input features and decode the predicted token ids.

* Step-by-step Optical Character Recognition (OCR)


```
>>> from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
>>> import requests
>>> from PIL import Image

>>> processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
>>> model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')

>>> # load image from the IIIT-5k dataset
>>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> pixel_values = processor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(pixel_values)

>>> generated_text = processor.batch_decode(outputs.logits)['generated_text']
```

## MgpstrConfig

### class transformers.MgpstrConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mgp_str/configuration_mgp_str.py#L24)

( image\_size = [32, 128] patch\_size = 4 num\_channels = 3 max\_token\_length = 27 num\_character\_labels = 38 num\_bpe\_labels = 50257 num\_wordpiece\_labels = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 mlp\_ratio = 4.0 qkv\_bias = True distilled = False layer\_norm\_eps = 1e-05 drop\_rate = 0.0 attn\_drop\_rate = 0.0 drop\_path\_rate = 0.0 output\_a3\_attentions = False initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **image\_size** (`list[int]`, *optional*, defaults to `[32, 128]`) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 4) —
  The size (resolution) of each patch.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **max\_token\_length** (`int`, *optional*, defaults to 27) —
  The max number of output tokens.
* **num\_character\_labels** (`int`, *optional*, defaults to 38) —
  The number of classes for character head .
* **num\_bpe\_labels** (`int`, *optional*, defaults to 50257) —
  The number of classes for bpe head .
* **num\_wordpiece\_labels** (`int`, *optional*, defaults to 30522) —
  The number of classes for wordpiece head .
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  The embedding dimension.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **mlp\_ratio** (`float`, *optional*, defaults to 4.0) —
  The ratio of mlp hidden dim to embedding dim.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to add a bias to the queries, keys and values.
* **distilled** (`bool`, *optional*, defaults to `False`) —
  Model includes a distillation token and head as in DeiT models.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **drop\_rate** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder.
* **attn\_drop\_rate** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.0) —
  The stochastic depth rate.
* **output\_a3\_attentions** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should returns A^3 module attentions.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of an [MgpstrModel](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrModel). It is used to instantiate an
MGP-STR model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the MGP-STR
[alibaba-damo/mgp-str-base](https://huggingface.co/alibaba-damo/mgp-str-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MgpstrConfig, MgpstrForSceneTextRecognition

>>> # Initializing a Mgpstr mgp-str-base style configuration
>>> configuration = MgpstrConfig()

>>> # Initializing a model (with random weights) from the mgp-str-base style configuration
>>> model = MgpstrForSceneTextRecognition(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MgpstrTokenizer

### class transformers.MgpstrTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mgp_str/tokenization_mgp_str.py#L30)

( vocab\_file unk\_token = '[GO]' bos\_token = '[GO]' eos\_token = '[s]' pad\_token = '[GO]' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **unk\_token** (`str`, *optional*, defaults to `"[GO]"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **bos\_token** (`str`, *optional*, defaults to `"[GO]"`) —
  The beginning of sequence token.
* **eos\_token** (`str`, *optional*, defaults to `"[s]"`) —
  The end of sequence token.
* **pad\_token** (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"[GO]"`) —
  A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
  attention mechanisms or loss computation.

Construct a MGP-STR char tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mgp_str/tokenization_mgp_str.py#L90)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## MgpstrProcessor

### class transformers.MgpstrProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mgp_str/processing_mgp_str.py#L41)

( image\_processor = None tokenizer = None \*\*kwargs  )

Parameters

* **image\_processor** (`ViTImageProcessor`, *optional*) —
  An instance of `ViTImageProcessor`. The image processor is a required input.
* **tokenizer** ([MgpstrTokenizer](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrTokenizer), *optional*) —
  The tokenizer is a required input.

Constructs a MGP-STR processor which wraps an image processor and MGP-STR tokenizers into a single

[MgpstrProcessor](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrProcessor) offers all the functionalities of `ViTImageProcessor`] and [MgpstrTokenizer](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrTokenizer). See the
[**call**()](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrProcessor.__call__) and [batch\_decode()](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrProcessor.batch_decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mgp_str/processing_mgp_str.py#L81)

( text = None images = None return\_tensors = None \*\*kwargs  )

When used in normal mode, this method forwards all its arguments to ViTImageProcessor’s
[**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) and returns its output. This method also forwards the `text` and `kwargs`
arguments to MgpstrTokenizer’s [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) if `text` is not `None` to encode the text. Please
refer to the docstring of the above methods for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mgp_str/processing_mgp_str.py#L104)

( sequences  ) → `dict[str, any]`

Parameters

* **sequences** (`torch.Tensor`) —
  List of tokenized input ids.

Returns

`dict[str, any]`

Dictionary of all the outputs of the decoded results.
generated\_text (`list[str]`): The final results after fusion of char, bpe, and wp. scores
(`list[float]`): The final scores after fusion of char, bpe, and wp. char\_preds (`list[str]`): The list
of character decoded sentences. bpe\_preds (`list[str]`): The list of bpe decoded sentences. wp\_preds
(`list[str]`): The list of wp decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

## MgpstrModel

### class transformers.MgpstrModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mgp_str/modeling_mgp_str.py#L309)

( config: MgpstrConfig  )

Parameters

* **config** ([MgpstrConfig](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Mgp Str Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mgp_str/modeling_mgp_str.py#L322)

( pixel\_values: FloatTensor output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([MgpstrProcessor](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrProcessor) uses
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MgpstrConfig](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MgpstrModel](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## MgpstrForSceneTextRecognition

### class transformers.MgpstrForSceneTextRecognition

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mgp_str/modeling_mgp_str.py#L363)

( config: MgpstrConfig  )

Parameters

* **config** ([MgpstrConfig](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MGP-STR Model transformer with three classification heads on top (three A^3 modules and three linear layer on top
of the transformer encoder output) for scene text recognition (STR) .

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mgp_str/modeling_mgp_str.py#L384)

( pixel\_values: FloatTensor output\_attentions: typing.Optional[bool] = None output\_a3\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.mgp_str.modeling_mgp_str.MgpstrModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([MgpstrProcessor](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrProcessor) uses
  [ViTImageProcessor](/docs/transformers/v4.56.2/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_a3\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of a3 modules. See `a3_attentions` under returned tensors
  for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.mgp_str.modeling_mgp_str.MgpstrModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.mgp_str.modeling_mgp_str.MgpstrModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MgpstrConfig](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrConfig)) and inputs.

* **logits** (`tuple(torch.FloatTensor)` of shape `(batch_size, config.num_character_labels)`) — Tuple of `torch.FloatTensor` (one for the output of character of shape `(batch_size, config.max_token_length, config.num_character_labels)`, + one for the output of bpe of shape `(batch_size, config.max_token_length, config.num_bpe_labels)`, + one for the output of wordpiece of shape `(batch_size, config.max_token_length, config.num_wordpiece_labels)`) .

  Classification scores (before SoftMax) of character, bpe and wordpiece.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **a3\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_a3_attentions=True` is passed or when `config.output_a3_attentions=True`) — Tuple of `torch.FloatTensor` (one for the attention of character, + one for the attention of bpe`, + one for the attention of wordpiece) of shape` (batch\_size, config.max\_token\_length, sequence\_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MgpstrForSceneTextRecognition](/docs/transformers/v4.56.2/en/model_doc/mgp-str#transformers.MgpstrForSceneTextRecognition) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import (
...     MgpstrProcessor,
...     MgpstrForSceneTextRecognition,
... )
>>> import requests
>>> from PIL import Image

>>> # load image from the IIIT-5k dataset
>>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> processor = MgpstrProcessor.from_pretrained("alibaba-damo/mgp-str-base")
>>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

>>> model = MgpstrForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base")

>>> # inference
>>> outputs = model(pixel_values)
>>> out_strs = processor.batch_decode(outputs.logits)
>>> out_strs["generated_text"]
'["ticket"]'
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mgp-str.md)
