# MGP-STR

## Overview

The MGP-STR model was proposed in [Multi-Granularity Prediction for Scene Text Recognition](https://huggingface.co/papers/2209.03592) by Peng Wang, Cheng Da, and Cong Yao. MGP-STR is a conceptually **simple** yet **powerful** vision Scene Text Recognition (STR) model, which is built upon the [Vision Transformer (ViT)](vit). To integrate linguistic knowledge, Multi-Granularity Prediction (MGP) strategy is proposed to inject information from the language modality into the model in an implicit way.

The abstract from the paper is the following:

*Scene text recognition (STR) has been an active research topic in computer vision for years. To tackle this challenging problem, numerous innovative methods have been successively proposed and incorporating linguistic knowledge into STR models has recently become a prominent trend. In this work, we first draw inspiration from the recent progress in Vision Transformer (ViT) to construct a conceptually simple yet powerful vision STR model, which is built upon ViT and outperforms previous state-of-the-art models for scene text recognition, including both pure vision models and language-augmented methods. To integrate linguistic knowledge, we further propose a Multi-Granularity Prediction strategy to inject information from the language modality into the model in an implicit way, i.e. , subword representations (BPE and WordPiece) widely-used in NLP are introduced into the output space, in addition to the conventional character level representation, while no independent language model (LM) is adopted. The resultant algorithm (termed MGP-STR) is able to push the performance envelop of STR to an even higher level. Specifically, it achieves an average recognition accuracy of 93.35% on standard benchmarks.*

 MGP-STR architecture. Taken from the original paper. 

MGP-STR is trained on two synthetic datasets [MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/) (MJ) and [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) (ST) without fine-tuning on other datasets. It achieves state-of-the-art results on six standard Latin scene text benchmarks, including 3 regular text datasets (IC13, SVT, IIIT) and 3 irregular ones (IC15, SVTP, CUTE).
This model was contributed by [yuekun](https://huggingface.co/yuekun). The original code can be found [here](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR).

## Inference example

[MgpstrModel](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrModel) accepts images as input and generates three types of predictions, which represent textual information at different granularities.
The three types of predictions are fused to give the final prediction result.

The [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) class is responsible for preprocessing the input image and
[MgpstrTokenizer](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrTokenizer) decodes the generated character tokens to the target string. The
[MgpstrProcessor](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrProcessor) wraps [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) and [MgpstrTokenizer](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrTokenizer)
into a single instance to both extract the input features and decode the predicted token ids.

- Step-by-step Optical Character Recognition (OCR)

```py
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

## MgpstrConfig[[transformers.MgpstrConfig]]

#### transformers.MgpstrConfig[[transformers.MgpstrConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mgp_str/configuration_mgp_str.py#L24)

This is the configuration class to store the configuration of an [MgpstrModel](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrModel). It is used to instantiate an
MGP-STR model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the MGP-STR
[alibaba-damo/mgp-str-base](https://huggingface.co/alibaba-damo/mgp-str-base) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import MgpstrConfig, MgpstrForSceneTextRecognition

>>> # Initializing a Mgpstr mgp-str-base style configuration
>>> configuration = MgpstrConfig()

>>> # Initializing a model (with random weights) from the mgp-str-base style configuration
>>> model = MgpstrForSceneTextRecognition(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

image_size (`list[int]`, *optional*, defaults to `[32, 128]`) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 4) : The size (resolution) of each patch.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

max_token_length (`int`, *optional*, defaults to 27) : The max number of output tokens.

num_character_labels (`int`, *optional*, defaults to 38) : The number of classes for character head .

num_bpe_labels (`int`, *optional*, defaults to 50257) : The number of classes for bpe head .

num_wordpiece_labels (`int`, *optional*, defaults to 30522) : The number of classes for wordpiece head .

hidden_size (`int`, *optional*, defaults to 768) : The embedding dimension.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

mlp_ratio (`float`, *optional*, defaults to 4.0) : The ratio of mlp hidden dim to embedding dim.

qkv_bias (`bool`, *optional*, defaults to `True`) : Whether to add a bias to the queries, keys and values.

distilled (`bool`, *optional*, defaults to `False`) : Model includes a distillation token and head as in DeiT models.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

drop_rate (`float`, *optional*, defaults to 0.0) : The dropout probability for all fully connected layers in the embeddings, encoder.

attn_drop_rate (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

drop_path_rate (`float`, *optional*, defaults to 0.0) : The stochastic depth rate.

output_a3_attentions (`bool`, *optional*, defaults to `False`) : Whether or not the model should returns A^3 module attentions.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

## MgpstrTokenizer[[transformers.MgpstrTokenizer]]

#### transformers.MgpstrTokenizer[[transformers.MgpstrTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mgp_str/tokenization_mgp_str.py#L30)

Construct a MGP-STR char tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

save_vocabularytransformers.MgpstrTokenizer.save_vocabularyhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/mgp_str/tokenization_mgp_str.py#L91[{"name": "save_directory", "val": ": str"}, {"name": "filename_prefix", "val": ": typing.Optional[str] = None"}]

**Parameters:**

vocab_file (`str`) : Path to the vocabulary file.

unk_token (`str`, *optional*, defaults to `"[GO]"`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

bos_token (`str`, *optional*, defaults to `"[GO]"`) : The beginning of sequence token.

eos_token (`str`, *optional*, defaults to `"[s]"`) : The end of sequence token.

pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"[GO]"`) : A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.

## MgpstrProcessor[[transformers.MgpstrProcessor]]

#### transformers.MgpstrProcessor[[transformers.MgpstrProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mgp_str/processing_mgp_str.py#L39)

Constructs a MGP-STR processor which wraps an image processor and MGP-STR tokenizers into a single

[MgpstrProcessor](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrProcessor) offers all the functionalities of `ViTImageProcessor`] and [MgpstrTokenizer](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrTokenizer). See the
[__call__()](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrProcessor.__call__) and [batch_decode()](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrProcessor.batch_decode) for more information.

__call__transformers.MgpstrProcessor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/mgp_str/processing_mgp_str.py#L60[{"name": "text", "val": " = None"}, {"name": "images", "val": " = None"}, {"name": "return_tensors", "val": " = None"}, {"name": "**kwargs", "val": ""}]

When used in normal mode, this method forwards all its arguments to ViTImageProcessor's
[__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) and returns its output. This method also forwards the `text` and `kwargs`
arguments to MgpstrTokenizer's [__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) if `text` is not `None` to encode the text. Please
refer to the docstring of the above methods for more information.

**Parameters:**

image_processor (`ViTImageProcessor`, *optional*) : An instance of `ViTImageProcessor`. The image processor is a required input.

tokenizer ([MgpstrTokenizer](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrTokenizer), *optional*) : The tokenizer is a required input.
#### batch_decode[[transformers.MgpstrProcessor.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mgp_str/processing_mgp_str.py#L83)

Convert a list of lists of token ids into a list of strings by calling decode.

This method forwards all its arguments to PreTrainedTokenizer's [batch_decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

**Parameters:**

sequences (`torch.Tensor`) : List of tokenized input ids.

**Returns:**

``dict[str, any]``

Dictionary of all the outputs of the decoded results.
generated_text (`list[str]`): The final results after fusion of char, bpe, and wp. scores
(`list[float]`): The final scores after fusion of char, bpe, and wp. char_preds (`list[str]`): The list
of character decoded sentences. bpe_preds (`list[str]`): The list of bpe decoded sentences. wp_preds
(`list[str]`): The list of wp decoded sentences.

## MgpstrModel[[transformers.MgpstrModel]]

#### transformers.MgpstrModel[[transformers.MgpstrModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mgp_str/modeling_mgp_str.py#L305)

The bare Mgp Str Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.MgpstrModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/mgp_str/modeling_mgp_str.py#L318[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([MgpstrProcessor](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrProcessor) uses
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MgpstrConfig](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [MgpstrModel](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([MgpstrConfig](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MgpstrConfig](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## MgpstrForSceneTextRecognition[[transformers.MgpstrForSceneTextRecognition]]

#### transformers.MgpstrForSceneTextRecognition[[transformers.MgpstrForSceneTextRecognition]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mgp_str/modeling_mgp_str.py#L360)

MGP-STR Model transformer with three classification heads on top (three A^3 modules and three linear layer on top
of the transformer encoder output) for scene text recognition (STR) .

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.MgpstrForSceneTextRecognition.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/mgp_str/modeling_mgp_str.py#L381[{"name": "pixel_values", "val": ": FloatTensor"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_a3_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([MgpstrProcessor](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrProcessor) uses
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_a3_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of a3 modules. See `a3_attentions` under returned tensors
  for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.mgp_str.modeling_mgp_str.MgpstrModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.mgp_str.modeling_mgp_str.MgpstrModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MgpstrConfig](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrConfig)) and inputs.

- **logits** (`tuple(torch.FloatTensor)` of shape `(batch_size, config.num_character_labels)`) -- Tuple of `torch.FloatTensor` (one for the output of character of shape `(batch_size,
  config.max_token_length, config.num_character_labels)`, + one for the output of bpe of shape `(batch_size,
  config.max_token_length, config.num_bpe_labels)`, + one for the output of wordpiece of shape `(batch_size,
  config.max_token_length, config.num_wordpiece_labels)`) .

  Classification scores (before SoftMax) of character, bpe and wordpiece.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **a3_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_a3_attentions=True` is passed or when `config.output_a3_attentions=True`) -- Tuple of `torch.FloatTensor` (one for the attention of character, + one for the attention of bpe`, + one
  for the attention of wordpiece) of shape `(batch_size, config.max_token_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [MgpstrForSceneTextRecognition](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrForSceneTextRecognition) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
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

**Parameters:**

config ([MgpstrConfig](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.mgp_str.modeling_mgp_str.MgpstrModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.mgp_str.modeling_mgp_str.MgpstrModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MgpstrConfig](/docs/transformers/main/en/model_doc/mgp-str#transformers.MgpstrConfig)) and inputs.

- **logits** (`tuple(torch.FloatTensor)` of shape `(batch_size, config.num_character_labels)`) -- Tuple of `torch.FloatTensor` (one for the output of character of shape `(batch_size,
  config.max_token_length, config.num_character_labels)`, + one for the output of bpe of shape `(batch_size,
  config.max_token_length, config.num_bpe_labels)`, + one for the output of wordpiece of shape `(batch_size,
  config.max_token_length, config.num_wordpiece_labels)`) .

  Classification scores (before SoftMax) of character, bpe and wordpiece.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **a3_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_a3_attentions=True` is passed or when `config.output_a3_attentions=True`) -- Tuple of `torch.FloatTensor` (one for the attention of character, + one for the attention of bpe`, + one
  for the attention of wordpiece) of shape `(batch_size, config.max_token_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
