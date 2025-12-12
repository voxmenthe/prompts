*This model was released on 2024-11-21 and added to Hugging Face Transformers on 2025-07-08.*

# AIMv2

## Overview

The AIMv2 model was proposed in [Multimodal Autoregressive Pre-training of Large Vision Encoders](https://huggingface.co/papers/2411.14402) by Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang, Joshua M. Susskind, Alaaeldin El-Nouby.

The abstract from the paper is the following:

*We introduce a novel method for pre-training of large-scale vision encoders. Building on recent advancements in autoregressive pre-training of vision models, we extend this framework to a multimodal setting, i.e., images and text. In this paper, we present AIMV2, a family of generalist vision encoders characterized by a straightforward pre-training process, scalability, and remarkable performance across a range of downstream tasks. This is achieved by pairing the vision encoder with a multimodal decoder that autoregressively generates raw image patches and text tokens. Our encoders excel not only in multimodal evaluations but also in vision benchmarks such as localization, grounding, and classification. Notably, our AIMV2-3B encoder achieves 89.5% accuracy on ImageNet-1k with a frozen trunk. Furthermore, AIMV2 consistently outperforms state-of-the-art contrastive models (e.g., CLIP, SigLIP) in multimodal image understanding across diverse settings.*

This model was contributed by [Yaswanth Gali](https://huggingface.co/yaswanthgali).
The original code can be found [here](https://github.com/apple/ml-aim).

## Usage Example

Here is an example of Image Feature Extraction using specific checkpoints on resized images and native resolution images:


```
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-native")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-native")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

Here is an example of a checkpoint performing zero-shot classification:


```
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = ["Picture of a dog.", "Picture of a cat.", "Picture of a horse."]

processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")
model = AutoModel.from_pretrained("apple/aimv2-large-patch14-224-lit")

inputs = processor(
    images=image,
    text=text,
    add_special_tokens=True,
    truncation=True,
    padding=True,
    return_tensors="pt",
)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)
```

## Aimv2Config

### class transformers.Aimv2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aimv2/configuration_aimv2.py#L215)

( text\_config = None vision\_config = None projection\_dim = 512 logit\_scale\_init\_value = 2.6592 \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Aimv2TextConfig](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2TextConfig).
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [Aimv2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2VisionConfig).
* **projection\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of text and vision projection layers.
* **logit\_scale\_init\_value** (`float`, *optional*, defaults to 2.6592) —
  The initial value of the *logit\_scale* parameter.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[Aimv2Config](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Config) is the configuration class to store the configuration of a [Aimv2Model](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Model). It is used to
instantiate a AIMv2 model according to the specified arguments, defining the text model and vision model configs.
Instantiating a configuration with the defaults will yield a similar configuration to that of the AIMv2
[apple/aimv2-large-patch14-224-lit](https://huggingface.co/apple/aimv2-large-patch14-224-lit) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Aimv2Config, Aimv2Model

>>> # Initializing a Aimv2Config with apple/aimv2-large-patch14-224-lit style configuration
>>> configuration = Aimv2Config()

>>> # Initializing a Aimv2Model (with random weights) from the apple/aimv2-large-patch14-224-lit style configuration
>>> model = Aimv2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a Aimv2Config from a Aimv2TextConfig and a Aimv2VisionConfig
>>> from transformers import Aimv2TextConfig, Aimv2VisionConfig

>>> # Initializing a AIMv2Text and AIMv2Vision configuration
>>> config_text = Aimv2TextConfig()
>>> config_vision = Aimv2VisionConfig()

>>> config = Aimv2Config(text_config=config_text, vision_config=config_vision)
```

## Aimv2TextConfig

### class transformers.Aimv2TextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aimv2/configuration_aimv2.py#L130)

( vocab\_size: int = 49408 hidden\_size: int = 768 intermediate\_size: int = 2048 num\_hidden\_layers: int = 12 num\_attention\_heads: int = 6 rms\_norm\_eps: float = 1e-05 attention\_dropout: float = 0.0 qkv\_bias: bool = False mlp\_bias: bool = False hidden\_act: str = 'silu' pad\_token\_id: typing.Optional[int] = None bos\_token\_id: typing.Optional[int] = None eos\_token\_id: int = 49407 max\_position\_embeddings: int = 77 initializer\_range: bool = 0.02 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 49408) —
  Vocabulary size of the AIMv2 text model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [Aimv2Model](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Model).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 6) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **qkv\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to add a bias to the queries, keys and values.
* **mlp\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to add a bias to the Linear layers or Not.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **pad\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the padding token in the vocabulary.
* **bos\_token\_id** (`int`, *optional*, defaults to 49406) —
  The id of the beginning-of-sequence token in the vocabulary.
* **eos\_token\_id** (`int`, *optional*, defaults to 49407) —
  The id of the end-of-sequence token in the vocabulary.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 77) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the for initializing all weight matrices.

This is the configuration class to store the configuration of a [Aimv2TextModel](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2TextModel). It is used to instantiate a
AIMv2 text encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the text encoder of the AIMv2
[apple/aimv2-large-patch14-224-lit](https://huggingface.co/apple/aimv2-large-patch14-224-lit) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## Aimv2VisionConfig

### class transformers.Aimv2VisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aimv2/configuration_aimv2.py#L31)

( hidden\_size: int = 1024 intermediate\_size: int = 2816 num\_hidden\_layers: int = 24 num\_attention\_heads: int = 8 num\_channels: int = 3 image\_size: int = 224 patch\_size: int = 14 rms\_norm\_eps: float = 1e-05 attention\_dropout: float = 0.0 qkv\_bias: bool = False mlp\_bias: bool = False hidden\_act: str = 'silu' initializer\_range: float = 0.02 use\_head: bool = True is\_native: bool = False \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 2816) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  Number of channels in the input images.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The size (resolution) of each patch.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **qkv\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to add a bias to the queries, keys and values.
* **mlp\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to add a bias to the Linear layers or Not.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the for initializing all weight matrices.
* **use\_head** (`str`, *optional*, defaults to `True`) —
  Whether to use Attention Pooling Head or Not.
* **is\_native** (`str`, *optional*, defaults to `False`) —
  Whether to use ckpt trained for image native resolution or not.

This is the configuration class to store the configuration of a [Aimv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2VisionModel). It is used to instantiate a
AIMv2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the vision encoder of the AIMv2
[apple/aimv2-large-patch14-224](https://huggingface.co/apple/aimv2-large-patch14-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import SiglipVisionConfig, SiglipVisionModel

>>> # Initializing a Aimv2VisionConfig with apple/aimv2-large-patch14-224 style configuration
>>> configuration = Aimv2VisionConfig()

>>> # Initializing a Aimv2VisionModel (with random weights) from the apple/aimv2-large-patch14-224 style configuration
>>> model = Aimv2VisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Aimv2Model

### class transformers.Aimv2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aimv2/modeling_aimv2.py#L630)

( config: Aimv2Config  )

Parameters

* **config** ([Aimv2Config](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Aimv2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aimv2/modeling_aimv2.py#L745)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → `transformers.models.aimv2.modeling_aimv2.Aimv2Output` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
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

Returns

`transformers.models.aimv2.modeling_aimv2.Aimv2Output` or `tuple(torch.FloatTensor)`

A `transformers.models.aimv2.modeling_aimv2.Aimv2Output` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
* **logits\_per\_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [Aimv2TextModel](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2TextModel).
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of [Aimv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2VisionModel).
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [Aimv2TextModel](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2TextModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output`, defaults to `None`) — The output of the [Aimv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2VisionModel).

The [Aimv2Model](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Aimv2Model

>>> model = Aimv2Model.from_pretrained("apple/aimv2-large-patch14-224-lit")
>>> processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

## Aimv2VisionModel

### class transformers.Aimv2VisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aimv2/modeling_aimv2.py#L464)

( config: Aimv2VisionConfig  )

Parameters

* **config** ([Aimv2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2VisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Vision model from AIMv2 without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aimv2/modeling_aimv2.py#L485)

( pixel\_values attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (``of shape`(batch\_size, num\_channels, image\_size, image\_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPProcessor) uses
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).
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

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Aimv2Config](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Config)) and inputs.

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

The [Aimv2VisionModel](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2VisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Siglip2VisionModel

>>> model = Aimv2VisionModel.from_pretrained("apple/aimv2-large-patch14-native")
>>> processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-native")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled features
```

## Aimv2TextModel

### class transformers.Aimv2TextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aimv2/modeling_aimv2.py#L545)

( config: Aimv2TextConfig  )

Parameters

* **config** ([Aimv2TextConfig](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2TextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The text model from AIMv2 without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aimv2/modeling_aimv2.py#L565)

( input\_ids attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (``of shape`(batch\_size, sequence\_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
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

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Aimv2Config](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2Config)) and inputs.

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

The [Aimv2TextModel](/docs/transformers/v4.56.2/en/model_doc/aimv2#transformers.Aimv2TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/aimv2.md)
