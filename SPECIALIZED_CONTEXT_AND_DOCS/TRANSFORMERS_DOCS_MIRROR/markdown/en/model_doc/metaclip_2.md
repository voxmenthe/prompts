*This model was released on {release\_date} and added to Hugging Face Transformers on 2025-08-20.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# MetaCLIP 2

## Overview

MetaCLIP 2 is a replication of the original CLIP model trained on 300+ languages. It achieves state-of-the-art (SOTA) results on multilingual benchmarks (e.g., XM3600, CVQA, Babel‑ImageNet), surpassing previous SOTA such as [mSigLIP](siglip) and [SigLIP‑2](siglip2). The authors show that English and non-English worlds can mutually benefit and elevate each other.

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/facebookresearch/MetaCLIP).

You can find all the MetaCLIP 2 checkpoints under the [Meta](https://huggingface.co/facebook?search_models=metaclip-2) organization.

Click on the MetaCLIP 2 models in the right sidebar for more examples of how to apply MetaCLIP 2 to different image and language tasks.

The example below demonstrates how to calculate similarity scores between multiple text descriptions and an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class. Usage of the MetaCLIP 2 models is identical to the CLIP models, you just need the `MetaClip2Model` class instead of `CLIPModel`.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

clip = pipeline(
   task="zero-shot-image-classification",
   model="facebook/metaclip-2-worldwide-huge-quickgelu",
   dtype=torch.bfloat16,
   device=0
)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
clip("http://images.cocodataset.org/val2017/000000039769.jpg", candidate_labels=labels)
```

## MetaClip2Config

### class transformers.MetaClip2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/configuration_metaclip_2.py#L208)

( text\_config = None vision\_config = None projection\_dim = 512 logit\_scale\_init\_value = 2.6592 \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [MetaClip2TextConfig](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2TextConfig).
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [MetaClip2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2VisionConfig).
* **projection\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of text and vision projection layers.
* **logit\_scale\_init\_value** (`float`, *optional*, defaults to 2.6592) —
  The initial value of the *logit\_scale* parameter. Default is used as per the original METACLIP\_2 implementation.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config) is the configuration class to store the configuration of a [MetaClip2Model](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Model). It is used to instantiate
a METACLIP\_2 model according to the specified arguments, defining the text model and vision model configs. Instantiating
a configuration with the defaults will yield a similar configuration to that of the METACLIP\_2
[openai/metaclip\_2-vit-base-patch32](https://huggingface.co/openai/metaclip_2-vit-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MetaClip2Config, MetaClip2Model

>>> # Initializing a MetaClip2Config with openai/metaclip_2-vit-base-patch32 style configuration
>>> configuration = MetaClip2Config()

>>> # Initializing a MetaClip2Model (with random weights) from the openai/metaclip_2-vit-base-patch32 style configuration
>>> model = MetaClip2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a MetaClip2Config from a MetaClip2TextConfig and a MetaClip2VisionConfig
>>> from transformers import MetaClip2TextConfig, MetaClip2VisionConfig

>>> # Initializing a MetaClip2Text and MetaClip2Vision configuration
>>> config_text = MetaClip2TextConfig()
>>> config_vision = MetaClip2VisionConfig()

>>> config = MetaClip2Config.from_text_vision_configs(config_text, config_vision)
```

#### from\_text\_vision\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/configuration_utils.py#L1254)

( text\_config vision\_config \*\*kwargs  ) → `PreTrainedConfig`

Returns

`PreTrainedConfig`

An instance of a configuration object

Instantiate a model config (or a derived class) from text model configuration and vision model
configuration.

## MetaClip2TextConfig

### class transformers.MetaClip2TextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/configuration_metaclip_2.py#L15)

( vocab\_size = 49408 hidden\_size = 512 intermediate\_size = 2048 projection\_dim = 512 num\_hidden\_layers = 12 num\_attention\_heads = 8 max\_position\_embeddings = 77 hidden\_act = 'quick\_gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 pad\_token\_id = 1 bos\_token\_id = 49406 eos\_token\_id = 49407 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 49408) —
  Vocabulary size of the METACLIP\_2 text model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [MetaClip2Model](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Model).
* **hidden\_size** (`int`, *optional*, defaults to 512) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **projection\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of text and vision projection layers.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 77) —
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
* **pad\_token\_id** (`int`, *optional*, defaults to 1) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 49406) —
  Beginning of stream token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 49407) —
  End of stream token id.

This is the configuration class to store the configuration of a [MetaClip2TextModel](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2TextModel). It is used to instantiate a METACLIP\_2
text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the text encoder of the METACLIP\_2
[openai/metaclip\_2-vit-base-patch32](https://huggingface.co/openai/metaclip_2-vit-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MetaClip2TextConfig, MetaClip2TextModel

>>> # Initializing a MetaClip2TextConfig with openai/metaclip_2-vit-base-patch32 style configuration
>>> configuration = MetaClip2TextConfig()

>>> # Initializing a MetaClip2TextModel (with random weights) from the openai/metaclip_2-vit-base-patch32 style configuration
>>> model = MetaClip2TextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MetaClip2VisionConfig

### class transformers.MetaClip2VisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/configuration_metaclip_2.py#L116)

( hidden\_size = 768 intermediate\_size = 3072 projection\_dim = 512 num\_hidden\_layers = 12 num\_attention\_heads = 12 num\_channels = 3 image\_size = 224 patch\_size = 32 hidden\_act = 'quick\_gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **projection\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of text and vision projection layers.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 32) —
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

This is the configuration class to store the configuration of a [MetaClip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2VisionModel). It is used to instantiate a
METACLIP\_2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the vision encoder of the METACLIP\_2
[openai/metaclip\_2-vit-base-patch32](https://huggingface.co/openai/metaclip_2-vit-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import MetaClip2VisionConfig, MetaClip2VisionModel

>>> # Initializing a MetaClip2VisionConfig with openai/metaclip_2-vit-base-patch32 style configuration
>>> configuration = MetaClip2VisionConfig()

>>> # Initializing a MetaClip2VisionModel (with random weights) from the openai/metaclip_2-vit-base-patch32 style configuration
>>> model = MetaClip2VisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MetaClip2Model

### class transformers.MetaClip2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L742)

( config: MetaClip2Config  )

Parameters

* **config** ([MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Metaclip 2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L874)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None return\_loss: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  ) → `transformers.models.metaclip_2.modeling_metaclip_2.MetaClip2Output` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPProcessor) uses
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
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

Returns

`transformers.models.metaclip_2.modeling_metaclip_2.MetaClip2Output` or `tuple(torch.FloatTensor)`

A `transformers.models.metaclip_2.modeling_metaclip_2.MetaClip2Output` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
* **logits\_per\_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [MetaClip2TextModel](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2TextModel).
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of [MetaClip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2VisionModel).
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [MetaClip2TextModel](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2TextModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output`, defaults to `None`) — The output of the [MetaClip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2VisionModel).

The [MetaClip2Model](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, MetaClip2Model

>>> model = MetaClip2Model.from_pretrained("openai/metaclip_2-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/metaclip_2-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

#### get\_text\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L782)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by
applying the projection layer to the pooled output of [MetaClip2TextModel](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2TextModel).

Examples:


```
>>> from transformers import AutoTokenizer, MetaClip2Model

>>> model = MetaClip2Model.from_pretrained("openai/metaclip_2-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/metaclip_2-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L826)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPProcessor) uses
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by
applying the projection layer to the pooled output of [MetaClip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2VisionModel).

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, MetaClip2Model

>>> model = MetaClip2Model.from_pretrained("openai/metaclip_2-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/metaclip_2-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```

## MetaClip2TextModel

### class transformers.MetaClip2TextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L544)

( config: MetaClip2TextConfig  )

Parameters

* **config** ([MetaClip2TextConfig](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2TextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The text model from METACLIP\_2 without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L562)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
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
elements depending on the configuration ([MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config)) and inputs.

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

The [MetaClip2TextModel](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, MetaClip2TextModel

>>> model = MetaClip2TextModel.from_pretrained("openai/metaclip_2-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/metaclip_2-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
```

## MetaClip2TextModelWithProjection

### class transformers.MetaClip2TextModelWithProjection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L616)

( config: MetaClip2TextConfig  )

Parameters

* **config** ([MetaClip2TextConfig](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2TextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Metaclip 2 Model with a projection layer on top (a linear layer on top of the pooled output).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L638)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → `transformers.models.metaclip_2.modeling_metaclip_2.MetaClip2TextModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.models.metaclip_2.modeling_metaclip_2.MetaClip2TextModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.metaclip_2.modeling_metaclip_2.MetaClip2TextModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config)) and inputs.

* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) — The text embeddings obtained by applying the projection layer to the pooler\_output.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MetaClip2TextModelWithProjection](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2TextModelWithProjection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, MetaClip2TextModelWithProjection

>>> model = MetaClip2TextModelWithProjection.from_pretrained("openai/metaclip_2-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/metaclip_2-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> text_embeds = outputs.text_embeds
```

## MetaClip2VisionModelWithProjection

### class transformers.MetaClip2VisionModelWithProjection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L1088)

( config: MetaClip2VisionConfig  )

Parameters

* **config** ([MetaClip2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2VisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Metaclip 2 Model with a projection layer on top (a linear layer on top of the pooled output).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L1106)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  ) → `transformers.models.metaclip_2.modeling_metaclip_2.MetaClip2VisionModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPProcessor) uses
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.

Returns

`transformers.models.metaclip_2.modeling_metaclip_2.MetaClip2VisionModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.metaclip_2.modeling_metaclip_2.MetaClip2VisionModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config)) and inputs.

* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) — The image embeddings obtained by applying the projection layer to the pooler\_output.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MetaClip2VisionModelWithProjection](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2VisionModelWithProjection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, MetaClip2VisionModelWithProjection

>>> model = MetaClip2VisionModelWithProjection.from_pretrained("openai/metaclip_2-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/metaclip_2-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> image_embeds = outputs.image_embeds
```

## MetaClip2VisionModel

### class transformers.MetaClip2VisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L1017)

( config: MetaClip2VisionConfig  )

Parameters

* **config** ([MetaClip2VisionConfig](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2VisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The vision model from METACLIP\_2 without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L1031)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPProcessor) uses
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **interpolate\_pos\_encoding** (`bool`, defaults to `False`) —
  Whether to interpolate the pre-trained position encodings.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config)) and inputs.

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

The [MetaClip2VisionModel](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2VisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, MetaClip2VisionModel

>>> model = MetaClip2VisionModel.from_pretrained("openai/metaclip_2-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/metaclip_2-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled CLS states
```

## MetaClip2ForImageClassification

### class transformers.MetaClip2ForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L1158)

( config: MetaClip2Config  )

Parameters

* **config** ([MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

METACLIP\_2 vision encoder with an image classification head on top (a linear layer on top of the pooled final hidden states of
the patch tokens) e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/metaclip_2/modeling_metaclip_2.py#L1176)

( pixel\_values: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPProcessor) uses
  [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

[transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MetaClip2Config](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
  (also called feature maps) of the model at the output of each stage.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [MetaClip2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/metaclip_2#transformers.MetaClip2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, MetaClip2ForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("openai/metaclip_2-vit-base-patch32")
>>> model = MetaClip2ForImageClassification.from_pretrained("openai/metaclip_2-vit-base-patch32")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/metaclip_2.md)
