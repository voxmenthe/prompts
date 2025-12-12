*This model was released on 2022-11-12 and added to Hugging Face Transformers on 2023-01-04.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# AltCLIP

[AltCLIP](https://huggingface.co/papers/2211.06679) replaces the [CLIP](./clip) text encoder with a multilingual XLM-R encoder and aligns image and text representations with teacher learning and contrastive learning.

You can find all the original AltCLIP checkpoints under the [AltClip](https://huggingface.co/collections/BAAI/alt-clip-diffusion-66987a97de8525205f1221bf) collection.

Click on the AltCLIP models in the right sidebar for more examples of how to apply AltCLIP to different tasks.

The examples below demonstrates how to calculate similarity scores between an image and one or more captions with the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

AutoModel


```
import torch
import requests
from PIL import Image
from transformers import AltCLIPModel, AltCLIPProcessor

model = AltCLIPModel.from_pretrained("BAAI/AltCLIP", dtype=torch.bfloat16)
processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

labels = ["a photo of a cat", "a photo of a dog"]
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.


```
# !pip install torchao
import torch
import requests
from PIL import Image
from transformers import AltCLIPModel, AltCLIPProcessor, TorchAoConfig

model = AltCLIPModel.from_pretrained(
    "BAAI/AltCLIP",
    quantization_config=TorchAoConfig("int4_weight_only", group_size=128),
    dtype=torch.bfloat16,
)

processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

labels = ["a photo of a cat", "a photo of a dog"]
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
```

## Notes

* AltCLIP uses bidirectional attention instead of causal attention and it uses the `[CLS]` token in XLM-R to represent a text embedding.
* Use [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) to resize (or rescale) and normalize images for the model.
* [AltCLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPProcessor) combines [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) and [XLMRobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) into a single instance to encode text and prepare images.

## AltCLIPConfig

### class transformers.AltCLIPConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/configuration_altclip.py#L235)

( text\_config = None vision\_config = None projection\_dim = 768 logit\_scale\_init\_value = 2.6592 \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [AltCLIPTextConfig](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPTextConfig).
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [AltCLIPVisionConfig](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPVisionConfig).
* **projection\_dim** (`int`, *optional*, defaults to 768) —
  Dimensionality of text and vision projection layers.
* **logit\_scale\_init\_value** (`float`, *optional*, defaults to 2.6592) —
  The initial value of the *logit\_scale* parameter. Default is used as per the original CLIP implementation.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

This is the configuration class to store the configuration of a [AltCLIPModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPModel). It is used to instantiate an
AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the AltCLIP
[BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import AltCLIPConfig, AltCLIPModel

>>> # Initializing a AltCLIPConfig with BAAI/AltCLIP style configuration
>>> configuration = AltCLIPConfig()

>>> # Initializing a AltCLIPModel (with random weights) from the BAAI/AltCLIP style configuration
>>> model = AltCLIPModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a AltCLIPConfig from a AltCLIPTextConfig and a AltCLIPVisionConfig

>>> # Initializing a AltCLIPText and AltCLIPVision configuration
>>> config_text = AltCLIPTextConfig()
>>> config_vision = AltCLIPVisionConfig()

>>> config = AltCLIPConfig.from_text_vision_configs(config_text, config_vision)
```

## AltCLIPTextConfig

### class transformers.AltCLIPTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/configuration_altclip.py#L24)

( vocab\_size = 250002 hidden\_size = 1024 num\_hidden\_layers = 24 num\_attention\_heads = 16 intermediate\_size = 4096 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 514 type\_vocab\_size = 1 initializer\_range = 0.02 initializer\_factor = 0.02 layer\_norm\_eps = 1e-05 pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 position\_embedding\_type = 'absolute' use\_cache = True project\_dim = 768 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 250002) —
  Vocabulary size of the AltCLIP model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [AltCLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPTextModel).
* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 514) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **type\_vocab\_size** (`int`, *optional*, defaults to 1) —
  The vocabulary size of the `token_type_ids` passed when calling [AltCLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPTextModel)
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **initializer\_factor** (`float`, *optional*, defaults to 0.02) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **pad\_token\_id** (`int`, *optional*, defaults to 1) — The id of the *padding* token.
* **bos\_token\_id** (`int`, *optional*, defaults to 0) — The id of the *beginning-of-sequence* token.
* **eos\_token\_id** (`Union[int, list[int]]`, *optional*, defaults to 2) —
  The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"absolute"`) —
  Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
  positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
  [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
  For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
  with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **project\_dim** (`int`, *optional*, defaults to 768) —
  The dimensions of the teacher model before the mapping layer.

This is the configuration class to store the configuration of a [AltCLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPTextModel). It is used to instantiate a
AltCLIP text model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the AltCLIP
[BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import AltCLIPTextModel, AltCLIPTextConfig

>>> # Initializing a AltCLIPTextConfig with BAAI/AltCLIP style configuration
>>> configuration = AltCLIPTextConfig()

>>> # Initializing a AltCLIPTextModel (with random weights) from the BAAI/AltCLIP style configuration
>>> model = AltCLIPTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## AltCLIPVisionConfig

### class transformers.AltCLIPVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/configuration_altclip.py#L142)

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

This is the configuration class to store the configuration of a [AltCLIPModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPModel). It is used to instantiate an
AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the AltCLIP
[BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import AltCLIPVisionConfig, AltCLIPVisionModel

>>> # Initializing a AltCLIPVisionConfig with BAAI/AltCLIP style configuration
>>> configuration = AltCLIPVisionConfig()

>>> # Initializing a AltCLIPVisionModel (with random weights) from the BAAI/AltCLIP style configuration
>>> model = AltCLIPVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## AltCLIPModel

### class transformers.AltCLIPModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/modeling_altclip.py#L1171)

( config: AltCLIPConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/modeling_altclip.py#L1302)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None return\_loss: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → `transformers.models.altclip.modeling_altclip.AltCLIPOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([AltCLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPProcessor) uses
  `image_processor_class` for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
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
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.altclip.modeling_altclip.AltCLIPOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.altclip.modeling_altclip.AltCLIPOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AltCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
* **logits\_per\_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [AltCLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPTextModel).
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of [AltCLIPVisionModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPVisionModel).
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [AltCLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPTextModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output`, defaults to `None`) — The output of the [AltCLIPVisionModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPVisionModel).

The [AltCLIPModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AltCLIPModel

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
... )
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/modeling_altclip.py#L1254)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([AltCLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPProcessor) uses
  `image_processor_class` for processing images).
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
applying the projection layer to the pooled output of [AltCLIPVisionModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPVisionModel).

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AltCLIPModel

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt")
>>> image_features = model.get_image_features(**inputs)
```

#### get\_text\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/modeling_altclip.py#L1207)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None token\_type\_ids = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

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
* **token\_type\_ids** (``of shape`(batch\_size, sequence\_length)`) -- Segment token indices to indicate first and second portions of the inputs. Indices are selected in` [0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
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
applying the projection layer to the pooled output of [AltCLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPTextModel).

Examples:


```
>>> from transformers import AutoProcessor, AltCLIPModel

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

## AltCLIPTextModel

### class transformers.AltCLIPTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/modeling_altclip.py#L1088)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/modeling_altclip.py#L1107)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection` or `tuple(torch.FloatTensor)`

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
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AltCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPConfig)) and inputs.

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
* **projection\_state** (`tuple(torch.FloatTensor)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` of shape `(batch_size,config.project_dim)`.

  Text embeddings before the projection layer, used to mimic the last hidden state of the teacher encoder.

The [AltCLIPTextModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoProcessor, AltCLIPTextModel

>>> model = AltCLIPTextModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

>>> texts = ["it's a cat", "it's a dog"]

>>> inputs = processor(text=texts, padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled CLS states
```

## AltCLIPVisionModel

### class transformers.AltCLIPVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/modeling_altclip.py#L916)

( config: AltCLIPVisionConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/modeling_altclip.py#L929)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None interpolate\_pos\_encoding: bool = False return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([AltCLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPProcessor) uses
  `image_processor_class` for processing images).
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
elements depending on the configuration ([AltCLIPConfig](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPConfig)) and inputs.

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

The [AltCLIPVisionModel](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AltCLIPVisionModel

>>> model = AltCLIPVisionModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled CLS states
```

## AltCLIPProcessor

### class transformers.AltCLIPProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/altclip/processing_altclip.py#L31)

( image\_processor = None tokenizer = None  )

Parameters

* **image\_processor** ([CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([XLMRobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizerFast), *optional*) —
  The tokenizer is a required input.

Constructs a AltCLIP processor which wraps a CLIP image processor and a XLM-Roberta tokenizer into a single
processor.

[AltCLIPProcessor](/docs/transformers/v4.56.2/en/model_doc/altclip#transformers.AltCLIPProcessor) offers all the functionalities of [CLIPImageProcessor](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPImageProcessor) and [XLMRobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizerFast). See
the `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/altclip.md)
