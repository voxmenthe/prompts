# AltCLIP

[AltCLIP](https://huggingface.co/papers/2211.06679) replaces the [CLIP](./clip) text encoder with a multilingual XLM-R encoder and aligns image and text representations with teacher learning and contrastive learning.

You can find all the original AltCLIP checkpoints under the [AltClip](https://huggingface.co/collections/BAAI/alt-clip-diffusion-66987a97de8525205f1221bf) collection.

> [!TIP]
> Click on the AltCLIP models in the right sidebar for more examples of how to apply AltCLIP to different tasks.

The examples below demonstrates how to calculate similarity scores between an image and one or more captions with the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

```python
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

```python
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

- AltCLIP uses bidirectional attention instead of causal attention and it uses the `[CLS]` token in XLM-R to represent a text embedding.
- Use [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor) to resize (or rescale) and normalize images for the model.
- [AltCLIPProcessor](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPProcessor) combines [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor) and [XLMRobertaTokenizer](/docs/transformers/main/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer) into a single instance to encode text and prepare images.

## AltCLIPConfig[[transformers.AltCLIPConfig]]

#### transformers.AltCLIPConfig[[transformers.AltCLIPConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/configuration_altclip.py#L227)

This is the configuration class to store the configuration of a [AltCLIPModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPModel). It is used to instantiate an
AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the AltCLIP
[BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
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

>>> config = AltCLIPConfig(text_config=config_text, vision_config=config_vision)
```

**Parameters:**

text_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [AltCLIPTextConfig](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPTextConfig).

vision_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [AltCLIPVisionConfig](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPVisionConfig).

projection_dim (`int`, *optional*, defaults to 768) : Dimensionality of text and vision projection layers.

logit_scale_init_value (`float`, *optional*, defaults to 2.6592) : The initial value of the *logit_scale* parameter. Default is used as per the original CLIP implementation.

kwargs (*optional*) : Dictionary of keyword arguments.

## AltCLIPTextConfig[[transformers.AltCLIPTextConfig]]

#### transformers.AltCLIPTextConfig[[transformers.AltCLIPTextConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/configuration_altclip.py#L24)

This is the configuration class to store the configuration of a [AltCLIPTextModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPTextModel). It is used to instantiate a
AltCLIP text model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the AltCLIP
[BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Examples:

```python
>>> from transformers import AltCLIPTextModel, AltCLIPTextConfig

>>> # Initializing a AltCLIPTextConfig with BAAI/AltCLIP style configuration
>>> configuration = AltCLIPTextConfig()

>>> # Initializing a AltCLIPTextModel (with random weights) from the BAAI/AltCLIP style configuration
>>> model = AltCLIPTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 250002) : Vocabulary size of the AltCLIP model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [AltCLIPTextModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPTextModel).

hidden_size (`int`, *optional*, defaults to 1024) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 24) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 4096) : Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to 514) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

type_vocab_size (`int`, *optional*, defaults to 1) : The vocabulary size of the `token_type_ids` passed when calling [AltCLIPTextModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPTextModel)

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

initializer_factor (`float`, *optional*, defaults to 0.02) : A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

pad_token_id (`int`, *optional*, defaults to 1) : The id of the *padding* token.

bos_token_id (`int`, *optional*, defaults to 0) : The id of the *beginning-of-sequence* token.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to 2) : The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.

project_dim (`int`, *optional*, defaults to 768) : The dimensions of the teacher model before the mapping layer.

## AltCLIPVisionConfig[[transformers.AltCLIPVisionConfig]]

#### transformers.AltCLIPVisionConfig[[transformers.AltCLIPVisionConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/configuration_altclip.py#L134)

This is the configuration class to store the configuration of a [AltCLIPModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPModel). It is used to instantiate an
AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the AltCLIP
[BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import AltCLIPVisionConfig, AltCLIPVisionModel

>>> # Initializing a AltCLIPVisionConfig with BAAI/AltCLIP style configuration
>>> configuration = AltCLIPVisionConfig()

>>> # Initializing a AltCLIPVisionModel (with random weights) from the BAAI/AltCLIP style configuration
>>> model = AltCLIPVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

projection_dim (`int`, *optional*, defaults to 512) : Dimensionality of text and vision projection layers.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

image_size (`int`, *optional*, defaults to 224) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 32) : The size (resolution) of each patch.

hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

initializer_factor (`float`, *optional*, defaults to 1.0) : A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

## AltCLIPModel[[transformers.AltCLIPModel]]

#### transformers.AltCLIPModel[[transformers.AltCLIPModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/modeling_altclip.py#L1117)

forwardtransformers.AltCLIPModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/modeling_altclip.py#L1229[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "return_loss", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AltCLIPProcessor](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPProcessor) uses
  [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **return_loss** (`bool`, *optional*) --
  Whether or not to return the contrastive loss.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.altclip.modeling_altclip.AltCLIPOutput` or `tuple(torch.FloatTensor)`A `transformers.models.altclip.modeling_altclip.AltCLIPOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AltCLIPConfig](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [AltCLIPTextModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The image embeddings obtained by applying the projection layer to the pooled output of [AltCLIPVisionModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [AltCLIPTextModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [AltCLIPVisionModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPVisionModel).
The [AltCLIPModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
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

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AltCLIPProcessor](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPProcessor) uses [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

token_type_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

return_loss (`bool`, *optional*) : Whether or not to return the contrastive loss.

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

interpolate_pos_encoding (`bool`, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

``transformers.models.altclip.modeling_altclip.AltCLIPOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.altclip.modeling_altclip.AltCLIPOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AltCLIPConfig](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [AltCLIPTextModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The image embeddings obtained by applying the projection layer to the pooled output of [AltCLIPVisionModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [AltCLIPTextModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [AltCLIPVisionModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPVisionModel).
#### get_image_features[[transformers.AltCLIPModel.get_image_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/modeling_altclip.py#L1191)

Examples:

```python
>>> import torch
>>> from transformers import AutoProcessor, AltCLIPModel
>>> from transformers.image_utils import load_image

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = load_image(url)

>>> inputs = processor(images=image, return_tensors="pt")
>>> with torch.inference_mode():
...     image_features = model.get_image_features(**inputs)
```

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AltCLIPProcessor](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPProcessor) uses [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).

interpolate_pos_encoding (`bool`, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

**Returns:**

`image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The image embeddings obtained by
applying the projection layer to the pooled output of [AltCLIPVisionModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPVisionModel).
#### get_text_features[[transformers.AltCLIPModel.get_text_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/modeling_altclip.py#L1153)

Examples:

```python
>>> import torch
>>> from transformers import AutoProcessor, AltCLIPModel

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> with torch.inference_mode():
...     text_features = model.get_text_features(**inputs)
```

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

token_type_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

**Returns:**

`text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The text embeddings obtained by
applying the projection layer to the pooled output of [AltCLIPTextModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPTextModel).

## AltCLIPTextModel[[transformers.AltCLIPTextModel]]

#### transformers.AltCLIPTextModel[[transformers.AltCLIPTextModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/modeling_altclip.py#L1034)

forwardtransformers.AltCLIPTextModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/modeling_altclip.py#L1054[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.0`transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection` or `tuple(torch.FloatTensor)`A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AltCLIPConfig](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **projection_state** (`tuple(torch.FloatTensor)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` of shape `(batch_size,config.project_dim)`.

  Text embeddings before the projection layer, used to mimic the last hidden state of the teacher encoder.
The [AltCLIPTextModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoProcessor, AltCLIPTextModel

>>> model = AltCLIPTextModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

>>> texts = ["it's a cat", "it's a dog"]

>>> inputs = processor(text=texts, padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled CLS states
```

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

**Returns:**

``transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection` or `tuple(torch.FloatTensor)``

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndProjection` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AltCLIPConfig](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **projection_state** (`tuple(torch.FloatTensor)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` of shape `(batch_size,config.project_dim)`.

  Text embeddings before the projection layer, used to mimic the last hidden state of the teacher encoder.

## AltCLIPVisionModel[[transformers.AltCLIPVisionModel]]

#### transformers.AltCLIPVisionModel[[transformers.AltCLIPVisionModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/modeling_altclip.py#L872)

forwardtransformers.AltCLIPVisionModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/modeling_altclip.py#L886[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AltCLIPProcessor](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPProcessor) uses
  [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AltCLIPConfig](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [AltCLIPVisionModel](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
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

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor). See [CLIPImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AltCLIPProcessor](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPProcessor) uses [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor) for processing images).

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

interpolate_pos_encoding (`bool`, defaults to `False`) : Whether to interpolate the pre-trained position encodings.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AltCLIPConfig](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## AltCLIPProcessor[[transformers.AltCLIPProcessor]]

#### transformers.AltCLIPProcessor[[transformers.AltCLIPProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/altclip/processing_altclip.py#L22)

Constructs a AltCLIP processor which wraps a CLIP image processor and a XLM-Roberta tokenizer into a single
processor.

[AltCLIPProcessor](/docs/transformers/main/en/model_doc/altclip#transformers.AltCLIPProcessor) offers all the functionalities of [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor) and [XLMRobertaTokenizerFast](/docs/transformers/main/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer). See
the [__call__()](/docs/transformers/main/en/model_doc/bros#transformers.BrosProcessor.__call__) and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

**Parameters:**

image_processor ([CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor), *optional*) : The image processor is a required input.

tokenizer ([XLMRobertaTokenizerFast](/docs/transformers/main/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer), *optional*) : The tokenizer is a required input.
