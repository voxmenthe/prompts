# ALIGN

[ALIGN](https://huggingface.co/papers/2102.05918) is pretrained on a noisy 1.8 billion alt‑text and image pair dataset to show that scale can make up for the noise. It uses a dual‑encoder architecture, [EfficientNet](./efficientnet) for images and [BERT](./bert) for text, and a contrastive loss to align similar image–text embeddings together while pushing different embeddings apart. Once trained, ALIGN can encode any image and candidate captions into a shared vector space for zero‑shot retrieval or classification without requiring extra labels. This scale‑first approach reduces dataset curation costs and powers state‑of‑the‑art image–text retrieval and zero‑shot ImageNet classification.

You can find all the original ALIGN checkpoints under the [Kakao Brain](https://huggingface.co/kakaobrain?search_models=align) organization.

> [!TIP]
> Click on the ALIGN models in the right sidebar for more examples of how to apply ALIGN to different vision and text related tasks.

The example below demonstrates zero-shot image classification with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

  

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="zero-shot-image-classification",
    model="kakaobrain/align-base",
    device=0,
    dtype=torch.bfloat16
)

candidate_labels = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a person"
]

pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", candidate_labels=candidate_labels)
```

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
model = AutoModelForZeroShotImageClassification.from_pretrained("kakaobrain/align-base", device_map="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = requests.get(url, stream=True)
inputs = Image.open(image.raw).convert("RGB")

image_inputs = processor(images=inputs, return_tensors="pt").to(model.device)
with torch.no_grad():
    image_embeds = model.get_image_features(**image_inputs)

candidate_labels = ["a photo of a dog", "a photo of a cat", "a photo of a person"]
text_inputs = processor(text=candidate_labels, padding=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    text_embeds = model.get_text_features(**text_inputs)

image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds  = text_embeds  / text_embeds.norm(p=2, dim=-1, keepdim=True)

logits = (image_embeds @ text_embeds.T) * 100.0
probs  = logits.softmax(dim=-1).cpu().squeeze()

for label, score in zip(candidate_labels, probs):
    print(f"{label:20s} → {score.item():.4f}")
```

## Notes

- ALIGN projects the text and visual features into latent space and the dot product between the projected image and text features is used as the similarity score. The example below demonstrates how to calculate the image-text similarity score with [AlignProcessor](/docs/transformers/main/en/model_doc/align#transformers.AlignProcessor) and [AlignModel](/docs/transformers/main/en/model_doc/align#transformers.AlignModel).

  ```py
  # Example of using ALIGN for image-text similarity
  from transformers import AlignProcessor, AlignModel
  import torch
  from PIL import Image
  import requests
  from io import BytesIO
  
  # Load processor and model
  processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
  model = AlignModel.from_pretrained("kakaobrain/align-base")
  
  # Download image from URL
  url = "https://huggingface.co/roschmid/dog-races/resolve/main/images/Golden_Retriever.jpg"
  response = requests.get(url)
  image = Image.open(BytesIO(response.content))  # Convert the downloaded bytes to a PIL Image
  
  texts = ["a photo of a cat", "a photo of a dog"]
  
  # Process image and text inputs
  inputs = processor(images=image, text=texts, return_tensors="pt")
  
  # Get the embeddings
  with torch.no_grad():
      outputs = model(**inputs)
  
  image_embeds = outputs.image_embeds
  text_embeds = outputs.text_embeds
  
  # Normalize embeddings for cosine similarity
  image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
  text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
  
  # Calculate similarity scores
  similarity_scores = torch.matmul(text_embeds, image_embeds.T)
  
  # Print raw scores
  print("Similarity scores:", similarity_scores)
  
  # Convert to probabilities
  probs = torch.nn.functional.softmax(similarity_scores, dim=0)
  print("Probabilities:", probs)
  
  # Get the most similar text
  most_similar_idx = similarity_scores.argmax().item()
  print(f"Most similar text: '{texts[most_similar_idx]}'")
  ```

## Resources

- Refer to the [Kakao Brain’s Open Source ViT, ALIGN, and the New COYO Text-Image Dataset](https://huggingface.co/blog/vit-align) blog post for more details.

## AlignConfig[[transformers.AlignConfig]]

#### transformers.AlignConfig[[transformers.AlignConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/align/configuration_align.py#L245)

[AlignConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignConfig) is the configuration class to store the configuration of a [AlignModel](/docs/transformers/main/en/model_doc/align#transformers.AlignModel). It is used to
instantiate a ALIGN model according to the specified arguments, defining the text model and vision model configs.
Instantiating a configuration with the defaults will yield a similar configuration to that of the ALIGN
[kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import AlignConfig, AlignModel

>>> # Initializing a AlignConfig with kakaobrain/align-base style configuration
>>> configuration = AlignConfig()

>>> # Initializing a AlignModel (with random weights) from the kakaobrain/align-base style configuration
>>> model = AlignModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a AlignConfig from a AlignTextConfig and a AlignVisionConfig
>>> from transformers import AlignTextConfig, AlignVisionConfig

>>> # Initializing ALIGN Text and Vision configurations
>>> config_text = AlignTextConfig()
>>> config_vision = AlignVisionConfig()

>>> config = AlignConfig(text_config=config_text, vision_config=config_vision)
```

**Parameters:**

text_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [AlignTextConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignTextConfig).

vision_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [AlignVisionConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignVisionConfig).

projection_dim (`int`, *optional*, defaults to 640) : Dimensionality of text and vision projection layers.

temperature_init_value (`float`, *optional*, defaults to 1.0) : The initial value of the *temperature* parameter. Default is used as per the original ALIGN implementation.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

kwargs (*optional*) : Dictionary of keyword arguments.

## AlignTextConfig[[transformers.AlignTextConfig]]

#### transformers.AlignTextConfig[[transformers.AlignTextConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/align/configuration_align.py#L24)

This is the configuration class to store the configuration of a [AlignTextModel](/docs/transformers/main/en/model_doc/align#transformers.AlignTextModel). It is used to instantiate a
ALIGN text encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the text encoder of the ALIGN
[kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values here are
copied from BERT.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import AlignTextConfig, AlignTextModel

>>> # Initializing a AlignTextConfig with kakaobrain/align-base style configuration
>>> configuration = AlignTextConfig()

>>> # Initializing a AlignTextModel (with random weights) from the kakaobrain/align-base style configuration
>>> model = AlignTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 30522) : Vocabulary size of the Align Text model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [AlignTextModel](/docs/transformers/main/en/model_doc/align#transformers.AlignTextModel).

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

type_vocab_size (`int`, *optional*, defaults to 2) : The vocabulary size of the `token_type_ids` passed when calling [AlignTextModel](/docs/transformers/main/en/model_doc/align#transformers.AlignTextModel).

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

pad_token_id (`int`, *optional*, defaults to 0) : Padding token id.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.

## AlignVisionConfig[[transformers.AlignVisionConfig]]

#### transformers.AlignVisionConfig[[transformers.AlignVisionConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/align/configuration_align.py#L123)

This is the configuration class to store the configuration of a [AlignVisionModel](/docs/transformers/main/en/model_doc/align#transformers.AlignVisionModel). It is used to instantiate a
ALIGN vision encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the vision encoder of the ALIGN
[kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values are copied
from EfficientNet (efficientnet-b7)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import AlignVisionConfig, AlignVisionModel

>>> # Initializing a AlignVisionConfig with kakaobrain/align-base style configuration
>>> configuration = AlignVisionConfig()

>>> # Initializing a AlignVisionModel (with random weights) from the kakaobrain/align-base style configuration
>>> model = AlignVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

num_channels (`int`, *optional*, defaults to 3) : The number of input channels.

image_size (`int`, *optional*, defaults to 600) : The input image size.

width_coefficient (`float`, *optional*, defaults to 2.0) : Scaling coefficient for network width at each stage.

depth_coefficient (`float`, *optional*, defaults to 3.1) : Scaling coefficient for network depth at each stage.

depth_divisor `int`, *optional*, defaults to 8) : A unit of network width.

kernel_sizes (`list[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`) : List of kernel sizes to be used in each block.

in_channels (`list[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`) : List of input channel sizes to be used in each block for convolutional layers.

out_channels (`list[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`) : List of output channel sizes to be used in each block for convolutional layers.

depthwise_padding (`list[int]`, *optional*, defaults to `[]`) : List of block indices with square padding.

strides (`list[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`) : List of stride sizes to be used in each block for convolutional layers.

num_block_repeats (`list[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`) : List of the number of times each block is to repeated.

expand_ratios (`list[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`) : List of scaling coefficient of each block.

squeeze_expansion_ratio (`float`, *optional*, defaults to 0.25) : Squeeze expansion ratio.

hidden_act (`str` or `function`, *optional*, defaults to `"silu"`) : The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`, `"selu", `"gelu_new"`, `"silu"` and `"mish"` are supported.

hidden_dim (`int`, *optional*, defaults to 1280) : The hidden dimension of the layer before the classification head.

pooling_type (`str` or `function`, *optional*, defaults to `"mean"`) : Type of final pooling to be applied before the dense classification head. Available options are [`"mean"`, `"max"`]

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

batch_norm_eps (`float`, *optional*, defaults to 1e-3) : The epsilon used by the batch normalization layers.

batch_norm_momentum (`float`, *optional*, defaults to 0.99) : The momentum used by the batch normalization layers.

drop_connect_rate (`float`, *optional*, defaults to 0.2) : The drop rate for skip connections.

## AlignProcessor[[transformers.AlignProcessor]]

#### transformers.AlignProcessor[[transformers.AlignProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/align/processing_align.py#L32)

Constructs an ALIGN processor which wraps [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) and
[BertTokenizer](/docs/transformers/main/en/model_doc/electra#transformers.BertTokenizer)/[BertTokenizerFast](/docs/transformers/main/en/model_doc/electra#transformers.BertTokenizer) into a single processor that inherits both the image processor and
tokenizer functionalities. See the [__call__()](/docs/transformers/main/en/model_doc/bros#transformers.BrosProcessor.__call__) and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more
information.
The preferred way of passing kwargs is as a dictionary per modality, see usage example below.

```python
from transformers import AlignProcessor
from PIL import Image
model_id = "kakaobrain/align-base"
processor = AlignProcessor.from_pretrained(model_id)

processor(
    images=your_pil_image,
    text=["What is that?"],
    images_kwargs = {"crop_size": {"height": 224, "width": 224}},
    text_kwargs = {"padding": "do_not_pad"},
    common_kwargs = {"return_tensors": "pt"},
)
```

**Parameters:**

image_processor ([EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor)) : The image processor is a required input.

tokenizer ([`BertTokenizer`, `BertTokenizerFast`]) : The tokenizer is a required input.

## AlignModel[[transformers.AlignModel]]

#### transformers.AlignModel[[transformers.AlignModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/align/modeling_align.py#L1057)

The bare Align Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.AlignModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/align/modeling_align.py#L1159[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "return_loss", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor). See [EfficientNetImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AlignProcessor](/docs/transformers/main/en/model_doc/align#transformers.AlignProcessor) uses
  [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) for processing images).
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
- **return_loss** (`bool`, *optional*) --
  Whether or not to return the contrastive loss.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.align.modeling_align.AlignOutput` or `tuple(torch.FloatTensor)`A `transformers.models.align.modeling_align.AlignOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlignConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [AlignTextModel](/docs/transformers/main/en/model_doc/align#transformers.AlignTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The output of [AlignVisionModel](/docs/transformers/main/en/model_doc/align#transformers.AlignVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [AlignTextModel](/docs/transformers/main/en/model_doc/align#transformers.AlignTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [AlignVisionModel](/docs/transformers/main/en/model_doc/align#transformers.AlignVisionModel).
The [AlignModel](/docs/transformers/main/en/model_doc/align#transformers.AlignModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> import torch
>>> from transformers import AutoProcessor, AlignModel
>>> from transformers.image_utils import load_image

>>> model = AlignModel.from_pretrained("kakaobrain/align-base")
>>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = load_image(url)

>>> inputs = processor(
...     images=image, text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding=True
... )

>>> with torch.inference_mode():
...     outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

**Parameters:**

config ([AlignConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.align.modeling_align.AlignOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.align.modeling_align.AlignOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlignConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [AlignTextModel](/docs/transformers/main/en/model_doc/align#transformers.AlignTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The output of [AlignVisionModel](/docs/transformers/main/en/model_doc/align#transformers.AlignVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [AlignTextModel](/docs/transformers/main/en/model_doc/align#transformers.AlignTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [AlignVisionModel](/docs/transformers/main/en/model_doc/align#transformers.AlignVisionModel).
#### get_text_features[[transformers.AlignModel.get_text_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/align/modeling_align.py#L1090)

Examples:

```python
>>> import torch
>>> from transformers import AutoTokenizer, AlignModel

>>> model = AlignModel.from_pretrained("kakaobrain/align-base")
>>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> with torch.inference_mode():
...     text_features = model.get_text_features(**inputs)
```

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

**Returns:**

`text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The text embeddings obtained by
applying the projection layer to the pooled output of [AlignTextModel](/docs/transformers/main/en/model_doc/align#transformers.AlignTextModel).
#### get_image_features[[transformers.AlignModel.get_image_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/align/modeling_align.py#L1130)

Examples:

```python
>>> import torch
>>> from transformers import AutoProcessor, AlignModel
>>> from transformers.image_utils import load_image

>>> model = AlignModel.from_pretrained("kakaobrain/align-base")
>>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = load_image(url)

>>> inputs = processor(images=image, return_tensors="pt")
>>> with torch.inference_mode():
...     image_features = model.get_image_features(**inputs)
```

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor). See [EfficientNetImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AlignProcessor](/docs/transformers/main/en/model_doc/align#transformers.AlignProcessor) uses [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) for processing images).

**Returns:**

`image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The image embeddings obtained by
applying the projection layer to the pooled output of [AlignVisionModel](/docs/transformers/main/en/model_doc/align#transformers.AlignVisionModel).

## AlignTextModel[[transformers.AlignTextModel]]

#### transformers.AlignTextModel[[transformers.AlignTextModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/align/modeling_align.py#L854)

The text model from ALIGN without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.AlignTextModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/align/modeling_align.py#L881[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
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
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlignConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignConfig)) and inputs.

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
The [AlignTextModel](/docs/transformers/main/en/model_doc/align#transformers.AlignTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoTokenizer, AlignTextModel

>>> model = AlignTextModel.from_pretrained("kakaobrain/align-base")
>>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
```

**Parameters:**

config ([AlignTextConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignTextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

add_pooling_layer (`bool`, *optional*, defaults to `True`) : Whether to add a pooling layer

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlignConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignConfig)) and inputs.

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

## AlignVisionModel[[transformers.AlignVisionModel]]

#### transformers.AlignVisionModel[[transformers.AlignVisionModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/align/modeling_align.py#L974)

The vision model from ALIGN without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.AlignVisionModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/align/modeling_align.py#L1000[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor). See [EfficientNetImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AlignProcessor](/docs/transformers/main/en/model_doc/align#transformers.AlignProcessor) uses
  [EfficientNetImageProcessor](/docs/transformers/main/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) for processing images).
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlignConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state after a pooling operation on the spatial dimensions.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
The [AlignVisionModel](/docs/transformers/main/en/model_doc/align#transformers.AlignVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AlignVisionModel

>>> model = AlignVisionModel.from_pretrained("kakaobrain/align-base")
>>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled CLS states
```

**Parameters:**

config ([AlignVisionConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignVisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)``

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlignConfig](/docs/transformers/main/en/model_doc/align#transformers.AlignConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state after a pooling operation on the spatial dimensions.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
