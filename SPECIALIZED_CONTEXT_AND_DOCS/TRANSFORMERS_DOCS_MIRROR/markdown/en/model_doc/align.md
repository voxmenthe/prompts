*This model was released on 2021-02-11 and added to Hugging Face Transformers on 2023-03-01.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![Transformers](https://img.shields.io/badge/Transformers-6B5B95?style=flat&logo=transformers&logoColor=white)

# ALIGN

[ALIGN](https://huggingface.co/papers/2102.05918) is pretrained on a noisy 1.8 billion alt‑text and image pair dataset to show that scale can make up for the noise. It uses a dual‑encoder architecture, [EfficientNet](./efficientnet) for images and [BERT](./bert) for text, and a contrastive loss to align similar image–text embeddings together while pushing different embeddings apart. Once trained, ALIGN can encode any image and candidate captions into a shared vector space for zero‑shot retrieval or classification without requiring extra labels. This scale‑first approach reduces dataset curation costs and powers state‑of‑the‑art image–text retrieval and zero‑shot ImageNet classification.

You can find all the original ALIGN checkpoints under the [Kakao Brain](https://huggingface.co/kakaobrain?search_models=align) organization.

Click on the ALIGN models in the right sidebar for more examples of how to apply ALIGN to different vision and text related tasks.

The example below demonstrates zero-shot image classification with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
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

## Notes

* ALIGN projects the text and visual features into latent space and the dot product between the projected image and text features is used as the similarity score. The example below demonstrates how to calculate the image-text similarity score with [AlignProcessor](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignProcessor) and [AlignModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignModel).


  ```
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

* Refer to the [Kakao Brain’s Open Source ViT, ALIGN, and the New COYO Text-Image Dataset](https://huggingface.co/blog/vit-align) blog post for more details.

## AlignConfig

### class transformers.AlignConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/configuration_align.py#L253)

( text\_config = None vision\_config = None projection\_dim = 640 temperature\_init\_value = 1.0 initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [AlignTextConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignTextConfig).
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [AlignVisionConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignVisionConfig).
* **projection\_dim** (`int`, *optional*, defaults to 640) —
  Dimensionality of text and vision projection layers.
* **temperature\_init\_value** (`float`, *optional*, defaults to 1.0) —
  The initial value of the *temperature* parameter. Default is used as per the original ALIGN implementation.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[AlignConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignConfig) is the configuration class to store the configuration of a [AlignModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignModel). It is used to
instantiate a ALIGN model according to the specified arguments, defining the text model and vision model configs.
Instantiating a configuration with the defaults will yield a similar configuration to that of the ALIGN
[kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
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

>>> config = AlignConfig.from_text_vision_configs(config_text, config_vision)
```

#### from\_text\_vision\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/configuration_utils.py#L1254)

( text\_config vision\_config \*\*kwargs  ) → `PreTrainedConfig`

Returns

`PreTrainedConfig`

An instance of a configuration object

Instantiate a model config (or a derived class) from text model configuration and vision model
configuration.

## AlignTextConfig

### class transformers.AlignTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/configuration_align.py#L24)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 position\_embedding\_type = 'absolute' use\_cache = True \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the Align Text model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [AlignTextModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignTextModel).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the `token_type_ids` passed when calling [AlignTextModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignTextModel).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  Padding token id.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"absolute"`) —
  Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
  positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
  [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
  For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
  with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.

This is the configuration class to store the configuration of a [AlignTextModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignTextModel). It is used to instantiate a
ALIGN text encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the text encoder of the ALIGN
[kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values here are
copied from BERT.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import AlignTextConfig, AlignTextModel

>>> # Initializing a AlignTextConfig with kakaobrain/align-base style configuration
>>> configuration = AlignTextConfig()

>>> # Initializing a AlignTextModel (with random weights) from the kakaobrain/align-base style configuration
>>> model = AlignTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## AlignVisionConfig

### class transformers.AlignVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/configuration_align.py#L131)

( num\_channels: int = 3 image\_size: int = 600 width\_coefficient: float = 2.0 depth\_coefficient: float = 3.1 depth\_divisor: int = 8 kernel\_sizes: list = [3, 3, 5, 3, 5, 5, 3] in\_channels: list = [32, 16, 24, 40, 80, 112, 192] out\_channels: list = [16, 24, 40, 80, 112, 192, 320] depthwise\_padding: list = [] strides: list = [1, 2, 2, 2, 1, 2, 1] num\_block\_repeats: list = [1, 2, 2, 3, 3, 4, 1] expand\_ratios: list = [1, 6, 6, 6, 6, 6, 6] squeeze\_expansion\_ratio: float = 0.25 hidden\_act: str = 'swish' hidden\_dim: int = 2560 pooling\_type: str = 'mean' initializer\_range: float = 0.02 batch\_norm\_eps: float = 0.001 batch\_norm\_momentum: float = 0.99 drop\_connect\_rate: float = 0.2 \*\*kwargs  )

Parameters

* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of input channels.
* **image\_size** (`int`, *optional*, defaults to 600) —
  The input image size.
* **width\_coefficient** (`float`, *optional*, defaults to 2.0) —
  Scaling coefficient for network width at each stage.
* **depth\_coefficient** (`float`, *optional*, defaults to 3.1) —
  Scaling coefficient for network depth at each stage.
* **depth\_divisor** `int`, *optional*, defaults to 8) —
  A unit of network width.
* **kernel\_sizes** (`list[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`) —
  List of kernel sizes to be used in each block.
* **in\_channels** (`list[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`) —
  List of input channel sizes to be used in each block for convolutional layers.
* **out\_channels** (`list[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`) —
  List of output channel sizes to be used in each block for convolutional layers.
* **depthwise\_padding** (`list[int]`, *optional*, defaults to `[]`) —
  List of block indices with square padding.
* **strides** (`list[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`) —
  List of stride sizes to be used in each block for convolutional layers.
* **num\_block\_repeats** (`list[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`) —
  List of the number of times each block is to repeated.
* **expand\_ratios** (`list[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`) —
  List of scaling coefficient of each block.
* **squeeze\_expansion\_ratio** (`float`, *optional*, defaults to 0.25) —
  Squeeze expansion ratio.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
  `"selu",` “gelu\_new”`,` “silu”`and`“mish”` are supported.
* **hidden\_dim** (`int`, *optional*, defaults to 1280) —
  The hidden dimension of the layer before the classification head.
* **pooling\_type** (`str` or `function`, *optional*, defaults to `"mean"`) —
  Type of final pooling to be applied before the dense classification head. Available options are [`"mean"`,
  `"max"`]
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **batch\_norm\_eps** (`float`, *optional*, defaults to 1e-3) —
  The epsilon used by the batch normalization layers.
* **batch\_norm\_momentum** (`float`, *optional*, defaults to 0.99) —
  The momentum used by the batch normalization layers.
* **drop\_connect\_rate** (`float`, *optional*, defaults to 0.2) —
  The drop rate for skip connections.

This is the configuration class to store the configuration of a [AlignVisionModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignVisionModel). It is used to instantiate a
ALIGN vision encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the vision encoder of the ALIGN
[kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values are copied
from EfficientNet (efficientnet-b7)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import AlignVisionConfig, AlignVisionModel

>>> # Initializing a AlignVisionConfig with kakaobrain/align-base style configuration
>>> configuration = AlignVisionConfig()

>>> # Initializing a AlignVisionModel (with random weights) from the kakaobrain/align-base style configuration
>>> model = AlignVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## AlignProcessor

### class transformers.AlignProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/processing_align.py#L36)

( image\_processor tokenizer  )

Parameters

* **image\_processor** ([EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor)) —
  The image processor is a required input.
* **tokenizer** ([`BertTokenizer`, `BertTokenizerFast`]) —
  The tokenizer is a required input.

Constructs an ALIGN processor which wraps [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) and
[BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer)/[BertTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast) into a single processor that inherits both the image processor and
tokenizer functionalities. See the `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more
information.
The preferred way of passing kwargs is as a dictionary per modality, see usage example below.


```
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

## AlignModel

### class transformers.AlignModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/modeling_align.py#L1095)

( config: AlignConfig  )

Parameters

* **config** ([AlignConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Align Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/modeling_align.py#L1226)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None return\_loss: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.align.modeling_align.AlignOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor). See [EfficientNetImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AlignProcessor](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignProcessor) uses
  [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) for processing images).
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
* **return\_loss** (`bool`, *optional*) —
  Whether or not to return the contrastive loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.align.modeling_align.AlignOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.align.modeling_align.AlignOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlignConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
* **logits\_per\_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [AlignTextModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignTextModel).
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The output of [AlignVisionModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignVisionModel).
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [AlignTextModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignTextModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPoolingAndNoAttention'>.vision_model_output`, defaults to `None`) — The output of the [AlignVisionModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignVisionModel).

The [AlignModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AlignModel

>>> model = AlignModel.from_pretrained("kakaobrain/align-base")
>>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     images=image, text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

#### get\_text\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/modeling_align.py#L1128)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

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
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by
applying the projection layer to the pooled output of [AlignTextModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignTextModel).

Examples:


```
>>> from transformers import AutoTokenizer, AlignModel

>>> model = AlignModel.from_pretrained("kakaobrain/align-base")
>>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/modeling_align.py#L1181)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor). See [EfficientNetImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AlignProcessor](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignProcessor) uses
  [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by
applying the projection layer to the pooled output of [AlignVisionModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignVisionModel).

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AlignModel

>>> model = AlignModel.from_pretrained("kakaobrain/align-base")
>>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```

## AlignTextModel

### class transformers.AlignTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/modeling_align.py#L886)

( config: AlignTextConfig add\_pooling\_layer: bool = True  )

Parameters

* **config** ([AlignTextConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The text model from ALIGN without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/modeling_align.py#L912)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

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
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
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
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlignConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignConfig)) and inputs.

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

The [AlignTextModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, AlignTextModel

>>> model = AlignTextModel.from_pretrained("kakaobrain/align-base")
>>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
```

## AlignVisionModel

### class transformers.AlignVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/modeling_align.py#L1014)

( config: AlignVisionConfig  )

Parameters

* **config** ([AlignVisionConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignVisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The vision model from ALIGN without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/align/modeling_align.py#L1039)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor). See [EfficientNetImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([AlignProcessor](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignProcessor) uses
  [EfficientNetImageProcessor](/docs/transformers/v4.56.2/en/model_doc/efficientnet#transformers.EfficientNetImageProcessor) for processing images).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([AlignConfig](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state after a pooling operation on the spatial dimensions.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

The [AlignVisionModel](/docs/transformers/v4.56.2/en/model_doc/align#transformers.AlignVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/align.md)
