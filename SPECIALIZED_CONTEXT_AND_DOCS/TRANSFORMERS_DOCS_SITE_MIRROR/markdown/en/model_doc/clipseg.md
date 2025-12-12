# CLIPSeg

## Overview

The CLIPSeg model was proposed in [Image Segmentation Using Text and Image Prompts](https://huggingface.co/papers/2112.10003) by Timo Lüddecke
and Alexander Ecker. CLIPSeg adds a minimal decoder on top of a frozen [CLIP](clip) model for zero-shot and one-shot image segmentation.

The abstract from the paper is the following:

*Image segmentation is usually addressed by training a
model for a fixed set of object classes. Incorporating additional classes or more complex queries later is expensive
as it requires re-training the model on a dataset that encompasses these expressions. Here we propose a system
that can generate image segmentations based on arbitrary
prompts at test time. A prompt can be either a text or an
image. This approach enables us to create a unified model
(trained once) for three common segmentation tasks, which
come with distinct challenges: referring expression segmentation, zero-shot segmentation and one-shot segmentation.
We build upon the CLIP model as a backbone which we extend with a transformer-based decoder that enables dense
prediction. After training on an extended version of the
PhraseCut dataset, our system generates a binary segmentation map for an image based on a free-text prompt or on
an additional image expressing the query. We analyze different variants of the latter image-based prompts in detail.
This novel hybrid input allows for dynamic adaptation not
only to the three segmentation tasks mentioned above, but
to any binary segmentation task where a text or image query
can be formulated. Finally, we find our system to adapt well
to generalized queries involving affordances or properties*

 CLIPSeg overview. Taken from the original paper. 

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/timojl/clipseg).

## Usage tips

- [CLIPSegForImageSegmentation](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegForImageSegmentation) adds a decoder on top of [CLIPSegModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegModel). The latter is identical to [CLIPModel](/docs/transformers/main/en/model_doc/clip#transformers.CLIPModel).
- [CLIPSegForImageSegmentation](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegForImageSegmentation) can generate image segmentations based on arbitrary prompts at test time. A prompt can be either a text
(provided to the model as `input_ids`) or an image (provided to the model as `conditional_pixel_values`). One can also provide custom
conditional embeddings (provided to the model as `conditional_embeddings`).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with CLIPSeg. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- A notebook that illustrates [zero-shot image segmentation with CLIPSeg](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb).

## CLIPSegConfig[[transformers.CLIPSegConfig]]

#### transformers.CLIPSegConfig[[transformers.CLIPSegConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/configuration_clipseg.py#L207)

[CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig) is the configuration class to store the configuration of a [CLIPSegModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegModel). It is used to
instantiate a CLIPSeg model according to the specified arguments, defining the text model and vision model configs.
Instantiating a configuration with the defaults will yield a similar configuration to that of the CLIPSeg
[CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import CLIPSegConfig, CLIPSegModel

>>> # Initializing a CLIPSegConfig with CIDAS/clipseg-rd64 style configuration
>>> configuration = CLIPSegConfig()

>>> # Initializing a CLIPSegModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
>>> model = CLIPSegModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a CLIPSegConfig from a CLIPSegTextConfig and a CLIPSegVisionConfig

>>> # Initializing a CLIPSegText and CLIPSegVision configuration
>>> config_text = CLIPSegTextConfig()
>>> config_vision = CLIPSegVisionConfig()

>>> config = CLIPSegConfig(text_config=config_text, vision_config=config_vision)
```

**Parameters:**

text_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [CLIPSegTextConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextConfig).

vision_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [CLIPSegVisionConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionConfig).

projection_dim (`int`, *optional*, defaults to 512) : Dimensionality of text and vision projection layers.

logit_scale_init_value (`float`, *optional*, defaults to 2.6592) : The initial value of the *logit_scale* parameter. Default is used as per the original CLIPSeg implementation.

extract_layers (`list[int]`, *optional*, defaults to `[3, 6, 9]`) : Layers to extract when forwarding the query image through the frozen visual backbone of CLIP.

reduce_dim (`int`, *optional*, defaults to 64) : Dimensionality to reduce the CLIP vision embedding.

decoder_num_attention_heads (`int`, *optional*, defaults to 4) : Number of attention heads in the decoder of CLIPSeg.

decoder_attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

decoder_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.

decoder_intermediate_size (`int`, *optional*, defaults to 2048) : Dimensionality of the "intermediate" (i.e., feed-forward) layers in the Transformer decoder.

conditional_layer (`int`, *optional*, defaults to 0) : The layer to use of the Transformer encoder whose activations will be combined with the condition embeddings using FiLM (Feature-wise Linear Modulation). If 0, the last layer is used.

use_complex_transposed_convolution (`bool`, *optional*, defaults to `False`) : Whether to use a more complex transposed convolution in the decoder, enabling more fine-grained segmentation.

kwargs (*optional*) : Dictionary of keyword arguments.

## CLIPSegTextConfig[[transformers.CLIPSegTextConfig]]

#### transformers.CLIPSegTextConfig[[transformers.CLIPSegTextConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/configuration_clipseg.py#L24)

This is the configuration class to store the configuration of a [CLIPSegModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegModel). It is used to instantiate an
CLIPSeg model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the CLIPSeg
[CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import CLIPSegTextConfig, CLIPSegTextModel

>>> # Initializing a CLIPSegTextConfig with CIDAS/clipseg-rd64 style configuration
>>> configuration = CLIPSegTextConfig()

>>> # Initializing a CLIPSegTextModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
>>> model = CLIPSegTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 49408) : Vocabulary size of the CLIPSeg text model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [CLIPSegModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegModel).

hidden_size (`int`, *optional*, defaults to 512) : Dimensionality of the encoder layers and the pooler layer.

intermediate_size (`int`, *optional*, defaults to 2048) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 8) : Number of attention heads for each attention layer in the Transformer encoder.

max_position_embeddings (`int`, *optional*, defaults to 77) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

initializer_factor (`float`, *optional*, defaults to 1.0) : A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

pad_token_id (`int`, *optional*, defaults to 1) : Padding token id.

bos_token_id (`int`, *optional*, defaults to 49406) : Beginning of stream token id.

eos_token_id (`int`, *optional*, defaults to 49407) : End of stream token id.

## CLIPSegVisionConfig[[transformers.CLIPSegVisionConfig]]

#### transformers.CLIPSegVisionConfig[[transformers.CLIPSegVisionConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/configuration_clipseg.py#L119)

This is the configuration class to store the configuration of a [CLIPSegModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegModel). It is used to instantiate an
CLIPSeg model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the CLIPSeg
[CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import CLIPSegVisionConfig, CLIPSegVisionModel

>>> # Initializing a CLIPSegVisionConfig with CIDAS/clipseg-rd64 style configuration
>>> configuration = CLIPSegVisionConfig()

>>> # Initializing a CLIPSegVisionModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
>>> model = CLIPSegVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

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

## CLIPSegProcessor[[transformers.CLIPSegProcessor]]

#### transformers.CLIPSegProcessor[[transformers.CLIPSegProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/processing_clipseg.py#L23)

Constructs a CLIPSeg processor which wraps a CLIPSeg image processor and a CLIP tokenizer into a single processor.

[CLIPSegProcessor](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegProcessor) offers all the functionalities of [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) and [CLIPTokenizerFast](/docs/transformers/main/en/model_doc/clip#transformers.CLIPTokenizer). See the
`__call__()` and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

**Parameters:**

image_processor ([ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor), *optional*) : The image processor is a required input.

tokenizer ([CLIPTokenizerFast](/docs/transformers/main/en/model_doc/clip#transformers.CLIPTokenizer), *optional*) : The tokenizer is a required input.

## CLIPSegModel[[transformers.CLIPSegModel]]

#### transformers.CLIPSegModel[[transformers.CLIPSegModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py#L812)

The bare Clipseg Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.CLIPSegModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py#L926[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "return_loss", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = True"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPSegProcessor](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegProcessor) uses
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **return_loss** (`bool`, *optional*) --
  Whether or not to return the contrastive loss.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **interpolate_pos_encoding** (`bool`, defaults to `True`) --
  Whether to interpolate the pre-trained position encodings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.clipseg.modeling_clipseg.CLIPSegOutput` or `tuple(torch.FloatTensor)`A `transformers.models.clipseg.modeling_clipseg.CLIPSegOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [CLIPSegTextModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The image embeddings obtained by applying the projection layer to the pooled output of [CLIPSegVisionModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [CLIPSegTextModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [CLIPSegVisionModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionModel).
The [CLIPSegModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> import torch
>>> from transformers import AutoProcessor, CLIPSegModel
>>> from transformers.image_utils import load_image

>>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = load_image(url)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
... )

>>> with torch.inference_mode():
...     outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

**Parameters:**

config ([CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.clipseg.modeling_clipseg.CLIPSegOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.clipseg.modeling_clipseg.CLIPSegOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [CLIPSegTextModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The image embeddings obtained by applying the projection layer to the pooled output of [CLIPSegVisionModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [CLIPSegTextModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [CLIPSegVisionModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionModel).
#### get_text_features[[transformers.CLIPSegModel.get_text_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py#L851)

Examples:

```python
>>> import torch
>>> from transformers import AutoTokenizer, CLIPSegModel

>>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> with torch.inference_mode():
...     text_features = model.get_text_features(**inputs)
```

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

**Returns:**

`text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The text embeddings obtained by
applying the projection layer to the pooled output of [CLIPSegTextModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextModel).
#### get_image_features[[transformers.CLIPSegModel.get_image_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py#L887)

Examples:

```python
>>> import torch
>>> from transformers import AutoProcessor, CLIPSegModel
>>> from transformers.image_utils import load_image

>>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = load_image(url)

>>> inputs = processor(images=image, return_tensors="pt")

>>> with torch.inference_mode():
...     image_features = model.get_image_features(**inputs)
```

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPSegProcessor](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegProcessor) uses [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).

interpolate_pos_encoding (`bool`, defaults to `True`) : Whether to interpolate the pre-trained position encodings.

**Returns:**

`image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The image embeddings obtained by
applying the projection layer to the pooled output of [CLIPSegVisionModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionModel).

## CLIPSegTextModel[[transformers.CLIPSegTextModel]]

#### transformers.CLIPSegTextModel[[transformers.CLIPSegTextModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py#L652)

forwardtransformers.CLIPSegTextModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py#L670[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig)) and inputs.

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
The [CLIPSegTextModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoTokenizer, CLIPSegTextModel

>>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegTextModel.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
```

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig)) and inputs.

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

## CLIPSegVisionModel[[transformers.CLIPSegVisionModel]]

#### transformers.CLIPSegVisionModel[[transformers.CLIPSegVisionModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py#L758)

forwardtransformers.CLIPSegVisionModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py#L772[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": typing.Optional[bool] = True"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPSegProcessor](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegProcessor) uses
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **interpolate_pos_encoding** (`bool`, *optional*, defaults to `True`) --
  Whether to interpolate the pre-trained position encodings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig)) and inputs.

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
The [CLIPSegVisionModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, CLIPSegVisionModel

>>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled CLS states
```

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPSegProcessor](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegProcessor) uses [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

interpolate_pos_encoding (`bool`, *optional*, defaults to `True`) : Whether to interpolate the pre-trained position encodings.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig)) and inputs.

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

## CLIPSegForImageSegmentation[[transformers.CLIPSegForImageSegmentation]]

#### transformers.CLIPSegForImageSegmentation[[transformers.CLIPSegForImageSegmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py#L1187)

CLIPSeg model with a Transformer-based decoder on top for zero-shot and one-shot image segmentation.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.CLIPSegForImageSegmentation.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py#L1232[{"name": "input_ids", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "conditional_pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "conditional_embeddings", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = True"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([CLIPSegProcessor](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegProcessor) uses
  [ViTImageProcessor](/docs/transformers/main/en/model_doc/vit#transformers.ViTImageProcessor) for processing images).
- **conditional_pixel_values** (`torch.FloatTensor`, *optional*) --
  The pixel values of the conditional images.
- **conditional_embeddings** (`torch.FloatTensor` of shape `(batch_size, config.projection_dim)`, *optional*) --
  The conditional embeddings for the query images. If provided, the model will use this instead of computing
  the embeddings from the conditional_pixel_values.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **interpolate_pos_encoding** (`bool`, defaults to `True`) --
  Whether to interpolate the pre-trained position encodings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.clipseg.modeling_clipseg.CLIPSegOutput` or `tuple(torch.FloatTensor)`A `transformers.models.clipseg.modeling_clipseg.CLIPSegOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [CLIPSegTextModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The image embeddings obtained by applying the projection layer to the pooled output of [CLIPSegVisionModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [CLIPSegTextModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [CLIPSegVisionModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionModel).
The [CLIPSegForImageSegmentation](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegForImageSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> import torch
>>> from transformers import AutoProcessor, CLIPSegForImageSegmentation
>>> from transformers.image_utils import load_image

>>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
>>> model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = load_image(url)

>>> texts = ["a cat", "a remote", "a blanket"]
>>> inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

>>> with torch.inference_mode():
...     outputs = model(**inputs)

>>> logits = outputs.logits
>>> print(logits.shape)
torch.Size([3, 352, 352])
```

**Parameters:**

config ([CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.clipseg.modeling_clipseg.CLIPSegOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.clipseg.modeling_clipseg.CLIPSegOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CLIPSegConfig](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for image-text similarity.
- **logits_per_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) -- The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) -- The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [CLIPSegTextModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextModel).
- **image_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The image embeddings obtained by applying the projection layer to the pooled output of [CLIPSegVisionModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [CLIPSegTextModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [CLIPSegVisionModel](/docs/transformers/main/en/model_doc/clipseg#transformers.CLIPSegVisionModel).
