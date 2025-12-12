*This model was released on 2022-02-22 and added to Hugging Face Transformers on 2022-06-28.*

# GroupViT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The GroupViT model was proposed in [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://huggingface.co/papers/2202.11094) by Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang.
Inspired by [CLIP](clip), GroupViT is a vision-language model that can perform zero-shot semantic segmentation on any given vocabulary categories.

The abstract from the paper is the following:

*Grouping and recognition are important components of visual scene understanding, e.g., for object detection and semantic segmentation. With end-to-end deep learning systems, grouping of image regions usually happens implicitly via top-down supervision from pixel-level recognition labels. Instead, in this paper, we propose to bring back the grouping mechanism into deep networks, which allows semantic segments to emerge automatically with only text supervision. We propose a hierarchical Grouping Vision Transformer (GroupViT), which goes beyond the regular grid structure representation and learns to group image regions into progressively larger arbitrary-shaped segments. We train GroupViT jointly with a text encoder on a large-scale image-text dataset via contrastive losses. With only text supervision and without any pixel-level annotations, GroupViT learns to group together semantic regions and successfully transfers to the task of semantic segmentation in a zero-shot manner, i.e., without any further fine-tuning. It achieves a zero-shot accuracy of 52.3% mIoU on the PASCAL VOC 2012 and 22.4% mIoU on PASCAL Context datasets, and performs competitively to state-of-the-art transfer-learning methods requiring greater levels of supervision.*

This model was contributed by [xvjiarui](https://huggingface.co/xvjiarui). The original code can be found [here](https://github.com/NVlabs/GroupViT).

## Usage tips

* You may specify `output_segmentation=True` in the forward of `GroupViTModel` to get the segmentation logits of input texts.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with GroupViT.

* The quickest way to get started with GroupViT is by checking the [example notebooks](https://github.com/xvjiarui/GroupViT/blob/main/demo/GroupViT_hf_inference_notebook.ipynb) (which showcase zero-shot segmentation inference).
* One can also check out the [HuggingFace Spaces demo](https://huggingface.co/spaces/xvjiarui/GroupViT) to play with GroupViT.

## GroupViTConfig

### class transformers.GroupViTConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/configuration_groupvit.py#L234)

( text\_config = None vision\_config = None projection\_dim = 256 projection\_intermediate\_dim = 4096 logit\_scale\_init\_value = 2.6592 \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [GroupViTTextConfig](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTTextConfig).
* **vision\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [GroupViTVisionConfig](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTVisionConfig).
* **projection\_dim** (`int`, *optional*, defaults to 256) —
  Dimensionality of text and vision projection layers.
* **projection\_intermediate\_dim** (`int`, *optional*, defaults to 4096) —
  Dimensionality of intermediate layer of text and vision projection layers.
* **logit\_scale\_init\_value** (`float`, *optional*, defaults to 2.6592) —
  The initial value of the *logit\_scale* parameter. Default is used as per the original GroupViT
  implementation.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[GroupViTConfig](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTConfig) is the configuration class to store the configuration of a [GroupViTModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTModel). It is used to
instantiate a GroupViT model according to the specified arguments, defining the text model and vision model
configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the GroupViT
[nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

#### from\_text\_vision\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/configuration_utils.py#L1254)

( text\_config vision\_config \*\*kwargs  ) → `PreTrainedConfig`

Returns

`PreTrainedConfig`

An instance of a configuration object

Instantiate a model config (or a derived class) from text model configuration and vision model
configuration.

## GroupViTTextConfig

### class transformers.GroupViTTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/configuration_groupvit.py#L34)

( vocab\_size = 49408 hidden\_size = 256 intermediate\_size = 1024 num\_hidden\_layers = 12 num\_attention\_heads = 4 max\_position\_embeddings = 77 hidden\_act = 'quick\_gelu' layer\_norm\_eps = 1e-05 dropout = 0.0 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 pad\_token\_id = 1 bos\_token\_id = 49406 eos\_token\_id = 49407 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 49408) —
  Vocabulary size of the GroupViT text model. Defines the number of different tokens that can be represented
  by the `inputs_ids` passed when calling [GroupViTModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTModel).
* **hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 4) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 77) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"quick_gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-5) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).

This is the configuration class to store the configuration of a [GroupViTTextModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTTextModel). It is used to instantiate an
GroupViT model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the GroupViT
[nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import GroupViTTextConfig, GroupViTTextModel

>>> # Initializing a GroupViTTextModel with nvidia/groupvit-gcc-yfcc style configuration
>>> configuration = GroupViTTextConfig()

>>> model = GroupViTTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## GroupViTVisionConfig

### class transformers.GroupViTVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/configuration_groupvit.py#L126)

( hidden\_size = 384 intermediate\_size = 1536 depths = [6, 3, 3] num\_hidden\_layers = 12 num\_group\_tokens = [64, 8, 0] num\_output\_groups = [64, 8, 8] num\_attention\_heads = 6 image\_size = 224 patch\_size = 16 num\_channels = 3 hidden\_act = 'gelu' layer\_norm\_eps = 1e-05 dropout = 0.0 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 assign\_eps = 1.0 assign\_mlp\_ratio = [0.5, 4] \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 384) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 1536) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **depths** (`list[int]`, *optional*, defaults to [6, 3, 3]) —
  The number of layers in each encoder block.
* **num\_group\_tokens** (`list[int]`, *optional*, defaults to [64, 8, 0]) —
  The number of group tokens for each stage.
* **num\_output\_groups** (`list[int]`, *optional*, defaults to [64, 8, 8]) —
  The number of output groups for each stage, 0 means no group.
* **num\_attention\_heads** (`int`, *optional*, defaults to 6) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 16) —
  The size (resolution) of each patch.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-5) —
  The epsilon used by the layer normalization layers.
* **dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).

This is the configuration class to store the configuration of a [GroupViTVisionModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTVisionModel). It is used to instantiate
an GroupViT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the GroupViT
[nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import GroupViTVisionConfig, GroupViTVisionModel

>>> # Initializing a GroupViTVisionModel with nvidia/groupvit-gcc-yfcc style configuration
>>> configuration = GroupViTVisionConfig()

>>> model = GroupViTVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## GroupViTModel

### class transformers.GroupViTModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/modeling_groupvit.py#L1176)

( config: GroupViTConfig  )

Parameters

* **config** ([GroupViTConfig](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Groupvit Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/modeling_groupvit.py#L1318)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None return\_loss: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_segmentation: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.groupvit.modeling_groupvit.GroupViTModelOutput` or `tuple(torch.FloatTensor)`

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
* **output\_segmentation** (`bool`, *optional*) —
  Whether or not to return the segmentation logits.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.groupvit.modeling_groupvit.GroupViTModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.groupvit.modeling_groupvit.GroupViTModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GroupViTConfig](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
* **logits\_per\_image** (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
  similarity scores.
* **segmentation\_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`) — Classification scores for each pixel.

  The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
  to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
  original image size as post-processing. You should always check your logits shape and resize as needed.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of
  [GroupViTTextModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTTextModel).
* **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of
  [GroupViTVisionModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTVisionModel).
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [GroupViTTextModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTTextModel).
* **vision\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.vision_model_output`, defaults to `None`) — The output of the [GroupViTVisionModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTVisionModel).

The [GroupViTModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, GroupViTModel

>>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/modeling_groupvit.py#L1222)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

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
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by
applying the projection layer to the pooled output of [GroupViTTextModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTTextModel).

Examples:


```
>>> from transformers import CLIPTokenizer, GroupViTModel

>>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/modeling_groupvit.py#L1269)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

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
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by
applying the projection layer to the pooled output of [GroupViTVisionModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTVisionModel).

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, GroupViTModel

>>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```

## GroupViTTextModel

### class transformers.GroupViTTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/modeling_groupvit.py#L1023)

( config: GroupViTTextConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/modeling_groupvit.py#L1038)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

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
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GroupViTConfig](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTConfig)) and inputs.

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

The [GroupViTTextModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import CLIPTokenizer, GroupViTTextModel

>>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> model = GroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
```

## GroupViTVisionModel

### class transformers.GroupViTVisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/modeling_groupvit.py#L1126)

( config: GroupViTVisionConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/groupvit/modeling_groupvit.py#L1139)

( pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

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
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GroupViTConfig](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTConfig)) and inputs.

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

The [GroupViTVisionModel](/docs/transformers/v4.56.2/en/model_doc/groupvit#transformers.GroupViTVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, GroupViTVisionModel

>>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> model = GroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  # pooled CLS states
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/groupvit.md)
