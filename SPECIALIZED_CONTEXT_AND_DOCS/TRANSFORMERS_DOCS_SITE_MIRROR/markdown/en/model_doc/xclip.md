# X-CLIP

## Overview

The X-CLIP model was proposed in [Expanding Language-Image Pretrained Models for General Video Recognition](https://huggingface.co/papers/2208.02816) by Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, Haibin Ling.
X-CLIP is a minimal extension of [CLIP](clip) for video. The model consists of a text encoder, a cross-frame vision encoder, a multi-frame integration Transformer, and a video-specific prompt generator.

The abstract from the paper is the following:

*Contrastive language-image pretraining has shown great success in learning visual-textual joint representation from web-scale data, demonstrating remarkable "zero-shot" generalization ability for various image tasks. However, how to effectively expand such new language-image pretraining methods to video domains is still an open problem. In this work, we present a simple yet effective approach that adapts the pretrained language-image models to video recognition directly, instead of pretraining a new model from scratch. More concretely, to capture the long-range dependencies of frames along the temporal dimension, we propose a cross-frame attention mechanism that explicitly exchanges information across frames. Such module is lightweight and can be plugged into pretrained language-image models seamlessly. Moreover, we propose a video-specific prompting scheme, which leverages video content information for generating discriminative textual prompts. Extensive experiments demonstrate that our approach is effective and can be generalized to different video recognition scenarios. In particular, under fully-supervised settings, our approach achieves a top-1 accuracy of 87.1% on Kinectics-400, while using 12 times fewer FLOPs compared with Swin-L and ViViT-H. In zero-shot experiments, our approach surpasses the current state-of-the-art methods by +7.6% and +14.9% in terms of top-1 accuracy under two popular protocols. In few-shot scenarios, our approach outperforms previous best methods by +32.1% and +23.1% when the labeled data is extremely limited.*

Tips:

- Usage of X-CLIP is identical to [CLIP](clip).

 X-CLIP architecture. Taken from the original paper. 

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/microsoft/VideoX/tree/master/X-CLIP).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with X-CLIP.

- Demo notebooks for X-CLIP can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/X-CLIP).

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## XCLIPProcessor[[transformers.XCLIPProcessor]]

#### transformers.XCLIPProcessor[[transformers.XCLIPProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/processing_x_clip.py#L22)

Constructs an X-CLIP processor which wraps a VideoMAE image processor and a CLIP tokenizer into a single processor.

[XCLIPProcessor](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPProcessor) offers all the functionalities of [CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor) and [CLIPTokenizerFast](/docs/transformers/main/en/model_doc/clip#transformers.CLIPTokenizer). See the
[__call__()](/docs/transformers/main/en/model_doc/bros#transformers.BrosProcessor.__call__) and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

**Parameters:**

image_processor ([CLIPImageProcessor](/docs/transformers/main/en/model_doc/clip#transformers.CLIPImageProcessor), *optional*) : The image processor is a required input.

tokenizer ([CLIPTokenizerFast](/docs/transformers/main/en/model_doc/clip#transformers.CLIPTokenizer), *optional*) : The tokenizer is a required input.

## XCLIPConfig[[transformers.XCLIPConfig]]

#### transformers.XCLIPConfig[[transformers.XCLIPConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/configuration_x_clip.py#L224)

[XCLIPConfig](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPConfig) is the configuration class to store the configuration of a [XCLIPModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPModel). It is used to
instantiate X-CLIP model according to the specified arguments, defining the text model and vision model configs.
Instantiating a configuration with the defaults will yield a similar configuration to that of the X-CLIP
[microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

**Parameters:**

text_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [XCLIPTextConfig](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPTextConfig).

vision_config (`dict`, *optional*) : Dictionary of configuration options used to initialize [XCLIPVisionConfig](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPVisionConfig).

projection_dim (`int`, *optional*, defaults to 512) : Dimensionality of text and vision projection layers.

prompt_layers (`int`, *optional*, defaults to 2) : Number of layers in the video specific prompt generator.

prompt_alpha (`float`, *optional*, defaults to 0.1) : Alpha value to use in the video specific prompt generator.

prompt_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`) : The non-linear activation function (function or string) in the video specific prompt generator. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.

prompt_num_attention_heads (`int`, *optional*, defaults to 8) : Number of attention heads in the cross-attention of the video specific prompt generator.

prompt_attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout probability for the attention layers in the video specific prompt generator.

prompt_projection_dropout (`float`, *optional*, defaults to 0.0) : The dropout probability for the projection layers in the video specific prompt generator.

logit_scale_init_value (`float`, *optional*, defaults to 2.6592) : The initial value of the *logit_scale* parameter. Default is used as per the original XCLIP implementation.

kwargs (*optional*) : Dictionary of keyword arguments.

## XCLIPTextConfig[[transformers.XCLIPTextConfig]]

#### transformers.XCLIPTextConfig[[transformers.XCLIPTextConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/configuration_x_clip.py#L24)

This is the configuration class to store the configuration of a [XCLIPModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPModel). It is used to instantiate an X-CLIP
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the X-CLIP
[microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import XCLIPTextModel, XCLIPTextConfig

>>> # Initializing a XCLIPTextModel with microsoft/xclip-base-patch32 style configuration
>>> configuration = XCLIPTextConfig()

>>> # Initializing a XCLIPTextConfig from the microsoft/xclip-base-patch32 style configuration
>>> model = XCLIPTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 49408) : Vocabulary size of the X-CLIP text model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [XCLIPModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPModel).

hidden_size (`int`, *optional*, defaults to 512) : Dimensionality of the encoder layers and the pooler layer.

intermediate_size (`int`, *optional*, defaults to 2048) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 8) : Number of attention heads for each attention layer in the Transformer encoder.

max_position_embeddings (`int`, *optional*, defaults to 77) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.

layer_norm_eps (`float`, *optional*, defaults to 1e-5) : The epsilon used by the layer normalization layers.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

initializer_factor (`float`, *optional*, defaults to 1) : A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

## XCLIPVisionConfig[[transformers.XCLIPVisionConfig]]

#### transformers.XCLIPVisionConfig[[transformers.XCLIPVisionConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/configuration_x_clip.py#L114)

This is the configuration class to store the configuration of a [XCLIPModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPModel). It is used to instantiate an X-CLIP
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the X-CLIP
[microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import XCLIPVisionModel, XCLIPVisionConfig

>>> # Initializing a XCLIPVisionModel with microsoft/xclip-base-patch32 style configuration
>>> configuration = XCLIPVisionConfig()

>>> # Initializing a XCLIPVisionModel model from the microsoft/xclip-base-patch32 style configuration
>>> model = XCLIPVisionModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

mit_hidden_size (`int`, *optional*, defaults to 512) : Dimensionality of the encoder layers of the Multiframe Integration Transformer (MIT).

mit_intermediate_size (`int`, *optional*, defaults to 2048) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Multiframe Integration Transformer (MIT).

mit_num_hidden_layers (`int`, *optional*, defaults to 1) : Number of hidden layers in the Multiframe Integration Transformer (MIT).

mit_num_attention_heads (`int`, *optional*, defaults to 8) : Number of attention heads for each attention layer in the Multiframe Integration Transformer (MIT).

image_size (`int`, *optional*, defaults to 224) : The size (resolution) of each image.

patch_size (`int`, *optional*, defaults to 32) : The size (resolution) of each patch.

hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"`, `"gelu_new"` and `"quick_gelu"` are supported.

layer_norm_eps (`float`, *optional*, defaults to 1e-5) : The epsilon used by the layer normalization layers.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

initializer_factor (`float`, *optional*, defaults to 1) : A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

drop_path_rate (`float`, *optional*, defaults to 0.0) : Stochastic depth rate.

## XCLIPModel[[transformers.XCLIPModel]]

#### transformers.XCLIPModel[[transformers.XCLIPModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/modeling_x_clip.py#L1153)

The bare X Clip Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.XCLIPModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/modeling_x_clip.py#L1333[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "return_loss", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "interpolate_pos_encoding", "val": ": bool = False"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
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
- **interpolate_pos_encoding** (`bool`, defaults to `False`) --
  Whether to interpolate the pre-trained position encodings.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0`transformers.models.x_clip.modeling_x_clip.XCLIPOutput` or `tuple(torch.FloatTensor)`A `transformers.models.x_clip.modeling_x_clip.XCLIPOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XCLIPConfig](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for video-text similarity.
- **logits_per_video** (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`) -- The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`) -- The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [XCLIPTextModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPTextModel).
- **video_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The video embeddings obtained by applying the projection layer to the pooled output of
  [XCLIPVisionModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [XCLIPTextModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [XCLIPVisionModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPVisionModel).
- **mit_output** (`.mit_output`, defaults to `None`) -- The output of `XCLIPMultiframeIntegrationTransformer` (MIT for short).
The [XCLIPModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> import av
>>> import torch
>>> import numpy as np

>>> from transformers import AutoProcessor, AutoModel
>>> from huggingface_hub import hf_hub_download

>>> np.random.seed(0)

>>> def read_video_pyav(container, indices):
...     '''
...     Decode the video with PyAV decoder.
...     Args:
...         container (`av.container.input.InputContainer`): PyAV container.
...         indices (`list[int]`): List of frame indices to decode.
...     Returns:
...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
...     '''
...     frames = []
...     container.seek(0)
...     start_index = indices[0]
...     end_index = indices[-1]
...     for i, frame in enumerate(container.decode(video=0)):
...         if i > end_index:
...             break
...         if i >= start_index and i in indices:
...             frames.append(frame)
...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

>>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
...     '''
...     Sample a given number of frame indices from the video.
...     Args:
...         clip_len (`int`): Total number of frames to sample.
...         frame_sample_rate (`int`): Sample every n-th frame.
...         seg_len (`int`): Maximum allowed index of sample's last frame.
...     Returns:
...         indices (`list[int]`): List of sampled frame indices
...     '''
...     converted_len = int(clip_len * frame_sample_rate)
...     end_idx = np.random.randint(converted_len, seg_len)
...     start_idx = end_idx - converted_len
...     indices = np.linspace(start_idx, end_idx, num=clip_len)
...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
...     return indices

>>> # video clip consists of 300 frames (10 seconds at 30 FPS)
>>> file_path = hf_hub_download(
...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )
>>> container = av.open(file_path)

>>> # sample 8 frames
>>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
>>> video = read_video_pyav(container, indices)

>>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
>>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

>>> inputs = processor(
...     text=["playing sports", "eating spaghetti", "go shopping"],
...     videos=list(video),
...     return_tensors="pt",
...     padding=True,
... )

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
>>> probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
>>> print(probs)
tensor([[1.9496e-04, 9.9960e-01, 2.0825e-04]])
```

**Parameters:**

config ([XCLIPConfig](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.x_clip.modeling_x_clip.XCLIPOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.x_clip.modeling_x_clip.XCLIPOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XCLIPConfig](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) -- Contrastive loss for video-text similarity.
- **logits_per_video** (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`) -- The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
  similarity scores.
- **logits_per_text** (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`) -- The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
  similarity scores.
- **text_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The text embeddings obtained by applying the projection layer to the pooled output of [XCLIPTextModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPTextModel).
- **video_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) -- The video embeddings obtained by applying the projection layer to the pooled output of
  [XCLIPVisionModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPVisionModel).
- **text_model_output** (`.text_model_output`, defaults to `None`) -- The output of the [XCLIPTextModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPTextModel).
- **vision_model_output** (`.vision_model_output`, defaults to `None`) -- The output of the [XCLIPVisionModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPVisionModel).
- **mit_output** (`.mit_output`, defaults to `None`) -- The output of `XCLIPMultiframeIntegrationTransformer` (MIT for short).
#### get_text_features[[transformers.XCLIPModel.get_text_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/modeling_x_clip.py#L1203)

Examples:

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
>>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> with torch.inference_mode():
...     text_features = model.get_text_features(**inputs)
```

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

**Returns:**

`text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The text embeddings obtained by
applying the projection layer to the pooled output of [XCLIPTextModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPTextModel).
#### get_video_features[[transformers.XCLIPModel.get_video_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/modeling_x_clip.py#L1237)

Examples:

```python
>>> import av
>>> import torch
>>> import numpy as np

>>> from transformers import AutoProcessor, AutoModel
>>> from huggingface_hub import hf_hub_download

>>> np.random.seed(0)

>>> def read_video_pyav(container, indices):
...     '''
...     Decode the video with PyAV decoder.
...     Args:
...         container (`av.container.input.InputContainer`): PyAV container.
...         indices (`list[int]`): List of frame indices to decode.
...     Returns:
...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
...     '''
...     frames = []
...     container.seek(0)
...     start_index = indices[0]
...     end_index = indices[-1]
...     for i, frame in enumerate(container.decode(video=0)):
...         if i > end_index:
...             break
...         if i >= start_index and i in indices:
...             frames.append(frame)
...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

>>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
...     '''
...     Sample a given number of frame indices from the video.
...     Args:
...         clip_len (`int`): Total number of frames to sample.
...         frame_sample_rate (`int`): Sample every n-th frame.
...         seg_len (`int`): Maximum allowed index of sample's last frame.
...     Returns:
...         indices (`list[int]`): List of sampled frame indices
...     '''
...     converted_len = int(clip_len * frame_sample_rate)
...     end_idx = np.random.randint(converted_len, seg_len)
...     start_idx = end_idx - converted_len
...     indices = np.linspace(start_idx, end_idx, num=clip_len)
...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
...     return indices

>>> # video clip consists of 300 frames (10 seconds at 30 FPS)
>>> file_path = hf_hub_download(
...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )
>>> container = av.open(file_path)

>>> # sample 8 frames
>>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
>>> video = read_video_pyav(container, indices)

>>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
>>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

>>> inputs = processor(videos=list(video), return_tensors="pt")

>>> video_features = model.get_video_features(**inputs)
```

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

**Returns:**

`video_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)`

The video embeddings obtained by
applying the projection layer to the pooled output of [XCLIPVisionModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPVisionModel) and
`XCLIPMultiframeIntegrationTransformer`.

## XCLIPTextModel[[transformers.XCLIPTextModel]]

#### transformers.XCLIPTextModel[[transformers.XCLIPTextModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/modeling_x_clip.py#L715)

forwardtransformers.XCLIPTextModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/modeling_x_clip.py#L731[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
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
elements depending on the configuration ([XCLIPConfig](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPConfig)) and inputs.

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
The [XCLIPTextModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoTokenizer, XCLIPTextModel

>>> model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

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
elements depending on the configuration ([XCLIPConfig](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPConfig)) and inputs.

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

## XCLIPVisionModel[[transformers.XCLIPVisionModel]]

#### transformers.XCLIPVisionModel[[transformers.XCLIPVisionModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/modeling_x_clip.py#L910)

forwardtransformers.XCLIPVisionModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/x_clip/modeling_x_clip.py#L924[{"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses
  `image_processor_class` for processing images).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XCLIPConfig](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPConfig)) and inputs.

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
The [XCLIPVisionModel](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> import av
>>> import torch
>>> import numpy as np

>>> from transformers import AutoProcessor, XCLIPVisionModel
>>> from huggingface_hub import hf_hub_download

>>> np.random.seed(0)

>>> def read_video_pyav(container, indices):
...     '''
...     Decode the video with PyAV decoder.
...     Args:
...         container (`av.container.input.InputContainer`): PyAV container.
...         indices (`list[int]`): List of frame indices to decode.
...     Returns:
...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
...     '''
...     frames = []
...     container.seek(0)
...     start_index = indices[0]
...     end_index = indices[-1]
...     for i, frame in enumerate(container.decode(video=0)):
...         if i > end_index:
...             break
...         if i >= start_index and i in indices:
...             frames.append(frame)
...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

>>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
...     '''
...     Sample a given number of frame indices from the video.
...     Args:
...         clip_len (`int`): Total number of frames to sample.
...         frame_sample_rate (`int`): Sample every n-th frame.
...         seg_len (`int`): Maximum allowed index of sample's last frame.
...     Returns:
...         indices (`list[int]`): List of sampled frame indices
...     '''
...     converted_len = int(clip_len * frame_sample_rate)
...     end_idx = np.random.randint(converted_len, seg_len)
...     start_idx = end_idx - converted_len
...     indices = np.linspace(start_idx, end_idx, num=clip_len)
...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
...     return indices

>>> # video clip consists of 300 frames (10 seconds at 30 FPS)
>>> file_path = hf_hub_download(
...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )
>>> container = av.open(file_path)

>>> # sample 16 frames
>>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
>>> video = read_video_pyav(container, indices)

>>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
>>> model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")

>>> pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values

>>> batch_size, num_frames, num_channels, height, width = pixel_values.shape
>>> pixel_values = pixel_values.reshape(-1, num_channels, height, width)

>>> outputs = model(pixel_values)
>>> last_hidden_state = outputs.last_hidden_state
```

**Parameters:**

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPooling](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([XCLIPConfig](/docs/transformers/main/en/model_doc/xclip#transformers.XCLIPConfig)) and inputs.

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
