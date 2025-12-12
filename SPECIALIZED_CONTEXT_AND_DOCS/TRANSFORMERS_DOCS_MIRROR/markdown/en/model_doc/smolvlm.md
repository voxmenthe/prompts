*This model was released on 2025-02-20 and added to Hugging Face Transformers on 2025-02-20.*

# SmolVLM

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

[SmolVLM2](https://huggingface.co/papers/2504.05299) ([blog post](https://huggingface.co/blog/smolvlm2)) is an adaptation of the Idefics3 model with two main differences:

* It uses SmolLM2 for the text model.
* It supports multi-image and video inputs

## Usage tips

Input images are processed either by upsampling (if resizing is enabled) or at their original resolution. The resizing behavior depends on two parameters: do\_resize and size.

Videos should not be upsampled.

If `do_resize` is set to `True`, the model resizes images so that the longest edge is 4*512 pixels by default.
The default resizing behavior can be customized by passing a dictionary to the `size` parameter. For example, `{“longest\_edge”: 4* 512}` is the default, but you can change it to a different value if needed.

Here’s how to control resizing and set a custom size:


```
image_processor = SmolVLMImageProcessor(do_resize=True, size={"longest_edge": 2 * 512}, max_image_size=512)
```

Additionally, the `max_image_size` parameter, which controls the size of each square patch the image is decomposed into, is set to 512 by default but can be adjusted as needed. After resizing (if applicable), the image processor decomposes the images into square patches based on the `max_image_size` parameter.

This model was contributed by [orrzohar](https://huggingface.co/orrzohar).

## Usage example

### Single Media inference

The model can accept both images and videos as input, but you should use only one of the modalities at a time. Here’s an example code for that.


```
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    dtype=torch.bfloat16,
    device_map="auto"
)

conversation = [
    {
        "role": "user",
        "content":[
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

output_ids = model.generate(**inputs, max_new_tokens=128)
generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
print(generated_texts)


# Video
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
```

### Batch Mixed Media Inference

The model can batch inputs composed of several images/videos and text. Here is an example.


```
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    dtype=torch.bfloat16,
    device_map="auto"
)

# Conversation for the first image
conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# Conversation with two images
conversation2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "text", "text": "What is written in the pictures?"}
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {"role": "user","content": "who are you?"}
]


conversations = [conversation1, conversation2, conversation3]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
```

## SmolVLMConfig

### class transformers.SmolVLMConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/configuration_smolvlm.py#L115)

( use\_cache = True image\_token\_id = 128257 tie\_word\_embeddings = False vision\_config = None text\_config = None scale\_factor = 2 pad\_token\_id = 128002 \*\*kwargs  )

Parameters

* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should cache the key/value pairs of the attention mechanism. Only
  relevant if `config.is_decoder=True`.
* **image\_token\_id** (`int`, *optional*, defaults to 128257) —
  The id of the “image” token.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether or not to tie the word embeddings with the token embeddings.
* **vision\_config** (`IdeficsVisionConfig` or `dict`, *optional*, defaults to `IdeficsVisionConfig`) —
  Custom vision config or dict for the vision tower
* **text\_config** (`PretrainedConfig` or `dict`, *optional*, defaults to `LlamaConfig`) —
  Custom text config or dict for the text model
* **scale\_factor** (`int`, *optional*, defaults to 2) —
  The scale factor for the image encoder.
* **pad\_token\_id** (`int`, *optional*, defaults to 128002) —
  The id of the padding token.

This is the configuration class to store the configuration of a [SmolVLMModel](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMModel). It is used to instantiate a
SmolVLM model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the model of the SmolVLM
[HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import SmolVLMModel, SmolVLMConfig
>>> # Initializing configuration
>>> configuration = SmolVLMConfig()
>>> # Initializing a model from the configuration
>>> model = SmolVLMModel(configuration)
>>> # Accessing the model configuration
>>> configuration = model.config
```

## SmolVLMVisionConfig

### class transformers.SmolVLMVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/configuration_smolvlm.py#L30)

( hidden\_size = 1152 intermediate\_size = 3072 num\_hidden\_layers = 12 num\_attention\_heads = 16 num\_channels = 3 image\_size = 224 patch\_size = 32 hidden\_act = 'gelu\_pytorch\_tanh' layer\_norm\_eps = 1e-06 attention\_dropout = 0.0 initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 1152) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  Number of channels in the input images.
* **image\_size** (`int`, *optional*, defaults to 224) —
  The size (resolution) of each image.
* **patch\_size** (`int`, *optional*, defaults to 32) —
  The size (resolution) of each patch.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a `SmolVLMVisionModel`. It is used to instantiate a
SmolVLM vision encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the SigLIP checkpoint
[google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) used in SmolVLM
[HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionTransformer
>>> from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig

>>> # Initializing a SmolVLMVisionConfig with google/siglip-so400m-patch14-384 style configuration
>>> configuration = SmolVLMVisionConfig()

>>> # Initializing a SmolVLMVisionTransformer (with random weights) from the google/siglip-so400m-patch14-384 style configuration
>>> model = SmolVLMVisionTransformer(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Idefics3VisionTransformer

### class transformers.SmolVLMVisionTransformer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/modeling_smolvlm.py#L400)

( config: SmolVLMVisionConfig  )

Parameters

* **config** ([SmolVLMVisionConfig](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMVisionConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The SmolVLM Vision Transformer Model outputting raw image embedding.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

## SmolVLMModel

### class transformers.SmolVLMModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/modeling_smolvlm.py#L552)

( config: SmolVLMConfig  )

Parameters

* **config** ([SmolVLMConfig](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SmolVLM model consisting of a SIGLIP vision encoder and Llama3 language decoder

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/modeling_smolvlm.py#L700)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None pixel\_attention\_mask: typing.Optional[torch.BoolTensor] = None image\_hidden\_states: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.smolvlm.modeling_smolvlm.SmolVLMBaseModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SmolVLMImageProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMImageProcessor). See [SmolVLMImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([SmolVLMProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMProcessor) uses
  [SmolVLMImageProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMImageProcessor) for processing images).
* **pixel\_attention\_mask** (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*) —
  Mask to avoid performing attention on padding pixel indices.
* **image\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The hidden states of the image encoder after modality projection.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.models.smolvlm.modeling_smolvlm.SmolVLMBaseModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.smolvlm.modeling_smolvlm.SmolVLMBaseModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SmolVLMConfig](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
  `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images, sequence_length, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder

Inputs fed to the model can have an arbitrary number of images. To account for this, pixel\_values fed to
the model have image padding -> (batch\_size, max\_num\_images, 3, max\_heights, max\_widths) where
max\_num\_images is the maximum number of images among the batch\_size samples in the batch.
Padding images are not needed beyond padding the pixel\_values at the entrance of the model.
For efficiency, we only pass through the vision\_model’s forward the real images by
discarding the padding images i.e. pixel\_values of size (image\_batch\_size, 3, height, width) where
image\_batch\_size would be 7 when num\_images\_per\_sample=[1, 3, 1, 2] and max\_num\_images would be 3.

## SmolVLMForConditionalGeneration

### class transformers.SmolVLMForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/modeling_smolvlm.py#L838)

( config  )

Parameters

* **config** ([SmolVLMForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMForConditionalGeneration)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The SmolVLM Model with a language modeling head. It is made up a SigLIP vision encoder, with a language modeling head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/modeling_smolvlm.py#L879)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None pixel\_attention\_mask: typing.Optional[torch.BoolTensor] = None image\_hidden\_states: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None return\_dict: typing.Optional[bool] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.smolvlm.modeling_smolvlm.SmolVLMCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SmolVLMImageProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMImageProcessor). See [SmolVLMImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([SmolVLMProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMProcessor) uses
  [SmolVLMImageProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMImageProcessor) for processing images).
* **pixel\_attention\_mask** (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*) —
  Mask to avoid performing attention on padding pixel indices.
* **image\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The hidden states of the image encoder after modality projection.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
  ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.smolvlm.modeling_smolvlm.SmolVLMCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.smolvlm.modeling_smolvlm.SmolVLMCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SmolVLMConfig](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images, sequence_length, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder

The [SmolVLMForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import requests
>>> import torch
>>> from PIL import Image
>>> from io import BytesIO

>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> from transformers.image_utils import load_image

>>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
>>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
>>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
>>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

>>> processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
>>> model = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", dtype=torch.bfloat16, device_map="auto")

>>> # Create inputs
>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "video", "path": path/to/video},
...             {"type": "text", "text": "What is happening in this video?"},
...         ]
...     }
... ]

>>> inputs = processor.apply_chat_template([messages], add_generation_prompt=True)

>>> # Generate
>>> generated_ids = model.generate(**inputs, max_new_tokens=256)
>>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

>>> print(generated_texts)
```

## SmolVLMImageProcessor

### class transformers.SmolVLMImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/image_processing_smolvlm.py#L249)

( do\_convert\_rgb: bool = True do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.LANCZOS: 1> do\_image\_splitting: bool = True max\_image\_size: typing.Optional[dict[str, int]] = None do\_rescale: bool = True rescale\_factor: float = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: bool = True \*\*kwargs  )

Parameters

* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB. This is useful if the input image is of a different format e.g. RGBA.
  Only has an effect if the input image is in the PIL format.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image. The longest edge of the image is resized to be <= `size["longest_edge"]`, with the
  shortest edge resized to keep the input aspect ratio.
* **size** (`Dict`, *optional*, defaults to `{"longest_edge" -- 4 * 364}`):
  Controls the size of the output image. This is a dictionary containing the key “longest\_edge”.
  The image will be resized such that the longest edge is <= `size["longest_edge"]` and the shortest edge is resized
  to keep the input aspect ratio.
* **resample** (`Resampling`, *optional*, defaults to `Resampling.LANCZOS`) —
  Resampling filter to use when resizing the image.
* **do\_image\_splitting** (`bool`, *optional*, defaults to `True`) —
  Whether to split the image into sub-images concatenated with the original image. They are split into patches
  such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.
* **max\_image\_size** (`Dict`, *optional*, defaults to `{"longest_edge" -- 364}`):
  Maximum resolution of the patches of images accepted by the model. This is a dictionary containing the key “longest\_edge”.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image. If set to `True`, the image is rescaled to have pixel values between 0 and 1.
* **rescale\_factor** (`float`, *optional*, defaults to `1/255`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. If set to `True`, the image is normalized to have a mean of `image_mean` and
  a standard deviation of `image_std`.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether or not to pad the images to the largest height and width in the batch and number of images per
  sample in the batch, such that the returned tensor is of shape (batch\_size, max\_num\_images, num\_channels, max\_height, max\_width).

Constructs a SmolVLM image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/image_processing_smolvlm.py#L600)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_convert\_rgb: typing.Optional[bool] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_image\_splitting: typing.Optional[bool] = None do\_rescale: typing.Optional[bool] = None max\_image\_size: typing.Optional[dict[str, int]] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_row\_col\_info: bool = False data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  A list of images to preprocess.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing. With the longest edge resized to keep the input aspect ratio.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_image\_splitting** (`bool`, *optional*, defaults to `self.do_image_splitting`) —
  Whether to split the image into sub-images concatenated with the original image. They are split into patches
  such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.
* **max\_image\_size** (`Dict`, *optional*, defaults to `self.max_image_size`) —
  Maximum resolution of the images. If the image is larger than this size, the image is split into patches.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether or not to pad the images to the largest height and width in the batch.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **return\_row\_col\_info** (`bool`, *optional*, default to `False`) —
  Whether to return the number of rows and columns of the split images. This is used for the
  `SmolVLMProcessor` to generate prompt strings based on the number of rows and columns.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess a batch of images.

## SmolVLMImageProcessorFast

### class transformers.SmolVLMImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/image_processing_smolvlm_fast.py#L183)

( \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  )

Constructs a fast Smolvlm image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/image_processing_smolvlm_fast.py#L363)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.smolvlm.image\_processing\_smolvlm\_fast.SmolVLMFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*) —
  Describes the maximum input dimensions to the model.
* **default\_to\_square** (`bool`, *optional*) —
  Whether to default to a square image when resizing, if size is an int.
* **resample** (`Union[PILImageResampling, F.InterpolationMode, NoneType]`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*) —
  Size of the output image after applying `center_crop`.
* **do\_rescale** (`bool`, *optional*) —
  Whether to rescale the image.
* **rescale\_factor** (`Union[int, float, NoneType]`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*) —
  Whether to normalize the image.
* **image\_mean** (`Union[float, list[float], NoneType]`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`Union[float, list[float], NoneType]`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_convert\_rgb** (`bool`, *optional*) —
  Whether to convert the image to RGB.
* **return\_tensors** (`Union[str, ~utils.generic.TensorType, NoneType]`) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`~image_utils.ChannelDimension`, *optional*) —
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
* **input\_data\_format** (`Union[str, ~image_utils.ChannelDimension, NoneType]`) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the images on. If unset, the device is inferred from the input images.
* **disable\_grouping** (`bool`, *optional*) —
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: <https://github.com/huggingface/transformers/pull/38157>
* **do\_pad** (`bool`, *optional*) —
  Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
  number of patches in the batch. Padding will be applied to the bottom and right with zeros.
* **do\_image\_splitting** (`bool`, *optional*, defaults to `True`) —
  Whether to split the image into sub-images concatenated with the original image. They are split into patches
  such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.
* **max\_image\_size** (`Dict`, *optional*, defaults to `{"longest_edge" -- 364}`):
  Maximum resolution of the patches of images accepted by the model. This is a dictionary containing the key “longest\_edge”.
* **return\_row\_col\_info** (`bool`, *optional*, defaults to `False`) —
  Whether to return the row and column information of the images.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## SmolVLMVideoProcessor

### class transformers.SmolVLMVideoProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/video_processing_smolvlm.py#L128)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.smolvlm.video\_processing\_smolvlm.SmolVLMVideoProcessorInitKwargs]  )

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/video_processing_utils.py#L355)

( videos: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]]] \*\*kwargs: typing\_extensions.Unpack[transformers.processing\_utils.VideosKwargs]  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the video’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `self.size`) —
  Size of the output video after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **size\_divisor** (`int`, *optional*, defaults to `self.size_divisor`) —
  The size by which to make sure both the height and width can be divided.
* **default\_to\_square** (`bool`, *optional*, defaults to `self.default_to_square`) —
  Whether to default to a square video when resizing, if size is an int.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
* **do\_center\_crop** (`bool`, *optional*, defaults to `self.do_center_crop`) —
  Whether to center crop the video to the specified `crop_size`. Can be overridden by `do_center_crop` in the
  `preprocess` method.
* **do\_pad** (`bool`, *optional*) —
  Whether to pad the video to the `(max_height, max_width)` of the videos in the batch.
* **crop\_size** (`dict[str, int]` *optional*, defaults to `self.crop_size`) —
  Size of the output video after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
  method.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `self.rescale_factor`) —
  Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Mean to use if normalizing the video. This is a float or list of floats the length of the number of
  channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
  number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.image_std`) —
  Whether to convert the video to RGB.
* **video\_metadata** (`VideoMetadata`, *optional*) —
  Metadata of the video containing information about total duration, fps and total number of frames.
* **do\_sample\_frames** (`int`, *optional*, defaults to `self.do_sample_frames`) —
  Whether to sample frames from the video before processing or to process the whole video.
* **num\_frames** (`int`, *optional*, defaults to `self.num_frames`) —
  Maximum number of frames to sample when `do_sample_frames=True`.
* **fps** (`int` or `float`, *optional*, defaults to `self.fps`) —
  Target frames to sample per second when `do_sample_frames=True`.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input video.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input video. If unset, the channel dimension format is inferred
  from the input video. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: video in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the videos on. If unset, the device is inferred from the input videos.
* **return\_metadata** (`bool`, *optional*) —
  Whether to return video metadata or not.

## SmolVLMProcessor

### class transformers.SmolVLMProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/processing_smolvlm.py#L129)

( image\_processor tokenizer video\_processor image\_seq\_len: int = 169 chat\_template: typing.Optional[str] = None \*\*kwargs  )

Parameters

* **image\_processor** (`SmolVLMImageProcessor`) —
  An instance of [SmolVLMImageProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMImageProcessor). The image processor is a required input.
* **tokenizer** (`PreTrainedTokenizerBase`) —
  An instance of [PreTrainedTokenizerBase](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase). This should correspond with the model’s text model. The tokenizer is a required input.
* **video\_processor** (`SmolVLMImageProcessor`) —
  n instance of [SmolVLMImageProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMImageProcessor). The video processor is a required input.
* **image\_seq\_len** (`int`, *optional*, defaults to 169) —
  The length of the image sequence i.e. the number of  tokens per image in the input.
  This parameter is used to build the string from the input prompt and image tokens and should match the
  value the model used. It is computed as: image\_seq\_len = int(((image\_size // patch\_size)  **2) / (scale\_factor**2))
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a SmolVLM processor which wraps a LLama tokenizer and SmolVLM image processor into a single processor.

[SmolVLMProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMProcessor) offers all the functionalities of [SmolVLMImageProcessor](/docs/transformers/v4.56.2/en/model_doc/smolvlm#transformers.SmolVLMImageProcessor) and `SmolVLMTokenizerFast`. See
the docstring of [**call**()](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smolvlm/processing_smolvlm.py#L247)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]], list[list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]]]] = None text: typing.Union[str, ForwardRef('PreTokenizedInput'), list[str], list['PreTokenizedInput']] = None audio = None videos: typing.Union[list['PIL.Image.Image'], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list['np.ndarrray']], list[list['torch.Tensor']], transformers.video\_utils.URL, list[transformers.video\_utils.URL], list[list[transformers.video\_utils.URL]], transformers.video\_utils.Path, list[transformers.video\_utils.Path], list[list[transformers.video\_utils.Path]]] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.smolvlm.processing\_smolvlm.SmolVLMProcessorKwargs]  )

Parameters

* **images** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`, *optional*) —
  The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
  tensor. If is of type `list[ImageInput]`, it’s assumed that this is for a single prompt i.e. of batch size 1.
* **text** (`Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
  Wherever an image token, `<image>` is encountered it is expanded to
  `<fake_token_around_image>` + `<row_x_col_y>` + `<image>`  *`image_seq_len`*  `.
* **videos** (`list[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`, *optional*) —
  The video or batch of videos to be prepared. Each video can be a list of PIL frames, NumPy array or PyTorch
  tensor. If is of type `list[VideoInput]`, it’s assumed that this is for a single prompt i.e. of batch size 1.
* **return\_tensors** (`Union[str, TensorType]`, *optional*) —
  If set, will return tensors of a particular framework. See [PreTrainedTokenizerFast.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for more
  information.

Processes the input prompts and returns a BatchEncoding.

Example:


```
>>> import requests
>>> from transformers import SmolVLMProcessor
>>> from transformers.image_utils import load_image

>>> processor = SmolVLMProcessor.from_pretrained("HuggingFaceM4/SmolVLM2-256M-Video-Instruct")
>>> processor.image_processor.do_image_splitting = False  # Force as False to simplify the example

>>> url1 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
>>> url2 = "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"

>>> image1, image2 = load_image(url1), load_image(url2)
>>> images = [[image1], [image2]]

>>> text = [
...     "<image>In this image, we see",
...     "bla bla bla<image>",
... ]
>>> outputs = processor(images=images, text=text, return_tensors="pt", padding=True)
>>> input_ids = outputs.input_ids
>>> input_tokens = processor.tokenizer.batch_decode(input_ids)
>>> print(input_tokens)
['<|begin_of_text|><fake_token_around_image><global-img>((<image>)*169)<fake_token_around_image> In this image, we see', '<|reserved_special_token_0|><|reserved_special_token_0|><|reserved_special_token_0|><|begin_of_text|>bla bla bla<fake_token_around_image><global-img>((<image>)*169)<fake_token_around_image>']
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/smolvlm.md)
