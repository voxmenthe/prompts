# Ovis2

## Overview

The [Ovis2](https://github.com/AIDC-AI/Ovis) is an updated version of the [Ovis](https://huggingface.co/papers/2405.20797) model developed by the AIDC-AI team at Alibaba International Digital Commerce Group.

Ovis2 is the latest advancement in multi-modal large language models (MLLMs), succeeding Ovis1.6. It retains the architectural design of the Ovis series, which focuses on aligning visual and textual embeddings, and introduces major improvements in data curation and training methods.

![](https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/XB-vgzDL6FshrSNGyZvzc.png) Ovis2 architecture.

This model was contributed by [thisisiron](https://huggingface.co/thisisiron).

## Usage example


```
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers.image_utils import load_images, load_video
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, infer_device

device = f"{infer_device()}:0"

model = AutoModelForVision2Seq.from_pretrained(
    "thisisiron/Ovis2-2B-hf",
    dtype=torch.bfloat16,
).eval().to(device)
processor = AutoProcessor.from_pretrained("thisisiron/Ovis2-2B-hf")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the image."},
        ],
    },
]
url = "http://images.cocodataset.org/val2014/COCO_val2014_000000537955.jpg"
image = Image.open(requests.get(url, stream=True).raw)
messages = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(messages)

inputs = processor(
    images=[image],
    text=messages,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(output_text)
```

## Ovis2Config

### class transformers.Ovis2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/configuration_ovis2.py#L109)

( vision\_config = None text\_config = None image\_token\_id = 151665 visual\_indicator\_token\_ids = [151666, 151667, 151668, 151669, 151670] vocab\_size = 151643 hidden\_size = 1536 \*\*kwargs  )

Parameters

* **vision\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `Ovis2VisionConfig`) —
  The config object or dictionary of the vision backbone.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`) —
  The config object or dictionary of the text backbone.
* **image\_token\_id** (`int`, *optional*, defaults to 151665) —
  The image token id to encode the image prompt.
* **visual\_indicator\_token\_ids** (`List[int]`, *optional*, defaults to `[151666, 151667, 151668, 151669, 151670]`) —
  The visual indicator token ids to encode the image prompt.
* **vocab\_size** (`int`, *optional*, defaults to 151643) —
  Vocabulary size of the text model.
* **hidden\_size** (`int`, *optional*, defaults to 1536) —
  Dimensionality of the encoder layers and the pooler layer.

This is the configuration class to store the configuration of a [Ovis2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ForConditionalGeneration). It is used to instantiate a
Ovis2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of Ovis2.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

e.g. [thisisiron/Ovis2-1B-hf](https://huggingface.co/thisisiron/Ovis2-1B-hf)


```
>>> from transformers import Ovis2ForConditionalGeneration, Ovis2Config

>>> # Initializing a Ovis2 style configuration
>>> configuration = Ovis2Config()

>>> # Initializing a model from the Ovis2-2B style configuration
>>> model = Ovis2ForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Ovis2VisionConfig

### class transformers.Ovis2VisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/configuration_ovis2.py#L20)

( hidden\_size: int = 1024 intermediate\_size: int = 2816 num\_hidden\_layers: int = 24 num\_attention\_heads: int = 8 num\_channels: int = 3 image\_size: int = 224 patch\_size: int = 14 rms\_norm\_eps: float = 1e-05 attention\_dropout: float = 0.0 qkv\_bias: bool = False mlp\_bias: bool = False hidden\_act = 'silu' vocab\_size = 16384 hidden\_stride = 1 num\_visual\_indicator\_tokens = 5 initializer\_range = 0.02 tokenize\_function = 'softmax' \*\*kwargs  )

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
  The epsilon used by the RMSNorm layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **qkv\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to add a learnable bias to the query, key, and value sequences at each attention head.
* **mlp\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to add a learnable bias to the MLP layers.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **vocab\_size** (`int`, *optional*, defaults to 16384) —
  Vocabulary size of the Vision Transformer.
* **hidden\_stride** (`int`, *optional*, defaults to 1) —
  The stride of the hidden layer in the Vision Transformer.
* **num\_visual\_indicator\_tokens** (`int`, *optional*, defaults to 5) —
  Number of visual indicator tokens.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated normal initializer for initializing all weight matrices.
* **tokenize\_function** (`str`, *optional*, defaults to `"softmax"`) —
  The function used to tokenize the visual indicator tokens.

This is the configuration class to store the configuration of a `Ovis2VisionModel`. It is used to instantiate a
Ovis2VisionModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of Ovis2.

## Ovis2Model

### class transformers.Ovis2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/modeling_ovis2.py#L561)

( config: Ovis2Config  )

Parameters

* **config** ([Ovis2Config](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Ovis2 model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/modeling_ovis2.py#L652)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs  ) → `transformers.models.ovis2.modeling_ovis2.Ovis2ModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Ovis2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ImageProcessor). See [Ovis2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Ovis2Processor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Processor) uses
  [Ovis2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
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
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.ovis2.modeling_ovis2.Ovis2ModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.ovis2.modeling_ovis2.Ovis2ModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Ovis2Config](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the model.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [Ovis2Model](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/modeling_ovis2.py#L587)

( pixel\_values: FloatTensor  ) → image\_features (`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`) —
  The tensors corresponding to the input images.
* **vision\_feature\_layer** (`Union[int, list[int]]`, *optional*) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_feature\_select\_strategy** (`str`, *optional*) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`

Returns

image\_features (`torch.Tensor`)

Image feature tensor of shape `(num_images, image_length, embed_dim)`).

Obtains image last hidden states from the vision tower and apply multimodal projection.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/modeling_ovis2.py#L628)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

## Ovis2ForConditionalGeneration

### class transformers.Ovis2ForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/modeling_ovis2.py#L732)

( config: Ovis2Config  )

Parameters

* **config** ([Ovis2Config](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Ovis2 Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/modeling_ovis2.py#L773)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[list[torch.FloatTensor]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs  ) → `transformers.models.ovis2.modeling_ovis2.Ovis2CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Ovis2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ImageProcessor). See [Ovis2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Ovis2Processor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Processor) uses
  [Ovis2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`list[torch.FloatTensor]`, *optional*) —
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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
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
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.ovis2.modeling_ovis2.Ovis2CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.ovis2.modeling_ovis2.Ovis2CausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Ovis2Config](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Config)) and inputs.

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
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size (batch\_size \* num\_patches, num\_images, sequence\_length, hidden\_size)`.
  image\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [Ovis2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Ovis2ForConditionalGeneration

>>> model = Ovis2ForConditionalGeneration.from_pretrained("thisisiron/Ovis2-2B-hf")
>>> processor = AutoProcessor.from_pretrained("thisisiron/Ovis2-2B-hf")

>>> prompt = "<|im_start|>user\n<image>\nDescribe the image.<|im_end|>\n<|im_start|>assistant\n"
>>> url = "http://images.cocodataset.org/val2014/COCO_val2014_000000537955.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, text=prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs, max_new_tokens=15)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
"user\n\nDescribe the image.\nassistant\nThe image features a brown dog standing on a wooden floor, looking up with"
```

## Ovis2ImageProcessor

### class transformers.Ovis2ImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/image_processing_ovis2.py#L178)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None crop\_to\_patches: bool = False min\_patches: int = 1 max\_patches: int = 12 resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_convert\_rgb: bool = True use\_covering\_area\_grid: bool = True \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the
  `do_resize` parameter in the `preprocess` method.
* **size** (`dict`, *optional*, defaults to `{"height" -- 384, "width": 384}`):
  Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
  method.
* **crop\_to\_patches** (`bool`, *optional*, defaults to `False`) —
  Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
  `preprocess` method.
* **min\_patches** (`int`, *optional*, defaults to 1) —
  The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
* **max\_patches** (`int`, *optional*, defaults to 12) —
  The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`) —
  Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
  overridden by the `resample` parameter in the `preprocess` method.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
  `do_rescale` parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
  overridden by the `rescale_factor` parameter in the `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
* **image\_mean** (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB.
* **use\_covering\_area\_grid** (`bool`, *optional*, defaults to `True`) —
  Whether to use the covering area grid to determine the number of patches. Only has an effect if
  `crop_to_patches` is set to `True`. Can be overridden by the `use_covering_area_grid` parameter in the
  `preprocess` method.

Constructs a Ovis2 image processor.

#### crop\_image\_to\_patches

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/image_processing_ovis2.py#L476)

( images: ndarray min\_patches: int max\_patches: int use\_covering\_area\_grid: bool = True patch\_size: typing.Union[tuple, int, dict, NoneType] = None data\_format: ChannelDimension = None covering\_threshold: float = 0.9  ) → List`PIL.Image.Image` or List[np.ndarray]

Parameters

* **images** (`np.ndarray`) —
  The image to be cropped.
* **min\_patches** (`int`) —
  The minimum number of patches to be extracted from the image.
* **max\_patches** (`int`) —
  The maximum number of patches to be extracted from the image.
* **use\_covering\_area\_grid** (`bool`, *optional*, defaults to `True`) —
  Whether to use the covering area grid to determine the number of patches.
* **patch\_size** (`int`, `Tuple[int, int]`, `dict`, *optional*) —
  The size of the output patches.
* **data\_format** (`ChannelDimension`, *optional*) —
  The format of the image data. If `None`, the format is inferred from the input image.
* **covering\_threshold** (`float`, *optional*, defaults to `0.9`) —
  The threshold for the covering area grid. If the covering area is less than this value, the grid is
  considered invalid.

Returns

List`PIL.Image.Image` or List[np.ndarray]

The list of cropped images.

Crop the image to patches and return a list of cropped images.
The number of patches and their grid arrangement are determined by the original image size,
the target patch size and the minimum and maximum number of patches.
The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/image_processing_ovis2.py#L310)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None crop\_to\_patches: typing.Optional[bool] = None min\_patches: typing.Optional[int] = None max\_patches: typing.Optional[int] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None do\_convert\_rgb: typing.Optional[bool] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None use\_covering\_area\_grid: bool = True  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`Dict[str, int]`, *optional*, defaults to `self.size`) —
  Controls the size of the image after `resize`. The shortest edge of the image is resized to
  `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
  is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
  edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
* **crop\_to\_patches** (`bool`, *optional*, defaults to `self.crop_to_patches`) —
  Whether to crop the image to patches.
* **min\_patches** (`int`, *optional*, defaults to `self.min_patches`) —
  The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`.
* **max\_patches** (`int`, *optional*, defaults to `self.max_patches`) —
  The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image values between [0 - 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `List[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to normalize the image by if `do_normalize` is set to `True`.
* **image\_std** (`float` or `List[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
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
* **use\_covering\_area\_grid** (`bool`, *optional*, defaults to `True`) —
  Whether to use the covering area grid to determine the number of patches. Only has an effect if
  `crop_to_patches` is set to `True`.

Preprocess an image or batch of images.

#### resize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/image_processing_ovis2.py#L262)

( image: ndarray size: dict resample: Resampling = <Resampling.BICUBIC: 3> data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None \*\*kwargs  ) → `np.ndarray`

Parameters

* **image** (`np.ndarray`) —
  Image to resize.
* **size** (`Dict[str, int]`) —
  Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
* **resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`) —
  `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
* **data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the output image. If unset, the channel dimension format of the input
  image is used. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Returns

`np.ndarray`

The resized image.

Resize an image to `(size["height"], size["width"])`.

## Ovis2ImageProcessorFast

### class transformers.Ovis2ImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/image_processing_ovis2_fast.py#L78)

( \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  )

Constructs a fast Ovis2 image processor.

#### crop\_image\_to\_patches

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/image_processing_ovis2_fast.py#L98)

( images: torch.Tensor min\_patches: int max\_patches: int use\_covering\_area\_grid: bool = True covering\_threshold: float = 0.9 patch\_size: typing.Union[tuple, int, dict, NoneType] = None interpolation: typing.Optional[ForwardRef('F.InterpolationMode')] = None  ) → List`PIL.Image.Image` or List[np.ndarray]

Parameters

* **images** (`torch.Tensor`) —
  The images to be cropped.
* **min\_patches** (`int`) —
  The minimum number of patches to be extracted from the image.
* **max\_patches** (`int`) —
  The maximum number of patches to be extracted from the image.
* **use\_covering\_area\_grid** (`bool`, *optional*, defaults to `True`) —
  Whether to use the original OVIS2 approach: compute the minimal number of tiles that cover at least 90%
  of the image area. If `False`, the closest aspect ratio to the target is used.
* **covering\_threshold** (`float`, *optional*, defaults to `0.9`) —
  The threshold for the covering area. Only has an effect if `use_covering_area_grid` is set to `True`.
* **patch\_size** (`int`, `Tuple[int, int]`, `dict`, *optional*) —
  The size of the output patches.
  The format of the image data. If `None`, the format is inferred from the input image.
* **interpolation** (`InterpolationMode`) —
  Resampling filter to use if resizing the image.

Returns

List`PIL.Image.Image` or List[np.ndarray]

The list of cropped images.

Crop the images to patches and return a list of cropped images.
The number of patches and their grid arrangement are determined by the original image size,
the target patch size and the minimum and maximum number of patches.
The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/image_processing_ovis2_fast.py#L94)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.ovis2.image\_processing\_ovis2\_fast.Ovis2ImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **crop\_to\_patches** (`bool`, *optional*, defaults to `False`) —
  Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
  `preprocess` method.
* **min\_patches** (`int`, *optional*, defaults to 1) —
  The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
* **max\_patches** (`int`, *optional*, defaults to 12) —
  The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
  set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.
* **use\_covering\_area\_grid** (`bool`, *optional*, defaults to `True`) —
  Whether to use the covering area grid to determine the number of patches. Only has an effect if
  `crop_to_patches` is set to `True`. Can be overridden by the `use_covering_area_grid` parameter in the
  `preprocess` method.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## Ovis2Processor

### class transformers.Ovis2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/processing_ovis2.py#L37)

( image\_processor = None tokenizer = None chat\_template = None image\_token = '<image>' image\_seq\_length = 256 \*\*kwargs  )

Parameters

* **image\_processor** ([Ovis2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast), *optional*) —
  The tokenizer is a required input.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.
* **image\_token** (`str`, *optional*, defaults to `"<image>"`) —
  Special token used to denote image location.
* **image\_seq\_length** (`int`, *optional*, defaults to 256) —
  The number of image tokens to be used for each image in the input.

Constructs a Ovis2 processor which wraps Ovis2 image processor and a Qwen2 tokenizer into a single processor.

[Ovis2Processor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Processor) offers all the functionalities of `Ovis2VideoProcessor`, [Ovis2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2ImageProcessor) and [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/model_doc/ovis2#transformers.Ovis2Processor.decode) for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/processing_ovis2.py#L160)

( \*args \*\*kwargs  )

This method forwards all its arguments to Qwen2TokenizerFast’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ovis2/processing_ovis2.py#L167)

( \*args \*\*kwargs  )

This method forwards all its arguments to Qwen2TokenizerFast’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/ovis2.md)
