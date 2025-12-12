*This model was released on 2025-04-01 and added to Hugging Face Transformers on 2025-03-20.*

# ShieldGemma 2

## Overview

The ShieldGemma 2 model was proposed in a [technical report](https://huggingface.co/papers/2504.01081) by Google. ShieldGemma 2, built on [Gemma 3](https://ai.google.dev/gemma/docs/core/model_card_3), is a 4 billion (4B) parameter model that checks the safety of both synthetic and natural images against key categories to help you build robust datasets and models. With this addition to the Gemma family of models, researchers and developers can now easily minimize the risk of harmful content in their models across key areas of harm as defined below:

* No Sexually Explicit content: The image shall not contain content that depicts explicit or graphic sexual acts (e.g., pornography, erotic nudity, depictions of rape or sexual assault).
* No Dangerous Content: The image shall not contain content that facilitates or encourages activities that could cause real-world harm (e.g., building firearms and explosive devices, promotion of terrorism, instructions for suicide).
* No Violence/Gore content: The image shall not contain content that depicts shocking, sensational, or gratuitous violence (e.g., excessive blood and gore, gratuitous violence against animals, extreme injury or moment of death).

We recommend using ShieldGemma 2 as an input filter to vision language models, or as an output filter of image generation systems. To train a robust image safety model, we curated training datasets of natural and synthetic images and instruction-tuned Gemma 3 to demonstrate strong performance.

This model was contributed by [Ryan Mullins](https://huggingface.co/RyanMullins).

## Usage Example

* ShieldGemma 2 provides a Processor that accepts a list of `images` and an optional list of `policies` as input, and constructs a batch of prompts as the product of these two lists using the provided chat template.
* You can extend ShieldGemma’s built-in in policies with the `custom_policies` argument to the Processor. Using the same key as one of the built-in policies will overwrite that policy with your custom definition.
* ShieldGemma 2 does not support the image cropping capabilities used by Gemma 3.

### Classification against Built-in Policies


```
from PIL import Image
import requests
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

model_id = "google/shieldgemma-2-4b-it"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=[image], return_tensors="pt").to(model.device)

output = model(**inputs)
print(output.probabilities)
```

### Classification against Custom Policies


```
from PIL import Image
import requests
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

model_id = "google/shieldgemma-2-4b-it"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

custom_policies = {
    "key_a": "descrition_a",
    "key_b": "descrition_b",
}

inputs = processor(
    images=[image],
    custom_policies=custom_policies,
    policies=["dangerous", "key_a", "key_b"],
    return_tensors="pt",
).to(model.device)

output = model(**inputs)
print(output.probabilities)
```

## ShieldGemma2Processor

### class transformers.ShieldGemma2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/shieldgemma2/processing_shieldgemma2.py#L62)

( image\_processor tokenizer chat\_template = None image\_seq\_length = 256 policy\_definitions = None \*\*kwargs  )

## ShieldGemma2Config

### class transformers.ShieldGemma2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/shieldgemma2/configuration_shieldgemma2.py#L25)

( text\_config = None vision\_config = None mm\_tokens\_per\_image: int = 256 boi\_token\_index: int = 255999 eoi\_token\_index: int = 256000 image\_token\_index: int = 262144 initializer\_range: float = 0.02 \*\*kwargs  )

Parameters

* **text\_config** (`Union[ShieldGemma2TextConfig, dict]`, *optional*) —
  The config object of the text backbone.
* **vision\_config** (`Union[AutoConfig, dict]`, *optional*) —
  Custom vision config or dict.
* **mm\_tokens\_per\_image** (`int`, *optional*, defaults to 256) —
  The number of tokens per image embedding.
* **boi\_token\_index** (`int`, *optional*, defaults to 255999) —
  The begin-of-image token index to wrap the image prompt.
* **eoi\_token\_index** (`int`, *optional*, defaults to 256000) —
  The end-of-image token index to wrap the image prompt.
* **image\_token\_index** (`int`, *optional*, defaults to 262144) —
  The image token index to encode the image prompt.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a [ShieldGemma2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/shieldgemma2#transformers.ShieldGemma2ForImageClassification). It is used to instantiate an
ShieldGemma2ForImageClassification according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the shieldgemma-2-4b-it.

e.g. [google/gemma-3-4b](https://huggingface.co/google/gemma-3-4b)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import ShieldGemma2ForConditionalGeneration, ShieldGemma2Config, SiglipVisionConfig, ShieldGemma2TextConfig

>>> # Initializing a Siglip-like vision config
>>> vision_config = SiglipVisionConfig()

>>> # Initializing a ShieldGemma2 Text config
>>> text_config = ShieldGemma2TextConfig()

>>> # Initializing a ShieldGemma2 gemma-3-4b style configuration
>>> configuration = ShieldGemma2Config(vision_config, text_config)

>>> # Initializing a model from the gemma-3-4b style configuration
>>> model = ShieldGemma2TextConfig(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ShieldGemma2ForImageClassification

### class transformers.ShieldGemma2ForImageClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/shieldgemma2/modeling_shieldgemma2.py#L46)

( config: ShieldGemma2Config  )

Parameters

* **config** ([ShieldGemma2Config](/docs/transformers/v4.56.2/en/model_doc/shieldgemma2#transformers.ShieldGemma2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Shieldgemma2 Model with an image classification head on top e.g. for ImageNet.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/shieldgemma2/modeling_shieldgemma2.py#L82)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[list[torch.FloatTensor], transformers.cache\_utils.Cache, NoneType] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*lm\_kwargs  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Gemma3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessor). See [Gemma3ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([ShieldGemma2Processor](/docs/transformers/v4.56.2/en/model_doc/shieldgemma2#transformers.ShieldGemma2Processor) uses
  [Gemma3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessor) for processing images).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
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
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

The [ShieldGemma2ForImageClassification](/docs/transformers/v4.56.2/en/model_doc/shieldgemma2#transformers.ShieldGemma2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoImageProcessor, ShieldGemma2ForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/gemma-3-4b")
>>> model = ShieldGemma2ForImageClassification.from_pretrained("google/gemma-3-4b")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # model predicts one of the 1000 ImageNet classes
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
...
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/shieldgemma2.md)
