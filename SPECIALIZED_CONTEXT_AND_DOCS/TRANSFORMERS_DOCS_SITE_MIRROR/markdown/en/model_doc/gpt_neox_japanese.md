# GPT-NeoX-Japanese

GPT-NeoX-Japanese, a Japanese language model based on [GPT-NeoX](./gpt_neox).
Japanese uses three types of characters (hiragana, katakana, kanji) and has a huge vocabulary. This model uses [BPEEncoder V2](https://github.com/tanreinama/Japanese-BPEEncoder_V2), a sub-word tokenizer to handle the different characters.

The model also removes some bias parameters for better performance.

You can find all the original GPT-NeoX-Japanese checkpoints under the [ABEJA](https://huggingface.co/abeja/models?search=gpt-neo-x) organization.

> [!TIP]
> This model was contributed by [Shinya Otani](https://github.com/SO0529), [Takayoshi Makabe](https://github.com/spider-man-tm), [Anuj Arora](https://github.com/Anuj040), and [Kyo Hattori](https://github.com/go5paopao) from [ABEJA, Inc.](https://www.abejainc.com/).
>
> Click on the GPT-NeoX-Japanese models in the right sidebar for more examples of how to apply GPT-NeoX-Japanese to different language tasks.

The example below demonstrates how to generate text with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel), and from the command line.

```python
import torch
from transformers import pipeline
pipeline = pipeline(task="text-generation", 
                    model="abeja/gpt-neox-japanese-2.7b", dtype=torch.float16, device=0)
pipeline("人とAIが協調するためには、")
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b", dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
input_ids = tokenizer("人とAIが協調するためには、", return_tensors="pt").input_ids.to(model.device)
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

```bash
echo -e "人とAIが協調するためには、" | transformers run --task text-generation --model abeja/gpt-neox-japanese-2.7b --device 0
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)
model = AutoModelForCausalLM.from_pretrained(
    "abeja/gpt-neox-japanese-2.7b",
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
input_ids = tokenizer.encode("人とAIが協調するためには、", return_tensors="pt").to(model.device)
output = model.generate(input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```py
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("abeja/gpt-neox-japanese-2.7b")
visualizer("What is shown in this image?")
```

    

## Resources

Refer to the [Training a better GPT model: Learnings from PaLM](https://medium.com/ml-abeja/training-a-better-gpt-2-93b157662ae4) blog post for more details about how ABEJA trained GPT-NeoX-Japanese.

## GPTNeoXJapaneseConfig[[transformers.GPTNeoXJapaneseConfig]]

#### transformers.GPTNeoXJapaneseConfig[[transformers.GPTNeoXJapaneseConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox_japanese/configuration_gpt_neox_japanese.py#L27)

This is the configuration class to store the configuration of a `GPTNeoXModelJapanese`. It is used to instantiate
a GPTNeoX model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the GPTNeoXJapanese
[abeja/gpt-neox-japanese-2.7b](https://huggingface.co/abeja/gpt-neox-japanese-2.7b) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information. Default configs is set as 2.7B model

```python
>>> from transformers import GPTNeoXJapaneseConfig, GPTNeoXJapaneseModel

>>> # Initializing a GPTNeoXJapanese gpt-neox-japanese-2.7b style configuration
>>> configuration = GPTNeoXJapaneseConfig()

>>> # Initializing a model (with random weights) from the gpt-neox-japanese-2.7b style configuration
>>> model = GPTNeoXJapaneseModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 32000) : Vocabulary size of the GPTNeoXJapanese model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `GPTNeoXJapanese`.

hidden_size (`int`, *optional*, defaults to 2560) : Dimension of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 32) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 32) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_multiple_size (`int`, *optional*, defaults to 4) : Dimension of the "intermediate" layer in the Transformer encoder is calculated by hidden_size * intermediate_multiple_size.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler.

max_position_embeddings (`int`, *optional*, defaults to 2048) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-5) : The epsilon used by the layer normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.

rope_parameters (`RopeParameters`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_dropout (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention.

hidden_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the hidden layer.

Example --

## GPTNeoXJapaneseTokenizer[[transformers.GPTNeoXJapaneseTokenizer]]

#### transformers.GPTNeoXJapaneseTokenizer[[transformers.GPTNeoXJapaneseTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox_japanese/tokenization_gpt_neox_japanese.py#L55)

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) and is based on Japanese special Sub-Word-Encoding that is
used in this repository (https://github.com/tanreinama/Japanese-BPEEncoder_V2). Check the repository for details.
Japanese has a relatively large vocabulary and there is no separation between words. Furthermore, the language is a
combination of hiragana, katakana, and kanji, and variants such as "1" and "①" are often used. In order to cope
with these, this tokenizer has the following features
- Subword-by-subword segmentation, which is intermediate between byte strings and morphological analysis.
- BPEs are created for each Kanji, Hiragana, and Katakana character, and there are no BPEs that cross character
  types, such as Kanji + Hiragana or Hiragana + Katakana.
- All-byte encoding that does not require .
- Independent of UTF codes such as 2-byte and 3-byte characters
- Conversion of heterographs to the same token_id
- Emoji and Emoticon are grouped into 12 types as special tags.

Example:

```python
>>> from transformers import GPTNeoXJapaneseTokenizer

>>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> # You can confirm both 慶応 and 慶應 are encoded to 17749
>>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
[30014, 26883, 26638, 27228, 25, 26650, 31732, 31679, 27809, 26638, 17749, 31592, 17749, 31593, 321, 1281]

>>> # Both 慶応 and 慶應 are decoded to 慶応
>>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
'吾輩は猫である🐯。実は慶応(慶応)大学出身'
```

convert_tokens_to_stringtransformers.GPTNeoXJapaneseTokenizer.convert_tokens_to_stringhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox_japanese/tokenization_gpt_neox_japanese.py#L161[{"name": "tokens", "val": ""}]
Converts a sequence of tokens (string) in a single string.

**Parameters:**

vocab_file (`str`) : File containing the vocabulary.

emoji_file (`str`) : File containing the emoji.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token.

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.

do_clean_text (`bool`, *optional*, defaults to `False`) : Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.

## GPTNeoXJapaneseModel[[transformers.GPTNeoXJapaneseModel]]

#### transformers.GPTNeoXJapaneseModel[[transformers.GPTNeoXJapaneseModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py#L401)

The bare Gpt Neox Japanese Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.GPTNeoXJapaneseModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py#L422[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.0[transformers.modeling_outputs.BaseModelOutputWithPast](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutputWithPast](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GPTNeoXJapaneseConfig](/docs/transformers/main/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [GPTNeoXJapaneseModel](/docs/transformers/main/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, GPTNeoXJapaneseModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> model = GPTNeoXJapaneseModel.from_pretrained("abeja/gpt-neox-japanese-2.7b")

>>> inputs = tokenizer("日本語のGPT-neoxがHugging Faceで使えます😀", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

**Parameters:**

config ([GPTNeoXJapaneseModel](/docs/transformers/main/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutputWithPast](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutputWithPast](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GPTNeoXJapaneseConfig](/docs/transformers/main/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## GPTNeoXJapaneseForCausalLM[[transformers.GPTNeoXJapaneseForCausalLM]]

#### transformers.GPTNeoXJapaneseForCausalLM[[transformers.GPTNeoXJapaneseForCausalLM]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py#L652)

GPTNeoXJapanese Model with a `language modeling` head on top for Classifier Model fine-tuning.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.GPTNeoXJapaneseForCausalLM.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py#L671[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "logits_to_keep", "val": ": typing.Union[int, torch.Tensor] = 0"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
  `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
  ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **cache_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
- **logits_to_keep** (`Union[int, torch.Tensor]`, defaults to `0`) --
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).0[transformers.modeling_outputs.CausalLMOutputWithPast](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.CausalLMOutputWithPast](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GPTNeoXJapaneseConfig](/docs/transformers/main/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [GPTNeoXJapaneseForCausalLM](/docs/transformers/main/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseConfig
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> config = GPTNeoXJapaneseConfig.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> config.is_decoder = True
>>> model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b", config=config)

>>> inputs = tokenizer("日本語のGPT-neoxがHugging Faceで使えます😀", return_tensors="pt")
>>> outputs = model(**inputs)

>>> prediction_logits = outputs.logits
```

**Parameters:**

config ([GPTNeoXJapaneseForCausalLM](/docs/transformers/main/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseForCausalLM)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.CausalLMOutputWithPast](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.CausalLMOutputWithPast](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GPTNeoXJapaneseConfig](/docs/transformers/main/en/model_doc/gpt_neox_japanese#transformers.GPTNeoXJapaneseConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
