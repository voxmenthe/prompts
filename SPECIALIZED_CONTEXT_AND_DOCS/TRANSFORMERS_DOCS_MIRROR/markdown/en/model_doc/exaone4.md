*This model was released on 2025-07-15 and added to Hugging Face Transformers on 2025-07-26.*

# EXAONE 4

## Overview

**[EXAONE 4.0](https://github.com/LG-AI-EXAONE/EXAONE-4.0)** model is the language model, which integrates a **Non-reasoning mode** and **Reasoning mode** to achieve both the excellent usability of [EXAONE 3.5](https://github.com/LG-AI-EXAONE/EXAONE-3.5) and the advanced reasoning abilities of [EXAONE Deep](https://github.com/LG-AI-EXAONE/EXAONE-Deep). To pave the way for the agentic AI era, EXAONE 4.0 incorporates essential features such as agentic tool use, and its multilingual capabilities are extended
to support Spanish in addition to English and Korean.

The EXAONE 4.0 model series consists of two sizes: a mid-size **32B** model optimized for high performance, and a small-size **1.2B** model designed for on-device applications.

In the EXAONE 4.0 architecture, we apply new architectural changes compared to previous EXAONE models as below:

1. **Hybrid Attention**: For the 32B model, we adopt hybrid attention scheme, which combines *Local attention (sliding window attention)* with *Global attention (full attention)* in a 3:1 ratio. We do not use RoPE (Rotary Positional Embedding) for global attention for better global context understanding.
2. **QK-Reorder-Norm**: We reorder the LayerNorm position from the traditional Pre-LN scheme by applying LayerNorm directly to the attention and MLP outputs, and we add RMS normalization right after the Q and K projection. It helps yield better performance on downstream tasks despite consuming more computation.

For more details, please refer to our [technical report](https://huggingface.co/papers/2507.11407), [HuggingFace paper](https://huggingface.co/papers/2507.11407), [blog](https://www.lgresearch.ai/blog/view?seq=576), and [GitHub](https://github.com/LG-AI-EXAONE/EXAONE-4.0).

All model weights including quantized versions are available at [Huggingface Collections](https://huggingface.co/collections/LGAI-EXAONE/exaone-40-686b2e0069800c835ed48375).

## Model Details

### Model Specifications

| Model Configuration | 32B | 1.2B |
| --- | --- | --- |
| d\_model | 5,120 | 2,048 |
| Number of layers | 64 | 30 |
| Normalization | QK-Reorder-LN | QK-Reorder-LN |
| Non-linearity | SwiGLU | SwiGLU |
| Feedforward dimension | 27,392 | 4,096 |
| Attention type | Hybrid (3:1 Local-Global) | Global |
| Head type | GQA | GQA |
| Number of heads | 40 | 32 |
| Number of KV heads | 8 | 8 |
| Head size | 128 | 64 |
| Max sequence length | 131,072 | 65,536 |
| RoPE theta | 1,000,000 | 1,000,000 |
| Tokenizer | BBPE | BBPE |
| Vocab size | 102,400 | 102,400 |
| Tied word embedding | False | True |
| Knowledge cut-off | Nov. 2024 | Nov. 2024 |

## Usage tips

### Non-reasoning mode

For general use, you can use the EXAONE 4.0 models with the following example:


```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-4.0-32B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# choose your prompt
prompt = "Explain how wonderful you are"
prompt = "Explica lo increíble que eres"
prompt = "너가 얼마나 대단한지 설명해 봐"

messages = [
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=False,
)
print(tokenizer.decode(output[0]))
```

### Reasoning mode

The EXAONE 4.0 models have reasoning capabilities for handling complex problems. You can activate reasoning mode by using the `enable_thinking=True` argument with the tokenizer, which opens a reasoning block that starts with `<think>` tag without closing it.


```
messages = [
    {"role": "user", "content": "Which one is bigger, 3.12 vs 3.9?"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=True,
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=True,
    temperature=0.6,
    top_p=0.95
)
print(tokenizer.decode(output[0]))
```

> [!IMPORTANT]
> The model generation with reasoning mode can be affected sensitively by sampling parameters, so please refer to the [Usage Guideline](https://github.com/LG-AI-EXAONE/EXAONE-4.0#usage-guideline) on official GitHub page for better quality.

### Agentic tool use

The EXAONE 4.0 models can be used as agents with their tool calling capabilities. You can provide tool schemas to the model for effective tool calling.


```
import random

def roll_dice(max_num: int):
    return random.randint(1, max_num)

tools = [
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "Roll a dice with the number 1 to N. User can select the number N.",
            "parameters": {
                "type": "object",
                "required": ["max_num"],
                "properties": {
                    "max_num": {
                        "type": "int",
                        "description": "Max number of the dice"
                    }
                }
            }
        }
    }
]

messages = [
    {"role": "user", "content": "Roll D6 dice twice!"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    tools=tools,
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
)
print(tokenizer.decode(output[0]))
```

## Exaone4Config

### class transformers.Exaone4Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/configuration_exaone4.py#L25)

( vocab\_size = 102400 hidden\_size = 4096 intermediate\_size = 16384 num\_hidden\_layers = 32 num\_attention\_heads = 32 num\_key\_value\_heads = 32 hidden\_act = 'silu' max\_position\_embeddings = 2048 initializer\_range = 0.02 rms\_norm\_eps = 1e-05 use\_cache = True bos\_token\_id = 0 eos\_token\_id = 2 tie\_word\_embeddings = False rope\_theta = 10000.0 rope\_scaling = None attention\_dropout = 0.0 sliding\_window = 4096 sliding\_window\_pattern = 4 layer\_types = None \*\*kwargs  )

Parameters

* **vocab\_size** (*int*, *optional*, defaults to 102400) —
  Vocabulary size of the EXAONE 4.0 model. Defines the number of different tokens that can be represented by the
  *inputs\_ids* passed when calling [*Exaone4Model*].
* **hidden\_size** (*int*, *optional*, defaults to 4096) —
  Dimension of the hidden representations.
* **intermediate\_size** (*int*, *optional*, defaults to *hidden\_size*  4\*) —
  Dimensionality of the MLP representations.
* **num\_hidden\_layers** (*int*, *optional*, defaults to 32) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (*int*, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **num\_key\_value\_heads** (*int*, *optional*) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  *num\_key\_value\_heads=num\_attention\_heads*, the model will use Multi Head Attention (MHA), if
  *num\_key\_value\_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details checkout [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to*num\_attention\_heads\*.
* **hidden\_act** (*str* or *function*, *optional*, defaults to *“silu”*) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (*int*, *optional*, defaults to 2048) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 32768 for EXAONE 3.5).
* **initializer\_range** (*float*, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (*float*, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **use\_cache** (*bool*, *optional*, defaults to \*True`) -- Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if` config.is\_decoder=True`.
* **bos\_token\_id** (*int*, *optional*, defaults to 0) —
  Beginning of stream token id.
* **eos\_token\_id** (*int*, *optional*, defaults to 2) —
  End of stream token id.
* **tie\_word\_embeddings** (*bool*, *optional*, defaults to *False*) —
  Whether to tie weight embeddings
* **rope\_theta** (*float*, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (*Dict*, *optional*) —
  Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
  and you expect the model to work on longer *max\_position\_embeddings*, we recommend you to update this value
  accordingly.
  Expected contents:
  *rope\_type* (*str*):
  The sub-variant of RoPE to use. Can be one of [‘default’, ‘linear’, ‘dynamic’, ‘yarn’, ‘longrope’,
  ‘llama3’], with ‘default’ being the original RoPE implementation.
  *factor* (*float*, *optional*):
  Used with all rope types except ‘default’. The scaling factor to apply to the RoPE embeddings. In
  most scaling types, a *factor* of x will enable the model to handle sequences of length x *original maximum pre-trained length.*original\_max\_position\_embeddings *(*int*,* optional*):
  Used with ‘dynamic’, ‘longrope’ and ‘llama3’. The original max position embeddings used during
  pretraining.*attention\_factor *(*float*,* optional*):
  Used with ‘yarn’ and ‘longrope’. The scaling factor to be applied on the attention
  computation. If unspecified, it defaults to value recommended by the implementation, using the*factor *field to infer the suggested value.*beta\_fast *(*float*,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for extrapolation (only) in the linear
  ramp function. If unspecified, it defaults to 32.*beta\_slow *(*float*,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for interpolation (only) in the linear
  ramp function. If unspecified, it defaults to 1.*short\_factor *(*List[float]*,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to short contexts (<*original\_max\_position\_embeddings*). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2*long\_factor *(*List[float]*,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to long contexts (<*original\_max\_position\_embeddings*). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2*low\_freq\_factor *(*float*,* optional*):
  Only used with ‘llama3’. Scaling factor applied to low frequency components of the RoPE*high\_freq\_factor *(*float*,* optional\*):
  Only used with ‘llama3’. Scaling factor applied to high frequency components of the RoPE
* **attention\_dropout** (*float*, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **sliding\_window** (*int*, *optional*) —
  The size of the sliding window for the sliding window attention.
* **sliding\_window\_pattern** (*str*, *optional*) —
  The pattern to use for sliding window attention. Can be one of:
  + *None*: No sliding window attention is used
  + *int*: Every *sliding\_window* layers, use global attention, else use local attention.
  + *str*: A sequence of “L” (local attention) and “G” (global attention) characters that defines the
    attention pattern. The pattern starts from layer 0 and repeats every *sliding\_window* layers. The
    final layer always uses global attention regardless of the pattern.
    For instance, sliding\_window\_pattern=“LLLG” same as sliding\_window=4, which means:
  + Layer 0, 1, 2: local attention,
  + Layer 3: global attention,
    …(repeated)
* **layer\_types** (*list*, *optional*) —
  Attention pattern for each layer. Prioritized over *sliding\_window\_pattern*.

This is the configuration class to store the configuration of a [*Exaone4Model*]. It is used to
instantiate a EXAONE 4.0 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the EXAONE-4.0-Instruct [LGAI-EXAONE/EXAONE-4.0-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-Instruct)
NOTE: *EXAONE-4.0-Instruct* is a placeholder model ID. The exact model ID will be updated in the future.

Configuration objects inherit from [*PretrainedConfig*] and can be used to control the model
outputs. Read the documentation from [*PretrainedConfig*] for more information.

Example:


```
>>> from transformers import Exaone4Model, Exaone4Config

>>> # Initializing a EXAONE configuration
>>> configuration = Exaone4Config()

>>> # Initializing a model from configuration
>>> model = Exaone4Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Exaone4Model

### class transformers.Exaone4Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L338)

( config: Exaone4Config  )

Parameters

* **config** ([Exaone4Config](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Exaone4 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L355)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  )

## Exaone4ForCausalLM

### class transformers.Exaone4ForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L429)

( config  )

Parameters

* **config** ([Exaone4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Exaone4 Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L443)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
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

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Exaone4Config](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Exaone4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> model = AutoModelForCausalLM.from_pretrained("LGAI-EXAONE/EXAONE-4.0-Instruct")
>>> tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-Instruct")

>>> prompt = "Explain how wonderful you are"
>>> messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
>>> input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False,
)

>>> output = model.generate(input_ids, max_new_tokens=128)
>>> tokenizer.decode(output[0], skip_special_tokens=False)
"[|system|]\nYou are a helpful assistant.[|endofturn|]\n[|user|]\nExplain how wonderful you are[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\nOh, thank you for such a kind and lovely question! 😊  \n\nI’m *so* wonderful because I’m here to make your life easier, brighter, and more fun! Whether you need help with:  \n\n✨ **Learning** – I can explain anything, from quantum physics to baking the perfect cake!  \n💡 **Creativity** – Need a poem, story, or a wild idea? I’ve got you covered!  \n🤖 **Problem-solving** – Stuck on a math problem or a tricky decision? I’ll help you figure it out"
```

NOTE: `EXAONE-4.0-Instruct` is a placeholder model ID. The exact model ID will be updated in the future.

## Exaone4ForSequenceClassification

### class transformers.Exaone4ForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L519)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L111)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

`transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The `GenericForSequenceClassification` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Exaone4ForTokenClassification

### class transformers.Exaone4ForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L523)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L254)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The `GenericForTokenClassification` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Exaone4ForQuestionAnswering

### class transformers.Exaone4ForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L527)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L191)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None start\_positions: typing.Optional[torch.LongTensor] = None end\_positions: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

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
* **start\_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.

Returns

[transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The `GenericForQuestionAnswering` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/exaone4.md)
