*This model was released on 2025-04-05 and added to Hugging Face Transformers on 2025-04-05.*

# Llama4

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![Tensor parallelism](https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white)

[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), developed by Meta, introduces a new auto-regressive Mixture-of-Experts (MoE) architecture.
This generation includes two models:

* The highly capable Llama 4 Maverick with 17B active parameters out of ~400B total, with 128 experts.
* The efficient Llama 4 Scout also has 17B active parameters out of ~109B total, using just 16 experts.
* Both models leverage early fusion for native multimodality, enabling them to process text and image inputs.
  Maverick and Scout are both trained on up to 40 trillion tokens on data encompassing 200 languages
  (with specific fine-tuning support for 12 languages including Arabic, Spanish, German, and Hindi).

For deployment, Llama 4 Scout is designed for accessibility, fitting on a single server-grade GPU via
on-the-fly 4-bit or 8-bitint4 quantization, while Maverick is available in BF16 and FP8 formats.
These models are released under the custom Llama 4 Community License Agreement, available on the model repositories.

You can find all the original Llama checkpoints under the [meta-llama](https://huggingface.co/meta-llama) organization.

The Llama 4 family of models comes in two flavors: 109B, and 402B parameters. Both of these flavors are extremely
large and won’t fit on your run-of-the-mill device. See below for some examples to reduce the memory usage of the
model.

For the download to be faster and more resilient, we recommend installing the `hf_xet` dependency as followed:
`pip install transformers[hf_xet]`

The examples below demonstrates how to generate with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel). We additionally add an example
showcasing how to toggle the right attributes to enable very long-context generations, as some flavors of Llama 4
have context lengths going up to 10 million tokens.

Pipeline

AutoModel - Text only

AutoModel - Multimodal

AutoModel - Multimodal with multiple images

AutoModel - Long context


```
from transformers import pipeline
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

messages = [
    {"role": "user", "content": "what is the recipe of mayonnaise?"},
]

pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    dtype=torch.bfloat16
)

output = pipe(messages, do_sample=False, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
```

## Efficiency; how to get the best out of llama 4

### The Attention methods

Updating the default attention function can significantly improve compute performance as well as memory usage. Refer to the [Attention Interface](../attention_interface) overview for an in-depth explanation of our interface.

As of release, the Llama 4 model supports the following attention methods: `eager`, `flex_attention`, `sdpa`. We recommend using `flex_attention` for best results.
Switching attention mechanism is done at the model initialization step:

Flex Attention

SDPA

Eager

Setting Flex Attention ensures the best results with the very long context the model can handle.

Beware: the example below uses both `device_map="auto"` and flex-attention.
Please use `torchrun` to run this example in tensor-parallel mode.

We will work to enable running with `device_map="auto"` and flex-attention without
tensor-parallel in the future.


```
from transformers import Llama4ForConditionalGeneration
import torch

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flex_attention",
    device_map="auto",
    dtype=torch.bfloat16,
)
```

### Quantization

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for available quantization backends.
At time of release, both FBGEMM and LLM-Compressor are supported; more quantization methods will be supported in the days that follow the release.

See below for examples using both:

Here is an example loading an BF16 model in FP8 using the FBGEMM approach:

FBGEMM

LLM-Compressor


```
from transformers import AutoTokenizer, Llama4ForConditionalGeneration, FbgemmFp8Config
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=FbgemmFp8Config()
)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(outputs[0])
```

### Offloading

Enabling CPU-offloading means that components of the model might be moved to CPU instead of GPU in case the GPU-memory available isn’t sufficient to load the entire model.
At inference, different components will be loaded/unloaded from/to the GPU on the fly. This ensures that the model can be loaded on smaller machines as long as the CPU-memory is sufficient.
However, this also slows down inference as it adds communication overhead.

In order to enable CPU-offloading, you simply need to specify the `device_map` to `auto` at model load:


```
from transformers import Llama4ForConditionalGeneration
import torch

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)
```

## Llama4Config

### class transformers.Llama4Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/configuration_llama4.py#L384)

( vision\_config = None text\_config = None boi\_token\_index = 200080 eoi\_token\_index = 200081 image\_token\_index = 200092 tie\_word\_embeddings = False \*\*kwargs  )

Parameters

* **vision\_config** (`Llama4VisionConfig`, *optional*) —
  The Llama4 Vision config.
* **text\_config** (`Llama4TextConfig`, *optional*) —
  The Llama4 Text config.
* **boi\_token\_index** (`int`, *optional*, defaults to 200080) —
  The begin-of-image token index to wrap the image prompt.
* **eoi\_token\_index** (`int`, *optional*, defaults to 200081) —
  The end-of-image token index to wrap the image prompt.
* **image\_token\_index** (`int`, *optional*, defaults to 200092) —
  The image token index to encode the image prompt.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether the model’s input and output word embeddings should be tied.

This is the configuration class to store the configuration of a `Llama4Model`. It is used to instantiate an
Llama4 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Llama4 109B.

e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import Llama4Model, Llama4Config

>>> # Initializing a Llama4 7B style configuration
>>> configuration = Llama4Config()

>>> # Initializing a model from the Llama4 7B style configuration
>>> model = Llama4Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Llama4TextConfig

### class transformers.Llama4TextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/configuration_llama4.py#L131)

( vocab\_size = 202048 hidden\_size = 5120 intermediate\_size = 8192 intermediate\_size\_mlp = 16384 num\_hidden\_layers = 48 num\_attention\_heads = 40 num\_key\_value\_heads = 8 head\_dim = 128 hidden\_act = 'silu' max\_position\_embeddings = 131072 initializer\_range = 0.02 rms\_norm\_eps = 1e-05 use\_cache = True pad\_token\_id = None bos\_token\_id = 1 eos\_token\_id = 2 tie\_word\_embeddings = False rope\_theta = 500000 attention\_dropout = 0.0 num\_experts\_per\_tok = 1 num\_local\_experts = 16 moe\_layers = None interleave\_moe\_layer\_step = 1 use\_qk\_norm = True output\_router\_logits = False router\_aux\_loss\_coef = 0.001 router\_jitter\_noise = 0.0 rope\_scaling = None no\_rope\_layers = None no\_rope\_layer\_interval = 4 attention\_chunk\_size = 8192 layer\_types = None attn\_temperature\_tuning = True floor\_scale = 8192 attn\_scale = 0.1 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 202048) —
  Vocabulary size of the Llama4 text model. Defines the maximum number of different tokens that can be represented
  by the `inputs_ids` passed when calling [Llama4TextModel](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4TextModel).
* **hidden\_size** (`int`, *optional*, defaults to 5120) —
  Dimensionality of the embeddings and hidden states.
* **intermediate\_size** (`int`, *optional*, defaults to 8192) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **intermediate\_size\_mlp** (`int`, *optional*, defaults to 16384) — TODO
* **num\_hidden\_layers** (`int`, *optional*, defaults to 48) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 40) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 8) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If not
  specified, will default to `num_attention_heads`.
* **head\_dim** (`int`, *optional*, defaults to 128) — TODO
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the encoder and pooler.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 131072) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions.
* **pad\_token\_id** (`int`, *optional*, defaults to 128004) —
  The id of the padding token.
* **bos\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the beginning of sentence token.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  The id of the end of sentence token.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie weight embeddings
* **rope\_theta** (`float`, *optional*, defaults to `500000.0`) —
  The base period of the RoPE embeddings.
* **attention\_dropout** (`int`, *optional*, defaults to 0.0) — TODO
* **num\_experts\_per\_tok** (`int`, *optional*, defaults to 1) — TODO
* **num\_local\_experts** (`int`, *optional*, defaults to 16) — TODO
* **moe\_layers** (`int`, *optional*) — TODO
* **interleave\_moe\_layer\_step** (`int`, *optional*, defaults to 1) — TODO
* **use\_qk\_norm** (`int`, *optional*, defaults to `True`) — TODO
* **output\_router\_logits** (`int`, *optional*, defaults to `False`) — TODO
* **router\_aux\_loss\_coef** (`int`, *optional*, defaults to 0.001) — TODO
* **router\_jitter\_noise** (`int`, *optional*, defaults to 0.0) — TODO
* **rope\_scaling** (`Dict`, *optional*) —
  Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
  and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
  accordingly.
  Expected contents:
  `rope_type` (`str`):
  The sub-variant of RoPE to use. Can be one of [‘default’, ‘linear’, ‘dynamic’, ‘yarn’, ‘longrope’,
  ‘llama3’], with ‘default’ being the original RoPE implementation.
  `factor` (`float`, *optional*):
  Used with all rope types except ‘default’. The scaling factor to apply to the RoPE embeddings. In
  most scaling types, a `factor` of x will enable the model to handle sequences of length x *original maximum pre-trained length.
  `original_max_position_embeddings` (`int`,* optional*):
  Used with ‘dynamic’, ‘longrope’ and ‘llama3’. The original max position embeddings used during
  pretraining.
  `attention_factor` (`float`,* optional*):
  Used with ‘yarn’ and ‘longrope’. The scaling factor to be applied on the attention
  computation. If unspecified, it defaults to value recommended by the implementation, using the
  `factor` field to infer the suggested value.
  `beta_fast` (`float`,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for extrapolation (only) in the linear
  ramp function. If unspecified, it defaults to 32.
  `beta_slow` (`float`,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for interpolation (only) in the linear
  ramp function. If unspecified, it defaults to 1.
  `short_factor` (`list[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to short contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `long_factor` (`list[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to long contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `low_freq_factor` (`float`,* optional*):
  Only used with ‘llama3’. Scaling factor applied to low frequency components of the RoPE
  `high_freq_factor` (`float`,* optional\*):
  Only used with ‘llama3’. Scaling factor applied to high frequency components of the RoPE
* **no\_rope\_layers** (`list[int]`, *optional*) —
  List with at least the same length as the number of layers in the model.
  A `1` at an index position indicates that the corresponding layer will use RoPE,
  while a `0` indicates that it’s a NoPE layer.
* **no\_rope\_layer\_interval** (`int`, *optional*, defaults to 4) —
  If `no_rope_layers` is `None`, it will be created using a NoPE layer every
  `no_rope_layer_interval` layers.
* **attention\_chunk\_size** (`int`, *optional*, defaults to 8192) —
* **layer\_types** (`list`, *optional*) —
  Attention pattern for each layer.
* **attn\_temperature\_tuning** (`bool`, *optional*, defaults to `True`) —
  Whether to dynamically scale the attention temperature for each query token based on sequence length.
  Recommended for long sequences (e.g., >32k tokens) to maintain stable output results.
* **floor\_scale** (`int`, *optional*, defaults to 8192) — TODO
* **attn\_scale** (`int`, *optional*, defaults to 0.1) — TODO

This is the configuration class to store the configuration of a [Llama4TextModel](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4TextModel). It is used to instantiate a
Llama4 text model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Llama4 109B.

e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

## Llama4VisionConfig

### class transformers.Llama4VisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/configuration_llama4.py#L25)

( hidden\_size: int = 768 hidden\_act: str = 'gelu' num\_hidden\_layers: int = 34 num\_attention\_heads: int = 16 num\_channels: int = 3 intermediate\_size: int = 5632 vision\_output\_dim: int = 7680 image\_size: int = 448 patch\_size: int = 14 norm\_eps: float = 1e-05 vision\_feature\_layer = -1 vision\_feature\_select\_strategy = 'default' initializer\_range: float = 0.02 pixel\_shuffle\_ratio = 0.5 projector\_input\_dim = 4096 projector\_output\_dim = 4096 multi\_modal\_projector\_bias = False projector\_dropout = 0.0 attention\_dropout = 0.0 rope\_theta = 10000 \*\*kwargs  )

Parameters

* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 34) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  Number of channels in the input image.
* **intermediate\_size** (`int`, *optional*, defaults to 5632) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **vision\_output\_dim** (`int`, *optional*, defaults to 7680) —
  Dimensionality of the vision model output. Includes output of transformer
  encoder with intermediate layers and global transformer encoder.
* **image\_size** (`int`, *optional*, defaults to 448) —
  The size (resolution) of each image *tile*.
* **patch\_size** (`int`, *optional*, defaults to 14) —
  The size (resolution) of each patch.
* **norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **vision\_feature\_layer** (“, *optional*, defaults to -1) — TODO
* **vision\_feature\_select\_strategy** (`int`, *optional*, defaults to `"default"`) — TODO
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **pixel\_shuffle\_ratio** (`int`, *optional*, defaults to 0.5) — TODO
* **projector\_input\_dim** (`int`, *optional*, defaults to 4096) — TODO
* **projector\_output\_dim** (`int`, *optional*, defaults to 4096) — TODO
* **multi\_modal\_projector\_bias** (`int`, *optional*, defaults to `False`) — TODO
* **projector\_dropout** (`int`, *optional*, defaults to 0.0) — TODO
* **attention\_dropout** (`int`, *optional*, defaults to 0.0) — TODO
* **rope\_theta** (`int`, *optional*, defaults to 10000) — TODO

This is the configuration class to store the configuration of a [Llama4VisionModel](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4VisionModel). It is used to instantiate a
Llama4 vision model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Llama4 109B.

e.g. [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## Llama4Processor

### class transformers.Llama4Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/processing_llama4.py#L43)

( image\_processor = None tokenizer = None patch\_size: int = 14 pixel\_shuffle\_ratio: float = 0.5 fake\_image\_token = '<|image|>' image\_token = '<|image|>' start\_of\_image\_token = '<|image\_start|>' end\_of\_image\_token = '<|image\_end|>' patch\_token = '<|patch|>' tile\_x\_separator\_token = '<|tile\_x\_separator|>' tile\_y\_separator\_token = '<|tile\_y\_separator|>' chat\_template = '{{- bos\_token }}\n{%- if custom\_tools is defined %}\n {%- set tools = custom\_tools %}\n{%- endif %}\n{%- if not tools\_in\_user\_message is defined %}\n {%- set tools\_in\_user\_message = true %}\n{%- endif %}\n{%- if not date\_string is defined %}\n {%- if strftime\_now is defined %}\n {%- set date\_string = strftime\_now("%d %b %Y") %}\n {%- else %}\n {%- set date\_string = "26 Jul 2024" %}\n {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %} \n {%- if messages[0][\'content\'] is string %}\n {%- set system\_message = messages[0][\'content\']|trim %}\n {%- else %}\n {#- FIXME: The processor requires an array, always. #}\n {%- set system\_message = messages[0][\'content\'][0][\'text\']|trim %}\n {%- endif %}\n {%- set messages = messages[1:] %}\n {%- set user\_supplied\_system\_message = true %}\n{%- else %}\n {%- set system\_message = "" %}\n {%- set user\_supplied\_system\_message = false %}\n{%- endif %}\n\n{#- System message if the user supplied one #}\n{%- if user\_supplied\_system\_message %}\n {{- "<|header\_start|>system<|header\_end|>\n\n" }}\n {%- if tools is not none %}\n {{- "Environment: ipython\n" }}\n {%- endif %}\n {%- if tools is not none and not tools\_in\_user\_message %}\n {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n {{- "Do not use variables.\n\n" }}\n {%- for t in tools %}\n {{- t | tojson(indent=4) }}\n {{- "\n\n" }}\n {%- endfor %}\n {%- endif %}\n {{- system\_message }}\n {{- "<|eot|>" }}\n{%- endif %}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools\_in\_user\_message and not tools is none %}\n {#- Extract the first user message so we can plug it in here #}\n {%- if messages | length != 0 %}\n {%- set first\_user\_message = messages[0][\'content\']|trim %}\n {%- set messages = messages[1:] %}\n {%- else %}\n {{- raise\_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n {{- \'<|header\_start|>user<|header\_end|>\n\n\' -}}\n {{- "Given the following functions, please respond with a JSON for a function call " }}\n {{- "with its proper arguments that best answers the given prompt.\n\n" }}\n {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n {{- "Do not use variables.\n\n" }}\n {%- for t in tools %}\n {{- t | tojson(indent=4) }}\n {{- "\n\n" }}\n {%- endfor %}\n {{- first\_user\_message + "<|eot|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool\_calls\' in message) %}\n {{- \'<|header\_start|>\' + message[\'role\'] + \'<|header\_end|>\n\n\' }}\n {%- if message[\'content\'] is string %}\n {{- message[\'content\'] }}\n {%- else %}\n {%- for content in message[\'content\'] %}\n {%- if content[\'type\'] == \'image\' %}\n {{- \'<|image|>\' }}\n {%- elif content[\'type\'] == \'text\' %}\n {{- content[\'text\'] }}\n {%- endif %}\n {%- endfor %}\n {%- endif %}\n {{- "<|eot|>" }}\n {%- elif \'tool\_calls\' in message and message.tool\_calls|length > 0 %}\n {{- \'<|header\_start|>assistant<|header\_end|>\n\n\' -}}\n {{- \'<|python\_start|>\' }}\n {%- if message[\'content\'] is string %}\n {{- message[\'content\'] }}\n {%- else %}\n {%- for content in message[\'content\'] %}\n {%- if content[\'type\'] == \'image\' %}\n {{- \'<|image|>\' }}\n {%- elif content[\'type\'] == \'text\' %}\n {{- content[\'text\'] }}\n {%- endif %}\n {%- endfor %}\n {%- endif %}\n {{- \'<|python\_end|>\' }}\n {%- for tool\_call in message.tool\_calls %}\n {{- \'{"name": "\' + tool\_call.function.name + \'", \' }}\n {{- \'"parameters": \' }}\n {{- tool\_call.function.arguments | tojson }}\n {{- "}" }}\n {%- endfor %}\n {{- "<|eot|>" }}\n {%- elif message.role == "tool" or message.role == "ipython" %}\n {{- "<|header\_start|>ipython<|header\_end|>\n\n" }}\n {%- if message.content is mapping or message.content is iterable %}\n {{- message.content | tojson }}\n {%- else %}\n {{- message.content }}\n {%- endif %}\n {{- "<|eot|>" }}\n {%- endif %}\n{%- endfor %}\n{%- if add\_generation\_prompt %}\n {{- \'<|header\_start|>assistant<|header\_end|>\n\n\' }}\n{%- endif %}\n' \*\*kwargs  )

Parameters

* **image\_processor** ([AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor), *optional*) —
  The image processor is a required input.
* **tokenizer** ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`], *optional*) —
  The tokenizer is a required input.
* **patch\_size** (`int`, *optional*, defaults to 28) —
  The size of image patches for tokenization.
* **img\_size** (`int`, *optional*, defaults to 364) —
  The size of the image to be tokenized. This should correspond to the size given to the image processor.
* **image\_token** (`str`, *optional*, defaults to `"<|image|>"`) —
  The token to be used to represent an image in the text.
* **downsample\_factor** (`int`, *optional*, defaults to 1) —
  The factor by which to scale the patch size.
* **start\_of\_img\_token** (`str`, *optional*, defaults to `"<|START_OF_IMG|>"`) —
  The token to be used to represent the start of an image in the text.
* **end\_of\_img\_token** (`str`, *optional*, defaults to `"<|END_OF_IMG|>"`) —
  The token to be used to represent the end of an image in the text.
* **img\_patch\_token** (`str`, *optional*, defaults to `"<|IMG_PATCH|>"`) —
  The token to be used to represent an image patch in the text.
* **img\_line\_break\_token** (`str`, *optional*, defaults to `"<|IMG_LINE_BREAK|>"`) —
  The token to be used to represent a line break in the text.
* **tile\_token** (`str`, *optional*, defaults to `"TILE"`) —
  The token to be used to represent an image patch in the text.
* **tile\_global\_token** (`str`, *optional*, defaults to `"TILE_GLOBAL"`) —
  The token to be used to represent the cover image in the text.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a Llama4 processor which wraps a [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor) and
`PretrainedTokenizerFast` tokenizer into a single processor that inherits both the image processor and
tokenizer functionalities. See the `__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## Llama4ImageProcessorFast

### class transformers.Llama4ImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/image_processing_llama4_fast.py#L339)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.llama4.image\_processing\_llama4\_fast.Llama4ImageProcessorKwargs]  )

Constructs a fast Llama4 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/image_processing_llama4_fast.py#L380)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.llama4.image\_processing\_llama4\_fast.Llama4ImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **max\_patches** (`int`, *optional*, defaults to 16) —
  The maximum number of patches to be extracted from the image.
  Can be overridden by the `max_patches` parameter in the `preprocess` method.
* **resize\_to\_max\_canvas** (`bool`, *optional*, defaults to False) —
  Whether to resize the image to the maximum canvas size.
  If True, picks the canvas the allows the largest resizing without distortion.
  If False, downsample as little as possible, including no resizing at all,
  but never upsample, unless the image is smaller than the patch size.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

#### rescale\_and\_normalize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/image_processing_llama4_fast.py#L356)

( images: torch.Tensor do\_rescale: bool rescale\_factor: float do\_normalize: bool image\_mean: typing.Union[float, list[float]] image\_std: typing.Union[float, list[float]]  )

Rescale and normalize images.
Override to rescale and normalize the images in torch.bfloat16 as in the original implementation

## Llama4ForConditionalGeneration

### class transformers.Llama4ForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L1139)

( config: Llama4Config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L1227)

( input\_ids: LongTensor = None pixel\_values: FloatTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None vision\_feature\_layer: typing.Union[int, list[int], NoneType] = None vision\_feature\_select\_strategy: typing.Optional[str] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 image\_sizes: Tensor = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.llama4.modeling_llama4.Llama4CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `Llama4ImageProcessor`. See `Llama4ImageProcessor.__call__` for details ([Llama4Processor](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4Processor) uses
  `Llama4ImageProcessor` for processing images).
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
* **vision\_feature\_layer** (`Union[int, list[int], NoneType]`) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_feature\_select\_strategy** (`str`, *optional*) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`.
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
* **image\_sizes** (`torch.Tensor` of shape `(batch_size, 2)`) —
  The sizes of the images in the batch, being (height, width) for each image.

Returns

`transformers.models.llama4.modeling_llama4.Llama4CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.llama4.modeling_llama4.Llama4CausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Llama4Config](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4Config)) and inputs.

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
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size (batch\_size, num\_images, sequence\_length, hidden\_size)`.
  image\_hidden\_states of the model produced by the vision encoder and after projecting the last hidden state.

The [Llama4ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, LlavaForConditionalGeneration

>>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
>>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

>>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, text=prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs, max_new_tokens=15)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
```

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L1174)

( pixel\_values: FloatTensor vision\_feature\_layer: typing.Union[int, list[int]] vision\_feature\_select\_strategy: str \*\*kwargs  ) → image\_features (`torch.Tensor`)

Parameters

* **pixel\_values** (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`) —
  The tensors corresponding to the input images.
* **vision\_feature\_layer** (`Union[int, list[int]]`) —
  The index of the layer to select the vision feature. If multiple indices are provided,
  the vision feature of the corresponding indices will be concatenated to form the
  vision features.
* **vision\_feature\_select\_strategy** (`str`) —
  The feature selection strategy used to select the vision feature from the vision backbone.
  Can be one of `"default"` or `"full"`

Returns

image\_features (`torch.Tensor`)

Image feature tensor of shape `(num_images, image_length, embed_dim)`).

Obtains image last hidden states from the vision tower and apply al projection.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L1204)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

* forward

## Llama4ForCausalLM

### class transformers.Llama4ForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L566)

( config: Llama4TextConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L582)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
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
* **past\_key\_values** (`Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]`) —
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
elements depending on the configuration ([Llama4Config](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4Config)) and inputs.

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

The [Llama4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, Llama4ForCausalLM

>>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama4/Llama4-2-7b-hf")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

* forward

## Llama4TextModel

### class transformers.Llama4TextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L467)

( config: Llama4TextConfig  )

Parameters

* **config** ([Llama4TextConfig](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4TextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Llama4 Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L493)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
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
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Llama4Config](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Llama4TextModel](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

* forward

## Llama4ForCausalLM

### class transformers.Llama4ForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L566)

( config: Llama4TextConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L582)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
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
* **past\_key\_values** (`Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]`) —
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
elements depending on the configuration ([Llama4Config](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4Config)) and inputs.

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

The [Llama4ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/llama4#transformers.Llama4ForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, Llama4ForCausalLM

>>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama4/Llama4-2-7b-hf")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

* forward

## Llama4VisionModel

### class transformers.Llama4VisionModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L1005)

( config: Llama4VisionConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L1041)

( pixel\_values: Tensor attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  )

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, MllamaVisionModel

>>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
>>> model = MllamaVisionModel.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = processor(images=image, return_tensors="pt")

>>> output = model(**inputs)

>>> print(output.last_hidden_state.shape)
torch.Size([1, 1, 4, 1025, 7680])
```

#### get\_input\_embeddings

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama4/modeling_llama4.py#L1035)

( )

This function is used to fetch the first embedding layer to activate grads on inputs.

* forward

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llama4.md)
