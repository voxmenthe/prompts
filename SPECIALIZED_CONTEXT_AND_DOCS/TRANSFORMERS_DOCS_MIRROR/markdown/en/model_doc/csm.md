*This model was released on 2025-02-27 and added to Hugging Face Transformers on 2025-05-07.*

# Csm

## Overview

The Conversational Speech Model (CSM) is the first open-source contextual text-to-speech model [released by Sesame](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice). It is designed to generate natural-sounding speech with or without conversational context. This context typically consists of multi-turn dialogue between speakers, represented as sequences of text and corresponding spoken audio.

**Model Architecture:**
CSM is composed of two LLaMA-style auto-regressive transformer decoders: a backbone decoder that predicts the first codebook token and a depth decoder that generates the remaining tokens. It uses the pretrained codec model [Mimi](./mimi), introduced by Kyutai, to encode speech into discrete codebook tokens and decode them back into audio.

The original csm-1b checkpoint is available under the [Sesame](https://huggingface.co/sesame/csm-1b) organization on Hugging Face.

![](https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/csm_architecture.png)

## Usage Tips

### Without Conversational Context

CSM can be used to simply generate speech from a text prompt:


```
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, infer_device

model_id = "sesame/csm-1b"
device = infer_device()

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs
text = "[0]The past is just a story we tell ourselves." # `[0]` for speaker id 0
inputs = processor(text, add_special_tokens=True).to(device)

# another equivalent way to prepare the inputs
conversation = [
    {"role": "0", "content": [{"type": "text", "text": "The past is just a story we tell ourselves."}]},
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

# infer the model
audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "example_without_context.wav")
```

### With Conversational Context

CSM can be used to generate speech given a conversation, allowing consistency in the voices and content-aware generation:


```
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, infer_device
from datasets import load_dataset, Audio

model_id = "sesame/csm-1b"
device = infer_device()

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
# ensure the audio is 24kHz
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
conversation = []

# 1. context
for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
    conversation.append(
        {
            "role": f"{speaker_id}",
            "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
        }
    )

# 2. text prompt
conversation.append({"role": f"{ds[4]['speaker_id']}", "content": [{"type": "text", "text": ds[4]["text"]}]})

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

# infer the model
audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "example_with_context.wav")
```

### Batched Inference

CSM supports batched inference!


```
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, infer_device
from datasets import load_dataset, Audio

model_id = "sesame/csm-1b"
device = infer_device()

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs 
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
# ensure the audio is 24kHz
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
# here a batch with two prompts
conversation = [
    [
        {
            "role": f"{ds[0]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[0]["text"]},
                {"type": "audio", "path": ds[0]["audio"]["array"]},
            ],
        },
        {
            "role": f"{ds[1]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[1]["text"]},
            ],
        },
    ],
    [
        {
            "role": f"{ds[0]['speaker_id']}",
            "content": [
                {"type": "text", "text": ds[0]["text"]},
            ],
        }
    ],
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, [f"speech_batch_idx_{i}.wav" for i in range(len(audio))])
```

### Making The Model Go Brrr

CSM supports full-graph compilation with CUDA graphs!


```
import torch
import copy
from transformers import CsmForConditionalGeneration, AutoProcessor
from datasets import load_dataset

model_id = "sesame/csm-1b"
device = "cuda"

# set logs to ensure no recompilation and graph breaks
torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# use static cache, enabling automatically torch compile with fullgraph and reduce-overhead
model.generation_config.max_length = 250 # big enough to avoid recompilation
model.generation_config.max_new_tokens = None # would take precedence over max_length
model.generation_config.cache_implementation = "static"
model.depth_decoder.generation_config.cache_implementation = "static"

# generation kwargs
gen_kwargs = {
    "do_sample": False,
    "depth_decoder_do_sample": False,
    "temperature": 1.0,
    "depth_decoder_temperature": 1.0,
}

# Define a timing decorator
class TimerContext:
    def __init__(self, name="Execution"):
        self.name = name
        self.start_event = None
        self.end_event = None
        
    def __enter__(self):
        # Use CUDA events for more accurate GPU timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000.0
        print(f"{self.name} time: {elapsed_time:.4f} seconds")

# prepare the inputs 
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")

conversation = [
    {
        "role": f"{ds[0]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[0]["text"]},
            {"type": "audio", "path": ds[0]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[1]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[1]["text"]},
            {"type": "audio", "path": ds[1]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[2]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[2]["text"]},
        ],
    },
]

padded_inputs_1 = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

print("\n" + "="*50)
print("First generation - compiling and recording CUDA graphs...")
with TimerContext("First generation"):
    _ = model.generate(**padded_inputs_1, **gen_kwargs)
print("="*50)

print("\n" + "="*50)
print("Second generation - fast !!!")
with TimerContext("Second generation"):
    _ = model.generate(**padded_inputs_1, **gen_kwargs)
print("="*50)

# now with different inputs
conversation = [
    {
        "role": f"{ds[0]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[2]["text"]},
            {"type": "audio", "path": ds[2]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[1]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[3]["text"]},
            {"type": "audio", "path": ds[3]["audio"]["array"]},
        ],
    },
    {
        "role": f"{ds[2]['speaker_id']}",
        "content": [
            {"type": "text", "text": ds[4]["text"]},
        ],
    },
]
padded_inputs_2 = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
).to(model.device)

print("\n" + "="*50)
print("Generation with other inputs!")
with TimerContext("Generation with different inputs"):
    _ = model.generate(**padded_inputs_2, **gen_kwargs)
print("="*50)
```

### Training

CSM Transformers integration supports training!


```
from transformers import CsmForConditionalGeneration, AutoProcessor, infer_device
from datasets import load_dataset, Audio

model_id = "sesame/csm-1b"
device = infer_device()

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)
model.train()
model.codec_model.eval()

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
# ensure the audio is 24kHz
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
conversation = []

# context
for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
    conversation.append(
        {
            "role": f"{speaker_id}",
            "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
        }
    )

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    output_labels=True,
).to(model.device)

out = model(**inputs)
out.loss.backward()
```

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb).
The original code can be found [here](https://github.com/SesameAILabs/csm).

## CsmConfig

### class transformers.CsmConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/configuration_csm.py#L207)

( num\_codebooks = 32 vocab\_size = 2051 text\_vocab\_size = 128256 hidden\_size = 2048 intermediate\_size = 8192 num\_hidden\_layers = 16 num\_attention\_heads = 32 num\_key\_value\_heads = 8 hidden\_act = 'silu' max\_position\_embeddings = 2048 initializer\_range = 0.02 rms\_norm\_eps = 1e-05 use\_cache = True pad\_token\_id = 128002 codebook\_pad\_token\_id = 2050 codebook\_eos\_token\_id = 0 bos\_token\_id = 128000 eos\_token\_id = None audio\_token\_id = 128002 audio\_eos\_token\_id = 128003 rope\_theta = 500000 rope\_scaling = None attention\_bias = False attention\_dropout = 0.0 mlp\_bias = False head\_dim = None tie\_codebooks\_embeddings = True depth\_decoder\_config = None codec\_config = None \*\*kwargs  )

Parameters

* **num\_codebooks** (`int`, *optional*, defaults to 32) —
  Number of codebooks used in the underlying codec model responsible for tokenizing the audio.
* **vocab\_size** (`int`, *optional*, defaults to 2051) —
  Vocabulary size of the Csm model. Defines the number of different audio tokens that can be represented by each codebook.
* **text\_vocab\_size** (`int`, *optional*, defaults to 128256) —
  Vocabulary size of the text input for the Csm model. Defines the number of different text tokens that can be represented.
* **hidden\_size** (`int`, *optional*, defaults to 2048) —
  Dimension of the hidden representations of the backbone model.
* **intermediate\_size** (`int`, *optional*, defaults to 8192) —
  Dimension of the MLP representations of the backbone model.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 16) —
  Number of hidden layers in the backbone model Transformer decoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads for each attention layer in the backbone model Transformer decoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 8) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245).
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the backbone model Transformer decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 2048) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **pad\_token\_id** (`int`, *optional*, defaults to 128002) —
  Padding token id.
* **codebook\_pad\_token\_id** (`int`, *optional*, defaults to 2050) —
  Padding token id for codebook tokens.
* **codebook\_eos\_token\_id** (`int`, *optional*, defaults to 0) —
  End of stream token id for codebook tokens.
* **bos\_token\_id** (`int`, *optional*, defaults to 128000) —
  Beginning of stream token id.
* **eos\_token\_id** (`int`, *optional*) —
  End of stream token id.
* **audio\_token\_id** (`int`, *optional*, defaults to 128002) —
  Audio token id in the text input.
* **audio\_eos\_token\_id** (`int`, *optional*, defaults to 128003) —
  End of stream token id for audio in the text input.
* **rope\_theta** (`float`, *optional*, defaults to 500000) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (`Dict`, *optional*, defaults to `{'factor' -- 32.0, 'high_freq_factor': 0.5, 'low_freq_factor': 0.125, 'original_max_position_embeddings': 1024, 'rope_type': 'llama3'}`):
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
* **attention\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **mlp\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in up\_proj, down\_proj and gate\_proj layers in the MLP layers.
* **head\_dim** (`int`, *optional*) —
  The attention head dimension. If None, it will default to hidden\_size // num\_attention\_heads
* **tie\_codebooks\_embeddings** (`bool`, *optional*, defaults to `True`) —
  Whether to tie the codebook tokens embeddings of the backbone model to the codebook tokens embeddings of the depth decoder.
* **depth\_decoder\_config** (`CsmDepthDecoderConfig`, *optional*) —
  Configuration for the depth decoder.
* **codec\_config** (`PretrainedConfig`, *optional*) —
  Configuration for the codec.

This is the configuration class to store the configuration of a [CsmForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmForConditionalGeneration). It is used to instantiate an CSM
model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the csm-1b.

e.g. [sesame/csm-1b](https://huggingface.co/sesame/csm-1b)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import CsmForConditionalGeneration, CsmConfig

>>> # Initializing a CsmConfig
>>> configuration = CsmConfig()

>>> # Initializing a model
>>> model = CsmForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## CsmDepthDecoderConfig

### class transformers.CsmDepthDecoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/configuration_csm.py#L25)

( num\_codebooks = 32 backbone\_hidden\_size = 2048 vocab\_size = 2051 hidden\_size = 1024 intermediate\_size = 8192 num\_hidden\_layers = 4 num\_attention\_heads = 8 num\_key\_value\_heads = 2 hidden\_act = 'silu' max\_position\_embeddings = 33 initializer\_range = 0.02 rms\_norm\_eps = 1e-05 use\_cache = True pad\_token\_id = None bos\_token\_id = None eos\_token\_id = None rope\_theta = 500000 rope\_scaling = None attention\_bias = False attention\_dropout = 0.0 mlp\_bias = False head\_dim = None \*\*kwargs  )

Parameters

* **num\_codebooks** (`int`, *optional*, defaults to 32) —
  Number of codebooks used in the underlying codec model responsible for tokenizing the audio.
* **backbone\_hidden\_size** (`int`, *optional*, defaults to 2048) —
  Dimension of the hidden representations of the backbone model used with this depth decoder.
* **vocab\_size** (`int`, *optional*, defaults to 2051) —
  Vocabulary size of the CsmDepthDecoder model. Defines the number of different audio tokens that can be represented by each codebook.
* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 8192) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 4) —
  Number of hidden layers in the Transformer decoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 2) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
  `num_attention_heads`.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 33) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **pad\_token\_id** (`int`, *optional*, defaults to 2050) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*) —
  Beginning of stream token id.
* **eos\_token\_id** (`int`, *optional*) —
  End of stream token id.
* **rope\_theta** (`float`, *optional*, defaults to 500000) —
  The base period of the RoPE embeddings.
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
* **attention\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **mlp\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use a bias in up\_proj, down\_proj and gate\_proj layers in the MLP layers.
* **head\_dim** (`int`, *optional*) —
  The attention head dimension. If None, it will default to hidden\_size // num\_attention\_heads

This is the configuration class to store the configuration of a [CsmDepthDecoderModel](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmDepthDecoderModel). It is used to instantiate an CSM depth decoder
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
a similar configuration to that of the csm-1b.

e.g. [sesame/csm-1b](https://huggingface.co/sesame/csm-1b)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import CsmDepthDecoder, CsmDepthDecoderConfig

>>> # Initializing a CsmDepthDecoder
>>> configuration = CsmDepthDecoderConfig()
>>> model = CsmDepthDecoderModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## CsmProcessor

![](https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/fig1.jpg)

### class transformers.CsmProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/processing_csm.py#L62)

( feature\_extractor tokenizer chat\_template = None  )

Parameters

* **feature\_extractor** ([EncodecFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor)) —
  The feature extractor is a required input.
* **tokenizer** ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`]) —
  The tokenizer is a required input.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a Csm processor which wraps [EncodecFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor) and
`PretrainedTokenizerFast` into a single processor that inherits both the audio feature extraction and
tokenizer functionalities. See the [**call**()](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmProcessor.__call__) for more
information.
The preferred way of passing kwargs is as a dictionary per modality, see usage example below.


```
from transformers import CsmProcessor
from datasets import load_dataset

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
audio = ds[0]["audio"]["array"]

processor = CsmProcessor.from_pretrained("sesame/csm-1b")

processor(
    text=["<|begin_of_text|>[0]What are you working on?<|end_of_text|><|AUDIO|><|audio_eos|><|begin_of_text|>[1]I'm figuring out my budget.<|end_of_text|>"],
    audio=audio,
    text_kwargs = {"padding": False},
    audio_kwargs = {"sampling_rate": 16000},
    common_kwargs = {"return_tensors": "pt"},
)
# this should error out because EncodecFeatureExtractor expects a 24kHz audio :)
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/processing_csm.py#L197)

( text: typing.Union[str, list[str], list[list[str]], NoneType] audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), typing.Sequence[numpy.ndarray], typing.Sequence[ForwardRef('torch.Tensor')], NoneType] = None output\_labels: typing.Optional[bool] = False depth\_decoder\_labels\_ratio: typing.Optional[float] = 1.0 \*\*kwargs: typing\_extensions.Unpack[transformers.models.csm.processing\_csm.CsmProcessorKwargs]  ) → [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

Parameters

* **audio** (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`) —
  The audio or batch of audio to be prepared. Each audio can be a NumPy array or PyTorch
  tensor.
* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **output\_labels** (bool, *optional*, default=False) —
  Whether to return labels for training. Indices will be in `[config.audio_token_id, -100, -101]`.
  + `config.audio_token_id` indicates an audio frame (considering sequence length elements as frames)
  + `-100` will be ignored in the loss computation
  + `-101` indicates the audio frame will be used only for the backbone model (using the first codebook token as labels)
* **depth\_decoder\_labels\_ratio** (float, *optional*, default=1.0) —
  The ratio of audio frames to keep for the depth decoder labels.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors of a particular framework. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return NumPy `np.ndarray` objects.
  + `'jax'`: Return JAX `jnp.ndarray` objects.

Returns

[BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature)

A [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

* **input\_ids** — List of token ids to be fed to a model. Returned when `text` is not `None`.
* **input\_values** — List of audio values to be fed to a model. Returned when `audio` is not `None`.
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names` and if `text` is not
  `None`).
* **labels** — List of labels for the audio frames. Returned when `output_labels=True`.

Main method to prepare text(s) and audio to be fed as input to the model. This method forwards the `text`
arguments to PreTrainedTokenizerFast’s [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) to encode
the text. To prepare the audio, this method forwards the `audio` arguments to
EncodecFeatureExtractor’s [**call**()](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor.__call__). Please refer
to the docstring of the above two methods for more information.

## CsmForConditionalGeneration

### class transformers.CsmForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L745)

( config  )

Parameters

* **config** ([CsmForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmForConditionalGeneration)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Csm model consists of two llama-like auto-regressive transformer models: a backbone model that predicts the first codebook token and a depth decoder that predicts the other codebook tokens.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L924)

( input\_ids: LongTensor = None input\_values: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None input\_values\_cutoffs: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.csm.modeling_csm.CsmOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks) or (batch_size, sequence_length)`) —
  1. (batch\_size, sequence\_length): corresponds to the input sequence prepared with the processor from the text prompt. Such input
     requires `input_values` to be provided so that audio can be encoded in codebook tokens and then merged with the text tokens.
  2. (batch\_size, sequence\_length, num\_codebooks): codebook tokens generated during the autoregressive decoding. Such input is not meant to be used by end users.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **input\_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See `processor_class.__call__` for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **input\_values\_cutoffs** (`torch.Tensor` of shape `(batch_size, max_num_audio)`, *optional*) —
  Specify the end positions of audio segments within each batch entry, relative to the concatenated audio input.
  If a batch entry has fewer segments than the maximum, it is padded with -1. For example, in a batch of 2 sequences
  where the first contains 2 audio segments of length l1, and the second contains 1 audio segment of length l2,
  the input\_values\_cutoffs would be: [[l1, 2 \* l1], [l2, -1]].
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
  Labels for computing the masked language modeling loss. Indices should be in `[config.audio_token_id, -100, -101]`.
  Requires targeted `input_values` to be provided as audio tokens will be inferred from it using the `codec_model`.
  + `config.audio_token_id` indicates an audio frames (considering sequence length elements as frames)
  + `-100` will be ignored in the loss computation
  + `-101` indicates the audio frame will be used only for the backbone model (using the first codebook token as labels)

  Such labels can be prepared using `output_labels=True` when calling [CsmProcessor](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmProcessor).
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **logits\_to\_keep** (`int` or `torch.Tensor`, *optional*) —
  Kept for compatibility. Does not support another value than:
  1. `0`, which is equivalent to keeping all logits, used in the training regime
  2. `1`, which is equivalent to keeping only the last logit, used in the generation regime

Returns

`transformers.models.csm.modeling_csm.CsmOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.csm.modeling_csm.CsmOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CsmConfig](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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
* **depth\_decoder\_loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction) of the depth decoder model.
* **depth\_decoder\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the depth decoder (scores for each vocabulary token before SoftMax).
* **depth\_decoder\_past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
* **depth\_decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **depth\_decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
* **backbone\_loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction) of the backbone model.

The [CsmForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import CsmForConditionalGeneration, AutoProcessor
>>> from datasets import load_dataset, Audio

>>> model_id = "sesame/csm-1b"
>>> torch_device = "cuda" if torch.cuda.is_available() else "cpu"

>>> processor = AutoProcessor.from_pretrained(model_id)

>>> ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
>>> # ensure the audio is 24kHz
>>> ds = ds.cast_column("audio", Audio(sampling_rate=24000))

>>> conversation = []
>>> # prepare a conversation with text and corresponding audio
>>> for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
...     conversation.append(
...         {
...             "role": f"{speaker_id}",
...             "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
...         }
...     )

>>> inputs = processor.apply_chat_template(
...     conversation,
...     tokenize=True,
...     return_dict=True,
...     output_labels=True,
... ).to(torch_device)

>>> model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)
>>> output = model(**inputs)
>>> output.loss.backward()
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/generation_csm.py#L338)

( input\_ids: typing.Optional[torch.Tensor] = None input\_values: typing.Optional[torch.Tensor] = None input\_values\_cutoffs: typing.Optional[torch.Tensor] = None generation\_config: typing.Optional[transformers.generation.configuration\_utils.GenerationConfig] = None logits\_processor: typing.Optional[transformers.generation.logits\_process.LogitsProcessorList] = None stopping\_criteria: typing.Optional[transformers.generation.stopping\_criteria.StoppingCriteriaList] = None synced\_gpus: typing.Optional[bool] = None streamer: typing.Optional[ForwardRef('BaseStreamer')] = None output\_audio: typing.Optional[bool] = False \*\*kwargs  ) → `CsmGenerateOutput` or `torch.LongTensor` or `list[torch.FloatTensor]`

Parameters

* **inputs\_ids** (`torch.Tensor` of shape (batch\_size, seq\_length), *optional*) —
  The sequence used as a prompt for the backbone model.
* **input\_values** (`torch.Tensor` of shape (batch\_size, channels, max\_concatenated\_audio\_length), *optional*) —
  The batched audio input values, where each batch entry contains the concatenation of all audio segments for that entry.
  These values will be encoded into codebook tokens using the codec model and merged with the text input ids provided in `input_ids`.
* **input\_values\_cutoffs** (`torch.Tensor` of shape (batch\_size, max\_num\_audio), *optional*) —
  Specify the end positions of audio segments within each batch entry, relative to the concatenated audio input.
  If a batch entry has fewer segments than the maximum, it is padded with -1. For example, in a batch of 2 sequences
  where the first contains 2 audio segments of length l1, and the second contains 1 audio segment of length l2,
  the input\_values\_cutoffs would be: [[l1, 2 \* l1], [l2, -1]].
* **generation\_config** ([GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig), *optional*) —
  The generation configuration to be used as base parametrization for the generation call. `**kwargs`
  passed to generate matching the attributes of `generation_config` will override them. If
  `generation_config` is not provided, the default will be used, which has the following loading
  priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
  configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)’s
  default values, whose documentation should be checked to parameterize generation.
* **logits\_processor** (`LogitsProcessorList`, *optional*) —
  Custom logits processors that complement the default logits processors built from arguments and
  generation config. If a logit processor is passed that is already created with the arguments or a
  generation config an error is thrown. This feature is intended for advanced users.
* **stopping\_criteria** (`StoppingCriteriaList`, *optional*) —
  Custom stopping criteria that complements the default stopping criteria built from arguments and a
  generation config. If a stopping criteria is passed that is already created with the arguments or a
  generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
  sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
  intended for advanced users.
* **synced\_gpus** (`bool`, *optional*) —
  Whether to continue running the while loop until max\_length. Unless overridden, this flag will be set
  to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
  deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
* **streamer** (`BaseStreamer`, *optional*) —
  Streamer object that will be used to stream the generated sequences. Generated tokens are passed
  through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
* **output\_audio** (`bool`, *optional*) —
  Whether to return the generated audio.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
  forwarded to the `forward` function of the model. Depth decoder specific kwargs should be prefixed with *depth*decoder**.

Returns

`CsmGenerateOutput` or `torch.LongTensor` or `list[torch.FloatTensor]`

A `CsmGenerateOutput`
(if `return_dict_in_generate=True` or when `config.return_dict_in_generate=True`) or a `torch.LongTensor` when `output_audio=False`
or a `list[torch.FloatTensor]` otherwise.

This method overrides [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) to match the specifics of the Csm model.
Indeed, Csm model requires a custom generation sampling step:

1. Infer the backbone model to sample the first codebook token
2. Call generate on the depth decoder with the first codebook token as `input_ids` to sample the next codebook tokens
3. Use these generated codebook tokens as `input_ids` to sample the next first codebook token using the backbone model
4. Repeat until stopping criteria is met

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
model’s default generation configuration. You can override any `generation_config` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, do_sample=True)`.

Example:


```
>>> from transformers import CsmProcessor, CsmForConditionalGeneration
>>> from datasets import load_dataset, Audio

>>> model_id = "sesame/csm-1b"
>>> torch_device = "cuda" if torch.cuda.is_available() else "cpu"

>>> processor = AutoProcessor.from_pretrained(model_id)

>>> ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
>>> # ensure the audio is 24kHz
>>> ds = ds.cast_column("audio", Audio(sampling_rate=24000))

>>> conversation = []
>>> # prepare a conversation with text and corresponding audio
>>> for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
...     conversation.append(
...         {
...             "role": f"{speaker_id}",
...             "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
...         }
...     )

>>> # text prompt
>>> conversation.append({"role": f"{ds[4]['speaker_id']}", "content": [{"type": "text", "text": ds[4]["text"]}]})

>>> inputs = processor.apply_chat_template(
...     conversation,
...     tokenize=True,
...     return_dict=True,
... ).to(torch_device)

>>> model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)
>>> audio = model.generate(**inputs, output_audio=True)
>>> processor.save_audio(audio, "output.wav")
```

## CsmDepthDecoderForCausalLM

### class transformers.CsmDepthDecoderForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L531)

( config  )

Parameters

* **config** ([CsmDepthDecoderForCausalLM](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmDepthDecoderForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CsmDepthDecoder Model transformer, with a `CsmCodebooksHead` on top,
which can be seen a position-specific language modeling head, allowing to use a different linear layer for each codebook
(e.g. position 0 is the first codebook and uses the first codebook head, etc.)

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L545)

( input\_ids: LongTensor = None backbone\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[transformers.cache\_utils.Cache, list[torch.FloatTensor], NoneType] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **backbone\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, backbone_hidden_size)`, *optional*) —
  The last hidden state of the backbone model. Such input is required when the first codebook token (the one generated by the backbone model)
  is provided in the `input_ids` argument.
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
elements depending on the configuration ([CsmConfig](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmConfig)) and inputs.

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

The [CsmDepthDecoderForCausalLM](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmDepthDecoderForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## CsmDepthDecoderModel

### class transformers.CsmDepthDecoderModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L395)

( config  )

Parameters

* **config** ([CsmDepthDecoderModel](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmDepthDecoderModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Csm Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L414)

( input\_ids: LongTensor = None backbone\_last\_hidden\_state: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **backbone\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, backbone_hidden_size)`, *optional*) —
  The last hidden state of the backbone model. Such input is required when the first codebook token (the one generated by the backbone model)
  is provided in the `input_ids` argument.
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
elements depending on the configuration ([CsmConfig](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmConfig)) and inputs.

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

The [CsmDepthDecoderModel](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmDepthDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## CsmBackboneModel

### class transformers.CsmBackboneModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L651)

( config  )

Parameters

* **config** ([CsmBackboneModel](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmBackboneModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Csm Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L667)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks) or (batch_size, sequence_length)`) —
  1. (batch\_size, sequence\_length): corresponds to the input sequence prepared with the processor from the text prompt. Such input
     requires `input_values` to be provided so that audio can be encoded in codebook tokens and then merged with the text tokens.
  2. (batch\_size, sequence\_length, num\_codebooks): codebook tokens generated during the autoregressive decoding. Such input is not meant to be used by end users.

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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CsmConfig](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmConfig)) and inputs.

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

The [CsmBackboneModel](/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmBackboneModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/csm.md)
