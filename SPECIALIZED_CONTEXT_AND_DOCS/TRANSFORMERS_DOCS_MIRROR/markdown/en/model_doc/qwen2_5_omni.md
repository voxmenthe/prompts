*This model was released on 2025-03-26 and added to Hugging Face Transformers on 2025-04-14.*

# Qwen2.5-Omni

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The [Qwen2.5-Omni](https://qwenlm.github.io/blog/qwen2.5-omni/) model is a unified multiple modalities model proposed in [Qwen2.5-Omni Technical Report](https://huggingface.co/papers/2503.20215) from Qwen team, Alibaba Group.

The abstract from the technical report is the following:

*We present Qwen2.5-Omni, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. To enable the streaming of multimodal information inputs, both audio and visual encoders utilize a block-wise processing approach. This strategy effectively decouples the handling of long sequences of multimodal data, assigning the perceptual responsibilities to the multimodal encoder and entrusting the modeling of extended sequences to a large language model. Such a division of labor enhances the fusion of different modalities via the shared attention mechanism. To synchronize the timestamps of video inputs with audio, we organized the audio and video sequentially in an interleaved manner and propose a novel position embedding approach, named TMRoPE (Time-aligned Multimodal RoPE). To concurrently generate text and speech while avoiding interference between the two modalities, we propose Thinker-Talker architecture. In this framework, Thinker functions as a large language model tasked with text generation, while Talker is a dual-track autoregressive model that directly utilizes the hidden representations from the Thinker to produce audio tokens as output. Both the Thinker and Talker models are designed to be trained and inferred in an end-to-end manner. For decoding audio tokens in a streaming manner, we introduce a sliding-window DiT that restricts the receptive field, aiming to reduce the initial package delay. Qwen2.5-Omni outperforms the similarly sized Qwen2-VL and Qwen2-Audio in both image and audio capabilities. Furthermore, Qwen2.5-Omni achieves state-of-the-art performance on multimodal benchmarks like Omni-Bench. Notably, Qwen2.5-Omni is the first open-source model to achieve a level of performance in end-to-end speech instruction following that is comparable to its capabilities with text inputs, as evidenced by benchmarks such as MMLU and GSM8K. As for speech generation, Qwen2.5-Omni’s streaming Talker outperform most existing streaming and non-streaming alternatives in robustness and naturalness.*

## Notes

* Use [Qwen2\_5OmniForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniForConditionalGeneration) to generate audio and text output. To generate only one output type, use [Qwen2\_5OmniThinkerForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniThinkerForConditionalGeneration) for text-only and `Qwen2_5OmniTalkersForConditionalGeneration` for audio-only outputs.
* Audio generation with [Qwen2\_5OmniForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniForConditionalGeneration) supports only single batch size at the moment.
* In case out out-of-memory errors hwen working with video input, decrease `processor.max_pixels`. By default the maximum is set to a very arge value and high resolution visuals will not be resized, unless resolution exceeds `processor.max_pixels`.
* The processor has its own [apply\_chat\_template()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template) method to convert chat messages to model inputs.

## Usage example

`Qwen2.5-Omni` can be found on the [Huggingface Hub](https://huggingface.co/Qwen).

### Single Media inference

The model can accept text, images, audio and videos as input. Here’s an example code for inference.


```
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    dtype="auto",
    device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

conversations = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "text", "text": "What cant you hear and see in this video?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversations,
    load_audio_from_video=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    video_fps=1,

    # kwargs to be passed to `Qwen2-5-OmniProcessor`
    padding=True,
    use_audio_in_video=True,
).to(model.device)

# Generation params for audio or text can be different and have to be prefixed with `thinker_` or `talker_`
text_ids, audio = model.generate(**inputs, use_audio_in_video=True, thinker_do_sample=False, talker_do_sample=True)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
print(text)
```

### Text-only generation

To generate only text output and save compute by not loading the audio generation model, we can use `Qwen2_5OmniThinkerForConditionalGeneration` model.


```
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    dtype="auto",
    device_map="auto",
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

conversations = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "text", "text": "What cant you hear and see in this video?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversations,
    load_audio_from_video=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    video_fps=1,

    # kwargs to be passed to `Qwen2-5-OmniProcessor`
    padding=True,
    use_audio_in_video=True,
).to(model.device)


text_ids = model.generate(**inputs, use_audio_in_video=True)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
print(text)
```

### Batch Mixed Media Inference

The model can batch inputs composed of mixed samples of various types such as text, images, audio and videos as input when using `Qwen2_5OmniThinkerForConditionalGeneration` model. Here is an example.


```
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    dtype="auto",
    device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# Conversation with video only
conversation1 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
        ]
    }
]

# Conversation with audio only
conversation2 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "path": "/path/to/audio.wav"},
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "who are you?"}],
    }
]


# Conversation with mixed media
conversation4 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "audio", "path": "/path/to/audio.wav"},
            {"type": "text", "text": "What are the elements can you see and hear in these medias?"},
        ],
    }
]

conversations = [conversation1, conversation2, conversation3, conversation4]

inputs = processor.apply_chat_template(
    conversations,
    load_audio_from_video=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    video_fps=1,

    # kwargs to be passed to `Qwen2-5-OmniProcessor`
    padding=True,
    use_audio_in_video=True,
).to(model.thinker.device)

text_ids = model.generate(**inputs, use_audio_in_video=True)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(text)
```

### Usage Tips

#### Image Resolution trade-off

The model supports a wide range of resolution inputs. By default, it uses the native resolution for input, but higher resolutions can enhance performance at the cost of more computation. Users can set the minimum and maximum number of pixels to achieve an optimal configuration for their needs.


```
min_pixels = 128*28*28
max_pixels = 768*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B", min_pixels=min_pixels, max_pixels=max_pixels)
```

#### Prompt for audio output

If users need audio output, the system prompt must be set as “You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.”, otherwise the audio output may not work as expected.


```
{
    "role": "system",
    "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
}
```

#### Use audio output or not

The model supports both text and audio outputs, if users do not need audio outputs, they can set `enable_audio_output` in the `from_pretrained` function. This option will save about `~2GB` of GPU memory but the `return_audio` option for `generate` function will only allow to be set at `False`.


```
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    dtype="auto",
    device_map="auto",
    enable_audio_output=False,
)
```

In order to obtain a flexible experience, we recommend that users set `enable_audio_output` at `True` when initializing the model through `from_pretrained` function, and then decide whether to return audio when `generate` function is called. When `return_audio` is set to `False`, the model will only return text outputs to get text responses faster.


```
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    dtype="auto",
    device_map="auto",
    enable_audio_output=True,
)
...
text_ids = model.generate(**inputs, return_audio=False)
```

#### Change voice type of output audio

Qwen2.5-Omni supports the ability to change the voice of the output audio. Users can use the `spk` parameter of `generate` function to specify the voice type. The `"Qwen/Qwen2.5-Omni-7B"` checkpoint support two voice types: `Chelsie` and `Ethan`, while `Chelsie` is a female voice and `Ethan` is a male voice. By default, if `spk` is not specified, the default voice type is `Chelsie`.


```
text_ids, audio = model.generate(**inputs, spk="Chelsie")
```


```
text_ids, audio = model.generate(**inputs, spk="Ethan")
```

#### Flash-Attention 2 to speed up generation

First, make sure to install the latest version of Flash Attention 2:


```
pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using FlashAttention-2, add `attn_implementation="flash_attention_2"` when loading the model:


```
from transformers import Qwen2_5OmniForConditionalGeneration

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

## Qwen2\_5OmniConfig

### class transformers.Qwen2\_5OmniConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py#L995)

( thinker\_config = None talker\_config = None token2wav\_config = None enable\_audio\_output: bool = True \*\*kwargs  )

Parameters

* **thinker\_config** (`dict`, *optional*) — Configuration of the underlying thinker sub-model.
* **talker\_config** (`dict`, *optional*) — Configuration of the underlying talker sub-model.
* **token2wav\_config** (`dict`, *optional*) — Configuration of the underlying codec sub-model.
* **enable\_audio\_output** (`bool`, *optional*, defaults to `True`) — Whether enable audio output and load talker and token2wav module.

This is the configuration class to store the configuration of a [Qwen2\_5OmniForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniForConditionalGeneration). It is used to instantiate a Qwen2.5Omni
model according to the specified sub-models configurations, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the
[Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import (
...     Qwen2_5OmniThinkerConfig,
...     Qwen2_5OmniTalkerConfig,
...     Qwen2_5OmniToken2WavConfig,
...     Qwen2_5OmniForConditionalGeneration,
...     Qwen2_5OmniConfig,
... )

>>> # Initializing sub-modules configurations.
>>> thinker_config = Qwen2_5OmniThinkerConfig()
>>> talker_config = Qwen2_5OmniTalkerConfig()
>>> token2wav_config = Qwen2_5OmniToken2WavConfig()


>>> # Initializing a module style configuration
>>> configuration = Qwen2_5OmniConfig.from_sub_model_configs(
...     thinker_config, talker_config, token2wav_config
... )

>>> # Initializing a model (with random weights)
>>> model = Qwen2_5OmniForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

#### get\_text\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py#L1076)

( \*args \*\*kwargs  )

Parameters

* **decoder** (`Optional[bool]`, *optional*, defaults to `False`) —
  If set to `True`, then only search for decoder config names.

Returns the config that is meant to be used with text IO. On most models, it is the original config instance
itself. On specific composite models, it is under a set of valid names.

## Qwen2\_5OmniProcessor

### class transformers.Qwen2\_5OmniProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py#L76)

( image\_processor = None video\_processor = None feature\_extractor = None tokenizer = None chat\_template = None  )

Parameters

* **image\_processor** ([Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), *optional*) —
  The image processor.
* **video\_processor** ([Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor), *optional*) —
  The video processor.
* **feature\_extractor** ([WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor), *optional*) —
  The audio feature extractor.
* **tokenizer** ([Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast), *optional*) —
  The text tokenizer.
* **chat\_template** (`Optional[str]`, *optional*) —
  The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.

Constructs a Qwen2.5Omni processor.
[Qwen2\_5OmniProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniProcessor) offers all the functionalities of [Qwen2VLImageProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor), and [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### get\_chunked\_index

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py#L288)

( token\_indices: ndarray tokens\_per\_chunk: int  ) → `list[tuple[int, int]]`

Parameters

* **token\_indices** (`np.ndarray`) — A monotonically increasing list of token index values.
* **t\_ntoken\_per\_chunk** (`int`) — Number of tokens per chunk (used as the chunk size threshold).

Returns

`list[tuple[int, int]]`

A list of tuples, each representing the start (inclusive)
and end (exclusive) indices of a chunk in `token_indices`.

Splits token index list into chunks based on token value ranges.

Given a list of token indices, returns a list of (start, end) index tuples representing
slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:

* the first chunk contains token values < 1000,
* the second chunk contains values >= 1000 and < 2000, and so on.

## Qwen2\_5OmniForConditionalGeneration

### class transformers.Qwen2\_5OmniForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L3696)

( config  )

Parameters

* **config** ([Qwen2\_5OmniForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniForConditionalGeneration)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The full Qwen2.5Omni model, a multimodal model composed of 3 sub-models:

* [Qwen2\_5OmniThinkerForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniThinkerForConditionalGeneration):
  a causal auto-regressive transformer takes text, audio, image, video as input and predict text tokens.
* [Qwen2\_5OmniTalkerForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniTalkerForConditionalGeneration):
  a causal auto-regressive transformer takes thinker hidden states and response as input and predict speech tokens.
* [Qwen2\_5OmniToken2WavModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniToken2WavModel):
  a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### \_forward\_unimplemented

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/torch/nn/modules/module.py#L388)

( \*input: typing.Any  )

Define the computation performed at every call.

Should be overridden by all subclasses.

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

## Qwen2\_5OmniPreTrainedModelForConditionalGeneration

### class transformers.Qwen2\_5OmniPreTrainedModelForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L76)

( config: PretrainedConfig \*inputs \*\*kwargs  )

#### get\_chunked\_index

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L152)

( token\_indices: Tensor tokens\_per\_chunk: int remove\_index: int  ) → `list[tuple[int, int]]`

Parameters

* **token\_indices** (`torch.Tensor` of shape `(seq_len, )`) — A monotonically increasing list of
  token index values.
* **t\_ntoken\_per\_chunk** (`int`) — Number of tokens per chunk (used as the chunk size threshold).
* **remove\_index** (`int`) An index id to subtract from `token_indices` before chunking —

Returns

`list[tuple[int, int]]`

A list of tuples, each representing the start (inclusive)
and end (exclusive) indices of a chunk in `token_indices`.

Splits token index list into chunks based on token value ranges.

Given a list of token indices, returns a list of (start, end) index tuples representing
slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:

* the first chunk contains token values < 1000,
* the second chunk contains values >= 1000 and < 2000, and so on.

#### get\_rope\_index

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L189)

( input\_ids: typing.Optional[torch.LongTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None use\_audio\_in\_video: bool = False audio\_seqlens: typing.Optional[torch.LongTensor] = None second\_per\_grids: typing.Optional[torch.Tensor] = None  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
  it.
* **image\_grid\_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) —
  The temporal, height and width of feature shape of each image in LLM.
* **video\_grid\_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) —
  The temporal, height and width of feature shape of each video in LLM.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **use\_audio\_in\_video** (`bool`, *optional*) —
  If set to `True`, use the audio in video.
* **audio\_seqlens** (`torch.LongTensor` of shape `(num_audios)`, *optional*) —
  The length of feature shape of each audio in LLM.
* **second\_per\_grids** (`torch.LongTensor` of shape `(num_videos)`, *optional*) —
  The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

Calculate the 3D rope index based on image and video’s temporal, height and width in LLM.

Explanation:
Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
Examples:
input\_ids: [T T T T T], here T is for text.
temporal position\_ids: [0, 1, 2, 3, 4]
height position\_ids: [0, 1, 2, 3, 4]
width position\_ids: [0, 1, 2, 3, 4]

For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
and 1D rotary position embedding for text part.
Examples:
Temporal (Time): 3 patches, representing different segments of the video in time.
Height: 2 patches, dividing each frame vertically.
Width: 2 patches, dividing each frame horizontally.
We also have some important parameters:
fps (Frames Per Second): The video’s frame rate, set to 1. This means one frame is processed each second.
tokens\_per\_second: This is a crucial parameter. It dictates how many “time-steps” or “temporal tokens” are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
temporal\_patch\_size: The number of frames that compose one temporal patch. Here, it’s 2 frames.
interval: The step size for the temporal position IDs, calculated as tokens\_per\_second *temporal\_patch\_size / fps. In this case, 25* 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
input\_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
vision temporal position\_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
vision height position\_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
vision width position\_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
text temporal position\_ids: [101, 102, 103, 104, 105]
text height position\_ids: [101, 102, 103, 104, 105]
text width position\_ids: [101, 102, 103, 104, 105]
Here we calculate the text start position\_ids as the max vision position\_ids plus 1.

## Qwen2\_5OmniThinkerConfig

### class transformers.Qwen2\_5OmniThinkerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py#L413)

( audio\_config = None vision\_config = None text\_config = None audio\_token\_index = 151646 image\_token\_index = 151655 video\_token\_index = 151656 position\_id\_per\_seconds = 25 seconds\_per\_chunk = 2 audio\_start\_token\_id = 151647 audio\_end\_token\_id = 151648 user\_token\_id = 872 initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **audio\_config** (`dict`, *optional*) —
  The config dictionary of the audio backbone.
* **vision\_config** (`dict`, *optional*) —
  The config dictionary of the vision backbone.
* **text\_config** (`dict`, *optional*) —
  The config dictionary of the text backbone.
* **audio\_token\_index** (`int`, *optional*, defaults to 151646) —
  The audio token index to encode the audio prompt.
* **image\_token\_index** (`int`, *optional*, defaults to 151655) —
  The image token index to encode the image prompt.
* **video\_token\_index** (`int`, *optional*, defaults to 151656) —
  The video token index to encode the video prompt.
* **position\_id\_per\_seconds** (`int`, *optional*, defaults to 25) —
  The increment of position id per second.
* **seconds\_per\_chunk** (`int`, *optional*, defaults to 2) —
  The duration in seconds of the chunk of audio and video data.
* **audio\_start\_token\_id** (`int`, *optional*, defaults to 151647) —
  The audio start token index to encode the audio prompt.
* **audio\_end\_token\_id** (`int`, *optional*, defaults to 151648) —
  The audio end token index to encode the audio prompt.
* **user\_token\_id** (`int, *optional*, defaults to 872) —
  The user token index to encode the user token.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a [Qwen2\_5OmniThinkerForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniThinkerForConditionalGeneration). It is used to instantiate an
Qwen2.5-Omni-Thinker model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Qwen2.5-Omni-Thinker.

e.g. [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniThinkerConfig, Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniVisionEncoderConfig

>>> # Initializing a Qwen2_5OmniAudioEncoder config
>>> audio_config = Qwen2_5OmniAudioEncoderConfig()

>>> # Initializing a Qwen2_5OmniVisionEncoder config
>>> vision_config = Qwen2_5OmniVisionEncoderConfig()

>>> # Initializing a Qwen2_5OmniTextConfig config
>>> text_config = Qwen2_5OmniTextConfig()

>>> # Initializing a Qwen2.5OmniThinker configuration
>>> configuration = Qwen2_5OmniThinkerConfig(audio_config, vision_config, text_config)

>>> # Initializing a model from the Qwen-Omni style configuration
>>> model = Qwen2_5OmniThinkerForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen2\_5OmniThinkerForConditionalGeneration

### class transformers.Qwen2\_5OmniThinkerForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1663)

( config: Qwen2\_5OmniThinkerConfig  )

Parameters

* **config** ([Qwen2\_5OmniThinkerConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniThinkerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Qwen2.5OmniThinker model which consists of a audio backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1804)

( input\_ids: typing.Optional[torch.LongTensor] = None input\_features: typing.Optional[torch.FloatTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None pixel\_values\_videos: typing.Optional[torch.FloatTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None feature\_attention\_mask: typing.Optional[torch.Tensor] = None audio\_feature\_lengths: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None rope\_deltas: typing.Optional[torch.LongTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None use\_audio\_in\_video: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None video\_second\_per\_grid: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniThinkerCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  `feature_extractor_class`. See `feature_extractor_class.__call__` for details ([Qwen2\_5OmniProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniProcessor) uses
  `feature_extractor_class` for processing audios).
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  `image_processor_class`. See `image_processor_class.__call__` for details ([Qwen2\_5OmniProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniProcessor) uses
  `image_processor_class` for processing images).
* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) —
  The tensors corresponding to the input video. Pixel values for videos can be obtained using
  [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor). See `Qwen2VLVideoProcessor.__call__()` for details ([Qwen2\_5OmniProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniProcessor) uses
  [Qwen2VLVideoProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLVideoProcessor) for processing videos).
* **image\_grid\_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) —
  The temporal, height and width of feature shape of each image in LLM.
* **video\_grid\_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) —
  The temporal, height and width of feature shape of each video in LLM.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **feature\_attention\_mask** (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **audio\_feature\_lengths** (`torch.LongTensor` of shape `(num_audios)`, *optional*) —
  The length of feature shape of each audio in LLM.
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
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) —
  The rope index difference between sequence length and multimodal rope.
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
* **use\_audio\_in\_video** (`bool`, *optional*) —
  Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **video\_second\_per\_grid** (`torch.LongTensor` of shape `(num_videos)`, *optional*) —
  Number of seconds per grid for each video, used for temporal feature mapping.

Returns

`transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniThinkerCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniThinkerCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2\_5OmniConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) — The rope index difference between sequence length and multimodal rope.

The [Qwen2\_5OmniThinkerForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniThinkerForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from io import BytesIO
>>> from urllib.request import urlopen
>>> import librosa
>>> from qwen_vl_utils import process_vision_info
>>> from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

>>> thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
>>> processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

>>> conversations = [
>>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
>>>         {"role": "user", "content": [
>>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
>>>             {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
>>>         ]},
>>> ]

>>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
>>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
>>> images, videos = process_vision_info(conversations)
>>> inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

>>> # Generate
>>> inputs['use_audio_in_video'] = `True` or `False`
>>> generation = thinker.generate(**inputs, max_new_tokens=2048)
>>> generate_ids = generation[:, inputs.input_ids.size(1):]

>>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

#### get\_audio\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1717)

( input\_features: FloatTensor feature\_attention\_mask: typing.Optional[torch.LongTensor] = None audio\_feature\_lengths: typing.Optional[torch.LongTensor] = None  )

Parameters

* **input\_features** (`torch.FloatTensor`) —
  The tensors corresponding to the input audios.
* **feature\_attention\_mask** (`torch.LongTensor`, *optional*) —
  Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
* **audio\_feature\_lengths** (`torch.LongTensor` of shape `(num_audios)`, *optional*) —
  The length of feature shape of each audio in LLM.

Encodes audios into continuous embeddings that can be forwarded to the language model.

#### get\_image\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1703)

( pixel\_values: FloatTensor image\_grid\_thw: typing.Optional[torch.LongTensor] = None  )

Parameters

* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images.
* **image\_grid\_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) —
  The temporal, height and width of feature shape of each image in LLM.

Encodes images into continuous embeddings that can be forwarded to the language model.

#### get\_placeholder\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1756)

( input\_ids: LongTensor inputs\_embeds: FloatTensor image\_features: FloatTensor = None video\_features: FloatTensor = None  )

Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.

#### get\_video\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1687)

( pixel\_values\_videos: FloatTensor video\_grid\_thw: typing.Optional[torch.LongTensor] = None  )

Parameters

* **pixel\_values\_videos** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input videos.
* **video\_grid\_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) —
  The temporal, height and width of feature shape of each video in LLM.

Encodes videos into continuous embeddings that can be forwarded to the language model.

## Qwen2\_5OmniThinkerTextModel

### class transformers.Qwen2\_5OmniThinkerTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1501)

( config: Qwen2\_5OmniTextConfig  )

Parameters

* **config** (`Qwen2_5OmniTextConfig`) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Qwen2 5 Omni Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1523)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

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

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2\_5OmniConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniConfig)) and inputs.

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

The [Qwen2\_5OmniThinkerTextModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniThinkerTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Qwen2\_5OmniTalkerConfig

### class transformers.Qwen2\_5OmniTalkerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py#L533)

( audio\_token\_index = 151646 image\_token\_index = 151655 video\_token\_index = 151656 vocab\_size = 8448 tts\_text\_start\_token\_id = 151860 tts\_text\_end\_token\_id = 151861 tts\_text\_pad\_token\_id = 151859 tts\_codec\_start\_token\_id = 8293 tts\_codec\_end\_token\_id = 8294 tts\_codec\_pad\_token\_id = 8292 tts\_codec\_mask\_token\_id = 8296 vision\_start\_token\_id = 151652 vision\_end\_token\_id = 151653 embedding\_size = 3584 hidden\_size = 3584 intermediate\_size = 18944 num\_hidden\_layers = 28 num\_attention\_heads = 28 num\_key\_value\_heads = 4 hidden\_act = 'silu' max\_position\_embeddings = 32768 rms\_norm\_eps = 1e-06 head\_dim = 128 use\_cache = True tie\_word\_embeddings = False rope\_theta = 1000000.0 use\_sliding\_window = False sliding\_window = 32768 max\_window\_layers = 28 attention\_dropout = 0.0 rope\_scaling = None position\_id\_per\_seconds = 25 seconds\_per\_chunk = 2 audio\_start\_token\_id = 151647 audio\_end\_token\_id = 151648 initializer\_range = 0.02 spatial\_merge\_size = 2 layer\_types = None \*\*kwargs  )

Parameters

* **audio\_token\_index** (`int`, *optional*, defaults to 151646) —
  The audio token index to encode the audio prompt.
* **image\_token\_index** (`int`, *optional*, defaults to 151655) —
  The image token index to encode the image prompt.
* **video\_token\_index** (`int`, *optional*, defaults to 151656) —
  The video token index to encode the video prompt.
* **vocab\_size** (`int`, *optional*, defaults to 8448) —
  Vocabulary size of the QwenOmni model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [Qwen2VLModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLModel)
* **tts\_text\_start\_token\_id** (`int`, *optional*, defaults to 151860) —
  The tts text start token index to encode the start of tts text.
* **tts\_text\_end\_token\_id** (`int`, *optional*, defaults to 151861) —
  The tts text end token index to encode the end of tts text.
* **tts\_text\_pad\_token\_id** (`int`, *optional*, defaults to 151859) —
  The tts text pad token index to encode the pad of tts text.
* **tts\_codec\_start\_token\_id** (`int`, *optional*, defaults to 8293) —
  The tts codec start token index to encode the start of tts codec.
* **tts\_codec\_end\_token\_id** (`int`, *optional*, defaults to 8294) —
  The tts codec end token index to encode the end of tts codec.
* **tts\_codec\_pad\_token\_id** (`int`, *optional*, defaults to 8292) —
  The tts codec pad token index to encode the pad of tts codec.
* **tts\_codec\_mask\_token\_id** (`int`, *optional*, defaults to 8296) —
  The tts codec mask token index to encode the mask of tts codec.
* **vision\_start\_token\_id** (`int`, *optional*, defaults to 151652) —
  The tts vision start token index to encode the start of vision.
* **vision\_end\_token\_id** (`int`, *optional*, defaults to 151653) —
  The tts vision end token index to encode the end of vision.
* **embedding\_size** (`int`, *optional*, defaults to 3584) —
  Dimension of the embedding representations.
* **hidden\_size** (`int`, *optional*, defaults to 3584) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 18944) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 28) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 28) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 4) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details, check out [this
  paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 32768) —
  The maximum sequence length that this model might ever be used with.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the rms normalization layers.
* **head\_dim** (`int`, *optional*, defaults to 128) —
  The dimension of each attention head.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether the model’s input and output word embeddings should be tied.
* **rope\_theta** (`float`, *optional*, defaults to 1000000.0) —
  The base period of the RoPE embeddings.
* **use\_sliding\_window** (`bool`, *optional*, defaults to `False`) —
  Whether to use sliding window attention.
* **sliding\_window** (`int`, *optional*, defaults to 32768) —
  Sliding window attention (SWA) window size. If not specified, will default to `4096`.
* **max\_window\_layers** (`int`, *optional*, defaults to 28) —
  The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
  additional layer afterwards will use SWA (Sliding Window Attention).
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
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
* **position\_id\_per\_seconds** (`int`, *optional*, defaults to 25) —
  The increment of position id per second.
* **seconds\_per\_chunk** (`int`, *optional*, defaults to 2) —
  The duration in seconds of the chunk of audio and video data.
* **audio\_start\_token\_id** (`int`, *optional*, defaults to 151647) —
  The audio start token index to encode the audio prompt.
* **audio\_end\_token\_id** (`int`, *optional*, defaults to 151648) —
  The audio end token index to encode the audio prompt.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **spatial\_merge\_size** (`int`, *optional*, defaults to 2) —
  The size used for merging spatial dimensions.
* **layer\_types** (`list`, *optional*) —
  Attention pattern for each layer.

This is the configuration class to store the configuration of a [Qwen2\_5OmniTalkerForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniTalkerForConditionalGeneration). It is used to instantiate an
Qwen2.5-Omni-Talker model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Qwen2.5-Omni-Thinker.

e.g. [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Qwen2_5OmniTalkerForConditionalGeneration, Qwen2_5OmniThinkerConfig, Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniVisionEncoderConfig

>>> # Initializing a Qwen2_5OmniAudioEncoder config
>>> audio_config = Qwen2_5OmniAudioEncoderConfig()

>>> # Initializing a Qwen2 config
>>> text_config = Qwen2Config()

>>> # Initializing a Qwen2_5Omni configuration
>>> configuration = Qwen2_5OmniThinkerConfig(audio_config, text_config)

>>> # Initializing a model from the qwen2-audio style configuration
>>> model = Qwen2_5OmniTalkerForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen2\_5OmniTalkerForConditionalGeneration

### class transformers.Qwen2\_5OmniTalkerForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L2233)

( config: Qwen2\_5OmniTalkerConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L2266)

( input\_ids: LongTensor = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None thinker\_reply\_part: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None rope\_deltas: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None input\_text\_ids: typing.Optional[torch.LongTensor] = None image\_grid\_thw: typing.Optional[torch.LongTensor] = None video\_grid\_thw: typing.Optional[torch.LongTensor] = None use\_audio\_in\_video: typing.Optional[bool] = None audio\_feature\_lengths: typing.Optional[torch.LongTensor] = None video\_second\_per\_grid: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniTalkerCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

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
* **thinker\_reply\_part** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Hidden states from the thinker model’s output that represent the text reply part to be processed.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) —
  The rope index difference between sequence length and multimodal rope.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **input\_text\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Input token IDs for text-only content, used for position calculation in multimodal contexts.
* **image\_grid\_thw** (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) —
  The temporal, height and width of feature shape of each image in LLM.
* **video\_grid\_thw** (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) —
  The temporal, height and width of feature shape of each video in LLM.
* **use\_audio\_in\_video** (`bool`, *optional*) —
  Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
* **audio\_feature\_lengths** (`torch.LongTensor` of shape `(num_audios)`, *optional*) —
  The length of feature shape of each audio in LLM.
* **video\_second\_per\_grid** (`torch.LongTensor` of shape `(num_videos)`, *optional*) —
  Number of seconds per grid for each video, used for temporal feature mapping.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniTalkerCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniTalkerCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2\_5OmniConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniConfig)) and inputs.

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
* **rope\_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) — The rope index difference between sequence length and multimodal rope.
* **thinker\_reply\_part** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Hidden states from the thinker model that are used as input for the talker model. These represent the encoded
  response that the talker model will use to generate speech tokens.

The [Qwen2\_5OmniTalkerForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniTalkerForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from io import BytesIO
>>> from urllib.request import urlopen
>>> import librosa
>>> from transformers import AutoProcessor, Qwen2_5OmniTalkerForConditionalGeneration

>>> model = Qwen2_5OmniTalkerForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
>>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")

>>> prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
>>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
>>> audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

>>> inputs = processor(text=prompt, audios=audio, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs, max_length=30)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Generate the caption in English: Glass is breaking."
```

## Qwen2\_5OmniTalkerModel

### class transformers.Qwen2\_5OmniTalkerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L2077)

( config: Qwen2\_5OmniTalkerConfig  )

Parameters

* **config** ([Qwen2\_5OmniTalkerConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniTalkerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Qwen2 5 Omni Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L2098)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

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

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2\_5OmniConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniConfig)) and inputs.

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

The [Qwen2\_5OmniTalkerModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniTalkerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Qwen2\_5OmniToken2WavConfig

### class transformers.Qwen2\_5OmniToken2WavConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py#L936)

( dit\_config = None bigvgan\_config = None \*\*kwargs  )

Parameters

* **dit\_config** (`DiT_Args`, *optional*) —
  Configuration class for the Diffusion Transformer (DiT) module responsible for generating mel-spectrograms.
* **bigvgan\_config** (`BigVGAN_Args`, *optional*) —
  Configuration class for the BigVGAN module responsible for converting mel-spectrograms to waveforms.

This is the configuration class to store the configuration of a [Qwen2\_5OmniToken2WavModel](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniToken2WavModel).
It is used to instantiate the Qwen2.5-Omni-Token2Wav model which combines a Diffusion Transformer (DiT) for mel-spectrogram generation with a BigVGAN model for waveform synthesis. The configuration contains sub-configurations for both components.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Qwen2_5OmniToken2WavModel, DiT_Args, BigVGAN_Args

>>> # Initialize DiT configuration
>>> dit_config = DiT_Args(
...     dim=1024,
...     depth=22,
...     heads=16,
...     ff_mult=2
... )

>>> # Initialize BigVGAN configuration
>>> bigvgan_config = BigVGAN_Args(
...     mel_dim=80,
...     upsample_rates=[5,3,2,2,2,2]
... )

>>> # Initialize main configuration
>>> config = Qwen2_5OmniToken2WavConfig(dit_config, bigvgan_config)

>>> # Initialize model with config
>>> model = Qwen2_5OmniToken2Wav(config)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen2\_5OmniToken2WavModel

### class transformers.Qwen2\_5OmniToken2WavModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L3628)

( config: Qwen2\_5OmniToken2WavConfig  )

Parameters

* **config** ([Qwen2\_5OmniToken2WavConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_5_omni#transformers.Qwen2_5OmniToken2WavConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The full Qwen2.5Omni Token2Wav model. Consists a DiT model take speech tokens as input and predict mel spectrogram and a BigVGAN vocoder take mel spectrogram as input and predict waveform.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L3654)

( code conditioning reference\_mel num\_steps = 10 guidance\_scale = 0.5 sway\_coefficient = -1.0 \*\*kwargs  )

Generates a waveform from input code and conditioning parameters.

## Qwen2\_5OmniToken2WavDiTModel

### class transformers.Qwen2\_5OmniToken2WavDiTModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L3473)

( config: Qwen2\_5OmniDiTConfig  )

Parameters

* **config** (`Qwen2_5OmniDiTConfig`) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The full Qwen2.5Omni Token2WavDiT model. Which take speech tokens as input and predict mel spectrogram.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

## Qwen2\_5OmniToken2WavBigVGANModel

### class transformers.Qwen2\_5OmniToken2WavBigVGANModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L3338)

( config: Qwen2\_5OmniBigVGANConfig  )

Parameters

* **config** (`Qwen2_5OmniBigVGANConfig`) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The full Qwen2.5Omni Token2WavBigVGAN model. Which take mel spectrogram as input and predict waveform.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_5_omni.md)
