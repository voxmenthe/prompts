*This model was released on 2024-07-15 and added to Hugging Face Transformers on 2024-08-08.*

# Qwen2Audio

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Qwen2-Audio is the new model series of large audio-language models from the Qwen team. Qwen2-Audio is capable of accepting various audio signal inputs and performing audio analysis or direct textual responses with regard to speech instructions. We introduce two distinct audio interaction modes:

* voice chat: users can freely engage in voice interactions with Qwen2-Audio without text input
* audio analysis: users could provide audio and text instructions for analysis during the interaction

It was proposed in [Qwen2-Audio Technical Report](https://huggingface.co/papers/2407.10759) by Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, Jingren Zhou.

The abstract from the paper is the following:

*We introduce the latest progress of Qwen-Audio, a large-scale audio-language model called Qwen2-Audio, which is capable of accepting various audio signal inputs and performing audio analysis or direct textual responses with regard to speech instructions. In contrast to complex hierarchical tags, we have simplified the pre-training process by utilizing natural language prompts for different data and tasks, and have further expanded the data volume. We have boosted the instruction-following capability of Qwen2-Audio and implemented two distinct audio interaction modes for voice chat and audio analysis. In the voice chat mode, users can freely engage in voice interactions with Qwen2-Audio without text input. In the audio analysis mode, users could provide audio and text instructions for analysis during the interaction. Note that we do not use any system prompts to switch between voice chat and audio analysis modes. Qwen2-Audio is capable of intelligently comprehending the content within audio and following voice commands to respond appropriately. For instance, in an audio segment that simultaneously contains sounds, multi-speaker conversations, and a voice command, Qwen2-Audio can directly understand the command and provide an interpretation and response to the audio. Additionally, DPO has optimized the model’s performance in terms of factuality and adherence to desired behavior. According to the evaluation results from AIR-Bench, Qwen2-Audio outperformed previous SOTAs, such as Gemini-1.5-pro, in tests focused on audio-centric instruction-following capabilities. Qwen2-Audio is open-sourced with the aim of fostering the advancement of the multi-modal language community.*

## Usage tips

`Qwen2-Audio-7B` and `Qwen2-Audio-7B-Instruct` can be found on the [Huggingface Hub](https://huggingface.co/Qwen)

> [!NOTE]
> The `head_mask` argument is ignored when using all attention implementation other than “eager”. If you have a `head_mask` and want it to have effect, load the model with `XXXModel.from_pretrained(model_id, attn_implementation="eager")`

### Inference


```
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True, device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
audio, sr = librosa.load(BytesIO(urlopen(url).read()), sr=processor.feature_extractor.sampling_rate)
inputs = processor(text=prompt, audios=audio, return_tensors="pt").to(model.device)

generate_ids = model.generate(**inputs, max_length=256)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# We can also omit the audio_bos and audio_eos tokens
prompt = "<|AUDIO|>Generate the caption in English:"
inputs = processor(text=prompt, audios=audio, return_tensors="pt").to(model.device)

generate_ids = model.generate(**inputs, max_length=256)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

In the following, we demonstrate how to use `Qwen2-Audio-7B-Instruct` for the inference, supporting both voice chat and audio analysis modes. Note that we have used the ChatML format for dialog, in this demo we show how to leverage `apply_chat_template` for this purpose.

### Voice Chat Inference

In the voice chat mode, users can freely engage in voice interactions with Qwen2-Audio without text input:


```
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"},
    ]},
    {"role": "assistant", "content": "Yes, the speaker is female and in her twenties."},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav"},
    ]},
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audios.append(librosa.load(
                    BytesIO(urlopen(ele['audio_url']).read()),
                    sr=processor.feature_extractor.sampling_rate)[0]
                )

inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
inputs.input_ids = inputs.input_ids.to(model.device)

generate_ids = model.generate(**inputs, max_length=256)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

### Audio Analysis Inference

In the audio analysis, users could provide both audio and text instructions for analysis:


```
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        {"type": "text", "text": "What's that sound?"},
    ]},
    {"role": "assistant", "content": "It is the sound of glass shattering."},
    {"role": "user", "content": [
        {"type": "text", "text": "What can you do when you hear that?"},
    ]},
    {"role": "assistant", "content": "Stay alert and cautious, and check if anyone is hurt or if there is any damage to property."},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"},
        {"type": "text", "text": "What does the person say?"},
    ]},
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audios.append(
                    librosa.load(
                        BytesIO(urlopen(ele['audio_url']).read()),
                        sr=processor.feature_extractor.sampling_rate)[0]
                )

inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
inputs.input_ids = inputs.input_ids.to(model.device)

generate_ids = model.generate(**inputs, max_length=256)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

### Batch Inference

We also support batch inference:


```
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

conversation1 = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        {"type": "text", "text": "What's that sound?"},
    ]},
    {"role": "assistant", "content": "It is the sound of glass shattering."},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"},
        {"type": "text", "text": "What can you hear?"},
    ]}
]

conversation2 = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"},
        {"type": "text", "text": "What does the person say?"},
    ]},
]

conversations = [conversation1, conversation2]

text = [processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) for conversation in conversations]

audios = []
for conversation in conversations:
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            BytesIO(urlopen(ele['audio_url']).read()),
                            sr=processor.feature_extractor.sampling_rate)[0]
                    )

inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
inputs['input_ids'] = inputs['input_ids'].to(model.device)
inputs.input_ids = inputs.input_ids.to(model.device)

generate_ids = model.generate(**inputs, max_length=256)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

## Qwen2AudioConfig

### class transformers.Qwen2AudioConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_audio/configuration_qwen2_audio.py#L119)

( audio\_config = None text\_config = None audio\_token\_index = 151646 \*\*kwargs  )

Parameters

* **audio\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `CLIPVisionConfig`) —
  The config object or dictionary of the audio backbone.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`) —
  The config object or dictionary of the text backbone.
* **audio\_token\_index** (`int`, *optional*, defaults to 151646) —
  The image token index to encode the image prompt.

This is the configuration class to store the configuration of a [Qwen2AudioForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioForConditionalGeneration). It is used to instantiate an
Qwen2-Audio model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Qwen2-Audio.

e.g. [Qwen/Qwen2-Audio-7B](https://huggingface.co/Qwen/Qwen2-Audio-7B)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioConfig, Qwen2AudioEncoderConfig, Qwen2Config

>>> # Initializing a Qwen2AudioEncoder config
>>> audio_config = Qwen2AudioEncoderConfig()

>>> # Initializing a Qwen2 config
>>> text_config = Qwen2Config()

>>> # Initializing a Qwen2Audio configuration
>>> configuration = Qwen2AudioConfig(audio_config, text_config)

>>> # Initializing a model from the qwen2-audio style configuration
>>> model = Qwen2AudioForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen2AudioEncoderConfig

### class transformers.Qwen2AudioEncoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_audio/configuration_qwen2_audio.py#L24)

( num\_mel\_bins = 128 encoder\_layers = 32 encoder\_attention\_heads = 20 encoder\_ffn\_dim = 5120 encoder\_layerdrop = 0.0 d\_model = 1280 dropout = 0.0 attention\_dropout = 0.0 activation\_function = 'gelu' activation\_dropout = 0.0 scale\_embedding = False initializer\_range = 0.02 max\_source\_positions = 1500 \*\*kwargs  )

Parameters

* **num\_mel\_bins** (`int`, *optional*, defaults to 128) —
  Number of mel features used per input features. Should correspond to the value used in the
  `Qwen2AudioProcessor` class.
* **encoder\_layers** (`int`, *optional*, defaults to 32) —
  Number of encoder layers.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 20) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 5120) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in encoder.
* **encoder\_layerdrop** (`float`, *optional*, defaults to 0.0) —
  The LayerDrop probability for the encoder. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **d\_model** (`int`, *optional*, defaults to 1280) —
  Dimensionality of the layers.
* **dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **activation\_function** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **activation\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for activations inside the fully connected layer.
* **scale\_embedding** (`bool`, *optional*, defaults to `False`) —
  Scale embeddings by diving by sqrt(d\_model).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **max\_source\_positions** (`int`, *optional*, defaults to 1500) —
  The maximum sequence length of log-mel filter-bank features that this model might ever be used with.

This is the configuration class to store the configuration of a [Qwen2AudioEncoder](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioEncoder). It is used to instantiate a
Qwen2-Audio audio encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the audio encoder of the Qwen2-Audio
architecture.

e.g. [Qwen/Qwen2-Audio-7B](https://huggingface.co/Qwen/Qwen2-Audio-7B)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Qwen2AudioEncoderConfig, Qwen2AudioEncoder

>>> # Initializing a Qwen2AudioEncoderConfig
>>> configuration = Qwen2AudioEncoderConfig()

>>> # Initializing a Qwen2AudioEncoder (with random weights)
>>> model = Qwen2AudioEncoder(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen2AudioProcessor

### class transformers.Qwen2AudioProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L37)

( feature\_extractor = None tokenizer = None chat\_template = None audio\_token = '<|AUDIO|>' audio\_bos\_token = '<|audio\_bos|>' audio\_eos\_token = '<|audio\_eos|>'  )

Parameters

* **feature\_extractor** ([WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor), *optional*) —
  The feature extractor is a required input.
* **tokenizer** ([Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast), *optional*) —
  The tokenizer is a required input.
* **chat\_template** (`Optional[str]`, *optional*) —
  The Jinja template to use for formatting the conversation. If not provided, the default chat template
  is used.
* **audio\_token** (`str`, *optional*, defaults to `"<|AUDIO|>"`) —
  The token to use for audio tokens.
* **audio\_bos\_token** (`str`, *optional*, defaults to `"<|audio_bos|>"`) —
  The token to use for audio bos tokens.
* **audio\_eos\_token** (`str`, *optional*, defaults to `"<|audio_eos|>"`) —
  The token to use for audio eos tokens.

Constructs a Qwen2Audio processor which wraps a Qwen2Audio feature extractor and a Qwen2Audio tokenizer into a single processor.

[Qwen2AudioProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioProcessor) offers all the functionalities of [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor) and [Qwen2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## Qwen2AudioEncoder

### class transformers.Qwen2AudioEncoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L295)

( config: Qwen2AudioEncoderConfig  )

Parameters

* **config** ([Qwen2AudioEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioEncoderConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The audio model from Qwen2Audio without any head or projection on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L346)

( input\_features attention\_mask = None head\_mask = None output\_attentions = None output\_hidden\_states = None return\_dict = None  )

Parameters

* **attention\_mask** (`torch.Tensor`)`, *optional*) -- Qwen2Audio does not support masking of the` input\_features`, this argument is preserved for compatibility,
  but it is not used. By default the silence in the input log mel spectrogram are ignored.
* **head\_mask** (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under
  returned tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
  for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

## Qwen2AudioForConditionalGeneration

### class transformers.Qwen2AudioForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L474)

( config: Qwen2AudioConfig  )

Parameters

* **config** ([Qwen2AudioConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The QWEN2AUDIO model which consists of a audio backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L714)

( input\_ids: typing.Optional[torch.LongTensor] = None input\_features: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None feature\_attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None  ) → `transformers.models.qwen2_audio.modeling_qwen2_audio.Qwen2AudioCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  `feature_extractor_class`. See `feature_extractor_class.__call__` for details ([Qwen2AudioProcessor](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioProcessor) uses
  `feature_extractor_class` for processing audios).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **feature\_attention\_mask** (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`) —
  Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
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

`transformers.models.qwen2_audio.modeling_qwen2_audio.Qwen2AudioCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.qwen2_audio.modeling_qwen2_audio.Qwen2AudioCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen2AudioConfig](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Pre-computed hidden-states that can be used to speed up auto-regressive (sequential) decoding. There are
  two sets of pre-computed hidden-states: key and values states in the self-attention blocks.
  The `past_key_values` are returned when `use_cache=True` is passed or when `config.use_cache=True`.
  It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance.

  If `past_key_values` are used, the user can optionally input only the last `input_ids` (those
  that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of
  all `input_ids` of shape `(batch_size, sequence_length)`.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **attention\_mask** (`torch.FloatTensor`, *optional*) — Attentions mask, used to update attention mask and position\_ids.

The [Qwen2AudioForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#transformers.Qwen2AudioForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from io import BytesIO
>>> from urllib.request import urlopen
>>> import librosa
>>> from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

>>> model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_audio.md)
