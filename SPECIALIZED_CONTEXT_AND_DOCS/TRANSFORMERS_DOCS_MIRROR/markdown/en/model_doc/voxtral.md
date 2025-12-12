*This model was released on 2025-07-15 and added to Hugging Face Transformers on 2025-07-18.*

# Voxtral

Voxtral is an upgrade of [Ministral 3B and Mistral Small 3B](https://mistral.ai/news/ministraux), extending its language capabilities with audio input support. It is designed to handle tasks such as speech transcription, translation, and audio understanding.

You can read more in Mistral’s [realease blog post](https://mistral.ai/news/voxtral).

The model is available in two checkpoints:

* 3B: [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
* 24B: [mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)

## Key Features

Voxtral builds on Ministral-3B by adding audio processing capabilities:

* **Transcription mode**: Includes a dedicated mode for speech transcription. By default, Voxtral detects the spoken language and transcribes it accordingly.
* **Long-form context**: With a 32k token context window, Voxtral can process up to 30 minutes of audio for transcription or 40 minutes for broader audio understanding.
* **Integrated Q&A and summarization**: Supports querying audio directly and producing structured summaries without relying on separate ASR and language models.
* **Multilingual support**: Automatically detects language and performs well across several widely spoken languages, including English, Spanish, French, Portuguese, Hindi, German, Dutch, and Italian.
* **Function calling via voice**: Can trigger functions or workflows directly from spoken input based on detected user intent.
* **Text capabilities**: Maintains the strong text processing performance of its Ministral-3B foundation.

## Usage

### Audio Instruct Mode

The model supports audio-text instructions, including multi-turn and multi-audio interactions, all processed in batches.

➡️ audio + text instruction


```
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor, infer_device

device = infer_device()
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
            {"type": "text", "text": "What can you tell me about this audio?"},
        ],
    }
]

inputs = processor.apply_chat_template(conversation)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated response:")
print("=" * 80)
print(decoded_outputs[0])
print("=" * 80)
```

➡️ multi-audio + text instruction


```
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor, infer_device

device = infer_device()
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mary_had_lamb.mp3",
            },
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
            },
            {"type": "text", "text": "What sport and what nursery rhyme are referenced?"},
        ],
    }
]

inputs = processor.apply_chat_template(conversation)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated response:")
print("=" * 80)
print(decoded_outputs[0])
print("=" * 80)
```

➡️ multi-turn:


```
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor, infer_device

device = infer_device()
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
            },
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
            },
            {"type": "text", "text": "Describe briefly what you can hear."},
        ],
    },
    {
        "role": "assistant",
        "content": "The audio begins with the speaker delivering a farewell address in Chicago, reflecting on his eight years as president and expressing gratitude to the American people. The audio then transitions to a weather report, stating that it was 35 degrees in Barcelona the previous day, but the temperature would drop to minus 20 degrees the following day.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
            {"type": "text", "text": "Ok, now compare this new audio with the previous one."},
        ],
    },
]

inputs = processor.apply_chat_template(conversation)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated response:")
print("=" * 80)
print(decoded_outputs[0])
print("=" * 80)
```

➡️ text only:


```
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor, infer_device

device = infer_device()
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What if a cyber brain could possibly generate its own ghost, and create a soul all by itself?",
            },
        ],
    }
]

inputs = processor.apply_chat_template(conversation)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated response:")
print("=" * 80)
print(decoded_outputs[0])
print("=" * 80)
```

➡️ audio only:


```
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor, infer_device

device = infer_device()
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
        ],
    }
]

inputs = processor.apply_chat_template(conversation)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated response:")
print("=" * 80)
print(decoded_outputs[0])
print("=" * 80)
```

➡️ batched inference!


```
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor, infer_device()

device = infer_device()
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

conversations = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
                },
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
                },
                {
                    "type": "text",
                    "text": "Who's speaking in the speach and what city's weather is being discussed?",
                },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
                },
                {"type": "text", "text": "What can you tell me about this audio?"},
            ],
        }
    ],
]

inputs = processor.apply_chat_template(conversations)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated responses:")
print("=" * 80)
for decoded_output in decoded_outputs:
    print(decoded_output)
    print("=" * 80)
```

### Transcription Mode

Use the model to transcribe audio (supports English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian)!


```
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor, infer_device

device = infer_device()
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

inputs = processor.apply_transcription_request(language="en", audio="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3", model_id=repo_id)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated responses:")
print("=" * 80)
for decoded_output in decoded_outputs:
    print(decoded_output)
    print("=" * 80)
```

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb).

## VoxtralConfig

### class transformers.VoxtralConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/voxtral/configuration_voxtral.py#L119)

( audio\_config = None text\_config = None audio\_token\_id = None projector\_hidden\_act = 'gelu' \*\*kwargs  )

Parameters

* **audio\_config** (`Union[AutoConfig, dict]`, *optional*) —
  The config object or dictionary of the audio encoder.
* **text\_config** (`Union[AutoConfig, dict]`, *optional*) —
  The config object or dictionary of the text model.
* **audio\_token\_id** (`int`, *optional*) —
  The image token index to encode the image prompt.
* **projector\_hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The activation function (function or string) in the multi-modal projector.

This is the configuration class to store the configuration of a [VoxtralForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralForConditionalGeneration). It is used to instantiate an
Voxtral model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Voxtral-Mini-3B.

e.g. [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import VoxtralForConditionalGeneration, VoxtralConfig

>>> # Initializing a Voxtral configuration
>>> configuration = VoxtralConfig(audio_token_id=24, projector_hidden_act="gelu")

>>> # Initializing a 3B model with random weights
>>> model = VoxtralForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## VoxtralEncoderConfig

### class transformers.VoxtralEncoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/voxtral/configuration_voxtral.py#L20)

( vocab\_size = 51866 hidden\_size = 1280 intermediate\_size = 5120 num\_hidden\_layers = 32 num\_attention\_heads = 20 scale\_embedding = False activation\_function = 'gelu' num\_mel\_bins = 128 max\_source\_positions = 1500 initializer\_range = 0.02 attention\_dropout = 0.0 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 51866) —
  Vocabulary size of the model.
* **hidden\_size** (`int`, *optional*, defaults to 1280) —
  Dimensionality of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 5120) —
  Dimension of the MLP representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 32) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 20) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **scale\_embedding** (`bool`, *optional*, defaults to `False`) —
  Scale embeddings by dividing by sqrt(hidden\_size) if True.
* **activation\_function** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, “gelu”,
* **num\_mel\_bins** (`int`, *optional*, defaults to 128) —
  Number of mel features used per input features. Should correspond to the value used in the
  `VoxtralProcessor` class.
* **max\_source\_positions** (`int`, *optional*, defaults to 1500) —
  The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.

This is the configuration class to store the configuration of a [VoxtralEncoder](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralEncoder). It is used to instantiate a
Voxtral audio encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the audio encoder of the Voxtral
architecture.

e.g. [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import VoxtralEncoderConfig, VoxtralEncoder

>>> # Initializing a VoxtralEncoderConfig
>>> configuration = VoxtralEncoderConfig()

>>> # Initializing a VoxtralEncoder (with random weights)
>>> model = VoxtralEncoder(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## VoxtralProcessor

### class transformers.VoxtralProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/voxtral/processing_voxtral.py#L64)

( feature\_extractor tokenizer  )

Parameters

* **feature\_extractor** ([WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor)) —
  The feature extractor is a required input.
* **tokenizer** ([MistralCommonTokenizer](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer)) —
  The tokenizer is a required input.

Constructs a Voxtral processor which wraps [WhisperFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor) and
[MistralCommonTokenizer](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer) into a single processor that inherits both the audio feature extraction and
tokenizer functionalities.

#### apply\_chat\_template

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/voxtral/processing_voxtral.py#L108)

( conversation: typing.Union[list[dict[str, str]], list[list[dict[str, str]]]] \*\*kwargs: typing\_extensions.Unpack[transformers.processing\_utils.AllKwargsForChatTemplate]  )

Parameters

* **conversation** (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`) —
  The conversation to format.

This method applies the model’s chat completion template given a conversation. It relies on MistralCommonTokenizer’s
[apply\_chat\_template()](/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.apply_chat_template) to prepare input ids to the model and on WhisperFeatureExtractor’s
[**call**()](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor.__call__) to prepare input features to the model.

Note that audio is padded to the nearest 30-second multiple prior to mel feature extraction.

A `conversation` is a list of messages, where each message is a dictionary with a `role` and a `content` field.
For Voxtral, `role` can be `"user"` or `"assistant"`.
The `content` field can be a string or a list of dictionaries with a `type` field. See example below.


```
from huggingface_hub import hf_hub_download
from transformers.audio_utils import load_audio_as

audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"
audio_path = hf_hub_download(repo_id="hf-internal-testing/dummy-audio-samples", filename="bcn_weather.mp3", repo_type="dataset")
audio_base64 = load_audio_as(audio_path, return_format="base64", force_mono=True)

# audio + text
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": audio_url},
            {"type": "audio", "path": audio_path},
            {"type": "audio", "base64": audio_base64},
            {"type": "text", "text": "How many audio do you hear?"},
        ],
    },
]

processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")
inputs = processor.apply_chat_template(conversation)
```

#### apply\_transcription\_request

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/voxtral/processing_voxtral.py#L287)

( language: typing.Union[str, list[str]] audio: typing.Union[str, list[str], numpy.ndarray, ForwardRef('torch.Tensor'), typing.Sequence[numpy.ndarray], typing.Sequence[ForwardRef('torch.Tensor')]] model\_id: str sampling\_rate: typing.Optional[int] = None format: typing.Union[str, list[str], NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.voxtral.processing\_voxtral.VoxtralProcessorKwargs]  )

Parameters

* **language** (`str`, `list[str]`) —
  The language or languages of the audio. If provided as a string, will be applied uniformly to all audio.
  If provided as a list, will be applied to each audio individually with a one-to-one mapping.
* **audio** (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`) —
  The audio or batch of audio to be prepared. If provided as a string, it should correspond to the path or url of the audio file.
* **model\_id** (`str` —
  The hub model id of the model to use for transcription.
* **sampling\_rate** (`int`, *optional*) —
  The sampling rate of the audio. Necessary if it is provided as `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`.
  Used to avoid silent errors when passing audio that is not in the expected sampling rate.
* **format** (`str`, `list[str]`, *optional*) —
  The format of the audio, necessary if is provided as `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`.

This method applies the model’s transcription request template given a language and audio.
It relies on MistralCommonTokenizer and WhisperFeatureExtractor to prepare input ids and input features to the model.


```
from transformers import VoxtralProcessor

model_id = "mistralai/Voxtral-Mini-3B-2507"
processor = VoxtralProcessor.from_pretrained(model_id)

language = "en"
audio = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"

inputs = processor.apply_transcription_request(language=language, audio=audio, model_id=model_id)
```

## VoxtralEncoder

### class transformers.VoxtralEncoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/voxtral/modeling_voxtral.py#L268)

( config: VoxtralEncoderConfig  )

Parameters

* **config** ([VoxtralEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralEncoderConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Voxtral encoder, which is a Whisper encoder.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/voxtral/modeling_voxtral.py#L323)

( input\_features attention\_mask = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  )

Parameters

* **input\_features** (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`) —
  Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
  obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a
  `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
  `input_features`, the [AutoFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoFeatureExtractor) should be used for extracting the mel features, padding
  and conversion into a tensor of type `torch.FloatTensor`. See [**call**()](/docs/transformers/v4.56.2/en/model_doc/whisper#transformers.WhisperFeatureExtractor.__call__)
* **attention\_mask** (`torch.Tensor`)`, *optional*) -- Voxtral does not support masking of the` input\_features`, this argument is preserved for compatibility,
  but it is not used. By default the silence in the input log mel spectrogram are ignored.

## VoxtralForConditionalGeneration

### class transformers.VoxtralForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/voxtral/modeling_voxtral.py#L400)

( config  )

Parameters

* **config** ([VoxtralForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralForConditionalGeneration)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Voxtral model, which consists of Whisper encoder, a multi-modal projector and a LLama language model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/voxtral/modeling_voxtral.py#L455)

( input\_ids: typing.Optional[torch.LongTensor] = None input\_features: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  `feature_extractor_class`. See `feature_extractor_class.__call__` for details ([VoxtralProcessor](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralProcessor) uses
  `feature_extractor_class` for processing audios).
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
elements depending on the configuration ([VoxtralConfig](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralConfig)) and inputs.

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

The [VoxtralForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/voxtral#transformers.VoxtralForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import VoxtralForConditionalGeneration, AutoProcessor
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> repo_id = "mistralai/Voxtral-Mini-3B-2507"

>>> processor = AutoProcessor.from_pretrained(repo_id)
>>> model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

>>> conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
            },
            {"type": "text", "text": "What can you tell me about this audio?"},
        ],
    }
]

>>> inputs = processor.apply_chat_template(conversation)
>>> inputs = inputs.to(device, dtype=torch.bfloat16)

>>> outputs = model.generate(**inputs, max_new_tokens=30)
>>> processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
["This audio is a humorous conversation between two friends, likely in English, where one of them is trying to figure out what the other's tattoo says."]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/voxtral.md)
