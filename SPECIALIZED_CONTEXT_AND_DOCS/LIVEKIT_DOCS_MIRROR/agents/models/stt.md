LiveKit Docs › Models › Speech-to-text (STT) › Overview

---

# Speech-to-text (STT) models

> Models and plugins for realtime transcription in your voice agents.

## Overview

STT models, also known as Automated Speech Recognition (ASR) models, are used for realtime transcription or translation of spoken audio. In voice AI, they form the first of three models in the core pipeline: text is transcribed by an STT model, then processed by an [LLM](https://docs.livekit.io/agents/models/llm.md) model to generate a response which is turned backed to speech using a [TTS](https://docs.livekit.io/agents/models/tts.md) model.

You can choose a model served through LiveKit Inference, which is included in LiveKit Cloud, or you can use a plugin to connect directly to a wider range of model providers with your own account.

## LiveKit Inference

The following models are available in [LiveKit Inference](https://docs.livekit.io/agents/models.md#inference). Refer to the guide for each model for more details on additional configuration options.

| Provider | Model name |  | Languages |
| -------- | -------- | --------- |
| [AssemblyAI](https://docs.livekit.io/agents/models/stt/inference/assemblyai.md) | Universal-Streaming | English only |
| [Cartesia](https://docs.livekit.io/agents/models/stt/inference/cartesia.md) | Ink Whisper | 98 languages |
| [Deepgram](https://docs.livekit.io/agents/models/stt/inference/deepgram.md) | Nova-3 | Multilingual, 8 languages |
|   | Nova-3 Medical | English only |
|   | Nova-2 | Multilingual, 33 languages |
|   | Nova-2 Medical | English only |
|   | Nova-2 Conversational AI | English only |
|   | Nova-2 Phonecall | English only |

## Usage

To set up STT in an `AgentSession`, provide a descriptor with both the desired model and language. LiveKit Inference manages the connection to the model automatically. Consult the [models list](#inference) for available models and languages.

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    # AssemblyAI STT in English
    stt="assemblyai/universal-streaming:en",
    # ... llm, tts, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession, inference } from '@livekit/agents';

const session = new AgentSession({
    // AssemblyAI STT in English
    stt: "assemblyai/universal-streaming:en",
    // ... llm, tts, etc.
})

```

### Multilingual transcription

If you don't know the language of the input audio, or expect multiple languages to be used simultaneously, use `deepgram/nova-3` with the language set to `multi`. This model supports multilingual transcription.

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    stt="deepgram/nova-3:multi",
    # ... llm, tts, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

const session = new AgentSession({
    stt: "deepgram/nova-3:multi",
    // ... llm, tts, etc.
})

```

### Additional parameters

More configuration options, such as custom vocabulary, are available for each model. To set additional parameters, use the `STT` class from the `inference` module. Consult each model reference for examples and available parameters.

## Plugins

The LiveKit Agents framework also includes a variety of open source [plugins](https://docs.livekit.io/agents/models.md#plugins) for a wide range of STT providers. These plugins require authentication with the provider yourself, usually via an API key. You are responsible for setting up your own account and managing your own billing and credentials. The plugins are listed below, along with their availability for Python or Node.js.

| Provider | Python | Node.js |
| -------- | ------ | ------- |
| [Amazon Transcribe](https://docs.livekit.io/agents/models/stt/plugins/aws.md) | ✓ | — |
| [AssemblyAI](https://docs.livekit.io/agents/models/stt/plugins/assemblyai.md) | ✓ | — |
| [Azure AI Speech](https://docs.livekit.io/agents/models/stt/plugins/azure.md) | ✓ | — |
| [Azure OpenAI](https://docs.livekit.io/agents/models/stt/plugins/azure-openai.md) | ✓ | — |
| [Baseten](https://docs.livekit.io/agents/models/stt/plugins/baseten.md) | ✓ | — |
| [Cartesia](https://docs.livekit.io/agents/models/stt/plugins/cartesia.md) | ✓ | — |
| [Clova](https://docs.livekit.io/agents/models/stt/plugins/clova.md) | ✓ | — |
| [Deepgram](https://docs.livekit.io/agents/models/stt/plugins/deepgram.md) | ✓ | ✓ |
| [fal](https://docs.livekit.io/agents/models/stt/plugins/fal.md) | ✓ | — |
| [Gladia](https://docs.livekit.io/agents/models/stt/plugins/gladia.md) | ✓ | — |
| [Google Cloud](https://docs.livekit.io/agents/models/stt/plugins/google.md) | ✓ | — |
| [Groq](https://docs.livekit.io/agents/models/stt/plugins/groq.md) | ✓ | — |
| [Mistral AI](https://docs.livekit.io/agents/models/stt/plugins/mistralai.md) | ✓ | — |
| [OpenAI](https://docs.livekit.io/agents/models/stt/plugins/openai.md) | ✓ | ✓ |
| [Sarvam](https://docs.livekit.io/agents/models/stt/plugins/sarvam.md) | ✓ | — |
| [Soniox](https://docs.livekit.io/agents/models/stt/plugins/soniox.md) | ✓ | — |
| [Speechmatics](https://docs.livekit.io/agents/models/stt/plugins/speechmatics.md) | ✓ | — |
| [Spitch](https://docs.livekit.io/agents/models/stt/plugins/spitch.md) | ✓ | — |

Have another provider in mind? LiveKit is open source and welcomes [new plugin contributions](https://docs.livekit.io/agents/models.md#contribute).

## Advanced features

The following sections cover more advanced topics common to all STT providers. For more detailed reference on individual provider configuration, consult the model reference or plugin documentation for that provider.

### Automatic model selection

If you don't need to use any specific model features, and are only interested in the best model available for a given language, you can specify the language alone with the special model id `auto`. LiveKit Inference will choose the best model for the given language automatically.

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    # Use the best available model for Spanish
    stt="auto:es",   
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    // Use the best available model for Spanish
    stt: "auto:es",
})

```

LiveKit Inference supports the following languages:

- `en`: English
- `zh`: Chinese
- `de`: German
- `es`: Spanish
- `ru`: Russian
- `ko`: Korean
- `fr`: French
- `ja`: Japanese
- `pt`: Portuguese
- `tr`: Turkish
- `pl`: Polish
- `ca`: Catalan
- `nl`: Dutch
- `ar`: Arabic
- `sv`: Swedish
- `it`: Italian
- `id`: Indonesian
- `vi`: Vietnamese
- `he`: Hebrew
- `uk`: Ukrainian
- `el`: Greek
- `ms`: Malay
- `cs`: Czech
- `ro`: Romanian
- `da`: Danish
- `hu`: Hungarian
- `ta`: Tamil
- `no`: Norwegian
- `th`: Thai
- `ur`: Urdu
- `hr`: Croatian
- `bg`: Bulgarian
- `lt`: Lithuanian
- `la`: Latin
- `mi`: Maori
- `ml`: Malayalam
- `cy`: Welsh
- `sk`: Slovak
- `te`: Telugu
- `fa`: Farsi
- `lv`: Latvian
- `bn`: Bengali
- `sr`: Serbian
- `az`: Azerbaijani
- `sl`: Slovenian
- `kn`: Kannada
- `et`: Estonian
- `mk`: Macedonian
- `br`: Breton
- `eu`: Basque
- `is`: Icelandic
- `hy`: Armenian
- `ne`: Nepali
- `mn`: Mongolian
- `bs`: Bosnian
- `kk`: Kazakh
- `sq`: Albanian
- `sw`: Swahili
- `gl`: Galician
- `mr`: Marathi
- `pa`: Punjabi
- `si`: Sinhala
- `km`: Khmer
- `sn`: Shona
- `yo`: Yoruba
- `so`: Somali
- `af`: Afrikaans
- `oc`: Occitan
- `ka`: Georgian
- `be`: Belarusian
- `tg`: Tajik
- `sd`: Sindhi
- `gu`: Gujarati
- `am`: Amharic
- `yi`: Yiddish
- `lo`: Lao
- `uz`: Uzbek
- `fo`: Faroese
- `ht`: Haitian
- `ps`: Pashto
- `tk`: Turkmen
- `nn`: Norwegian Nynorsk
- `multi`: Multilingual (automatic)
- `mt`: Maltese
- `sa`: Sanskrit
- `lb`: Luxembourgish
- `my`: Myanmar
- `bo`: Tibetan
- `tl`: Tagalog
- `mg`: Malagasy
- `as`: Assamese
- `tt`: Tatar
- `haw`: Hawaiian
- `ln`: Lingala
- `ha`: Hausa
- `ba`: Bashkir
- `jw`: Javanese
- `su`: Sundanese
- `yue`: Cantonese
- `fi`: Finnish
- `hi`: Hindi
- `en-US`: English (United States)
- `en-AU`: English (Australia)
- `en-CA`: English (Canada)
- `en-GB`: English (United Kingdom)
- `en-IE`: English (Ireland)
- `en-IN`: English (India)
- `en-NZ`: English (New Zealand)
- `es-419`: Spanish (Latin America)
- `es-MX`: Spanish (Mexico)
- `de-CH`: German (Switzerland)
- `da-DK`: Danish (Denmark)
- `fr-CA`: French (Canada)
- `ko-KR`: Korean (South Korea)
- `nl-BE`: Dutch (Belgium)
- `pt-BR`: Portuguese (Brazil)
- `pt-PT`: Portuguese (Portugal)
- `sv-SE`: Swedish (Sweden)
- `zh-Hans`: Simplified Chinese
- `zh-Hant`: Traditional Chinese
- `zh-HK`: Traditional Chinese (Hong Kong)
- `th-TH`: Thai (Thailand)
- `zh-CN`: Simplified Chinese (China)
- `zh-TW`: Traditional Chinese (Taiwan)

### Custom STT

To create an entirely custom STT, implement the [STT node](https://docs.livekit.io/agents/build/nodes.md#stt_node) in your agent.

### Standalone usage

You can use an `STT` instance in a standalone fashion, without an `AgentSession`, using the streaming interface. Use `push_frame` to add [realtime audio frames](https://docs.livekit.io/home/client/tracks.md) to the stream, and then consume a stream of `SpeechEvent` events as output.

Here is an example of a standalone STT app:

** Filename: `agent.py`**

```python
import asyncio

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents.stt import SpeechEventType, SpeechEvent
from typing import AsyncIterable
from livekit.plugins import (
    deepgram,
)

load_dotenv()

async def entrypoint(ctx: agents.JobContext):
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.RemoteTrack):
        print(f"Subscribed to track: {track.name}")

        asyncio.create_task(process_track(track))

    async def process_track(track: rtc.RemoteTrack):
        stt = deepgram.STT(model="nova-2")
        stt_stream = stt.stream()
        audio_stream = rtc.AudioStream(track)

        async with asyncio.TaskGroup() as tg:
            # Create task for processing STT stream
            stt_task = tg.create_task(process_stt_stream(stt_stream))

            # Process audio stream
            async for audio_event in audio_stream:
                stt_stream.push_frame(audio_event.frame)

            # Indicates the end of the audio stream
            stt_stream.end_input()

            # Wait for STT processing to complete
            await stt_task

    async def process_stt_stream(stream: AsyncIterable[SpeechEvent]):
        try:
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    print(f"Final transcript: {event.alternatives[0].text}")
                elif event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    print(f"Interim transcript: {event.alternatives[0].text}")
                elif event.type == SpeechEventType.START_OF_SPEECH:
                    print("Start of speech")
                elif event.type == SpeechEventType.END_OF_SPEECH:
                    print("End of speech")
        finally:
            await stream.aclose()


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))


```

### VAD and StreamAdapter

Some STT providers or models, such as [Whisper](https://github.com/openai/whisper) don't support streaming input. In these cases, your app must determine when a chunk of audio represents a complete segment of speech. You can do this using VAD together with the `StreamAdapter` class.

The following example modifies the previous example to use VAD and `StreamAdapter` to buffer user speech until VAD detects the end of speech:

```python
from livekit import agents, rtc
from livekit.plugins import openai, silero

async def process_track(ctx: agents.JobContext, track: rtc.Track):
  whisper_stt = openai.STT()
  vad = silero.VAD.load(
    min_speech_duration=0.1,
    min_silence_duration=0.5,
  )
  vad_stream = vad.stream()
  # StreamAdapter will buffer audio until VAD emits END_SPEAKING event
  stt = agents.stt.StreamAdapter(whisper_stt, vad_stream)
  stt_stream = stt.stream()
  ...

```

## Additional resources

The following resources cover related topics that may be useful for your application.

- **[Text and transcriptions](https://docs.livekit.io/agents/build/text.md)**: Integrate realtime text features into your agent.

- **[Pipeline nodes](https://docs.livekit.io/agents/build/nodes.md)**: Learn how to customize the behavior of your agent by overriding nodes in the voice pipeline.

- **[Inference pricing](https://livekit.io/pricing/inference#stt)**: The latest pricing information for STT models in LiveKit Inference.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/stt.md](https://docs.livekit.io/agents/models/stt.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).