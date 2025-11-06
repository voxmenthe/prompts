LiveKit docs › Models › Text-to-speech (TTS) › Overview

---

# Text-to-speech (TTS) models

> Voices and plugins to add realtime speech to your voice agents.

## Overview

Voice agent speech is produced by a TTS model, configured with a voice profile that specifies tone, accent, and other qualitative characteristics of the speech. The TTS model runs on output from an [LLM](https://docs.livekit.io/agents/models/llm.md) model to speak the agent response to the user.

You can choose a voice model served through LiveKit Inference or you can use a plugin to connect directly to a wider range of model providers with your own account.

## LiveKit Inference

The following models are available in [LiveKit Inference](https://docs.livekit.io/agents/models.md#inference). Refer to the guide for each model for more details on additional configuration options. A limited selection of [Suggested voices](#voices) are available, as well as a wider selection through each provider's documentation.

- **[Cartesia](https://docs.livekit.io/agents/models/tts/inference/cartesia.md)**: Reference for Cartesia TTS in LiveKit Inference.

- **[ElevenLabs](https://docs.livekit.io/agents/models/tts/inference/elevenlabs.md)**: Reference for ElevenLabs TTS with LiveKit Inference.

- **[Inworld](https://docs.livekit.io/agents/models/tts/inference/inworld.md)**: Reference for Inworld TTS in LiveKit Inference.

- **[Rime](https://docs.livekit.io/agents/models/tts/inference/rime.md)**: Reference for Rime TTS in LiveKit Inference.

### Suggested voices

The following voices are good choices for overall quality and performance. Each provider has a much larger selection of voices to choose from, which you can find in their documentation. In addition to the voices below, you can choose to use other voices through LiveKit Inference.

Click the copy icon to copy the voice ID to use in your agent session.

| Provider | Name | Description | Language | ID |
| -------- | ---- | ----------- | -------- | -------- |
| Cartesia | Blake | Energetic American adult male | `en-US` | `cartesia/sonic-3:a167e0f3-df7e-4d52-a9c3-f949145efdab` |
| Cartesia | Daniela | Calm and trusting Mexican female | `es-MX` | `cartesia/sonic-3:5c5ad5e7-1020-476b-8b91-fdcbe9cc313c` |
| Cartesia | Jacqueline | Confident, young American adult female | `en-US` | `cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc` |
| Cartesia | Robyn | Neutral, mature Australian female | `en-AU` | `cartesia/sonic-3:f31cc6a7-c1e8-4764-980c-60a361443dd1` |
| ElevenLabs | Alice | Clear and engaging, friendly British woman | `en-GB` | `elevenlabs/eleven_turbo_v2_5:Xb7hH8MSUJpSbSDYk0k2` |
| ElevenLabs | Chris | Natural and real American male | `en-US` | `elevenlabs/eleven_turbo_v2_5:iP95p4xoKVk53GoZ742B` |
| ElevenLabs | Eric | A smooth tenor Mexican male | `es-MX` | `elevenlabs/eleven_turbo_v2_5:cjVigY5qzO86Huf0OWal` |
| ElevenLabs | Jessica | Young and popular, playful American female | `en-US` | `elevenlabs/eleven_turbo_v2_5:cgSgspJ2msm6clMCkdW9` |
| Rime | Astra | Chipper, upbeat American female | `en-US` | `rime/arcana:astra` |
| Rime | Celeste | Chill Gen-Z American female | `en-US` | `rime/arcana:celeste` |
| Rime | Luna | Chill but excitable American female | `en-US` | `rime/arcana:luna` |
| Rime | Ursa | Young, emo American male | `en-US` | `rime/arcana:ursa` |
| Inworld | Ashley | Warm, natural American female | `en-US` | `inworld/inworld-tts-1:Ashley` |
| Inworld | Diego | Soothing, gentle Mexican male | `es-MX` | `inworld/inworld-tts-1:Diego ` |
| Inworld | Edward | Fast-talking, emphatic American male | `en-US` | `inworld/inworld-tts-1:Edward` |
| Inworld | Olivia | Upbeat, friendly British female | `en-GB` | `inworld/inworld-tts-1:Olivia` |

## Usage

To set up TTS in an `AgentSession`, provide a descriptor with both the desired model and voice. LiveKit Inference manages the connection to the model automatically. Consult the [Suggested voices](#voices) list for suggeted voices, or view the model reference for more voices.

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    # ... llm, stt, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

const session = new AgentSession({
    tts: "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    // ... llm, stt, etc.
})

```

### Additional parameters

More configuration options, such as custom pronunciation, are available for each model. To set additional parameters, use the `TTS` class from the `inference` module. Consult each model reference for examples and available parameters.

## Plugins

The LiveKit Agents framework also includes a variety of open source [plugins](https://docs.livekit.io/agents/models.md#plugins) for a wide range of TTS providers. Plugins are especially useful if you need custom voices, including voice cloning support. These plugins require authentication with the provider yourself, usually via an API key. You are responsible for setting up your own account and managing your own billing and credentials. The plugins are listed below, along with their availability for Python or Node.js.

| Provider | Python | Node.js |
| -------- | ------ | ------- |
| [Amazon Polly](https://docs.livekit.io/agents/models/tts/plugins/aws.md) | ✓ | — |
| [Azure AI Speech](https://docs.livekit.io/agents/models/tts/plugins/azure.md) | ✓ | — |
| [Azure OpenAI](https://docs.livekit.io/agents/models/tts/plugins/azure-openai.md) | ✓ | — |
| [Baseten](https://docs.livekit.io/agents/models/tts/plugins/baseten.md) | ✓ | — |
| [Cartesia](https://docs.livekit.io/agents/models/tts/plugins/cartesia.md) | ✓ | ✓ |
| [Deepgram](https://docs.livekit.io/agents/models/tts/plugins/deepgram.md) | ✓ | — |
| [ElevenLabs](https://docs.livekit.io/agents/models/tts/plugins/elevenlabs.md) | ✓ | ✓ |
| [Gemini](https://docs.livekit.io/agents/models/tts/plugins/gemini.md) | ✓ | — |
| [Google Cloud](https://docs.livekit.io/agents/models/tts/plugins/google.md) | ✓ | — |
| [Groq](https://docs.livekit.io/agents/models/tts/plugins/groq.md) | ✓ | — |
| [Hume](https://docs.livekit.io/agents/models/tts/plugins/hume.md) | ✓ | — |
| [Inworld](https://docs.livekit.io/agents/models/tts/plugins/inworld.md) | ✓ | — |
| [LMNT](https://docs.livekit.io/agents/models/tts/plugins/lmnt.md) | ✓ | — |
| [MiniMax](https://docs.livekit.io/agents/models/tts/plugins/minimax.md) | ✓ | — |
| [Neuphonic](https://docs.livekit.io/agents/models/tts/plugins/neuphonic.md) | ✓ | ✓ |
| [OpenAI](https://docs.livekit.io/agents/models/tts/plugins/openai.md) | ✓ | ✓ |
| [Resemble AI](https://docs.livekit.io/agents/models/tts/plugins/resemble.md) | ✓ | ✓ |
| [Rime](https://docs.livekit.io/agents/models/tts/plugins/rime.md) | ✓ | ✓ |
| [Sarvam](https://docs.livekit.io/agents/models/tts/plugins/sarvam.md) | ✓ | — |
| [Smallest AI](https://docs.livekit.io/agents/models/tts/plugins/smallestai.md) | ✓ | — |
| [Speechify](https://docs.livekit.io/agents/models/tts/plugins/speechify.md) | ✓ | — |
| [Spitch](https://docs.livekit.io/agents/models/tts/plugins/spitch.md) | ✓ | — |

Have another provider in mind? LiveKit is open source and welcomes [new plugin contributions](https://docs.livekit.io/agents/models.md#contribute).

## Advanced features

The following sections cover more advanced topics common to all TTS providers. For more detailed reference on individual provider configuration, consult the model reference or plugin documentation for that provider.

### Custom TTS

To create an entirely custom TTS, implement the [TTS node](https://docs.livekit.io/agents/build/nodes.md#tts_node) in your agent.

### Standalone usage

You can use a `TTS` instance as a standalone component by creating a stream. Use `push_text` to add text to the stream, and then consume a stream of `SynthesizedAudio` to publish as [realtime audio](https://docs.livekit.io/home/client/tracks.md) to another participant.

Here is an example of a standalone TTS app:

** Filename: `agent.py`**

```python
from livekit import agents, rtc
from livekit.agents.tts import SynthesizedAudio
from livekit.plugins import cartesia
from typing import AsyncIterable

async def entrypoint(ctx: agents.JobContext):
    text_stream: AsyncIterable[str] = ... # you need to provide a stream of text
    audio_source = rtc.AudioSource(44100, 1)

    track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
    await ctx.room.local_participant.publish_track(track)

    tts = cartesia.TTS(model="sonic-english")
    tts_stream = tts.stream()

    # create a task to consume and publish audio frames
    ctx.create_task(send_audio(tts_stream))

    # push text into the stream, TTS stream will emit audio frames along with events
    # indicating sentence (or segment) boundaries.
    async for text in text_stream:
        tts_stream.push_text(text)
    tts_stream.end_input()

    async def send_audio(audio_stream: AsyncIterable[SynthesizedAudio]):
        async for a in audio_stream:
            await audio_source.capture_frame(a.audio.frame)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

```

## Additional resources

The following resources cover related topics that may be useful for your application.

- **[Agent speech docs](https://docs.livekit.io/agents/build/audio.md)**: Explore the speech capabilities and features of LiveKit Agents.

- **[Pipeline nodes](https://docs.livekit.io/agents/build/nodes.md)**: Learn how to customize the behavior of your agent by overriding nodes in the voice pipeline.

- **[Inference pricing](https://livekit.io/pricing/inference#tts)**: The latest pricing information for TTS models in LiveKit Inference.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts.md](https://docs.livekit.io/agents/models/tts.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).