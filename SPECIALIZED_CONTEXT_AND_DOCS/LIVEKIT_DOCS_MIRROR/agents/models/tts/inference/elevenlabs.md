LiveKit Docs › Models › Text-to-speech (TTS) › LiveKit Inference › ElevenLabs

---

# ElevenLabs TTS

> Reference for ElevenLabs TTS in LiveKit Inference.

## Overview

LiveKit Inference offers voice models powered by ElevenLabs. Pricing information is available on the [pricing page](https://livekit.io/pricing/inference#tts).

| Model ID | Languages |
| -------- | --------- |
| `elevenlabs/eleven_flash_v2` | `en` |
| `elevenlabs/eleven_flash_v2_5` | `en`, `ja`, `zh`, `de`, `hi`, `fr`, `ko`, `pt`, `it`, `es`, `id`, `nl`, `tr`, `fil`, `pl`, `sv`, `bg`, `ro`, `ar`, `cs`, `el`, `fi`, `hr`, `ms`, `sk`, `da`, `ta`, `uk`, `ru`, `hu`, `no`, `vi` |
| `elevenlabs/eleven_turbo_v2` | `en` |
| `elevenlabs/eleven_turbo_v2_5` | `en`, `ja`, `zh`, `de`, `hi`, `fr`, `ko`, `pt`, `it`, `es`, `id`, `nl`, `tr`, `fil`, `pl`, `sv`, `bg`, `ro`, `ar`, `cs`, `el`, `fi`, `hr`, `ms`, `sk`, `da`, `ta`, `uk`, `ru`, `hu`, `no`, `vi` |
| `elevenlabs/eleven_multilingual_v2` | `en`, `ja`, `zh`, `de`, `hi`, `fr`, `ko`, `pt`, `it`, `es`, `id`, `nl`, `tr`, `fil`, `pl`, `sv`, `bg`, `ro`, `ar`, `cs`, `el`, `fi`, `hr`, `ms`, `sk`, `da`, `ta`, `uk`, `ru` |

## Usage

To use ElevenLabs, pass a descriptor with the model and voice to the `tts` argument in your `AgentSession`:

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    tts="elevenlabs/eleven_turbo_v2_5:Xb7hH8MSUJpSbSDYk0k2",
    # ... llm, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    tts: "elevenlabs/eleven_turbo_v2_5:Xb7hH8MSUJpSbSDYk0k2",
    // ... tts, stt, vad, turn_detection, etc.
});

```

### Parameters

To customize additional parameters, use the `TTS` class from the `inference` module:

**Python**:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    tts=inference.TTS(
        model="elevenlabs/eleven_turbo_v2_5", 
        voice="Xb7hH8MSUJpSbSDYk0k2", 
        language="en"
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    tts: new inference.TTS({ 
        model: "elevenlabs/eleven_turbo_v2_5", 
        voice: "Xb7hH8MSUJpSbSDYk0k2", 
        language: "en" 
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model ID from the [models list](#models).

- **`voice`** _(string)_: See [voices](#voices) for guidance on selecting a voice.

- **`language`** _(string)_ (optional): Language code for the input text. If not set, the model default applies.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the ElevenLabs TTS API, including `inactivity_timeout`, and `apply_text_normalization`. See the provider's [documentation](#additional-resources) for more information.

## Voices

LiveKit Inference supports all of the default voices available in the ElevenLabs API. You can explore the available voices in the [ElevenLabs voice library](https://elevenlabs.io/app/default-voices) (free account required), and use the voice by copying its ID into your LiveKit agent session.

> ℹ️ **Custom & community voices unavailable**
> 
> Custom and community ElevenLabs voices, including voice cloning, are not yet supported in LiveKit Inference. To use these voices, create your own ElevenLabs account and use the [ElevenLabs plugin](https://docs.livekit.io/agents/models/tts/plugins/elevenlabs.md) for LiveKit Agents instead.

The following is a small sample of the ElevenLabs voices available in LiveKit Inference.

| Provider | Name | Description | Language | ID |
| -------- | ---- | ----------- | -------- | -------- |
| ElevenLabs | Alice | Clear and engaging, friendly British woman | `en-GB` | `elevenlabs/eleven_turbo_v2_5:Xb7hH8MSUJpSbSDYk0k2` |
| ElevenLabs | Chris | Natural and real American male | `en-US` | `elevenlabs/eleven_turbo_v2_5:iP95p4xoKVk53GoZ742B` |
| ElevenLabs | Eric | A smooth tenor Mexican male | `es-MX` | `elevenlabs/eleven_turbo_v2_5:cjVigY5qzO86Huf0OWal` |
| ElevenLabs | Jessica | Young and popular, playful American female | `en-US` | `elevenlabs/eleven_turbo_v2_5:cgSgspJ2msm6clMCkdW9` |

## Additional resources

The following links provide more information about ElevenLabs in LiveKit Inference.

- **[ElevenLabs Plugin](https://docs.livekit.io/agents/models/tts/plugins/elevenlabs.md)**: Plugin to use your own ElevenLabs account instead of LiveKit Inference.

- **[ElevenLabs docs](https://elevenlabs.io/docs)**: ElevenLabs's official API documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/inference/elevenlabs.md](https://docs.livekit.io/agents/models/tts/inference/elevenlabs.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).