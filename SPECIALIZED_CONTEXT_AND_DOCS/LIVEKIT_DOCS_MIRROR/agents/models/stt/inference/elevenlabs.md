LiveKit docs › Models › Speech-to-text (STT) › LiveKit Inference › ElevenLabs

---

# ElevenLabs STT

> Reference for ElevenLabs STT in LiveKit Inference.

## Overview

LiveKit Inference offers transcription powered by ElevenLabs. Pricing information is available on the [pricing page](https://livekit.io/pricing/inference#stt).

| Model name | Model ID | Languages |
| -------- | -------- | --------- |
| Scribe V2 Realtime | `elevenlabs/scribe_v2_realtime` | `en`, `en-US`, `en-GB`, `en-AU`, `en-CA`, `en-IN`, `en-NZ`, `es`, `es-ES`, `es-MX`, `es-AR`, `es-CO`, `es-CL`, `es-PE`, `es-VE`, `es-EC`, `es-GT`, `es-CU`, `es-BO`, `es-DO`, `es-HN`, `es-PY`, `es-SV`, `es-NI`, `es-CR`, `es-PA`, `es-UY`, `es-PR`, `fr`, `fr-FR`, `fr-CA`, `fr-BE`, `fr-CH`, `de`, `de-DE`, `de-AT`, `de-CH`, `it`, `it-IT`, `it-CH`, `pt`, `pt-BR`, `pt-PT`, `pl`, `pl-PL`, `ru`, `ru-RU`, `ja`, `ja-JP`, `zh`, `zh-CN`, `zh-TW`, `zh-HK`, `ko`, `ko-KR`, `ar`, `ar-SA`, `ar-EG`, `ar-AE`, `ar-IQ`, `ar-DZ`, `ar-MA`, `ar-KW`, `ar-JO`, `ar-LB`, `ar-OM`, `ar-QA`, `ar-BH`, `ar-TN`, `ar-YE`, `ar-SY`, `ar-SD`, `ar-LY`, `ar-MR`, `ar-SO`, `ar-DJ`, `ar-KM`, `ar-ER`, `ar-TD`, `hi`, `hi-IN`, `tr`, `tr-TR`, `nl`, `nl-NL`, `nl-BE`, `sv`, `sv-SE`, `id`, `id-ID`, `cs`, `cs-CZ`, `ro`, `ro-RO`, `hu`, `hu-HU`, `fi`, `fi-FI`, `da`, `da-DK`, `no`, `no-NO`, `th`, `th-TH`, `vi`, `vi-VN`, `uk`, `uk-UA`, `el`, `el-GR`, `he`, `he-IL`, `ms`, `ms-MY`, `sk`, `sk-SK`, `hr`, `hr-HR`, `bg`, `bg-BG`, `sr`, `sr-RS`, `sl`, `sl-SI`, `et`, `et-EE`, `lv`, `lv-LV`, `lt`, `lt-LT`, `is`, `is-IS`, `ga`, `ga-IE`, `mt`, `mt-MT`, `cy`, `cy-GB` |

## Usage

To use ElevenLabs, pass a descriptor with the model and language to the `stt` argument in your `AgentSession`:

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    stt="elevenlabs/scribe_v2_realtime:en",
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    stt: "elevenlabs/scribe_v2_realtime:en",
    // ... tts, stt, vad, turn_detection, etc.
});

```

### Multilingual transcription

ElevenLabs Scribe 2 Realtime supports multilingual transcription for over 90 languages with automatic language detection.

### Parameters

To customize additional parameters, including the language to use, use the `STT` class from the `inference` module:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    stt=inference.STT(
        model="elevenlabs/scribe_v2_realtime", 
        language="en"
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

- **`model`** _(string)_: The model to use for the STT.

- **`language`** _(string)_ (optional): Language code for the transcription.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the ElevenLabs STT API. For available parameters, see [provider's documentation](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime).

## Additional resources

The following links provide more information about Deepgram in LiveKit Inference.

- **[ElevenLabs Plugin](https://docs.livekit.io/agents/models/stt/plugins/elevenlabs.md)**: Plugin to use your own ElevenLabs account instead of LiveKit Inference.

- **[ElevenLabs docs](https://elevenlabs.io/docs/capabilities/speech-to-text)**: ElevenLabs STT API documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/stt/inference/elevenlabs.md](https://docs.livekit.io/agents/models/stt/inference/elevenlabs.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).