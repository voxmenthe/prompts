LiveKit docs › Models › Speech-to-text (STT) › LiveKit Inference › Deepgram

---

# Deepgram STT

> Reference for Deepgram STT in LiveKit Inference.

## Overview

LiveKit Inference offers transcription powered by Deepgram. Pricing information is available on the [pricing page](https://livekit.io/pricing/inference#stt).

| Model name | Model ID | Languages |
| -------- | -------- | --------- |
| Nova-3 | `deepgram/nova-3` | `en`, `en-US`, `en-AU`, `en-GB`, `en-IN`, `en-NZ`, `de`, `nl`, `sv`, `sv-SE`, `da`, `da-DK`, `es`, `es-419`, `fr`, `fr-CA`, `pt`, `pt-BR`, `pt-PT`, `multi` |
| Nova-3 Medical | `deepgram/nova-3-medical` | `en`, `en-US`, `en-AU`, `en-CA`, `en-GB`, `en-IE`, `en-IN`, `en-NZ` |
| Nova-2 | `deepgram/nova-2` | `multi`, `bg`, `ca`, `zh`, `zh-CN`, `zh-Hans`, `zh-TW`, `zh-Hant`, `zh-HK`, `cs`, `da`, `da-DK`, `nl`, `en`, `en-US`, `en-AU`, `en-GB`, `en-NZ`, `en-IN`, `et`, `fi`, `nl-BE`, `fr`, `fr-CA`, `de`, `de-CH`, `el`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `ko-KR`, `lv`, `lt`, `ms`, `no`, `pl`, `pt`, `pt-BR`, `pt-PT`, `ro`, `ru`, `sk`, `es`, `es-419`, `sv`, `sv-SE`, `th`, `th-TH`, `tr`, `uk`, `vi` |
| Nova-2 Medical | `deepgram/nova-2-medical` | `en`, `en-US` |
| Nova-2 Conversational AI | `deepgram/nova-2-conversationalai` | `en`, `en-US` |
| Nova-2 Phonecall | `deepgram/nova-2-phonecall` | `en`, `en-US` |

## Usage

To use Deepgram, pass a descriptor with the model and language to the `stt` argument in your `AgentSession`:

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    stt="deepgram/nova-3:en",
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    stt: "deepgram/nova-3:en",
    // ... tts, stt, vad, turn_detection, etc.
});

```

### Multilingual transcription

Deepgram Nova-3 and Nova-2 models support multilingual transcription. In this mode, the model automatically detects the language of each segment of speech and can accurately transcribe multiple languages in the same audio stream.

Multilingual transcription is billed at a different rate than monolingual transcription. Refer to the [pricing page](https://livekit.io/pricing/inference#stt) for more information.

To enable multilingual transcription on supported models, set the language to `multi`.

### Parameters

To customize additional parameters, including the language to use, use the `STT` class from the `inference` module:

**Python**:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    stt=inference.STT(
        model="deepgram/nova-3", 
        language="en"
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession, inference } from '@livekit/agents';

session = new AgentSession({
    stt: new inference.STT({ 
        model: "deepgram/nova-3", 
        language: "en" 
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model to use for the STT.

- **`language`** _(string)_ (optional): Language code for the transcription. If not set, the provider default applies. Set it to `multi` with supported models for multilingual transcription.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the Deepgram STT API, including `filler_words`, `interim_results`, `endpointing`, `punctuate`, `smart_format`, `keywords`, `keyterms`, `profanity_filter`, `numerals`, and `mip_opt_out`.

In Node.js this parameter is called `modelOptions`.

## Additional resources

The following links provide more information about Deepgram in LiveKit Inference.

- **[Deepgram Plugin](https://docs.livekit.io/agents/models/stt/plugins/deepgram.md)**: Plugin to use your own Deepgram account instead of LiveKit Inference.

- **[Deepgram docs](https://developers.deepgram.com/docs)**: Deepgram service documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/stt/inference/deepgram.md](https://docs.livekit.io/agents/models/stt/inference/deepgram.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).