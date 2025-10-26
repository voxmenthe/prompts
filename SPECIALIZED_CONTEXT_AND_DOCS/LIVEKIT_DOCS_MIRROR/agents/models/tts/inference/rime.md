LiveKit docs › Models › Text-to-speech (TTS) › LiveKit Inference › Rime

---

# Rime TTS

> Reference for Rime TTS in LiveKit Inference.

## Overview

LiveKit Inference offers voice models powered by Rime. Pricing information is available on the [pricing page](https://livekit.io/pricing/inference#tts).

| Model ID | Languages |
| -------- | --------- |
| `rime/arcana` | `en`, `es`, `fr`, `de` |
| `rime/mistv2` | `en`, `es`, `fr`, `de` |
| `rime/mist` | `en`, `es`, `fr`, `de` |

## Usage

The simplest way to use Rime TTS is to pass it to the `tts` argument in your `AgentSession`, including the model and voice to use:

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    tts="rime/arcana:astra",
    # ... llm, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    tts: "rime/arcana:astra",
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
        model="rime/arcana", 
        voice="astra", 
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
        model: "rime/arcana", 
        voice: "astra", 
        language: "en" 
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model ID from the [models list](#models).

- **`voice`** _(string)_: See [voices](#voices) for guidance on selecting a voice.

- **`language`** _(string)_ (optional): Two-letter language code for the input text. Note that the Rime API uses three-letter abbreviations (e.g. `eng` for English), but LiveKit Inference uses two-letter codes instead for consistency with other providers.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the Rime TTS API. See the provider's [documentation](#additional-resources) for more information.

In Node.js this parameter is called `modelOptions`.

## Voices

LiveKit Inference supports all of the voices available in the Rime API. You can view the default voices and explore the wider set in the API in the [Rime voices documentation](https://docs.rime.ai/api-reference/voices), and use the voice by copying its name into your LiveKit agent session.

The following is a small sample of the Rime voices available in LiveKit Inference.

| Provider | Name | Description | Language | ID |
| -------- | ---- | ----------- | -------- | -------- |
| Rime | Astra | Chipper, upbeat American female | `en-US` | `rime/arcana:astra` |
| Rime | Celeste | Chill Gen-Z American female | `en-US` | `rime/arcana:celeste` |
| Rime | Luna | Chill but excitable American female | `en-US` | `rime/arcana:luna` |
| Rime | Ursa | Young, emo American male | `en-US` | `rime/arcana:ursa` |

## Additional resources

The following links provide more information about Rime in LiveKit Inference.

- **[Rime Plugin](https://docs.livekit.io/agents/models/tts/plugins/rime.md)**: Plugin to use your own Rime account instead of LiveKit Inference.

- **[Rime TTS docs](https://docs.rime.ai/)**: Rime's official API documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/inference/rime.md](https://docs.livekit.io/agents/models/tts/inference/rime.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).