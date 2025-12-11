LiveKit docs › Models › Text-to-speech (TTS) › LiveKit Inference › Deepgram

---

# Deepgram TTS

> Reference for Deepgram TTS in LiveKit Inference.

## Overview

LiveKit Inference offers voice models powered by Deepgram. Pricing information is available on the [pricing page](https://livekit.io/pricing/inference#tts).

| Model ID | Languages |
| -------- | --------- |
| `deepgram/aura` | `en` |
| `deepgram/aura-2` | `en`, `es` |

## Usage

To use Deepgram, pass a descriptor with the model and voice to the `tts` argument in your `AgentSession`:

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    tts="deepgram/aura-2:athena",
    # ... stt, llm, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    tts: "deepgram/aura-2:athena",
    // ... stt, llm, vad, turn_detection, etc.
});

```

### Parameters

To customize additional parameters, use the `TTS` class from the `inference` module:

**Python**:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    tts=inference.TTS(
        model="deepgram/aura-2", 
        voice="athena", 
        language="en"
    ),
    # ... stt, llm, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    tts: new inference.TTS({ 
        model: "deepgram/aura-2", 
        voice: "athena", 
        language: "en" 
    }),
    // ... stt, llm, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model ID from the [models list](#models).

- **`voice`** _(string)_: See [voices](#voices) for guidance on selecting a voice.

- **`language`** _(string)_ (optional): Language code for the input text. If not set, the model default applies.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the Deepgram TTS API. See the provider's [documentation](#additional-resources) for more information.

In Node.js this parameter is called `modelOptions`.

## Voices

LiveKit Inference supports Deepgram Aura voices. You can explore the available voices in the [Deepgram voice library](https://developers.deepgram.com/docs/tts-models), and use the voice by copying its name into your LiveKit agent session.

The following is a small sample of the Deepgram voices available in LiveKit Inference.

| Provider | Name | Description | Language | ID |
| -------- | ---- | ----------- | -------- | -------- |
| Deepgram | Apollo | Comfortable, casual male | `en-US` | `deepgram/aura-2:apollo` |
| Deepgram | Athena | Smooth, professional female | `en-US` | `deepgram/aura-2:athena` |
| Deepgram | Odysseus | Calm, professional male | `en-US` | `deepgram/aura-2:odysseus` |
| Deepgram | Theia | Expressive, polite female | `en-AU` | `deepgram/aura-2:theia` |

## Additional resources

The following links provide more information about Deepgram in LiveKit Inference.

- **[Deepgram Plugin](https://docs.livekit.io/agents/models/tts/plugins/deepgram.md)**: Plugin to use your own Deepgram account instead of LiveKit Inference.

- **[Deepgram docs](https://developers.deepgram.com/docs)**: Deepgram's official API documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/inference/deepgram.md](https://docs.livekit.io/agents/models/tts/inference/deepgram.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).