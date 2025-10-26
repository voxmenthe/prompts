LiveKit docs › Models › Text-to-speech (TTS) › LiveKit Inference › Inworld

---

# Inworld TTS

> Reference for Inworld TTS in LiveKit Inference.

## Overview

LiveKit Inference offers voice models powered by Inworld. Pricing information is available on the [pricing page](https://livekit.io/pricing/inference#tts).

| Model ID | Languages |
| -------- | --------- |
| `inworld/inworld-tts-1` | `en`, `es`, `fr`, `ko`, `nl`, `zh`, `de`, `it`, `ja`, `pl`, `pt` |

## Usage

To use Inworld, pass a descriptor with the model and voice to the `tts` argument in your `AgentSession`:

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    tts="inworld/inworld-tts-1:ashley",
    # ... llm, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    tts: "inworld/inworld-tts-1:ashley",
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
        model="inworld/inworld-tts-1", 
        voice="ashley", 
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
        model: "inworld/inworld-tts-1", 
        voice: "ashley", 
        language: "en" 
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model ID from the [models list](#models).

- **`voice`** _(string)_: See [voices](#voices) for guidance on selecting a voice.

- **`language`** _(string)_ (optional): Language code for the input text. If not set, the model default applies.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the Inworld TTS API. See the provider's [documentation](#additional-resources) for more information.

In Node.js this parameter is called `modelOptions`.

## Voices

LiveKit Inference supports all of the default voices available in the Inworld API. You can explore the available voices in the [Inworld TTS Playground](https://docs.inworld.ai/docs/tts/tts-playground) (free account required), and use the voice by copying its name into your LiveKit agent session.

> ℹ️ **Cloned voices unavailable**
> 
> Cloned voices are not yet supported in LiveKit Inference. To use these voices, create your own Inworld account and use the [Inworld plugin](https://docs.livekit.io/agents/models/tts/plugins/inworld.md) for LiveKit Agents instead.

The following is a small sample of the Inworld voices available in LiveKit Inference.

| Provider | Name | Description | Language | ID |
| -------- | ---- | ----------- | -------- | -------- |
| Inworld | Ashley | Warm, natural American female | `en-US` | `inworld/inworld-tts-1:Ashley` |
| Inworld | Diego | Soothing, gentle Mexican male | `es-MX` | `inworld/inworld-tts-1:Diego ` |
| Inworld | Edward | Fast-talking, emphatic American male | `en-US` | `inworld/inworld-tts-1:Edward` |
| Inworld | Olivia | Upbeat, friendly British female | `en-GB` | `inworld/inworld-tts-1:Olivia` |

## Additional resources

The following links provide more information about Inworld in LiveKit Inference.

- **[Inworld Plugin](https://docs.livekit.io/agents/models/tts/plugins/inworld.md)**: Plugin to use your own Inworld account instead of LiveKit Inference.

- **[Inworld TTS docs](https://docs.inworld.ai/docs/tts/tts)**: Inworld's official API documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/inference/inworld.md](https://docs.livekit.io/agents/models/tts/inference/inworld.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).