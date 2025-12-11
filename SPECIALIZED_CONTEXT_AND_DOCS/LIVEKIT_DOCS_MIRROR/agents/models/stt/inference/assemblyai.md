LiveKit docs › Models › Speech-to-text (STT) › LiveKit Inference › AssemblyAI

---

# AssemblyAI STT

> Reference for AssemblyAI STT in LiveKit Inference.

## Overview

LiveKit Inference offers transcription powered by AssemblyAI. Pricing information is available on the [pricing page](https://livekit.io/pricing/inference#stt).

| Model name | Model ID | Languages |
| -------- | -------- | --------- |
| Universal-Streaming | `assemblyai/universal-streaming` | `en`, `en-US` |
| Universal-Streaming-Multilingual | `assemblyai/universal-streaming-multilingual` | `en`, `en-US`, `en-GB`, `en-AU`, `en-CA`, `en-IN`, `en-NZ`, `es`, `es-ES`, `es-MX`, `es-AR`, `es-CO`, `es-CL`, `es-PE`, `es-VE`, `es-EC`, `es-GT`, `es-CU`, `es-BO`, `es-DO`, `es-HN`, `es-PY`, `es-SV`, `es-NI`, `es-CR`, `es-PA`, `es-UY`, `es-PR`, `fr`, `fr-FR`, `fr-CA`, `fr-BE`, `fr-CH`, `de`, `de-DE`, `de-AT`, `de-CH`, `it`, `it-IT`, `it-CH`, `pt`, `pt-BR`, `pt-PT` |

## Usage

To use AssemblyAI, pass a descriptor with the model and language to the `stt` argument in your `AgentSession`:

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    stt="assemblyai/universal-streaming:en",
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    stt: "assemblyai/universal-streaming:en",
    // ... tts, stt, vad, turn_detection, etc.
});

```

### Parameters

To customize additional parameters, use the `STT` class from the `inference` module:

**Python**:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    stt=inference.STT(
        model="assemblyai/universal-streaming", 
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
        model: "assemblyai/universal-streaming", 
        language: "en" 
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model to use for the STT.

- **`language`** _(string)_ (optional): Language code for the transcription. If not set, the provider default applies.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the AssemblyAI Universal Streaming API, including `format_turns`, `end_of_turn_confidence_threshold`, `min_end_of_turn_silence_when_confident`, `max_turn_silence`, and `keyterms_prompt`. See the provider's [documentation](#additional-resources) for more information.

In Node.js this parameter is called `modelOptions`.

## Additional resources

The following links provide more information about AssemblyAI in LiveKit Inference.

- **[AssemblyAI Plugin](https://docs.livekit.io/agents/models/stt/plugins/assemblyai.md)**: Plugin to use your own AssemblyAI account instead of LiveKit Inference.

- **[AssemblyAI docs](https://www.assemblyai.com/docs/speech-to-text/universal-streaming)**: AssemblyAI's official documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/stt/inference/assemblyai.md](https://docs.livekit.io/agents/models/stt/inference/assemblyai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).