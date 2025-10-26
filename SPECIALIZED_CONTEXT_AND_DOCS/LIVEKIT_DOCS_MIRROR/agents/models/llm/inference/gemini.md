LiveKit docs › Partner spotlight › Google › Gemini in LiveKit Inference

---

# Google Gemini LLM

> Reference for the Google Gemini models served via LiveKit Inference.

## Overview

LiveKit Inference offers Gemini models through Google Vertex AI. Pricing is available on the [pricing page](https://livekit.io/pricing/inference#llm).

| Model name | Model ID | Providers |
| ---------- | -------- | -------- |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` | `google` |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` | `google` |
| Gemini 2.5 Flash Lite | `google/gemini-2.5-flash-lite` | `google` |
| Gemini 2.0 Flash | `google/gemini-2.0-flash` | `google` |
| Gemini 2.0 Flash Lite | `google/gemini-2.0-flash-lite` | `google` |

## Usage

To use Gemini, pass the model id to the `llm` argument in your `AgentSession`. LiveKit Inference manages the connection to the model automatically.

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    llm="google/gemini-2.5-flash-lite",
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    llm: "google/gemini-2.5-flash-lite",
    // ... tts, stt, vad, turn_detection, etc.
});

```

### Parameters

To customize additional parameters, use the `LLM` class from the `inference` module.

**Python**:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    llm=inference.LLM(
        model="google/gemini-2.5-flash-lite",
        extra_kwargs={
            "max_completion_tokens": 1000
        }
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession, inference } from '@livekit/agents';

session = new AgentSession({
    llm: new inference.LLM({ 
        model: "google/gemini-2.5-flash-lite", 
        modelOptions: { 
            max_completion_tokens: 1000 
        }
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model ID from the [models list](#models).

- **`provider`** _(string)_ (optional): Set a specific provider to use for the LLM. Refer to the [models list](#models) for available providers. If not set, LiveKit Inference uses the best available provider, and bills accordingly.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the Gemini Chat Completions API, such as `max_completion_tokens`.

In Node.js this parameter is called `modelOptions`.

## Additional resources

The following links provide more information about Gemini in LiveKit Inference.

- **[Gemini Plugin](https://docs.livekit.io/agents/models/llm/plugins/gemini.md)**: Plugin to use your own Gemini or Vertex AI account instead of LiveKit Inference.

- **[Gemini docs](https://ai.google.dev/gemini-api/docs/models/gemini)**: Gemini's official API documentation.

- **[Google ecosystem overview](https://docs.livekit.io/agents/integrations/google.md)**: Overview of the entire Google AI ecosystem and LiveKit Agents integration.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/llm/inference/gemini.md](https://docs.livekit.io/agents/models/llm/inference/gemini.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).