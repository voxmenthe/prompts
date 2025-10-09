LiveKit Docs › Models › Large language models (LLM) › LiveKit Inference › Kimi

---

# Kimi LLM

> Reference for Kimi models served via LiveKit Inference.

## Overview

LiveKit Inference offers Kimi models through Baseten. Pricing is available on the [pricing page](https://livekit.io/pricing/inference#llm).

| Model name | Model ID | Providers |
| ---------- | -------- | -------- |
| Kimi K2 Instruct | `moonshotai/kimi-k2-instruct` | `baseten` |

## Usage

To use Kimi, pass the model id to the `llm` argument in your `AgentSession`. LiveKit Inference manages the connection to the best available provider automatically.

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    llm="moonshotai/kimi-k2-instruct",
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    llm: "moonshotai/kimi-k2-instruct",
    // ... tts, stt, vad, turn_detection, etc.
});

```

### Parameters

To customize additional parameters, including the specific provider to use, use the `LLM` class from the `inference` module.

**Python**:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    llm=inference.LLM(
        model="moonshotai/kimi-k2-instruct", 
        provider="baseten",
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
        model: "moonshotai/kimi-k2-instruct", 
        provider: "baseten",
        extraKwargs: { 
            max_completion_tokens: 1000 
        }
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model ID from the [models list](#models).

- **`provider`** _(string)_ (optional): Set a specific provider to use for the LLM. Refer to the [models list](#models) for available providers. If not set, LiveKit Inference uses the best available provider, and bills accordingly.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the provider's Chat Completions API, such as `max_completion_tokens`. See the provider's [documentation](#additional-resources) for more information.

## Additional resources

The following links provide more information about Kimi in LiveKit Inference.

- **[Baseten Plugin](https://docs.livekit.io/agents/models/llm/plugins/baseten.md)**: Plugin to use your own Baseten account instead of LiveKit Inference.

- **[Baseten docs](https://docs.baseten.co/development/model-apis/overview)**: Baseten's official Model API documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/llm/inference/kimi.md](https://docs.livekit.io/agents/models/llm/inference/kimi.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).