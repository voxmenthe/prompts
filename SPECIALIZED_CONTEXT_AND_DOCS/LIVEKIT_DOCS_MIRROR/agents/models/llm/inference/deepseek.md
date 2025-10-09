LiveKit Docs › Models › Large language models (LLM) › LiveKit Inference › DeepSeek

---

# DeepSeek LLM

> Reference for DeepSeek models served via LiveKit Inference.

## Overview

LiveKit Inference offers DeepSeek models through Baseten. Pricing is available on the [pricing page](https://livekit.io/pricing/inference#llm).

| Model name | Model ID | Providers |
| ---------- | -------- | -------- |
| DeepSeek V3 | `deepseek-ai/deepseek-v3` | `baseten` |

## Usage

To use DeepSeek, pass the model id to the `llm` argument in your `AgentSession`. LiveKit Inference manages the connection to the best available provider automatically.

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    llm="deepseek-ai/deepseek-v3",
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    llm: "deepseek-ai/deepseek-v3",
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
        model="deepseek-ai/deepseek-v3", 
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
        model: "deepseek-ai/deepseek-v3", 
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

The following links provide more information about DeepSeek in LiveKit Inference.

- **[Baseten Plugin](https://docs.livekit.io/agents/models/llm/plugins/baseten.md)**: Plugin to use your own Baseten account instead of LiveKit Inference.

- **[DeepSeek Plugin](https://docs.livekit.io/agents/models/llm/plugins/deepseek.md)**: Plugin to use DeepSeek's official API instead of LiveKit Inference.

- **[Baseten docs](https://docs.baseten.co/development/model-apis/overview)**: Baseten's official Model API documentation.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/llm/inference/deepseek.md](https://docs.livekit.io/agents/models/llm/inference/deepseek.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).