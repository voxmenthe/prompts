LiveKit docs › Partner spotlight › Azure › Azure OpenAI in LiveKit Inference

---

# OpenAI LLM models

> Reference for OpenAI models served via LiveKit Inference.

## Overview

LiveKit Inference includes support for the following OpenAI models. Pricing information for each model and provider is available on the [pricing page](https://livekit.io/pricing/inference#proprietary-llms).

| Model name | Model ID | Providers |
| ---------- | -------- | -------- |
| GPT-4o | `openai/gpt-4o` | `azure`, `openai` |
| GPT-4o mini | `openai/gpt-4o-mini` | `azure`, `openai` |
| GPT-4.1 | `openai/gpt-4.1` | `azure`, `openai` |
| GPT-4.1 mini | `openai/gpt-4.1-mini` | `azure`, `openai` |
| GPT-4.1 nano | `openai/gpt-4.1-nano` | `azure`, `openai` |
| GPT-5 | `openai/gpt-5` | `azure`, `openai` |
| GPT-5 mini | `openai/gpt-5-mini` | `azure`, `openai` |
| GPT-5 nano | `openai/gpt-5-nano` | `azure`, `openai` |
| GPT OSS 120B | `openai/gpt-oss-120b` | `baseten`, `groq`, (cerebras coming soon) |

## Usage

To use OpenAI, pass the model id to the `llm` argument in your `AgentSession`. LiveKit Inference manages the connection to the model automatically and picks the best available provider.

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    llm="openai/gpt-4.1-mini",
    # ... tts, stt, vad, turn_detection, etc.
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    llm: "openai/gpt-4.1-mini",
    // ... tts, stt, vad, turn_detection, etc.
});

```

## Parameters

To customize additional parameters, or specify the exact provider to use, use the `LLM` class from the `inference` module.

**Python**:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    llm=inference.LLM(
        model="openai/gpt-5-mini", 
        provider="openai",
        extra_kwargs={
            "reasoning_effort": "low"
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
        model: "openai/gpt-5-mini", 
        provider: "openai",
        modelOptions: { 
            reasoning_effort: "low" 
        }
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

- **`model`** _(string)_: The model to use for the LLM. Must be a model from OpenAI.

- **`provider`** _(string)_: The provider to use for the LLM. Must be `openai` to use OpenAI models and other parameters.

- **`extra_kwargs`** _(dict)_ (optional): Additional parameters to pass to the provider's Chat Completions API, such as `reasoning_effort` or `max_completion_tokens`.

In Node.js this parameter is called `modelOptions`.

## Additional resources

The following links provide more information about OpenAI in LiveKit Inference.

- **[OpenAI Plugin](https://docs.livekit.io/agents/models/llm/plugins/openai.md)**: Plugin to use your own OpenAI account instead of LiveKit Inference.

- **[Azure OpenAI Plugin](https://docs.livekit.io/agents/models/llm/plugins/azure-openai.md)**: Plugin to use your own Azure OpenAI account instead of LiveKit Inference.

- **[OpenAI docs](https://platform.openai.com/docs)**: Official OpenAI platform documentation.

- **[Azure OpenAI docs](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/overview)**: Azure OpenAI documentation, for OpenAI proprietary models.

- **[Baseten docs](https://docs.baseten.co/development/model-apis/overview)**: Baseten's official Model API documentation, for GPT-OSS models.

- **[Groq docs](https://console.groq.com/docs/overview)**: Groq's official API documentation, for GPT-OSS models.

- **[OpenAI ecosystem overview](https://docs.livekit.io/agents/integrations/openai.md)**: Overview of the entire OpenAI ecosystem and LiveKit Agents integration.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/llm/inference/openai.md](https://docs.livekit.io/agents/models/llm/inference/openai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).