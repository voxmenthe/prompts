LiveKit docs › Partner spotlight › Azure › Overview

---

# Azure AI Services and LiveKit

> An overview of the Azure AI integrations with LiveKit Agents.

## Azure AI ecosystem support

Microsoft's [Azure AI Services](https://azure.microsoft.com/en-us/products/ai-services) is a large collection of cutting-edge production-ready AI services, which integrate with LiveKit in the following ways:

- **Azure OpenAI**: Run OpenAI models, including the Realtime API, with the security and reliability of Azure.
- **Azure Speech**: Speech-to-text and text-to-speech services.

LiveKit Inference and the LiveKit Agents OpenAI plugin support Azure OpenAI, and the Azure plugin supports Azure Speech.

## Getting started

Use the voice AI quickstart to build a voice AI app with Azure OpenAI. Select a realtime model type and add the following components to use the Azure OpenAI Realtime API:

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Build your first voice AI app with Azure OpenAI.

### LiveKit Inference

Use an Azure OpenAI model in your agent session:

**Python**:

```python
from livekit.agents import AgentSession, inference

session = AgentSession(
    llm=inference.LLM(
        model="openai/gpt-4.1-mini",
        provider="azure",
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
        model: "openai/gpt-4.1-mini",
        provider: "azure",
    }),
    // ... tts, stt, vad, turn_detection, etc.
});

```

### Azure OpenAI Realtime API

Install the OpenAI plugin:

```shell
uv add "livekit-agents[openai]~=1.2"

```

Add your Azure OpenAI endpoint and API key to your `.env.` file:

** Filename: `.env`**

```shell
AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>

```

Use the `with_azure` method to connect to Azure OpenAI:

** Filename: `agent.py`**

```python
from livekit.plugins import openai

# ...

# in your entrypoint function
session = AgentSession(
    llm=openai.realtime.RealtimeModel.with_azure(
        azure_deployment="<model-deployment>",
        api_version="2024-10-01-preview",
        voice="alloy",
    ),
    # ... vad, turn_detection, etc.
)

```

## LiveKit Agents overview

LiveKit Agents is an open source framework for building realtime AI apps in Python and Node.js. It supports complex voice AI [workflows](https://docs.livekit.io/agents/build/workflows.md) with multiple agents and discrete processing steps, and includes built-in load balancing.

LiveKit provides SIP support for [telephony integration](https://docs.livekit.io/agents/start/telephony.md) and full-featured [frontend SDKs](https://docs.livekit.io/agents/start/frontend.md) in multiple languages. It uses [WebRTC](https://docs.livekit.io/home/get-started/intro-to-livekit.md#what-is-webrtc) transport for end-user devices, enabling high-quality, low-latency realtime experiences. To learn more, see [LiveKit Agents](https://docs.livekit.io/agents.md).

## Azure plugin documentation

- **[Azure OpenAI in LiveKit Inference](https://docs.livekit.io/agents/models/llm/inference/openai.md)**: Azure OpenAI models in LiveKit Inference.

- **[Azure OpenAI Realtime API](https://docs.livekit.io/agents/models/realtime/plugins/azure-openai.md)**: Docs for Azure OpenAI Realtime API with the OpenAI plugin.

- **[Azure OpenAI LLM plugin](https://docs.livekit.io/agents/models/llm/plugins/azure-openai.md)**: Docs for Azure OpenAI LLMs with the OpenAI plugin.

- **[Azure OpenAI STT plugin](https://docs.livekit.io/agents/models/stt/plugins/azure-openai.md)**: Docs for Azure OpenAI STT with the OpenAI plugin.

- **[Azure OpenAI TTS plugin](https://docs.livekit.io/agents/models/tts/plugins/azure-openai.md)**: Docs for Azure OpenAI TTS with the OpenAI plugin.

- **[Azure Speech STT plugin](https://docs.livekit.io/agents/models/stt/plugins/azure.md)**: Docs for the Azure Speech STT plugin.

- **[Azure Speech TTS plugin](https://docs.livekit.io/agents/models/tts/plugins/azure.md)**: Docs for the Azure Speech TTS plugin.

---


For the latest version of this document, see [https://docs.livekit.io/agents/integrations/azure.md](https://docs.livekit.io/agents/integrations/azure.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).