LiveKit docs › Models › Overview

---

# Models

> Choose the right AI models for your voice agent.

## Overview

Voice agents require one or more AI models to provide understanding, intelligence, and speech. You can choose to use a high-performance STT-LLM-TTS voice pipeline constructed from multiple specialized models, or to use a realtime model with direct speech-to-speech capabilities.

LiveKit Agents includes support for a wide variety of AI providers, from the largest research companies to emerging startups. You can use LiveKit Inference to access many of these models [directly through LiveKit Cloud](#inference), or you can use the open source [plugins](#plugins) to connect directly to a wider range of model providers.

## Models

The following guides cover all models available in LiveKit Agents, both through LiveKit Inference and additional plugins. Refer to these guides for model availability, configuration options, and usage instructions.

- **[Large language models (LLM)](https://docs.livekit.io/agents/models/llm.md)**: Chat and reasoning models from the largest research companies and emerging startups.

- **[Speech-to-text (STT)](https://docs.livekit.io/agents/models/stt.md)**: Transcription models from providers including Deepgram and AssemblyAI.

- **[Text-to-speech (TTS)](https://docs.livekit.io/agents/models/tts.md)**: Speech models and custom voices from providers including Cartesia and ElevenLabs.

- **[Realtime models](https://docs.livekit.io/agents/models/realtime.md)**: Speech-to-speech models including the OpenAI Realtime API and Gemini Live.

- **[Virtual avatars](https://docs.livekit.io/agents/models/avatar.md)**: Realtime video avatars from providers including Hedra and Tavus.

## Usage

Use models with the `AgentSession` class. This class accepts models in the `stt`, `tts`, and `llm` arguments. You can pass a string descriptor for a model available on LiveKit Inference, or an instance of the `LLM`, `STT`, `TTS`, or `RealtimeModel` class from a plugin.

For instance, a simple `AgentSession` built on LiveKit Inference might look like the following:

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    stt="assemblyai/universal-streaming:en",
    llm="openai/gpt-4.1-mini",
    tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    stt: "assemblyai/universal-streaming:en",
    llm: "openai/gpt-4.1-mini",
    tts: "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
});

```

To use plugins instead, you can configure it like this:

**Python**:

```python
from livekit.agents import AgentSession
from livekit.plugins import openai, cartesia, assemblyai

session = AgentSession(
    llm=openai.LLM(model="gpt-4.1-mini"),
    tts=cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    stt=assemblyai.STT(language="en"),
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';
import * as openai from '@livekit/agents-plugin-openai';
import * as cartesia from '@livekit/agents-plugin-cartesia';
import * as assemblyai from '@livekit/agents-plugin-assemblyai';

session = new AgentSession({
    llm: new openai.LLM(model="gpt-4.1-mini"),
    tts: new cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    stt: new assemblyai.STT(language="en"),
});

```

You can use a combination of LiveKit Inference and plugins to build your voice agent. Additionally, you can change models during a session to optimize for different use cases or conversation phases. For more information, see [Workflows](https://docs.livekit.io/agents/build/workflows.md).

## LiveKit Inference

![Overview showing LiveKit Inference serving a STT-LLM-TTS pipeline for a voice agent.](/images/agents/inference.svg)

LiveKit Inference provides access to many of the best models and providers for voice agents, including models from OpenAI, Google, AssemblyAI, Deepgram, Cartesia, ElevenLabs and more. LiveKit Inference is included in LiveKit Cloud, and does not require any additional plugins. See the guides for [LLM](https://docs.livekit.io/agents/models/llm.md), [STT](https://docs.livekit.io/agents/models/stt.md), and [TTS](https://docs.livekit.io/agents/models/tts.md) for supported models and configuration options.

If you're interested in learning more about LiveKit Inference, see the blog post [Introducing LiveKit Inference: A unified model interface for voice AI](https://blog.livekit.io/introducing-livekit-inference/).

> ℹ️ **Agents SDK version**
> 
> LiveKit Inference requires the latest Agents SDK versions:
> 
> - Python SDK v1.2.13 or greater
> - Node.js SDK v1.0.7 or greater

### Billing

Inference billing is based on usage, with competitive rates for each supported model. Refer to the following articles for more information on quotas, limits, and billing for LiveKit Inference. The latest pricing is always available on the [LiveKit Inference pricing page](https://livekit.io/pricing/inference).

- **[Quotas and limits](https://docs.livekit.io/home/cloud/quotas-and-limits.md)**: Guide to quotas and limits for LiveKit Cloud plans.

- **[Billing](https://docs.livekit.io/home/cloud/billing.md)**: Guide to LiveKit Cloud invoices and billing cycles.

## Plugins

LiveKit Agents includes a large ecosystem of open source plugins for a variety of AI providers. Each plugin is designed to support a single provider, but may cover a range of functionality depending on the provider. For instance, the OpenAI plugin includes support for OpenAI language models, speech, transcription, and the Realtime API.

For Python, the plugins are offered as optional dependencies on the base SDK. For instance, to install the SDK with the OpenAI plugin, run the following command:

```shell
uv add "livekit-agents[openai]~=1.2"

```

For Node.js, the plugins are offered as individual packages. For instance, to install the OpenAI plugin, use the following command:

```shell
pnpm add "@livekit/agents-plugin-openai@1.x"

```

Each plugin requires that you have your own account with the provider, as well as an API key or other credentials. You can find authentication instructions in the documentation for each individual plugin.

### OpenAI API compatibility

Many providers have standardized around the OpenAI API format for chat completions and more. Support for a number of these providers is included out-of-the-box with the OpenAI plugin, and you can find specific instructions in the associated documentation. For any provider not included, you can override the API key and base URL at initialization for the LLM, STT, and TTS interfaces in the plugin.

**Python**:

```python
from livekit.plugins import openai
import os

session = AgentSession(
   llm=openai.LLM(
      model="model-name", 
      base_url="https://api.provider.com/v1", 
      api_key=os.getenv("PROVIDER_API_KEY")
   ),
    # ... stt, tts, etc ...
)

```

---

**Node.js**:

```typescript
import * as openai from '@livekit/agents-plugin-openai';

const session = new voice.AgentSession({
   llm: openai.LLM({ 
      model: "model-name", 
      baseURL: "https://api.provider.com/v1", 
      apiKey: process.env.PROVIDER_API_KEY
   }),
   // ... stt, tts, etc ...
});

```

### Contributing

The LiveKit Agents plugin framework is extensible and community-driven. Your plugin can integrate with new providers or directly load models for local inference. LiveKit especially welcomes new TTS, STT, and LLM plugins.

To learn more, see the guidelines for contributions to the [Python](https://github.com/livekit/agents/?tab=contributing-ov-file) and [Node.js](https://github.com/livekit/agents-js/?tab=contributing-ov-file) SDKs.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models.md](https://docs.livekit.io/agents/models.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).