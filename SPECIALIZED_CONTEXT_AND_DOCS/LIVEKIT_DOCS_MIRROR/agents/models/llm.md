LiveKit docs › Models › Large language models (LLM) › Overview

---

# Large language models (LLM)

> Conversational intelligence for your voice agents.

## Overview

The core reasoning, response, and orchestration of your voice agent is powered by an LLM. You can choose between a variety of models to balance performance, accuracy, and cost. In a voice agent, your LLM receives a transcript of the user's speech from an [STT](https://docs.livekit.io/agents/models/stt.md) model, and produces a text response which is turned into speech by a [TTS](https://docs.livekit.io/agents/models/tts.md) model.

You can choose a model served through LiveKit Inference, which is included in LiveKit Cloud, or you can use a plugin to connect directly to a wider range of model providers with your own account.

## LiveKit Inference

The following models are available in [LiveKit Inference](https://docs.livekit.io/agents/models.md#inference). Refer to the guide for each model for more details on additional configuration options.

| Model family | Model name | Provided by |
| ------------- | ---------- | ----------- |
| OpenAI | GPT-4o | Azure, OpenAI |
|   | GPT-4o mini | Azure, OpenAI |
|   | GPT-4.1 | Azure, OpenAI |
|   | GPT-4.1 mini | Azure, OpenAI |
|   | GPT-4.1 nano | Azure, OpenAI |
|   | GPT-5 | Azure, OpenAI |
|   | GPT-5 mini | Azure, OpenAI |
|   | GPT-5 nano | Azure, OpenAI |
|   | GPT OSS 120B | Baseten, Groq, Cerebras |
| Gemini | Gemini 2.5 Pro | Google |
|   | Gemini 2.5 Flash | Google |
|   | Gemini 2.5 Flash Lite | Google |
|   | Gemini 2.0 Flash | Google |
|   | Gemini 2.0 Flash Lite | Google |
| Qwen | Qwen3 235B A22B Instruct | Baseten |
| Kimi | Kimi K2 Instruct | Baseten |
| DeepSeek | DeepSeek V3 | Baseten |

## Usage

To set up an LLM in an `AgentSession`, provide the model id to the `llm` argument. LiveKit Inference manages the connection to the model automatically. Consult the [models list](#inference) for available models.

**Python**:

```python
from livekit.agents import AgentSession

session = AgentSession(
    llm="openai/gpt-4.1-mini",
)

```

---

**Node.js**:

```typescript
import { AgentSession } from '@livekit/agents';

session = new AgentSession({
    llm: "openai/gpt-4.1-mini",
});

```

### Additional parameters

More configuration options, such as reasoning effort, are available for each model. To set additional parameters, use the `LLM` class from the `inference` module. Consult each model reference for examples and available parameters.

## Plugins

The LiveKit Agents framework also includes a variety of open source [plugins](https://docs.livekit.io/agents/models.md#plugins) for a wide range of LLM providers. Plugins are especially useful if you need custom or fine-tuned models. These plugins require authentication with the provider yourself, usually via an API key. You are responsible for setting up your own account and managing your own billing and credentials. The plugins are listed below, along with their availability for Python or Node.js.

| Provider | Python | Node.js |
| -------- | ------ | ------- |
| [Amazon Bedrock](https://docs.livekit.io/agents/models/llm/plugins/aws.md) | ✓ | — |
| [Anthropic](https://docs.livekit.io/agents/models/llm/plugins/anthropic.md) | ✓ | — |
| [Baseten](https://docs.livekit.io/agents/models/llm/plugins/baseten.md) | ✓ | — |
| [Google Gemini](https://docs.livekit.io/agents/models/llm/plugins/gemini.md) | ✓ | ✓ |
| [Groq](https://docs.livekit.io/agents/models/llm/plugins/groq.md) | ✓ | ✓ |
| [LangChain](https://docs.livekit.io/agents/models/llm/plugins/langchain.md) | ✓ | — |
| [Mistral AI](https://docs.livekit.io/agents/models/llm/plugins/mistralai.md) | ✓ | — |
| [OpenAI](https://docs.livekit.io/agents/models/llm/plugins/openai.md) | ✓ | ✓ |
| [Azure OpenAI](https://docs.livekit.io/agents/models/llm/plugins/azure-openai.md) | ✓ | ✓ |
| [Cerebras](https://docs.livekit.io/agents/models/llm/plugins/cerebras.md) | ✓ | ✓ |
| [DeepSeek](https://docs.livekit.io/agents/models/llm/plugins/deepseek.md) | ✓ | ✓ |
| [Fireworks](https://docs.livekit.io/agents/models/llm/plugins/fireworks.md) | ✓ | ✓ |
| [Letta](https://docs.livekit.io/agents/models/llm/plugins/letta.md) | ✓ | — |
| [Ollama](https://docs.livekit.io/agents/models/llm/plugins/ollama.md) | ✓ | ✓ |
| [OpenRouter](https://docs.livekit.io/agents/models/llm/plugins/openrouter.md) | ✓ | — |
| [Perplexity](https://docs.livekit.io/agents/models/llm/plugins/perplexity.md) | ✓ | ✓ |
| [Telnyx](https://docs.livekit.io/agents/models/llm/plugins/telnyx.md) | ✓ | ✓ |
| [Together AI](https://docs.livekit.io/agents/models/llm/plugins/together.md) | ✓ | ✓ |
| [xAI](https://docs.livekit.io/agents/models/llm/plugins/xai.md) | ✓ | ✓ |

Have another provider in mind? LiveKit is open source and welcomes [new plugin contributions](https://docs.livekit.io/agents/models.md#contribute).

## Advanced features

The following sections cover more advanced topics common to all LLM providers. For more detailed reference on individual provider configuration, consult the model reference or plugin documentation for that provider.

### Custom LLM

To create an entirely custom LLM, implement the [LLM node](https://docs.livekit.io/agents/build/nodes.md#llm_node) in your agent.

### Standalone usage

You can use an `LLM` instance as a standalone component with its streaming interface. It expects a `ChatContext` object, which contains the conversation history. The return value is a stream of `ChatChunk` objects. This interface is the same across all LLM providers, regardless of their underlying API design:

```python
from livekit.agents import ChatContext
from livekit.plugins import openai

llm = openai.LLM(model="gpt-4o-mini")
    
chat_ctx = ChatContext()
chat_ctx.add_message(role="user", content="Hello, this is a test message!")
    
async with llm.chat(chat_ctx=chat_ctx) as stream:
    async for chunk in stream:
        print("Received chunk:", chunk.delta)

```

### Vision

LiveKit Agents supports image input from URL or from [realtime video frames](https://docs.livekit.io/home/client/tracks.md). Consult your model provider for details on compatible image types, external URL support, and other constraints. For more information, see [Vision](https://docs.livekit.io/agents/build/vision.md).

## Additional resources

The following resources cover related topics that may be useful for your application.

- **[Workflows](https://docs.livekit.io/agents/build/workflows.md)**: How to model repeatable, accurate tasks with multiple agents.

- **[Tool definition and usage](https://docs.livekit.io/agents/build/tools.md)**: Let your agents call external tools and more.

- **[Inference pricing](https://livekit.io/pricing/inference)**: The latest pricing information for all models in LiveKit Inference.

- **[Realtime models](https://docs.livekit.io/agents/models/realtime.md)**: Realtime models like the OpenAI Realtime API, Gemini Live, and Amazon Nova Sonic.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/llm.md](https://docs.livekit.io/agents/models/llm.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).