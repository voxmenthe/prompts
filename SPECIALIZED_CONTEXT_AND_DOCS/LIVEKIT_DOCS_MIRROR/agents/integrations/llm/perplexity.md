LiveKit Docs › Integration guides › Large language models (LLM) › Perplexity

---

# Perplexity LLM integration guide

> How to use Perplexity LLM with LiveKit Agents.

## Overview

[Perplexity](https://www.perplexity.ai/) provides access to their Sonar models, which are based on Llama 3.1 but fine-tuned for search, through their inference API. These models are multilingual and text-only, making them suitable for a variety of agent applications.

## Usage

Install the OpenAI plugin to add Perplexity support:

```shell
pip install "livekit-agents[openai]~=1.2"

```

Set the following environment variable in your `.env` file:

```shell
PERPLEXITY_API_KEY=<your-perplexity-api-key>

```

Create a Perplexity LLM using the `with_perplexity` method:

```python
from livekit.plugins import openai

session = AgentSession(
    llm=openai.LLM.with_perplexity(
        model="llama-3.1-sonar-small-128k-chat",
        temperature=0.7
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

## Parameters

This section describes some of the available parameters. For a complete reference of all available parameters, see the [method reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.LLM.with_perplexity).

- **`model`** _(str | PerplexityChatModels)_ (optional) - Default: `llama-3.1-sonar-small-128k-chat`: Model to use for inference. To learn more, see [supported models](https://docs.perplexity.ai/guides/model-cards).

- **`temperature`** _(float)_ (optional) - Default: `1.0`: Controls the randomness of the model's output. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more focused and deterministic.

Valid values are between `0` and `2`.

- **`parallel_tool_calls`** _(bool)_ (optional): Controls whether the model can make multiple tool calls in parallel. When enabled, the model can make multiple tool calls simultaneously, which can improve performance for complex tasks.

- **`tool_choice`** _(ToolChoice | Literal['auto', 'required', 'none'])_ (optional) - Default: `auto`: Controls how the model uses tools. Set to 'auto' to let the model decide, 'required' to force tool usage, or 'none' to disable tool usage.

## Links

The following links provide more information about the Perplexity LLM integration.

- **[Python package](https://pypi.org/project/livekit-plugins-openai/)**: The `livekit-plugins-openai` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.LLM.with_perplexity)**: Reference for the `with_perplexity` method of the OpenAI LLM plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai)**: View the source or contribute to the LiveKit OpenAI LLM plugin.

- **[Perplexity docs](https://docs.perplexity.ai/docs/overview)**: Perplexity API documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Perplexity.

---

This document was rendered at 2025-08-13T22:17:06.591Z.
For the latest version of this document, see [https://docs.livekit.io/agents/integrations/llm/perplexity.md](https://docs.livekit.io/agents/integrations/llm/perplexity.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).