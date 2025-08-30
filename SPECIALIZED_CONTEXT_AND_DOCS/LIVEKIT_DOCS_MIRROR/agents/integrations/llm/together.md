LiveKit Docs › Integration guides › Large language models (LLM) › Together AI

---

# Together AI LLM integration guide

> How to use Together AI Llama models with LiveKit Agents.

Available in:
- [x] Node.js
- [x] Python

## Overview

[Together AI](https://www.together.ai/) provides access to Llama 2 and Llama 3 models including instruction-tuned models through their inference API. These models are multilingual and text-only, making them suitable for a variety of agent applications.

## Usage

Install the OpenAI plugin to add Together AI support:

```shell
pip install "livekit-agents[openai]~=1.2"

```

Set the following environment variable in your `.env` file:

```shell
TOGETHER_API_KEY=<your-together-api-key>

```

Create a Together AI LLM using the `with_together` method:

```python
from livekit.plugins import openai

session = AgentSession(
    llm=openai.LLM.with_together(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.7
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

## Parameters

This section describes some of the available parameters. For a complete reference of all available parameters, see the [method reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.LLM.with_together).

- **`model`** _(str | TogetherChatModels)_ (optional) - Default: `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`: Model to use for inference. To learn more, see [supported models](https://docs.together.ai/docs/inference-models).

- **`temperature`** _(float)_ (optional) - Default: `1.0`: Controls the randomness of the model's output. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more focused and deterministic.

Valid values are between `0` and `1`.

- **`parallel_tool_calls`** _(bool)_ (optional): Controls whether the model can make multiple tool calls in parallel. When enabled, the model can make multiple tool calls simultaneously, which can improve performance for complex tasks.

- **`tool_choice`** _(ToolChoice | Literal['auto', 'required', 'none'])_ (optional) - Default: `auto`: Controls how the model uses tools. Set to 'auto' to let the model decide, 'required' to force tool usage, or 'none' to disable tool usage.

## Links

The following links provide more information about the Together AI LLM integration.

- **[Python package](https://pypi.org/project/livekit-plugins-openai/)**: The `livekit-plugins-openai` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.LLM.with_together)**: Reference for the `with_together` method of the OpenAI LLM plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai)**: View the source or contribute to the LiveKit OpenAI LLM plugin.

- **[Together AI docs](https://docs.together.ai/docs/overview)**: Together AI API documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Together AI.

---


For the latest version of this document, see [https://docs.livekit.io/agents/integrations/llm/together.md](https://docs.livekit.io/agents/integrations/llm/together.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).