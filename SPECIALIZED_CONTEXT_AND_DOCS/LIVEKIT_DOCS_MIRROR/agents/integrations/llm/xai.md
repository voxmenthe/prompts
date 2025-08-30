LiveKit Docs › Integration guides › Large language models (LLM) › xAI

---

# xAI LLM integration guide

> How to use xAI LLM with LiveKit Agents.

Available in:
- [x] Node.js
- [x] Python

## Overview

[xAI](https://x.ai/) provides access to Grok models through their OpenAI-compatible API. These models are multilingual and support multimodal capabilities, making them suitable for a variety of agent applications.

## Usage

Install the OpenAI plugin to add xAI support:

```shell
pip install "livekit-agents[openai]~=1.2"

```

Set the following environment variable in your `.env` file:

```shell
XAI_API_KEY=<your-xai-api-key>

```

Create a Grok LLM using the `with_x_ai` method:

```python
from livekit.plugins import openai

session = AgentSession(
    llm=openai.LLM.with_x_ai(
        model="grok-2-public",
        temperature=1.0,
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

## Parameters

This section describes some of the available parameters. For a complete reference of all available parameters, see the [method reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.LLM.with_x_ai).

- **`model`** _(str | XAIChatModels)_ (optional) - Default: `grok-2-public`: Grok model to use. To learn more, see the [xAI Grok models](https://docs.x.ai/docs/models) page.

- **`temperature`** _(float)_ (optional) - Default: `1.0`: Controls the randomness of the model's output. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more focused and deterministic.

Valid values are between `0` and `2`. To learn more, see the optional parameters for [Chat completions](https://docs.x.ai/docs/api-reference#chat-completions)

- **`parallel_tool_calls`** _(bool)_ (optional): Controls whether the model can make multiple tool calls in parallel. When enabled, the model can make multiple tool calls simultaneously, which can improve performance for complex tasks.

- **`tool_choice`** _(ToolChoice | Literal['auto', 'required', 'none'])_ (optional) - Default: `auto`: Controls how the model uses tools. Set to 'auto' to let the model decide, 'required' to force tool usage, or 'none' to disable tool usage.

## Links

The following links provide more information about the xAI Grok LLM integration.

- **[Python package](https://pypi.org/project/livekit-plugins-openai/)**: The `livekit-plugins-openai` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.LLM.with_x_ai)**: Reference for the `with_x_ai` method of the OpenAI LLM plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai)**: View the source or contribute to the LiveKit OpenAI LLM plugin.

- **[xAI docs](https://docs.x.ai/docs/overview)**: xAI Grok documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and xAI Grok.

---


For the latest version of this document, see [https://docs.livekit.io/agents/integrations/llm/xai.md](https://docs.livekit.io/agents/integrations/llm/xai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).