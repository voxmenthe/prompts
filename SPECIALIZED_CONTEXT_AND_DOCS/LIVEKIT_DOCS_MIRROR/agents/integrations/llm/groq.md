LiveKit Docs › Integration guides › Large language models (LLM) › Groq

---

# Groq LLM integration guide

> How to use the Groq LLM plugin for LiveKit Agents.

## Overview

[Groq](https://groq.com/) provides fast LLM inference using open models from Llama, DeepSeek, and more. With LiveKit's Groq integration and the Agents framework, you can build low-latency voice AI applications.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```bash
pip install "livekit-agents[groq]~=1.2"

```

### Authentication

The Groq plugin requires a [Groq API key](https://console.groq.com/keys).

Set `GROQ_API_KEY` in your `.env` file.

### Usage

Use a Groq LLM in your `AgentSession` or as a standalone LLM service. For example, you can use this LLM in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import groq

session = AgentSession(
    llm=groq.LLM(
        model="llama3-8b-8192"
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

### Parameters

This section describes some of the available parameters. For a complete reference of all available parameters, see the [plugin reference](https://docs.livekit.io/python/v1/livekit/plugins/groq/services.html.md#livekit.plugins.groq.services.LLM).

- **`model`** _(string)_ (optional) - Default: `llama-3.3-70b-versatile`: Name of the LLM model to use. For all options, see the [Groq model list](https://console.groq.com/docs/models).

- **`temperature`** _(float)_ (optional) - Default: `1.0`: A measure of randomness in output. A lower value results in more predictable output, while a higher value results in more creative output.

- **`parallel_tool_calls`** _(bool)_ (optional): Set to true to parallelize tool calls.

- **`tool_choice`** _(ToolChoice | Literal['auto', 'required', 'none'])_ (optional) - Default: `auto`: Specifies whether to use tools during response generation.

## Additional resources

The following resources provide more information about using Groq with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-groq/)**: The `livekit-plugins-groq` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/groq/index.html.md#livekit.plugins.groq.LLM)**: Reference for the Groq LLM plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-groq)**: View the source or contribute to the LiveKit Groq LLM plugin.

- **[Groq docs](https://console.groq.com/docs/overview)**: Groq docs.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Groq.

- **[Groq ecosystem overview](https://docs.livekit.io/agents/integrations/groq.md)**: Overview of the entire Groq and LiveKit Agents integration.

---

This document was rendered at 2025-08-13T22:17:06.297Z.
For the latest version of this document, see [https://docs.livekit.io/agents/integrations/llm/groq.md](https://docs.livekit.io/agents/integrations/llm/groq.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).