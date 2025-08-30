LiveKit Docs › Integration guides › Large language models (LLM) › OpenAI

---

# OpenAI LLM integration guide

> How to use the OpenAI LLM plugin for LiveKit Agents.

Available in:
- [x] Node.js
- [x] Python

## Overview

[OpenAI](https://openai.com/) provides powerful language models like `gpt-4o` and `o1`. With LiveKit's OpenAI integration and the Agents framework, you can build sophisticated voice AI applications using their industry-leading models.

> 💡 **Using Azure OpenAI?**
> 
> See our [Azure OpenAI LLM guide](https://docs.livekit.io/agents/integrations/llm/azure-openai.md).

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```bash
pip install "livekit-agents[openai]~=1.2"

```

### Authentication

The OpenAI plugin requires an [OpenAI API key](https://platform.openai.com/api-keys).

Set `OPENAI_API_KEY` in your `.env` file.

### Usage

Use OpenAI within an `AgentSession` or as a standalone LLM service. For example, you can use this LLM in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import openai

session = AgentSession(
    llm=openai.LLM(
        model="gpt-4o-mini"
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.LLM) for a complete list of all available parameters.

- **`model`** _(string)_ (optional) - Default: `gpt-4o-mini`: The model to use for the LLM. For more information, see the [OpenAI documentation](https://platform.openai.com/docs/models).

- **`temperature`** _(float)_ (optional) - Default: `0.8`: A measure of randomness in output. A lower value results in more predictable output, while a higher value results in more creative output.

- **`tool_choice`** _(ToolChoice | Literal['auto', 'required', 'none'])_ (optional) - Default: `auto`: Specifies whether to use tools during response generation.

## Additional resources

The following resources provide more information about using OpenAI with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-openai/)**: The `livekit-plugins-openai` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.LLM)**: Reference for the OpenAI LLM plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai)**: View the source or contribute to the LiveKit OpenAI LLM plugin.

- **[OpenAI docs](https://platform.openai.com/docs)**: OpenAI platform documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and OpenAI.

- **[OpenAI ecosystem overview](https://docs.livekit.io/agents/integrations/openai.md)**: Overview of the entire OpenAI and LiveKit Agents integration.

---


For the latest version of this document, see [https://docs.livekit.io/agents/integrations/llm/openai.md](https://docs.livekit.io/agents/integrations/llm/openai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).