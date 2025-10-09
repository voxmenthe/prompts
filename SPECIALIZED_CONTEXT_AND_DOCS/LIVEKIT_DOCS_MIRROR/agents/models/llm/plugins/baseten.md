LiveKit Docs › Models › Large language models (LLM) › Plugins › Baseten

---

# Baseten LLM plugin guide

> How to use the Baseten LLM plugin for LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use [Baseten](https://www.baseten.co/) as an LLM provider for your voice agents.

> 💡 **LiveKit Inference**
> 
> Some Baseten models are also available in LiveKit Inference, with billing and integration handled automatically. See [the docs](https://docs.livekit.io/agents/models/llm.md) for more information.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```bash
pip install "livekit-agents[baseten]~=1.2"

```

### Authentication

The Baseten plugin requires a [Baseten API key](https://app.baseten.co/settings/api-keys).

Set the following in your `.env` file:

```shell
BASETEN_API_KEY=<your-baseten-api-key>

```

### Model selection

LiveKit Agents integrates with Baseten's Model API, which supports the most popular open source LLMs with per-token billing. To use the Model API, you only need to activate the model and then copy its name.

1. Activate your desired model in the [Model API](https://app.baseten.co/model-apis/create)
2. Copy its name from your model API endpoint dialog in your [model library](https://app.baseten.co/model-apis)
3. Use the model name in the plugin (e.g. `"openai/gpt-oss-120b"`)

### Usage

Use a Baseten LLM in your `AgentSession` or as a standalone LLM service. For example, you can use this LLM in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import baseten

session = AgentSession(
    llm=baseten.LLM(
        model="openai/gpt-oss-120b"
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

### Parameters

This section describes some of the available parameters. For a complete reference of all available parameters, see the [plugin reference](https://docs.livekit.io/python/v1/livekit/plugins/groq/services.html.md#livekit.plugins.groq.services.LLM).

- **`model`** _(string)_ (optional) - Default: `meta-llama/Llama-4-Maverick-17B-128E-Instruct`: Name of the LLM model to use from the [Model API](https://www.baseten.co/model-apis). See [Model selection](#model-selection) for more information.

## Additional resources

The following resources provide more information about using Baseten with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-baseten/)**: The `livekit-plugins-baseten` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/baseten/index.html.md#livekit.plugins.baseten.LLM)**: Reference for the Baseten LLM plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-baseten)**: View the source or contribute to the LiveKit Baseten LLM plugin.

- **[Baseten docs](https://docs.baseten.co/)**: Baseten docs.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Baseten.

- **[Baseten TTS](https://docs.livekit.io/agents/models/tts/plugins/baseten.md)**: Baseten TTS integration guide.

- **[Baseten STT](https://docs.livekit.io/agents/models/stt/plugins/baseten.md)**: Baseten STT integration guide.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/llm/plugins/baseten.md](https://docs.livekit.io/agents/models/llm/plugins/baseten.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).