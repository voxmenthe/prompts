LiveKit docs › Models › Large language models (LLM) › Plugins › Mistral AI

---

# Mistral AI LLM plugin guide

> How to integrate Mistral AI's La Plateforme inference service with LiveKit Agents.

## Overview

This plugin allows you to use [Mistral AI](https://mistral.ai/) as an LLM provider for your voice agents.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the LiveKit Mistral AI plugin from PyPI:

```shell
uv add "livekit-agents[mistralai]~=1.2"

```

### Authentication

The Mistral AI integration requires a [Mistral AI API key](https://console.mistral.ai/api-keys/).

Set the `MISTRAL_API_KEY` in your `.env` file.

### Usage

Use Mistral AI within an `AgentSession` or as a standalone LLM service. For example, you can use this LLM in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import mistralai

session = AgentSession(
    llm=mistralai.LLM(
        model="mistral-medium-latest"
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/mistralai.md#livekit.plugins.mistralai.LLM) for a complete list of all available parameters.

- **`model`** _(string | ChatModels)_ (optional) - Default: `ministral-8b-2410`: Which Mistral AI model to use. You can pass a string or a typed enum from `ChatModels`.

- **`temperature`** _(float)_ (optional): Controls the randomness of the model's output. Higher values, for example 0.8, make the output more random, while lower values, for example 0.2, make it more focused and deterministic.

## Additional resources

The following resources provide more information about using Mistral AI with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-mistralai)**: The `livekit-plugins-mistralai` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/mistralai.md#livekit.plugins.mistralai.LLM)**: Reference for the Mistral AI LLM plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-mistralai)**: View the source or contribute to the LiveKit Mistral AI LLM plugin.

- **[Mistral AI STT docs](https://docs.livekit.io/agents/models/stt/plugins/mistralai.md)**: Mistral AI STT documentation.

- **[Mistral AI docs](https://docs.mistral.ai/)**: Mistral AI platform documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Mistral AI.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/llm/plugins/mistralai.md](https://docs.livekit.io/agents/models/llm/plugins/mistralai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).