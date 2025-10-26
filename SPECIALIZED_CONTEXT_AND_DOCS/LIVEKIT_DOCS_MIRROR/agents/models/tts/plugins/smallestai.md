LiveKit docs › Models › Text-to-speech (TTS) › Plugins › Smallest AI

---

# Smallest AI TTS plugin guide

> How to use the Smallest AI Waves TTS plugin for LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use the [Smallest AI](https://smallest.ai/text-to-speech) Waves platform as a TTS provider for your voice agents.

## Quick reference

This section includes a brief overview of the Smallest AI TTS plugin. For more information, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```shell
uv add "livekit-agents[smallestai]~=1.2"

```

### Authentication

The Smallest AI plugin requires an [API key](https://console.smallest.ai/apikeys).

Set `SMALLEST_API_KEY` in your `.env` file.

### Usage

Use Smallest AI TTS within an `AgentSession` or as a standalone speech generator. For example, you can use this TTS in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import smallestai

session = AgentSession(
    tts=smallestai.TTS(
        voice_id="irisha",
        sample_rate=24000,
        output_format="pcm",
    ),
    # ... llm, stt, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/smallestai/index.html.md) for a complete list of all available parameters.

- **`model`** _(TTSModels | string)_ (optional) - Default: `lightning`: Model to use.

- **`voice_id`** _(string)_ (optional): The voice ID for synthesis. Must be a valid Smallest AI voice identifier.

- **`sample_rate`** _(number)_ (optional) - Default: `24000`: Target audio sample rate in Hz. Match the rate to the rest of your audio pipeline to avoid resampling artifacts.

- **`output_format`** _(TTSOutputFormat | string)_ (optional) - Default: `pcm`: Encoding format for synthesized audio. Select a format based on if you want raw audio for streaming/processing or compressed for storage/playback.

## Additional resources

The following resources provide more information about using Smallest AI with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-smallestai/)**: The `livekit-plugins-smallestai` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/smallestai/index.html.md)**: Reference for the Smallest AI TTS plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-smallestai)**: View the source or contribute to the LiveKit Smallest AI TTS plugin.

- **[Smallest AI docs](https://waves-docs.smallest.ai/v3.0.1/content/introduction/introduction)**: Smallest AI's Waves TTS docs.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Smallest AI TTS.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/plugins/smallestai.md](https://docs.livekit.io/agents/models/tts/plugins/smallestai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).