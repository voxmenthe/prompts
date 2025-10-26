LiveKit docs › Models › Speech-to-text (STT) › Plugins › Cartesia

---

# Cartesia STT plugin guide

> How to use the Cartesia STT plugin for LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use [Cartesia](https://www.cartesia.ai/) as an STT provider for your voice agents.

> 💡 **LiveKit Inference**
> 
> Cartesia STT is also available in LiveKit Inference, with billing and integration handled automatically. See [the docs](https://docs.livekit.io/agents/models/stt/inference/cartesia.md) for more information.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```shell
uv add "livekit-agents[cartesia]~=1.2"

```

### Authentication

The Cartesia plugin requires a [Cartesia API key](https://play.cartesia.ai/keys).

Set `CARTESIA_API_KEY` in your `.env` file.

### Usage

Use Cartesia STT in an `AgentSession` or as a standalone transcription service. For example, you can use this STT in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import cartesia

session = AgentSession(
   stt = cartesia.STT(
      model="ink-whisper"
   ),
   # ... llm, tts, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/cartesia/index.html.md#livekit.plugins.cartesia.STT) for a complete list of all available parameters.

- **`model`** _(string)_ (optional) - Default: `ink-whisper`: Selected model to use for STT. See [Cartesia STT models](https://docs.cartesia.ai/build-with-cartesia/models/stt) for supported values.

- **`language`** _(string)_ (optional) - Default: `en`: Language of input audio in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) format. See [Cartesia STT models](https://docs.cartesia.ai/build-with-cartesia/models/stt) for supported values.

## Additional resources

The following resources provide more information about using Cartesia with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-cartesia/)**: The `livekit-plugins-cartesia` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/cartesia/index.html.md#livekit.plugins.cartesia.STT)**: Reference for the Cartesia STT plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-cartesia)**: View the source or contribute to the LiveKit Cartesia STT plugin.

- **[Cartesia docs](https://docs.cartesia.ai/build-with-cartesia/models/stt)**: Cartesia STT docs.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Cartesia STT.

- **[Cartesia TTS](https://docs.livekit.io/agents/models/tts/plugins/cartesia.md)**: Guide to the Cartesia TTS plugin with LiveKit Agents.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/stt/plugins/cartesia.md](https://docs.livekit.io/agents/models/stt/plugins/cartesia.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).