LiveKit docs › Models › Text-to-speech (TTS) › Plugins › Spitch

---

# Spitch TTS plugin guide

> How to use the Spitch TTS plugin for LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use [Spitch](https://spitch.app/) as a TTS provider for your voice agents.

## Quick reference

This section provides a quick reference for the Spitch TTS plugin. For more information, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```shell
uv add "livekit-agents[spitch]~=1.2"

```

### Authentication

The Spitch plugin requires a [Spitch API key](https://studio.spitch.app/api/keys).

Set `SPITCH_API_KEY` in your `.env` file.

### Usage

Use Spitch TTS within an `AgentSession` or as a standalone speech generator. For example, you can use this TTS in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import spitch

session = AgentSession(
   tts=spitch.TTS(
      language="en",
      voice="lina",
   )
   # ... llm, stt, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/spitch/index.html.md#livekit.plugins.spitch.TTS) for a complete list of all available parameters.

- **`language`** _(string)_ (optional) - Default: `en`: Language short code for the generated speech. For supported values, see [Spitch languages](https://docs.spitch.app/concepts/languages).

- **`voice`** _(string)_ (optional) - Default: `lina`: Voice to use for synthesis. For supported values, see [Spitch voices](https://docs.spitch.app/concepts/voices).

## Additional resources

The following resources provide more information about using Spitch with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-spitch/)**: The `livekit-plugins-spitch` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/spitch/index.html.md#livekit.plugins.spitch.TTS)**: Reference for the Spitch TTS plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-spitch)**: View the source or contribute to the LiveKit Spitch TTS plugin.

- **[Spitch docs](https://docs.spitch.app/)**: Spitch's official documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Spitch.

- **[Spitch STT](https://docs.livekit.io/agents/models/stt/plugins/spitch.md)**: Guide to the Spitch STT plugin with LiveKit Agents.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/plugins/spitch.md](https://docs.livekit.io/agents/models/tts/plugins/spitch.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).