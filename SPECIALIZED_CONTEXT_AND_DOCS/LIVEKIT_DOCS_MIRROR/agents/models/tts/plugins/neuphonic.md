LiveKit Docs › Models › Text-to-speech (TTS) › Plugins › Neuphonic

---

# Neuphonic TTS plugin guide

> How to use the Neuphonic TTS plugin for LiveKit Agents.

Available in:
- [x] Node.js
- [x] Python

## Overview

This plugin allows you to use [Neuphonic](https://neuphonic.com/) as a TTS provider for your voice agents.

## Quick reference

This section includes a brief overview of the Neuphonic TTS plugin. For more information, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```bash
pip install "livekit-agents[neuphonic]~=1.2"

```

### Authentication

The Neuphonic plugin requires a [Neuphonic API key](https://app.neuphonic.com/apikey).

Set `NEUPHONIC_API_TOKEN` in your `.env` file.

### Usage

Use Neuphonic TTS within an `AgentSession` or as a standalone speech generator. For example, you can use this TTS in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import neuphonic

session = AgentSession(
   tts=neuphonic.TTS(
      voice_id="fc854436-2dac-4d21-aa69-ae17b54e98eb"
   ),
   # ... llm, stt, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/neuphonic/index.html.md#livekit.plugins.neuphonic.TTS) for a complete list of all available parameters.

- **`voice_id`** _(string)_: ID of the voice to use for generation.

- **`speed`** _(float)_ (optional) - Default: `1`: Speed of generated speech.

- **`model`** _(string)_ (optional) - Default: `neu_hq`: ID of the model to use for generation.

- **`lang_code`** _(string)_ (optional) - Default: `en`: Language code for the generated speech.

## Additional resources

The following resources provide more information about using Neuphonic with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-neuphonic/)**: The `livekit-plugins-neuphonic` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/neuphonic/index.html.md#livekit.plugins.neuphonic.TTS)**: Reference for the Neuphonic TTS plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-neuphonic)**: View the source or contribute to the LiveKit Neuphonic TTS plugin.

- **[Neuphonic documentation](https://docs.neuphonic.com/)**: Neuphonic's full documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Neuphonic TTS.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/plugins/neuphonic.md](https://docs.livekit.io/agents/models/tts/plugins/neuphonic.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).