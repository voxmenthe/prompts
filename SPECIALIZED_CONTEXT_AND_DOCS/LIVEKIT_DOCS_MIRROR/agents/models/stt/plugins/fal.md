LiveKit Docs › Models › Speech-to-text (STT) › Plugins › fal

---

# fal STT plugin guide

> How to use the fal STT plugin for LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use [fal](https://fal.ai/) as an STT provider for your voice agents.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```bash
pip install "livekit-agents[fal]~=1.2"

```

### Authentication

The fal plugin requires a [fal API key](https://fal.ai/dashboard/keys).

Set `FAL_KEY` in your `.env` file.

### Usage

Use fal STT in an `AgentSession` or as a standalone transcription service. For example, you can use this STT in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import fal

session = AgentSession(
   stt = fal.STT(
      language="de",
   ),
   # ... llm, tts, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/fal/index.html.md#livekit.plugins.fal.STT) for a complete list of all available parameters.

- **`language`** _(str)_ (optional) - Default: `en`: Speech recognition language.

## Additional resources

The following resources provide more information about using fal with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-fal/)**: The `livekit-plugins-fal` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/fal/index.html.md#livekit.plugins.fal.STT)**: Reference for the fal STT plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-fal)**: View the source or contribute to the LiveKit fal STT plugin.

- **[fal docs](https://fal.ai/docs)**: fal's full docs site.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and fal.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/stt/plugins/fal.md](https://docs.livekit.io/agents/models/stt/plugins/fal.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).