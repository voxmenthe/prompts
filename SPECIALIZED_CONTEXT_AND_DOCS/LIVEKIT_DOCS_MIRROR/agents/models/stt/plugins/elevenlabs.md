LiveKit docs › Models › Speech-to-text (STT) › Plugins › ElevenLabs

---

# ElevenLabs STT plugin guide

> How to use the ElevenLabs STT plugin for LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use [ElevenLabs](https://elevenlabs.io/) as an STT provider for your voice agents.

> 💡 **LiveKit Inference**
> 
> ElevenLabs STT is also available in LiveKit Inference, with billing and integration handled automatically. See [the docs](https://docs.livekit.io/agents/models/stt/inference/elevenlabs.md) for more information.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```shell
uv add "livekit-agents[elevenlabs]~=1.2"

```

### Authentication

The ElevenLabs plugin requires an [ElevenLabs API key](https://elevenlabs.io/app/settings/api-keys).

Set `ELEVEN_API_KEY` in your `.env` file.

### Usage

Use ElevenLabs STT in an `AgentSession` or as a standalone transcription service. For example, you can use this STT in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import elevenlabs

session = AgentSession(
   stt=elevenlabs.STT(
      model="scribe_v2_realtime",
   ),
   # ... llm, tts, etc.
)

```

### Parameters

This section describes some of the available parameters. See the plugin reference links in the [Additional resources](#additional-resources) section for more details.

- **`model`** _(string)_ (optional) - Default: `scribe_v2_realtime`: The ElevenLabs model to use for speech recognition.

## Additional resources

The following resources provide more information about using ElevenLabs with LiveKit Agents.

- **[ElevenLabs docs](https://elevenlabs.io/docs)**: ElevenLabs' full docs site.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and ElevenLabs.

- **[ElevenLabs TTS](https://docs.livekit.io/agents/models/tts/plugins/elevenlabs.md)**: Guide to the ElevenLabs TTS plugin with LiveKit Agents.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/stt/plugins/elevenlabs.md](https://docs.livekit.io/agents/models/stt/plugins/elevenlabs.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).