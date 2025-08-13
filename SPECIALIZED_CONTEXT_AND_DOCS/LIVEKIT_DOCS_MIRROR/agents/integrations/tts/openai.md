LiveKit Docs › Integration guides › Text-to-speech (TTS) › OpenAI

---

# OpenAI TTS integration guide

> How to use the OpenAI TTS plugin for LiveKit Agents.

## Overview

[OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech) provides lifelike spoken audio through their latest model `gpt-4o-mini-tts` model or their well-tested `tts-1` and `tts-1-hd` models. With LiveKit's OpenAI TTS integration and the Agents framework, you can build voice AI applications that sound realistic and natural.

To learn more about TTS and generating agent speech, see [Agent speech](https://docs.livekit.io/agents/build/audio.md).

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

Use OpenAI TTS in an `AgentSession` or as a standalone speech generator. For example, you can use this TTS in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import openai

session = AgentSession(
  tts = openai.TTS(
    model="gpt-4o-mini-tts",
    voice="ash",
    instructions="Speak in a friendly and conversational tone.",
  ),
  # ... llm, stt, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.TTS) for a complete list of all available parameters.

- **`model`** _(TTSModels | string)_ (optional) - Default: `gpt-4o-mini-tts`: ID of the model to use for speech generation. To learn more, see [TTS models](https://platform.openai.com/docs/models#tts).

- **`voice`** _(TTSVoice | string)_ (optional) - Default: `ash`: ID of the voice used for speech generation. To learn more, see [TTS voice options](https://platform.openai.com/docs/guides/text-to-speech#voice-options).

- **`instructions`** _(string)_ (optional) - Default: ``: Instructions to control tone, style, and other characteristics of the speech. Does not work with `tts-1` or `tts-1-hd` models.

## Additional resources

The following resources provide more information about using OpenAI with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-openai/)**: The `livekit-plugins-openai` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.TTS)**: Reference for the OpenAI TTS plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai)**: View the source or contribute to the LiveKit OpenAI TTS plugin.

- **[OpenAI docs](https://platform.openai.com/docs/guides/text-to-speech)**: OpenAI TTS docs.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and OpenAI TTS.

- **[OpenAI ecosystem guide](https://docs.livekit.io/agents/integrations/openai.md)**: Overview of the entire OpenAI and LiveKit Agents integration.

---

This document was rendered at 2025-08-13T22:17:05.862Z.
For the latest version of this document, see [https://docs.livekit.io/agents/integrations/tts/openai.md](https://docs.livekit.io/agents/integrations/tts/openai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).