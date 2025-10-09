LiveKit Docs › Partner spotlight › OpenAI › OpenAI TTS Plugin

---

# OpenAI TTS plugin guide

> How to use the OpenAI TTS plugin for LiveKit Agents.

Available in:
- [x] Node.js
- [x] Python

## Overview

This plugin allows you to use [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech) as a TTS provider for your voice agents.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

**Python**:

```bash
pip install "livekit-agents[openai]~=1.2"

```

---

**Node.js**:

```bash
pnpm add @livekit/agents-plugin-openai@1.x

```

### Authentication

The OpenAI plugin requires an [OpenAI API key](https://platform.openai.com/api-keys).

Set `OPENAI_API_KEY` in your `.env` file.

### Usage

Use OpenAI TTS in an `AgentSession` or as a standalone speech generator. For example, you can use this TTS in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

**Python**:

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

---

**Node.js**:

```typescript
import * as openai from '@livekit/agents-plugin-openai';

const session = new voice.AgentSession({
    tts: new openai.TTS(
        model: "gpt-4o-mini-tts",
        voice: "ash",
        instructions: "Speak in a friendly and conversational tone.",
    ),
    // ... llm, stt, etc.
});

```

### Parameters

This section describes some of the available parameters. See the plugin reference links in the [Additional resources](#additional-resources) section for a complete list of all available parameters.

- **`model`** _(TTSModels | string)_ (optional) - Default: `gpt-4o-mini-tts`: ID of the model to use for speech generation. To learn more, see [TTS models](https://platform.openai.com/docs/models#tts).

- **`voice`** _(TTSVoice | string)_ (optional) - Default: `ash`: ID of the voice used for speech generation. To learn more, see [TTS voice options](https://platform.openai.com/docs/guides/text-to-speech#voice-options).

- **`instructions`** _(string)_ (optional) - Default: ``: Instructions to control tone, style, and other characteristics of the speech. Does not work with `tts-1` or `tts-1-hd` models.

## Additional resources

The following resources provide more information about using OpenAI with LiveKit Agents.

- **[OpenAI docs](https://platform.openai.com/docs/guides/text-to-speech)**: OpenAI TTS docs.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and OpenAI TTS.

- **[OpenAI ecosystem guide](https://docs.livekit.io/agents/integrations/openai.md)**: Overview of the entire OpenAI and LiveKit Agents integration.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/plugins/openai.md](https://docs.livekit.io/agents/models/tts/plugins/openai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).