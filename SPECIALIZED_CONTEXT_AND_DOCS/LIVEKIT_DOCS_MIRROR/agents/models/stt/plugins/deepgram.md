LiveKit Docs › Models › Speech-to-text (STT) › Plugins › Deepgram

---

# Deepgram STT plugin guide

> How to use the Deepgram STT plugin for LiveKit Agents.

Available in:
- [x] Node.js
- [x] Python

## Overview

This plugin allows you to use [Deepgram](https://deepgram.com/) as an STT provider for your voice agents.

> 💡 **LiveKit Inference**
> 
> Deepgram STT is also available in LiveKit Inference, with billing and integration handled automatically. See [the docs](https://docs.livekit.io/agents/models/stt/inference/deepgram.md) for more information.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

**Python**:

```bash
pip install "livekit-agents[deepgram]~=1.2"

```

---

**Node.js**:

```bash
pnpm add @livekit/agents-plugin-deepgram@1.x

```

### Authentication

The Deepgram plugin requires a [Deepgram API key](https://console.deepgram.com/).

Set `DEEPGRAM_API_KEY` in your `.env` file.

### Usage

Use Deepgram STT in an `AgentSession` or as a standalone transcription service. For example, you can use this STT in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

**Python**:

```python
from livekit.plugins import deepgram

session = AgentSession(
   stt=deepgram.STTv2(
      model="flux-general-en",
      eager_eot_threshold=0.4,
   ),
   # ... llm, tts, etc.
)

```

---

**Node.js**:

```typescript
import * as deepgram from '@livekit/agents-plugin-deepgram';

const session = new voice.AgentSession({
    stt: new deepgram.STT(
        model: "nova-3"
    ),
    // ... llm, tts, etc.
});

```

### Parameters

This section describes some of the available parameters. See the plugin reference links in the [Additional resources](#additional-resources) section for more details.

- **`model`** _(string)_ (optional) - Default: `nova-3`: The Deepgram model to use for speech recognition.

- **`keyterms`** _(list[string])_ (optional) - Default: `[]`: List of key terms to improve recognition accuracy. Supported by Nova-3 models.

## Additional resources

The following resources provide more information about using Deepgram with LiveKit Agents.

- **[Deepgram docs](https://developers.deepgram.com/docs)**: Deepgram's full docs site.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Deepgram.

- **[Deepgram TTS](https://docs.livekit.io/agents/models/tts/plugins/deepgram.md)**: Guide to the Deepgram TTS plugin with LiveKit Agents.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/stt/plugins/deepgram.md](https://docs.livekit.io/agents/models/stt/plugins/deepgram.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).