LiveKit docs › Models › Text-to-speech (TTS) › Plugins › Rime

---

# Rime TTS plugin guide

> How to use the Rime TTS plugin for LiveKit Agents.

Available in:
- [x] Node.js
- [x] Python

## Overview

This plugin allows you to use [Rime](https://rime.ai/) as a TTS provider for your voice agents.

> 💡 **LiveKit Inference**
> 
> Rime TTS is also available in LiveKit Inference, with billing and integration handled automatically. See [the docs](https://docs.livekit.io/agents/models/tts/inference/rime.md) for more information.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin:

**Python**:

```shell
uv add "livekit-agents[rime]~=1.2"

```

---

**Node.js**:

```shell
pnpm add @livekit/agents-plugin-rime@1.x

```

### Authentication

The Rime plugin requires a [Rime API key](https://rime.ai/).

Set `RIME_API_KEY` in your `.env` file.

### Usage

Use Rime TTS within an `AgentSession` or as a standalone speech generator. For example, you can use this TTS in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

**Python**:

```python
from livekit.plugins import rime

session = AgentSession(
   tts=rime.TTS(
      model="mistv2",
      speaker="rainforest",
      speed_alpha=0.9,
      reduce_latency=True,
   ),
   # ... llm, stt, etc.
)

```

---

**Node.js**:

```typescript
import * as rime from '@livekit/agents-plugin-rime';

const session = new voice.AgentSession({
    tts: new rime.TTS({
      modelId: "mistv2",
      speaker: "rainforest",
      speedAlpha: 0.9,
      reduceLatency: true,
    }),
    // ... llm, tts, etc.
});

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/rime/index.html.md#livekit.plugins.rime.TTS) for a complete list of all available parameters.

- **`model`** _(string)_ (optional) - Default: `mist`: ID of the model to use. To learn more, see [Models](https://docs.rime.ai/api-reference/models).

- **`speaker`** _(string)_ (optional) - Default: `lagoon`: ID of the voice to use for speech generation. To learn more, see [Voices](https://docs.rime.ai/api-reference/voices).

- **`audio_format`** _(TTSEncoding)_ (optional) - Default: `pcm`: Audio format to use. Valid values are: `pcm` and `mp3`.

- **`sample_rate`** _(integer)_ (optional) - Default: `16000`: Sample rate of the generated audio. Set this rate to best match your application needs. To learn more, see [Recommendations for reducing response time](https://docs.rime.ai/api-reference/latency#recommendations-for-reducing-response-time).

- **`speed_alpha`** _(float)_ (optional) - Default: `1.0`: Adjusts the speed of speech. Lower than `1.0` results in faster speech; higher than `1.0` results in slower speech.

- **`reduce_latency`** _(boolean)_ (optional) - Default: `false`: When set to `true`, turns off text normalization to reduce the amount of time spent preparing input text for TTS inference. This might result in the mispronunciation of digits and abbreviations. To learn more, see [Recommendations for reducing response time](https://docs.rime.ai/api-reference/latency#recommendations-for-reducing-response-time).

- **`phonemize_between_brackets`** _(boolean)_ (optional) - Default: `false`: When set to `true`, allows the use of custom pronunciation strings in text. To learn more, see [Custom pronunciation](https://docs.rime.ai/api-reference/custom-pronunciation).

- **`api_key`** _(string)_ (optional) - Environment: `RIME_API_KEY`: Rime API Key. Required if the environment variable isn't set.

## Customizing pronunciation

Rime TTS supports customizing pronunciation. To learn more, see [Custom Pronunciation guide](https://docs.rime.ai/api-reference/custom-pronunciation).

## Additional resources

The following resources provide more information about using Rime with LiveKit Agents.

- **[Rime docs](https://docs.rime.ai)**: Rime TTS docs.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Rime TTS.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/plugins/rime.md](https://docs.livekit.io/agents/models/tts/plugins/rime.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).