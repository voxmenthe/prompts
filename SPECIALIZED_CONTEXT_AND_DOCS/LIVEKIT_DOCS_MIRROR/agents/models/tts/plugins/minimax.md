LiveKit docs › Models › Text-to-speech (TTS) › Plugins › MiniMax

---

# MiniMax TTS plugin guide

> How to use the MiniMax TTS plugin for LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use [MiniMax](https://www.minimax.io/) as a TTS provider for your voice agents.

## Quick reference

This section includes a brief overview of the MiniMax TTS plugin. For more information, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```bash
pip install "livekit-agents[minimax]~=1.2"

```

### Authentication

The MiniMax plugin requires a [MiniMax API key](https://platform.minimax.io/user-center/basic-information/interface-key).

Set `MINIMAX_API_KEY` in your `.env` file.

### Usage

Use MiniMax TTS within an `AgentSession` or as a standalone speech generator. For example, you can use this TTS in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import minimax

session = AgentSession(
    tts=minimax.TTS(
    ),
    # ... llm, stt, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/minimax.md) for a complete list of all available parameters.

- **`model`** _(TTSModel | string)_ (optional) - Default: `DEFAULT_MODEL`: MiniMax TTS model to use. To learn more, see [TTS model options](https://platform.minimax.io/docs/guides/models-intro#audio).

- **`voice`** _(TTSVoice | string)_ (optional) - Default: `DEFAULT_VOICE_ID`: MiniMax TTS voice to use.

- **`emotion`** _(TTSEmotion | string)_ (optional) - Default: `None`: Control emotional tone for speech. Valid values are `happy`, `sad`, `angry`, `fearful`, `disgusted`, `surprised`, `neutral`.

- **`speed`** _(float)_ (optional): Speech speed where higher values speak faster. Range is `0.5` to `2.0`.

- **`pronunciation_dict`** _(dict[str, list[str]])_ (optional): Defines the pronunciation rules for specific characters or symbols. Read more in the [MiniMax API documentation](https://platform.minimax.io/docs/api-reference/speech-t2a-http#body-pronunciation-dict).

- **`english_normalization`** _(bool)_ (optional): Set to `true` to enable text normalization. This feature improves the model's pronunciation of numbers and dates, with a minor tradeoff in latency. Read more in the [MiniMax API FAQ](https://platform.minimax.io/docs/faq/about-apis#q%3A-the-function-of-the-english-normalization-parameter).

## Additional resources

The following resources provide more information about using MiniMax with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-minimax-ai/)**: The `livekit-plugins-minimax-ai` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/minimax.md)**: Reference for the MiniMax TTS plugin.

- **[MiniMax docs](https://platform.minimax.io/docs/guides/quickstart)**: MiniMax Open Platform documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and MiniMax TTS.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/plugins/minimax.md](https://docs.livekit.io/agents/models/tts/plugins/minimax.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).