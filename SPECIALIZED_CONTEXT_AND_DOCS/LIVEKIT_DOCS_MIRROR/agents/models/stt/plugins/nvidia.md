LiveKit docs › Models › Speech-to-text (STT) › Plugins › Nvidia

---

# NVIDIA Riva STT plugin guide

> How to use the NVIDIA Riva STT plugin for LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use [NVIDIA Riva](https://www.nvidia.com/en-us/ai-data-science/products/riva/) as an STT provider for your voice agents.

## Quick reference

This section provides a quick reference for the NVIDIA Riva STT plugin. For more information, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```shell
uv add "livekit-agents[nvidia]~=1.2"

```

### Authentication

The NVIDIA Riva plugin supports two authentication methods:

1. **NVIDIA API Key**: Set `NVIDIA_API_KEY` in your `.env` file to use NVIDIA's cloud services.
2. **Self-Hosted NVIDIA Riva Server**: Deploy your own NVIDIA Riva server and configure the plugin to communicate with it using the `riva_uri` parameter.

### Usage

Use NVIDIA Riva STT in an `AgentSession` or as a standalone transcription service. For example, you can use this STT in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import nvidia

session = AgentSession(
   stt=nvidia.STT(
      language="en-US",
      model="parakeet-rnnt-1.1b",
   ),
   # ... llm, tts, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/nvidia/index.html.md#livekit.plugins.nvidia.STT) for a complete list of all available parameters.

- **`language`** _(string)_ (optional): BCP-47 language code for the speech recognition language. Common values include `en-US`, `es-ES`, `fr-FR`, `de-DE`, `zh-CN`, `ja-JP`, and more. See [NVIDIA Riva documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html) for a complete list of supported languages.

- **`model`** _(string)_ (optional) - Default: `parakeet-rnnt-1.1b`: The NVIDIA Riva ASR model to use. Popular models include `parakeet-rnnt-1.1b` for streaming and `parakeet-ctc-1.1b` for batch processing. See [NVIDIA Riva documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html) for available models.

- **`riva_uri`** _(string)_ (optional): The URI of your self-hosted NVIDIA Riva server. If not provided, the plugin uses NVIDIA's cloud services (requires `NVIDIA_API_KEY`).

## Additional resources

The following resources provide more information about using NVIDIA Riva with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-nvidia/)**: The `livekit-plugins-nvidia` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/nvidia/index.html.md#livekit.plugins.nvidia.STT)**: Reference for the NVIDIA Riva STT plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-nvidia)**: View the source or contribute to the LiveKit NVIDIA Riva STT plugin.

- **[NVIDIA Riva docs](https://www.nvidia.com/en-us/ai-data-science/products/riva/)**: NVIDIA Riva's official documentation and product page.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and NVIDIA Riva.

- **[Example implementation](https://github.com/livekit/agents/blob/main/examples/voice_agents/nvidia_test.py)**: Example code showing how to use the NVIDIA Riva plugin with LiveKit Agents.

- **[NVIDIA Riva TTS](https://docs.livekit.io/agents/models/tts/plugins/nvidia.md)**: Guide to the NVIDIA Riva TTS plugin with LiveKit Agents.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/stt/plugins/nvidia.md](https://docs.livekit.io/agents/models/stt/plugins/nvidia.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).