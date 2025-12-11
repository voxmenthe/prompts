LiveKit docs › Models › Text-to-speech (TTS) › Plugins › Nvidia

---

# NVIDIA Riva TTS plugin guide

> How to use the NVIDIA Riva TTS plugin for LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use [NVIDIA Riva](https://www.nvidia.com/en-us/ai-data-science/products/riva/) as a TTS provider for your voice agents.

## Quick reference

This section provides a quick reference for the NVIDIA Riva TTS plugin. For more information, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```shell
uv add "livekit-agents[nvidia]~=1.2"

```

### Authentication

The NVIDIA Riva plugin supports two authentication methods:

1. **NVIDIA API Key**: Set `NVIDIA_API_KEY` in your `.env` file to use NVIDIA's cloud services.
2. **Self-Hosted NVIDIA Riva Server**: Deploy your own NVIDIA Riva server and configure the plugin to communicate with it using the `server` parameter and setting `use_ssl=False`.

### Usage

Use NVIDIA Riva TTS in an `AgentSession` or as a standalone speech generator. For example, you can use this TTS in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import nvidia

session = AgentSession(
   tts=nvidia.TTS(
      voice="Magpie-Multilingual.EN-US.Leo",
      language_code="en-US",
   ),
   # ... llm, stt, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/nvidia/index.html.md#livekit.plugins.nvidia.TTS) for a complete list of all available parameters.

- **`voice`** _(string)_ (optional) - Default: `Magpie-Multilingual.EN-US.Leo`: The NVIDIA Riva TTS voice to use. Use the `list_voices()` method to get available voices for your language. See [NVIDIA Riva documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html) for more information about available voices.

- **`language_code`** _(string)_ (optional) - Default: `en-US`: BCP-47 language code for the speech synthesis language. Common values include `en-US`, `es-ES`, `fr-FR`, `de-DE`, `zh-CN`, `ja-JP`, and more. See [NVIDIA Riva documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html) for a complete list of supported languages.

- **`server`** _(string)_ (optional) - Default: `grpc.nvcf.nvidia.com:443`: The URI of your NVIDIA Riva server. If you're using NVIDIA's cloud services, leave this as the default. For self-hosted servers, provide your server URI.

- **`function_id`** _(string)_ (optional) - Default: `877104f7-e885-42b9-8de8-f6e4c6303969`: The NVIDIA Cloud Functions function ID for TTS. Only required when using NVIDIA's cloud services.

- **`use_ssl`** _(boolean)_ (optional) - Default: `True`: Whether to use SSL/TLS for the connection. Set to `False` when using a self-hosted NVIDIA Riva server without SSL.

## Additional resources

The following resources provide more information about using NVIDIA Riva with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-nvidia/)**: The `livekit-plugins-nvidia` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/nvidia/index.html.md#livekit.plugins.nvidia.TTS)**: Reference for the NVIDIA Riva TTS plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-nvidia)**: View the source or contribute to the LiveKit NVIDIA Riva TTS plugin.

- **[NVIDIA Riva docs](https://www.nvidia.com/en-us/ai-data-science/products/riva/)**: NVIDIA Riva's official documentation and product page.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and NVIDIA Riva.

- **[Example implementation](https://github.com/livekit/agents/blob/main/examples/voice_agents/nvidia_test.py)**: Example code showing how to use the NVIDIA Riva plugin with LiveKit Agents.

- **[NVIDIA Riva STT](https://docs.livekit.io/agents/models/stt/plugins/nvidia.md)**: Guide to the NVIDIA Riva STT plugin with LiveKit Agents.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/tts/plugins/nvidia.md](https://docs.livekit.io/agents/models/tts/plugins/nvidia.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).