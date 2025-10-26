LiveKit docs › Models › Realtime models › Plugins › Ultravox Realtime API

---

# Ultravox plugin guide

> How to use the Ultravox Realtime model with LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

The Ultravox Realtime API combines STT, LLM, and TTS into a single connection. Use LiveKit's Ultravox plugin with this plugin to create an agent quickly without needing to wire up multiple providers.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the Ultravox plugin:

```shell
uv add "livekit-agents[ultravox]"

```

### Authentication

The Ultravox plugin requires an API key which can be accessed on the Ultravox console's [Settings](https://api.ultravox.ai/settings) page. Set `ULTRAVOX_API_KEY` as a variable in your `.env` file.

### Usage

Use the Ultravox Realtime API in an `AgentSession`. For example, you can use it in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import ultravox

async def entrypoint(ctx: agents.JobContext):
   session = AgentSession(
      llm=ultravox.realtime.RealtimeModel(),
   )

```

### Logging

You can optionally enable debug logs (disabled by default):

```shell
LK_ULTRAVOX_DEBUG=true uv run src/agent.py dev

```

### Parameters

This section describes some of the available parameters. For a complete reference of all available parameters, see the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/ultravox.md#livekit.plugins.ultravox).

- **`voice`** _(str)_ - Default: `Mark`: Ultravox voice to use from the [available voices](https://app.ultravox.ai/voices).

- **`time_exceeded_message`** _(str)_ (optional): Message to play when max duration is reached.

## Additional resources

The following resources provide more information about using Ultravox with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-ultravox/)**: The `livekit-plugins-ultravox` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/ultravox.md#livekit.plugins.ultravox)**: Reference for the Ultravox plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-ultravox)**: View the source or contribute to the LiveKit Ultravox plugin.

- **[Ultravox Realtime docs](https://docs.ultravox.ai/overview)**: Ultravox documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Ultravox.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/realtime/plugins/ultravox.md](https://docs.livekit.io/agents/models/realtime/plugins/ultravox.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).