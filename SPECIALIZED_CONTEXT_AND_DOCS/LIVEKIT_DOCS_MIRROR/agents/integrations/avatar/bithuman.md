LiveKit Docs › Integration guides › Virtual avatars › bitHuman

---

# bitHuman virtual avatar integration guide

> How to use the bitHuman virtual avatar plugin for LiveKit Agents.

## Overview

[bitHuman](https://www.bithuman.ai/) provides realtime virtual avatars that run locally on CPU only for low latency and high quality. You can use the open source bitHuman integration for LiveKit Agents to add virtual avatars to your voice AI app.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```bash
pip install "livekit-agents[bithuman]~=1.2"

```

### Authentication

The bitHuman plugin requires a [bitHuman API Secret](https://imaginex.bithuman.ai/#api).

Set `BITHUMAN_API_SECRET` in your `.env` file.

### Model installation

Each bitHuman avatar comes as a `.imx` file, which you must download locally for your agent. You can create and download avatar models from the [bitHuman ImagineX console](https://imaginex.bithuman.ai).

You can pass the model path to the avatar session, or set the `BITHUMAN_MODEL_PATH` environment variable.

### Usage

Use the plugin in an `AgentSession`. For example, you can use this avatar in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import bithuman

session = AgentSession(
   # ... stt, llm, tts, etc.
)

avatar = bithuman.AvatarSession(
    model_path="./albert_einstein.imx", # This example uses a demo model installed in the current directory
)

# Start the avatar and wait for it to join
await avatar.start(session, room=ctx.room)

# Start your agent session with the user
await session.start(
    room=ctx.room,
)

```

Preview the avatar in the [Agents Playground](https://docs.livekit.io/agents/start/playground.md) or a frontend [starter app](https://docs.livekit.io/agents/start/frontend.md#starter-apps) that you build.

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/bithuman/index.html.md#livekit.plugins.bithuman.AvatarSession) for a complete list of all available parameters.

- **`model_path`** _(string)_ - Environment: `BITHUMAN_MODEL_PATH`: Path to the bitHuman model to use. To learn more, see the [bitHuman docs](https://sdk.docs.bithuman.ai/#/getting-started/overview?id=quick-setup).

## Additional resources

The following resources provide more information about using bitHuman with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-bithuman/)**: The `livekit-plugins-bithuman` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/bithuman.html.md)**: Reference for the bitHuman avatar plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-bithuman)**: View the source or contribute to the LiveKit bitHuman avatar plugin.

- **[bitHuman docs](https://sdk.docs.bithuman.ai)**: bitHuman's full API docs site.

- **[Agents Playground](https://docs.livekit.io/agents/start/playground.md)**: A virtual workbench to test your avatar agent.

- **[Frontend starter apps](https://docs.livekit.io/agents/start/frontend.md#starter-apps)**: Ready-to-use frontend apps with avatar support.

---


For the latest version of this document, see [https://docs.livekit.io/agents/integrations/avatar/bithuman.md](https://docs.livekit.io/agents/integrations/avatar/bithuman.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).