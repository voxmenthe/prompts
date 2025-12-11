LiveKit docs › Models › Virtual avatars › Plugins › LiveAvatar

---

# LiveAvatar virtual avatar integration guide

> How to use the LiveAvatar virtual avatar plugin for LiveKit Agents.

Available in:
- [ ] Node.js
- [x] Python

## Overview

[LiveAvatar](https://www.liveavatar.com/) provides dynamic real-time avatars that naturally interact with users. You can use the open source LiveAvatar integration for LiveKit Agents to add virtual avatars to your voice AI app.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```shell
uv add "livekit-agents[liveavatar]~=1.3.6"

```

### Authentication

The LiveAvatar plugin requires a [LiveAvatar API key](https://docs.liveavatar.com/docs/api-key-configuration).

Set `LIVEAVATAR_API_KEY` in your `.env` file.

### Avatar setup

The LiveAvatar plugin requires an avatar ID, which can either be set as the `LIVEAVATAR_AVATAR_ID` environment variable or passed in the avatar session. You can choose either a public avatar or create your own on the LiveAvatar [dashboard](https://app.liveavatar.com/home).

Select an avatar ID for the following steps.

### Usage

Use the plugin in an `AgentSession`. For example, you can use this avatar in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit import agents
from livekit.agents import AgentServer, AgentSession, RoomOutputOptions
from livekit.plugins import liveavatar

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
   session = AgentSession(
      # ... stt, llm, tts, etc.
   )

   avatar = liveavatar.AvatarSession(
      avatar_id="...",  # ID of the LiveAvatar avatar to use
   )

   # Start the avatar and wait for it to join
   await avatar.start(session, room=ctx.room)

   # Start your agent session with the user
   await session.start(
      # ... room, agent, room_input_options, etc....
   )

```

Preview the avatar in the [Agents Playground](https://docs.livekit.io/agents/start/playground.md) or a frontend [starter app](https://docs.livekit.io/agents/start/frontend.md#starter-apps) that you build.

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/liveavatar/index.html.md#livekit.plugins.liveavatar.AvatarSession) for a complete list of all available parameters.

- **`avatar_id`** _(string)_: ID of the LiveAvatar avatar to use. See [Avatar setup](#avatar) for details.

## Additional resources

The following resources provide more information about using LiveAvatar with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-liveavatar/)**: The `livekit-plugins-liveavatar` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/liveavatar/index.html.md)**: Reference for the LiveAvatar avatar plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-liveavatar)**: View the source or contribute to the LiveKit LiveAvatar avatar plugin.

- **[LiveAvatar docs](https://docs.liveavatar.com/docs/getting-started)**: LiveAvatar's full docs site.

- **[Agents Playground](https://docs.livekit.io/agents/start/playground.md)**: A virtual workbench to test your avatar agent.

- **[Frontend starter apps](https://docs.livekit.io/agents/start/frontend.md#starter-apps)**: Ready-to-use frontend apps with avatar support.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/avatar/plugins/liveavatar.md](https://docs.livekit.io/agents/models/avatar/plugins/liveavatar.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).