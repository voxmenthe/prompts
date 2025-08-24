LiveKit Docs › Getting started › Voice AI quickstart

---

# Voice AI quickstart

> Build and deploy a simple voice assistant in less than 10 minutes.

## Overview

This guide walks you through the setup of your very first voice assistant using LiveKit Agents for Python. In less than 10 minutes, you'll have a voice assistant that you can speak to in your terminal, browser, telephone, or native app.

- **[Python starter project](https://github.com/livekit-examples/agent-starter-python)**: Prefer to just clone a repo? This repo is ready-to-go, will all the code you need to get started.

- **[Deeplearning.ai course](https://www.deeplearning.ai/short-courses/building-ai-voice-agents-for-production/)**: For a more in-depth guide, learn to build and deploy voice agents with LiveKit in this free course from Deeplearning.ai.

## Requirements

The following sections describe the minimum requirements to get started with LiveKit Agents.

### Python and uv

LiveKit Agents requires Python 3.9 or later.

This guide is written for for the [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager.

> ℹ️ **Looking for Node.js?**
> 
> The Node.js beta is still in development and has not yet reached v1.0. See the [v0.x documentation](https://docs.livekit.io/agents/v0.md) for Node.js reference and join the [LiveKit Community Slack](https://livekit.io/join-slack) to be the first to know when the next release is available.

### LiveKit Cloud

This guide assumes you have signed up for a free [LiveKit Cloud](https://cloud.livekit.io/) account. LiveKit Cloud offers realtime media transport and agent deployment. Create a free project and use the API keys in the following steps to get started.

While this guide assumes LiveKit Cloud, the instructions can be adapted for [self-hosting](https://docs.livekit.io/home/self-hosting/local.md) the open-source LiveKit server instead. You will need your own [custom deployment](https://docs.livekit.io/agents/ops/deployment/custom.md) environment in production, and should remove the [enhanced noise cancellation](https://docs.livekit.io/home/cloud/noise-cancellation.md) plugin from the agent code.

### LiveKit CLI

Use the LiveKit CLI to manage LiveKit API keys and deploy your agent to LiveKit Cloud.

1. Install the LiveKit CLI:

**macOS**:

Install the LiveKit CLI with [Homebrew](https://brew.sh/):

```text
brew install livekit-cli

```

---

**Linux**:

```text
curl -sSL https://get.livekit.io/cli | bash

```

> 💡 **Tip**
> 
> You can also download the latest precompiled binaries [here](https://github.com/livekit/livekit-cli/releases/latest).

---

**Windows**:

```text
winget install LiveKit.LiveKitCLI

```

> 💡 **Tip**
> 
> You can also download the latest precompiled binaries [here](https://github.com/livekit/livekit-cli/releases/latest).

---

**From Source**:

This repo uses [Git LFS](https://git-lfs.github.com/) for embedded video resources. Please ensure git-lfs is installed on your machine before proceeding.

```text
git clone github.com/livekit/livekit-cli
make install

```
2. Link your LiveKit Cloud project to the CLI:

```shell
lk cloud auth

```

This opens a browser window to authenticate and link your project to the CLI.

### AI providers

LiveKit Agents [integrates with most AI model providers](https://docs.livekit.io/agents/integrations.md) and supports both high-performance STT-LLM-TTS voice pipelines, as well as lifelike multimodal models.

The rest of this guide assumes you use one of the following two starter packs, which provide the best combination of value, features, and ease of setup.

**STT-LLM-TTS pipeline**:

Your agent strings together three specialized providers into a high-performance voice pipeline. You need accounts and API keys for each.

![Diagram showing STT-LLM-TTS pipeline.](/images/agents/stt-llm-tts-pipeline.svg)

| Component | Provider | Required Key | Alternatives |
| STT | [Deepgram](https://deepgram.com/) | `DEEPGRAM_API_KEY` | [STT integrations](https://docs.livekit.io/agents/integrations/stt.md#providers) |
| LLM | [OpenAI](https://platform.openai.com/) | `OPENAI_API_KEY` | [LLM integrations](https://docs.livekit.io/agents/integrations/llm.md#providers) |
| TTS | [Cartesia](https://cartesia.ai) | `CARTESIA_API_KEY` | [TTS integrations](https://docs.livekit.io/agents/integrations/tts.md#providers) |

---

**Realtime model**:

Your agent uses a single realtime model to provide an expressive and lifelike voice experience.

![Diagram showing realtime model.](/images/agents/realtime-model.svg)

| Component | Provider | Required Key | Alternatives |
| Realtime model | [OpenAI](https://platform.openai.com/docs/guides/realtime) | `OPENAI_API_KEY` | [Realtime models](https://docs.livekit.io/agents/integrations/realtime.md#providers) |

## Setup

Use the instructions in the following sections to set up your new project.

### Project initialization

Create a new project with uv. The following command creates a new folder and a `pyproject.toml` file inside of it.

```shell
uv init livekit-voice-agent --bare
cd livekit-voice-agent

```

### Install packages

**STT-LLM-TTS pipeline**:

Install the following packages to build a complete voice AI agent with your STT-LLM-TTS pipeline, noise cancellation, and [turn detection](https://docs.livekit.io/agents/build/turns.md):

```shell
uv add \
  "livekit-agents[deepgram,openai,cartesia,silero,turn-detector]~=1.2" \
  "livekit-plugins-noise-cancellation~=0.2" \
  "python-dotenv"

```

---

**Realtime model**:

Install the following packages to build a complete voice AI agent with your realtime model and noise cancellation.

```shell
uv add \
  "livekit-agents[openai]~=1.2" \
  "livekit-plugins-noise-cancellation~=0.2" \
  "python-dotenv"

```

### Environment variables

Run the following command to load your LiveKit Cloud API keys into a `.env.local` file:

```shell
lk app env -w

```

Now open this file and add keys for your selected AI provider. The file should look like this:

**STT-LLM-TTS pipeline**:

** Filename: `.env.local`**

```shell
DEEPGRAM_API_KEY=<Your Deepgram API Key>
OPENAI_API_KEY=<Your OpenAI API Key>
CARTESIA_API_KEY=<Your Cartesia API Key>
LIVEKIT_API_KEY=%{apiKey}%
LIVEKIT_API_SECRET=%{apiSecret}%
LIVEKIT_URL=%{wsURL}%

```

---

**Realtime model**:

** Filename: `.env.local`**

```shell
OPENAI_API_KEY=<Your OpenAI API Key>
LIVEKIT_API_KEY=%{apiKey}%
LIVEKIT_API_SECRET=%{apiSecret}%
LIVEKIT_URL=%{wsURL}%

```

### Agent code

Create a file named `agent.py` containing the following code for your first voice agent.

**STT-LLM-TTS pipeline**:

** Filename: `agent.py`**

```python
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))


```

---

**Realtime model**:

** Filename: `agent.py`**

```python
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
)

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral"
        )
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))


```

## Download model files

To use the `turn-detector`, `silero`, or `noise-cancellation` plugins, you first need to download the model files:

```shell
uv run agent.py download-files

```

## Speak to your agent

Start your agent in `console` mode to run inside your terminal:

```shell
uv run agent.py console

```

Your agent speaks to you in the terminal, and you can speak to it as well.

![Screenshot of the CLI console mode.](/images/agents/start/cli-console.png)

## Connect to playground

Start your agent in `dev` mode to connect it to LiveKit and make it available from anywhere on the internet:

```shell
uv run agent.py dev

```

Use the [Agents playground](https://docs.livekit.io/agents/start/playground.md) to speak with your agent and explore its full range of multimodal capabilities.

> 💡 **Agent CLI modes**
> 
> In the `console` mode, the agent runs locally and is only available within your terminal.
> 
> Run your agent in `dev` (development / debug) or `start` (production) mode to connect to LiveKit Cloud and join rooms.

## Deploy to LiveKit Cloud

From the root of your project, run the following command with the LiveKit CLI. Ensure you have [linked your LiveKit Cloud project](#cli).

```shell
lk agent create

```

The CLI creates `Dockerfile`, `.dockerignore`, and `livekit.toml` files in your current directory, then registers your agent with your LiveKit Cloud project and deploys it.

After the deployment completes, you can access your agent in the playground, or continue to use the `console` mode as you build and test your agent locally.

## Next steps

Follow these guides bring your voice AI app to life in the real world.

- **[Web and mobile frontends](https://docs.livekit.io/agents/start/frontend.md)**: Put your agent in your pocket with a custom web or mobile app.

- **[Telephony integration](https://docs.livekit.io/agents/start/telephony.md)**: Your agent can place and receive calls with LiveKit's SIP integration.

- **[Testing your agent](https://docs.livekit.io/agents/build/testing.md)**: Add behavioral tests to fine-tune your agent's behavior.

- **[Building voice agents](https://docs.livekit.io/agents/build.md)**: Comprehensive documentation to build advanced voice AI apps with LiveKit.

- **[Worker lifecycle](https://docs.livekit.io/agents/worker.md)**: Learn how to manage your agents with workers and jobs.

- **[Deploying to LiveKit Cloud](https://docs.livekit.io/agents/ops/deployment.md)**: Learn more about deploying and scaling your agent in production.

- **[Integration guides](https://docs.livekit.io/agents/integrations.md)**: Explore the full list of AI providers available for LiveKit Agents.

- **[Recipes](https://docs.livekit.io/recipes.md)**: A comprehensive collection of examples, guides, and recipes for LiveKit Agents.

---


For the latest version of this document, see [https://docs.livekit.io/agents/start/voice-ai.md](https://docs.livekit.io/agents/start/voice-ai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).