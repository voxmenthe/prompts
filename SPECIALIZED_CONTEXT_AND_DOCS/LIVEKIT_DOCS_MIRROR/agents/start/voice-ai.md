LiveKit docs › Getting started › Voice AI quickstart

---

# Voice AI quickstart

> Build and deploy a simple voice assistant in less than 10 minutes.

## Overview

This guide walks you through the setup of your very first voice assistant using LiveKit Agents for Python. In less than 10 minutes, you'll have a voice assistant that you can speak to in your terminal, browser, telephone, or native app.

> 💡 **LiveKit Agent Builder**
> 
> The LiveKit Agent Builder is a quick way to get started with voice agents in your browser, without writing any code. It's perfect for protototyping and exploring ideas, but doesn't have as many features as the full LiveKit Agents SDK. See the [Agent Builder](https://docs.livekit.io/agents/start/builder.md) guide for more details.

## Starter projects

The simplest way to get your first agent running with is with one of the following starter projects. Click "Use this template" in the top right to create a new repo on GitHub, then follow the instructions in the project's README.

These projects are constructed with best-practices, a complete working agent, tests, and an AGENTS.md optimized to turn your coding assistant into a LiveKit expert.

- **[Python starter project](https://github.com/livekit-examples/agent-starter-python)**: Ready-to-go Python starter project. Clone a repo with all the code you need to get started.

- **[Node.js starter project](https://github.com/livekit-examples/agent-starter-node)**: Ready-to-go Node.js starter project. Clone a repo with all the code you need to get started.

## Requirements

The following sections describe the minimum requirements to get started with LiveKit Agents.

**Python**:

- LiveKit Agents requires Python >= 3.9.
- This guide uses the [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager.

---

**Node.js**:

- LiveKit Agents for Node.js requires Node.js >= 20.
- This guide uses [pnpm](https://pnpm.io/installation) package manager and requires pnpm >= 10.15.0.

### LiveKit Cloud

This guide assumes you have signed up for a free [LiveKit Cloud](https://cloud.livekit.io/) account. LiveKit Cloud includes agent deployment, model inference, and realtime media transport. Create a free project and use the API keys in the following steps to get started.

While this guide assumes LiveKit Cloud, the instructions can be adapted for [self-hosting](https://docs.livekit.io/home/self-hosting/local.md) the open-source LiveKit server instead. For self-hosting in production, set up a [custom deployment](https://docs.livekit.io/agents/ops/deployment/custom.md) environment, and make the following changes: remove the [enhanced noise cancellation](https://docs.livekit.io/home/cloud/noise-cancellation.md) plugin from the agent code, and use [plugins](https://docs.livekit.io/agents/models.md#plugins) for your own AI providers.

### LiveKit Docs MCP server

If you're using an AI coding assistant, you should install the [LiveKit Docs MCP server](https://docs.livekit.io/home/get-started/mcp-server.md) to get the most out of it. This ensures your agent has access to the latest documentation and examples. The [starter projects](#starters) also include an `AGENTS.md` file with instructions for using the MCP server and other important information about building agents with LiveKit.

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

### AI models

Voice agents require one or more [AI models](https://docs.livekit.io/agents/models.md) to provide understanding, intelligence, and speech. LiveKit Agents supports both high-performance STT-LLM-TTS voice pipelines constructed from multiple specialized models, as well as realtime models with direct speech-to-speech capabilities.

The rest of this guide assumes you use one of the following two starter packs, which provide the best combination of value, features, and ease of setup.

**STT-LLM-TTS pipeline**:

Your agent strings together three specialized providers into a high-performance voice pipeline powered by LiveKit Inference. No additional setup is required.

![Diagram showing STT-LLM-TTS pipeline.](/images/agents/stt-llm-tts-pipeline.svg)

| Component | Model | Alternatives |
| STT | AssemblyAI Universal-Streaming | [STT models](https://docs.livekit.io/agents/models/stt.md) |
| LLM | OpenAI GPT-4.1 mini | [LLM models](https://docs.livekit.io/agents/models/llm.md) |
| TTS | Cartesia Sonic-3 | [TTS models](https://docs.livekit.io/agents/models/tts.md) |

---

**Realtime model**:

Your agent uses a single realtime model to provide an expressive and lifelike voice experience.

![Diagram showing realtime model.](/images/agents/realtime-model.svg)

| Model | Required Key | Alternatives |
| [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) | `OPENAI_API_KEY` | [Realtime models](https://docs.livekit.io/agents/models/realtime.md) |

## Setup

Use the instructions in the following sections to set up your new project.

### Project initialization

Create a new project for the voice agent.

**Python**:

Run the following commands to use uv to create a new project ready to use for your new voice agent.

```shell
uv init livekit-voice-agent --bare
cd livekit-voice-agent

```

---

**Node.js**:

Run the following commands to use pnpm to create a new TypeScript-based project ready to use for your new voice agent.

```shell
mkdir livekit-voice-agent
cd livekit-voice-agent
pnpm init --init-type module
pnpm add -D typescript tsx
pnpm exec tsc --init

```

### Install packages

**STT-LLM-TTS pipeline**:

Install the following packages to build a complete voice AI agent with your STT-LLM-TTS pipeline, noise cancellation, and [turn detection](https://docs.livekit.io/agents/build/turns.md):

** Filename: `Python`**

```shell
uv add \
  "livekit-agents[silero,turn-detector]~=1.2" \
  "livekit-plugins-noise-cancellation~=0.2" \
  "python-dotenv"

```

** Filename: `Node.js`**

```shell
pnpm add @livekit/agents@1.x \
    @livekit/agents-plugin-silero@1.x \
    @livekit/agents-plugin-livekit@1.x \
    @livekit/noise-cancellation-node@0.x \
    dotenv

```

---

**Realtime model**:

Install the following packages to build a complete voice AI agent with your realtime model and noise cancellation.

** Filename: `Python`**

```shell
uv add \
  "livekit-agents[openai]~=1.2" \
  "livekit-plugins-noise-cancellation~=0.2" \
  "python-dotenv"

```

** Filename: `Node.js`**

```shell
pnpm add @livekit/agents@1.x \
         @livekit/agents-plugin-openai@1.x \
         @livekit/noise-cancellation-node@0.x \
         dotenv

```

### Environment variables

Run the following command to load your LiveKit Cloud API keys into a `.env.local` file:

```shell
lk app env -w

```

The file should look like this:

**STT-LLM-TTS pipeline**:

```shell
LIVEKIT_API_KEY=%{apiKey}%
LIVEKIT_API_SECRET=%{apiSecret}%
LIVEKIT_URL=%{wsURL}%

```

---

**Realtime model**:

You must also set the `OPENAI_API_KEY` environment variable, using your own [OpenAI platform account](https://platform.openai.com/account/api-keys).

```shell
LIVEKIT_API_KEY=%{apiKey}%
LIVEKIT_API_SECRET=%{apiSecret}%
LIVEKIT_URL=%{wsURL}%
OPENAI_API_KEY=<Your OpenAI API Key>

```

### Agent code

Create a file with your agent code.

**STT-LLM-TTS pipeline**:

** Filename: `agent.py`**

```python
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer,AgentSession, Agent, room_io
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)


```

** Filename: `agent.ts`**

```typescript
import {
  type JobContext,
  type JobProcess,
  WorkerOptions,
  cli,
  defineAgent,
  voice,
} from '@livekit/agents';
import * as livekit from '@livekit/agents-plugin-livekit';
import * as silero from '@livekit/agents-plugin-silero';
import { BackgroundVoiceCancellation } from '@livekit/noise-cancellation-node';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load();
  },
  entry: async (ctx: JobContext) => {
    const vad = ctx.proc.userData.vad! as silero.VAD;
    
    const assistant = new voice.Agent({
	    instructions: 'You are a helpful voice AI assistant.',
    });

    const session = new voice.AgentSession({
      vad,
      stt: "assemblyai/universal-streaming:en",
      llm: "openai/gpt-4.1-mini",
      tts: "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
      turnDetection: new livekit.turnDetector.MultilingualModel(),
    });

    await session.start({
      agent: assistant,
      room: ctx.room,
      inputOptions: {
        // For telephony applications, use `TelephonyBackgroundVoiceCancellation` for best results
        noiseCancellation: BackgroundVoiceCancellation(),
      },
    });

    await ctx.connect();

    const handle = session.generateReply({
      instructions: 'Greet the user and offer your assistance.',
    });
  },
});

cli.runApp(new WorkerOptions({ agent: fileURLToPath(import.meta.url) }));

```

---

**Realtime model**:

** Filename: `agent.py`**

```python
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import (
    openai,
    noise_cancellation,
)

load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral"
        )
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance. You should start by speaking in English."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)


```

** Filename: `agent.ts`**

```typescript
import {
  type JobContext,
  WorkerOptions,
  cli,
  defineAgent,
  voice,
} from '@livekit/agents';
import * as openai from '@livekit/agents-plugin-openai';
import { BackgroundVoiceCancellation } from '@livekit/noise-cancellation-node';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

class Assistant extends voice.Agent {
  constructor() {
    super({
      instructions: 'You are a helpful voice AI assistant.',
    });
  }
}

export default defineAgent({
  entry: async (ctx: JobContext) => {
    const session = new voice.AgentSession({
      llm: new openai.realtime.RealtimeModel({
        voice: 'coral',
      }),
    });

    await session.start({
      agent: new Assistant(),
      room: ctx.room,
      inputOptions: {
        // For telephony applications, use `TelephonyBackgroundVoiceCancellation` for best results
        noiseCancellation: BackgroundVoiceCancellation(),
      },
    });

    await ctx.connect();

    const handle = session.generateReply({
      instructions: 'Greet the user and offer your assistance. You should start by speaking in English.',
    });
    await handle.waitForPlayout();
  },
});

cli.runApp(new WorkerOptions({ agent: fileURLToPath(import.meta.url) }));

```

## Download model files

To use the `turn-detector`, `silero`, and `noise-cancellation` plugins, you first need to download the model files:

**Python**:

```shell
uv run agent.py download-files

```

---

**Node.js**:

1. Add the `download-files` script to your `package.json` file:

```shell
pnpm pkg set "scripts.download-files=tsc && node agent.js download-files"

```
2. After you add the `download-files` script, run the following command:

```shell
pnpm download-files

```

## Speak to your agent

> ℹ️ **Python only**
> 
> If you're using Node.js, you can skip this setup and continue to [Connect to playground](#connect-to-playground).

Start your agent in `console` mode to run inside your terminal:

```shell
uv run agent.py console

```

Your agent speaks to you in the terminal, and you can speak to it as well.

![Screenshot of the CLI console mode.](/images/agents/start/cli-console.png)

## Connect to playground

Start your agent in `dev` mode to connect it to LiveKit and make it available from anywhere on the internet:

**Python**:

```shell
uv run agent.py dev

```

---

**Node.js**:

1. Add the dev script to your `package.json` file:

```shell
pnpm pkg set "scripts.dev=tsx agent.ts dev"

```
2. After you add the dev script, anytime you want to run your agent in development mode, run the following command:

```shell
pnpm dev

```

Use the [Agents playground](https://docs.livekit.io/agents/start/playground.md) to speak with your agent and explore its full range of multimodal capabilities.

## Agent CLI modes

In the `dev` and `start` modes, your agent connects to LiveKit Cloud and joins rooms:

- `dev` mode: Run your agent in development mode for testing and debugging.
- `start` mode: Run your agent in production mode.

**Python**:

For Python agents, run the following command to start your agent in production mode:

```shell
uv run agent.py start

```

---

**Node.js**:

For Node.js agents, you need to add the `build` and `start` scripts to your `package.json` file to use production mode.

```shell
pnpm pkg set "scripts.build=tsc"
pnpm pkg set "scripts.start=node agent.js start"

```

Now run the following commands to build and start your agent for production:

```shell
pnpm build
pnpm start

```

Python agents can also use `console` mode, which runs locally and is only available within your terminal.

## Deploy to LiveKit Cloud

From the root of your project, run the following command with the LiveKit CLI. Ensure you have [linked your LiveKit Cloud project](#cli) and added the [build and start scripts](#cli-modes).

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

- **[Agent server lifecycle](https://docs.livekit.io/agents/server.md)**: Learn how to manage your agents with agent servers and jobs.

- **[Deploying to LiveKit Cloud](https://docs.livekit.io/agents/ops/deployment.md)**: Learn more about deploying and scaling your agent in production.

- **[AI Models](https://docs.livekit.io/agents/models.md)**: Explore the full list of AI models available with LiveKit Agents.

- **[Recipes](https://docs.livekit.io/recipes.md)**: A comprehensive collection of examples, guides, and recipes for LiveKit Agents.

---


For the latest version of this document, see [https://docs.livekit.io/agents/start/voice-ai.md](https://docs.livekit.io/agents/start/voice-ai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).