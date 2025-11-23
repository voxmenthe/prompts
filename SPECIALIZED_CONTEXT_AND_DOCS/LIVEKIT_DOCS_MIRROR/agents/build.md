LiveKit docs › Building voice agents › Overview

---

# Building voice agents

> In-depth guide to voice AI with LiveKit Agents.

## Overview

Building a great voice AI app requires careful orchestration of multiple components. LiveKit Agents is built on top of the [Realtime SDK](https://github.com/livekit/python-sdks) to provide dedicated abstractions that simplify development while giving you full control over the underlying code.

## Voice AI providers

You can choose from a variety of providers for each part of the voice pipeline to fit your needs. The framework supports both high-performance STT-LLM-TTS pipelines and speech-to-speech models. In either case, the framework automatically manages interruptions, transcription forwarding, turn detection, and more.

You may add these components to the `AgentSession`, where they act as global defaults within the app, or to each individual `Agent` if needed.

- **[TTS](https://docs.livekit.io/agents/models/tts.md)**: Text-to-speech models

- **[STT](https://docs.livekit.io/agents/models/stt.md)**: Speech-to-text models

- **[LLM](https://docs.livekit.io/agents/models/llm.md)**: Language model models

- **[Realtime](https://docs.livekit.io/agents/models/realtime.md)**: Realtime models

## Capabilities

The following guides, in addition to others in this section, cover the core capabilities of the `AgentSession` and how to leverage them in your app.

- **[Workflows](https://docs.livekit.io/agents/build/workflows.md)**: Core constructs for building complex voice AI workflows.

- **[Agent sessions](https://docs.livekit.io/agents/build/sessions.md)**: An agent session orchestrates your voice AI app's lifecycle.

- **[Agents & handoffs](https://docs.livekit.io/agents/build/agents-handoffs.md)**: Define agents and agent handoffs to build multi-agent voice AI workflows.

- **[Tool definition & use](https://docs.livekit.io/agents/build/tools.md)**: Use tools to call external services, inject custom logic, and more.

- **[Tasks & task groups](https://docs.livekit.io/agents/build/tasks.md)**: Use tasks and task groups to execute discrete operations and build complex workflows.

- **[Pipeline nodes](https://docs.livekit.io/agents/build/nodes.md)**: Add custom behavior to any component of the voice pipeline.

---


For the latest version of this document, see [https://docs.livekit.io/agents/build.md](https://docs.livekit.io/agents/build.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).