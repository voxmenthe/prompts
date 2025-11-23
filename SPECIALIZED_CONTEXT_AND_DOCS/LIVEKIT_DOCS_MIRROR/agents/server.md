LiveKit docs › Agent server › Overview

---

# Agent server

> Guide to managing multiple concurrent sessions with the agent server.

## Overview

LiveKit Agents includes an agent server which is capable of serving multiple simultaneous agent sessions. LiveKit Cloud coordinates a pool of individual agent servers that can scale horizontally. When you start your app with `uv run agent.py dev` (on `pnpm run agent.ts dev` in Node.js), it starts the agent server.

A [programmatic participant](#programmatic-participants) is any code that joins a LiveKit room as a participant—this includes AI agents, media processors, or custom logic that processes realtime streams. This topic describes the agent server lifecycle for AI agents, but the same lifecycle applies to all programmatic participants.

## Server lifecycle

When a user connects to a [room](https://docs.livekit.io/home/get-started/api-primitives.md#room), LiveKit Cloud dispatches a request to available servers. The first available server accepts the job and starts the agent session. An overview of the server lifecycle is as follows:

1. **Agent server registration**: Your agent code registers itself as an "agent server" with LiveKit Cloud, then waits on standby for requests.
2. **Job request**: When a user connects to a room, LiveKit Cloud sends a request to an available agent server. An agent server accepts and starts a new process to handle the job. This is also known as [agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch.md).
3. **Job**: The job initiated by your entrypoint function. This is the bulk of the code and logic you write. To learn more, see [Job lifecycle](https://docs.livekit.io/agents/server/job.md).
4. **LiveKit session close**: By default, a room is automatically closed when the last non-agent participant leaves. Any remaining agents disconnect. You can also [end the session](https://docs.livekit.io/agents/server/job.md#ending-the-session) manually.

The following diagram shows the agent server lifecycle:

![Diagram describing the functionality of agent servers](/images/agents/agents-jobs-overview.svg)

Some additional features of agent servers include the following:

- Agent servers automatically exchange availability and capacity information with the LiveKit Cloud, enabling load balancing of incoming requests.
- Each agent server can run multiple jobs simultaneously, running each in its own process for isolation. If one crashes, it won't affect others running on the same agent server.
- When you deploy updates, agent servers gracefully drain active LiveKit sessions before shutting down, ensuring sessions aren't interrupted.

## Programmatic participants

The Agents framework isn't limited to AI agents. You can use it to deploy any code that needs to process realtime media and data streams. Some examples of what these participants can do include the following:

- **Process audio streams**: Analyze audio for patterns, quality metrics, or content detection.
- **Handle video processing**: Apply computer vision, video effects, or content moderation.
- **Manage data flows**: Aggregate, transform, or route realtime data between participants.
- **Provide services**: Act as bridges to external APIs, databases, or other systems.

The framework provides the same production-ready infrastructure for all types of programmatic participants, including automatic scaling and load balancing. To learn more, see [Processing raw media tracks](https://docs.livekit.io/home/client/tracks/raw-tracks.md).

## Agent server options

You can change the permissions, dispatch rules, add prewarm functions, and more through [Agent server options](https://docs.livekit.io/agents/server/options.md).

---


For the latest version of this document, see [https://docs.livekit.io/agents/server.md](https://docs.livekit.io/agents/server.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).