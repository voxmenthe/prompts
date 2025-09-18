LiveKit Docs › Worker lifecycle › Overview

---

# Worker lifecycle

> How the worker coordinates with LiveKit server to manage agent jobs.

## Overview

When you start your app with `python agent.py dev`, it registers itself as a **worker** with LiveKit server. LiveKit server manages dispatching your agents to rooms with users by sending requests to available workers.

A **LiveKit session** is one or more participants in a [room](https://docs.livekit.io/home/get-started/api-primitives.md#room). A LiveKit session is often referred to simply as a "room." When a user connects to a room, a worker fulfills the request to dispatch an agent to the room.

An overview of the worker lifecycle is as follows:

1. **Worker registration**: Your agent code registers itself as a "worker" with LiveKit server, then waits on standby for requests.
2. **Job request**: When a user connects to a room, LiveKit server sends a request to an available worker. A worker accepts and starts a new process to handle the job. This is also known as [agent dispatch](https://docs.livekit.io/agents/worker/agent-dispatch.md).
3. **Job**: The job initiated by your`entrypoint` function. This is the bulk of the code and logic you write. To learn more, see [Job lifecycle](https://docs.livekit.io/agents/worker/job.md).
4. **LiveKit session close**: By default, a room is automatically closed when the last non-agent participant leaves. Any remaining agents disconnect. You can also [end the session](https://docs.livekit.io/agents/worker/job.md#ending-the-session) manually.

The following diagram shows the worker lifecycle:

![Diagram describing the functionality of agent workers](/images/agents/agents-jobs-overview.svg)

Some additional features of workers include the following:

- Workers automatically exchange availability and capacity information with the LiveKit server, enabling load balancing of incoming requests.
- Each worker can run multiple jobs simultaneously, running each in its own process for isolation. If one crashes, it won’t affect others running on the same worker.
- When you deploy updates, workers gracefully drain active LiveKit sessions before shutting down, ensuring no sessions are interrupted mid-call.

## Worker options

You can change the permissions, dispatch rules, add prewarm functions, and more through [WorkerOptions](https://docs.livekit.io/agents/worker/options.md).

---

This document was rendered at 2025-08-04T02:30:05.713Z.
For the latest version of this document, see [https://docs.livekit.io/agents/worker.md](https://docs.livekit.io/agents/worker.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).