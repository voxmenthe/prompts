LiveKit docs › Agent observability › Insights in LiveKit Cloud

---

# Agent insights in LiveKit Cloud

> View transcripts, traces, logs, and audio recordings in LiveKit Cloud.

> ℹ️ **Beta feature**
> 
> Agent observability is currently in beta and free until the end of 2025.

## Overview

LiveKit Cloud includes a built-in observability stack optimized for voice agents. It includes transcripts, traces, and logs in a unified timeline with actual audio recordings for each of your agent sessions. This gives you access to comprehensive insights on your agent's behavior and user experience.

[Video: LiveKit Agents Observability](https://www.youtube.com/watch?v=LAXpS14bzW4)

## Availability

Agent observability is available on all LiveKit Cloud plans, and works for agents deployed to LiveKit Cloud and those with custom deployments. For complete information on pricing, see the [LiveKit Cloud pricing page](https://livekit.io/pricing).

To enable agent observability, ensure the following conditions are met:

1. The **Agent observability** feature is enabled within the **Data and privacy** section in your [project's settings](https://cloud.livekit.io/projects/p_/settings/project).
2. Your agent uses the latest version of the LiveKit Agents SDK- Python SDK version 1.3.0 or higher
- Node.js SDK version 1.0.18 or higher
- Or the [LiveKit Agent Builder](https://docs.livekit.io/agents/start/builder.md)

Agent observability is found in the **Agent insights** tab in your [project's sessions dashboard](https://cloud.livekit.io/projects/p_/sessions).

## Observation events

The timeline for each agent session combines transcripts, traces, logs, audio clips, and the per-event metrics emitted by the LiveKit Agents SDK. Trace data streams in while the session runs, while transcripts and recordings are uploaded once the session wraps up.

### Transcripts

Turn-by-turn transcripts for the user and agent. Tool calls and handoffs also appear in the timeline so you can correlate them with traces and logs. Thes events are enriched with additional metadata and metrics in the detail pane of the timeline.

### Session traces and metrics

Traces capture the execution flow of a session, broken into spans for every stage of the voice pipeline. Each span is enriched with metrics—token counts, durations, speech identifiers, and more—that you can inspect in the **Details** panel of the LiveKit Cloud timeline.

Session traces include events including user and agent turns, STT-LLM-TTS pipeline steps, tool calls, and more. Each event is enriched with relevant metrics and other metadata, available in the detail pane of the timeline.

### Logs

Runtime logs from the agent server are uploaded to LiveKit Cloud and available in the session timeline. The logs are collected according to the [log level](https://docs.livekit.io/agents/server/options.md#log-levels) configured for your agent server.

## Audio recordings

Audio recordings are collected for each agent session, and are available for playback in the browser, as well as for download. They are collected locally, and uploaded to LiveKit Cloud after the session ends along with the transcripts. Recordings include both the agent and the user audio.

If [noise cancellation](https://docs.livekit.io/home/cloud/noise-cancellation.md) is enabled, user audio recording is collected after noise cancellation is applied. The recording reflects what the STT or realtime model heard.

## Retention window

All agent observability data is subject to a **30-day retention window**. Data older than 30 days is automatically deleted from LiveKit Cloud.

### Model improvement program

Projects on the free LiveKit Cloud **Build** plan are included in the LiveKit model improvement program. This means that some anonymized session data may be retained by LiveKit for longer than the 30-day retention window, for the purposes of improving models such as the [LiveKit turn detector](https://docs.livekit.io/agents/build/turns/turn-detector.md). Projects on paid plans, including **Ship**, **Scale**, and **Enterprise**, are not included in the program and their data is fully deleted after the 30-day retention window.

## Disabling at the session level

To turn off recording for a specific session, pass `record=False` to the `start` method of the `AgentSession`. This disables upload of audio, transcripts, traces, and logs for the entire session.

**Python**:

```python
await session.start(
    # ... agent, room_options, etc.
    record=False
)

```

---

**Node.js**:

```typescript
await session.start({ 
    // ... agent, roomOptions, etc.
    record: false,
});

```

> ℹ️ **Enabling at the session level**
> 
> Passing `True` to this parameter has no effect. If the feature is turned off at the project level, you cannot enable it from an individual session. When the feature is on at the project level, the default behavior is to record each session.

## Data download

Complete data for each session is available for download within the timeline view. Click the **Download data** button in the top right corner of the timeline view to download a zip file containing the audio recording, transcripts, traces, and logs for the session.

## Custom data collection

To collect observability data within your agent itself, for export to external systems or custom logging, see the [Custom data collection](https://docs.livekit.io/agents/observability/data.md) guide.

---


For the latest version of this document, see [https://docs.livekit.io/agents/observability.md](https://docs.livekit.io/agents/observability.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).