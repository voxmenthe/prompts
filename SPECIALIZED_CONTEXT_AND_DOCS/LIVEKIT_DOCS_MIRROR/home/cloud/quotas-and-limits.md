LiveKit Docs › LiveKit Cloud › Quotas & limits

---

# Quotas and limits

> Guide to the quotas and limits for LiveKit Cloud plans.

## Overview

Each LiveKit Cloud plan includes resource quotas and limits on a per-project basis. The following guide includes precise definitions and information about these limits, why they exist, and how increase them.

## Quotas

Most features of LiveKit Cloud are metered, meaning that you are charged based on the amount of each resource that you use. Every plan includes a default allotment for each of these resources, referred to as a "quota". If you exceed this value, additional usage is billed incrementally based on the pricing for that plan.

For projects on the free Build plan, the quota is a hard limit. Additionally, this quota is shared among all of a user's free projects. Creating additional projects does not increase the total available quota. After you exceed your quota, new requests will fail.

Quotas for all plans resets on the first day of each calendar month. Unused quota does not roll over to the next month.

Refer to the latest [pricing page](https://livekit.io/pricing) for the current pricing and quotas for each plan.

### Metered resources

The following table includes a definition of each metered resource as well as the default quota included on the free Build plan.

| Resource | Definition | Free quota |
| Agent session minutes | Active time that an agent [deployed](https://docs.livekit.io/agents/ops/deployment.md) to LiveKit Cloud is connected to a WebRTC or Telephony session. | 1,000 minutes |
| SIP participant minutes | Time that a single caller is connected to LiveKit Cloud via [SIP](https://docs.livekit.io/sip.md). | 1,000 minutes |
| WebRTC participant minutes | Time that a single user is connected to LiveKit Cloud via a [LiveKit SDK](https://docs.livekit.io/home/client/connect.md). | 5,000 |
| Downstream data transfer GB | The total data transferred out of LiveKit Cloud during a session, including [media tracks](https://docs.livekit.io/home/client/tracks.md) and [data packets](https://docs.livekit.io/home/client/data.md). | 50 GB |
| Transcode minutes | Time spent transcoding an incoming stream with the [Ingress service](https://docs.livekit.io/home/ingress/overview.md) or a composite stream with the [Egress service](https://docs.livekit.io/home/egress/overview.md). | 60 minutes |
| Track egress minutes | Time spent transcoding a single track with the [Egress service](https://docs.livekit.io/home/egress/track.md). | 60 minutes |
| LiveKit Inference | Aggregated usage for all LiveKit Inference [models](https://docs.livekit.io/agents/models.md), at current [pricing](https://livekit.io/pricing/inference). | $2.50 |

> 💡 **Inference credits**
> 
> The monthly quota for LiveKit Inference is expressed in **credits**, measured in USD. These credits can be used for any combination of supported [models](https://docs.livekit.io/agents/models.md). Unused credits do not roll over to the next month.

## Limits

LiveKit Cloud places concurrency and/or rate limits on a number of services. These limits are designed to ensure the stability of the network and to prevent abuse, not to limit usage. As with quotas, these limits are higher on higher-priced plans and projects on the free Build plan share their limits with all of a user's free projects.

You can view the current limits on your project at any time in the [LiveKit Cloud](https://cloud.livekit.io) dashboard by navigating to **Settings** and selecting the **Project** tab.

### Concurrency limits

Many connections to LiveKit Cloud are persistent or long-lived, using WebRTC or WebSockets. These connections have a **concurrency limit**, which is the maximum number of simultaneous connections that can be established. When you exceed this limit, new connections of the same type fail until other connections are closed.

The following table shows the different types of persistent connection, and the default concurrency limits on the Build plan.

| Type | Definition | Free limit |
| Agent session | Actively connected agent sessions [running](https://docs.livekit.io/agents/ops/deployment.md) on LiveKit Cloud. | 5 sessions |
| LiveKit Inference STT | Active STT connections to LiveKit Inference [models](https://docs.livekit.io/agents/models/stt.md). | 5 connections |
| LiveKit Inference TTS | Active TTS connections to LiveKit Inference [models](https://docs.livekit.io/agents/models.md). | 5 connections |
| Participant | Total number of connected agents and end-users across all [rooms](https://docs.livekit.io/home/get-started/api-primitives.md). | 100 participants |
| Ingress request | An active session of the [Ingress service](https://docs.livekit.io/home/ingress/overview.md) transcoding an incoming stream. | 2 requests |
| Egress request | An active session of the [Egress service](https://docs.livekit.io/home/egress/overview.md) recording a composite stream or single track. | 2 requests |

### LiveKit Inference LLM limits

Unlike STT and TTS, which are served through WebSockets, LLM models are served through a stateless HTTP API. This allows for more flexibility in usage, but requires a different approach to limits. The goal is to support the same number of concurrent sessions as with STT and TTS, but due to application variance in terms of request rate and token usage, the service has two rate limits: requests per minute (RPM) and tokens per minute (TPM). If either limit is reached, new requests will fail. These limits are enforced in a sliding window of 60 seconds.

The following table shows the default rate limits on the Build plan. For rate limits on paid plans, refer to the latest [pricing](https://livekit.io/pricing).

| Limit type | Definition | Free limit |
| LLM requests | Individual requests to a LiveKit Inference [LLM model](https://docs.livekit.io/agents/models/llm.md), including [tool responses](https://docs.livekit.io/agents/build/tools.md) and [preemptive generations](https://docs.livekit.io/agents/build/llm.md#preemptive-generation). | 100 requests per minute |
| LLM tokens | Input and output tokens used in requests to a LiveKit Inference [LLM model](https://docs.livekit.io/agents/models/llm.md), including [tool responses](https://docs.livekit.io/agents/build/tools.md) and [preemptive generations](https://docs.livekit.io/agents/build/llm.md#preemptive-generation). | 600,000 tokens per minute |

### Egress time limits

The LiveKit Cloud [Egress service](https://docs.livekit.io/home/egress/overview.md) has time limits, which vary based on the output type. The following table shows the default limits for all plan types.

| Egress output | Time limit |
| File output (MP4, OGG, WebM) | 3 hours |
| HLS segments | 12 hours |
| HLS/RTMP streaming | 12 hours |
| Raw single stream (track) | 12 hours |

When these time limits are reached, any in-progress egress automatically ends with the status `LIMIT_REACHED`.

You can listen for this status change using the `egress_ended` [webhook](https://docs.livekit.io/home/server/webhooks.md).

### Media subscription limits

Each active participant can only subscribe to a limited number of individual media tracks at once. The following table shows the default limits for all plan types.

| Track type | Limit |
| Video | 100 |
| Audio | 100 |

For high volume video use cases, consider using pagination and [selective subscriptions](https://docs.livekit.io/home/client/receive.md#selective-subscription) to keep the number of subscriptions within these limits.

### Server API rate limits

All projects have a [Server API](https://docs.livekit.io/reference/server/server-apis.md) rate limit of 1,000 requests per minute. This applies to requests such as to the `RoomService` or `EgressService`, not to SDK methods like joining a room or sending data packets. Requests to [LiveKit Inference](https://docs.livekit.io/agents/inference.md) have their [own rate limits](#llm-rate-limits).

### Requesting increases

Customers on the Scale plan can request an increase for specific limits in their [project settings](https://cloud.livekit.io/projects/p_/settings/project).

## Agent cold starts

Projects on the Build plan might have their deployed agents shut down after all active sessions end. The agent automatically starts again when a new session begins. This can cause up to 10 to 20 seconds of delay before the agent joins the room.

## Custom plans

LiveKit can work with you to ensure your project has the capacity it needs. [Contact the sales team](https://livekit.io/contact-sales?plan=Enterprise) with your project details.

---


For the latest version of this document, see [https://docs.livekit.io/home/cloud/quotas-and-limits.md](https://docs.livekit.io/home/cloud/quotas-and-limits.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).