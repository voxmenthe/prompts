LiveKit Docs › Cloud › Quotas & limits

---

# Quotas and limits

> Guide to the quotas and limits for LiveKit Cloud plans.

## Free quotas

Every LiveKit project gets the following for free:

- 50GB data transfer
- 5,000 connection minutes
- 60 minutes of transcoding (for [Stream import (ingress)](https://docs.livekit.io/home/ingress/overview.md) or [Composite recording (egress)](https://docs.livekit.io/home/egress/overview.md))

## Connection limits

LiveKit projects have limitations on the number of concurrent connections of various types in order to ensure the stability of the network and to prevent abuse. This is similar to rate limiting for an HTTP service, but for a continuous service with long-lived connections. Like rate limiting, the primary purpose of these connection limits is to prevent abuse.

You can view the current connection limits on your project at any time in the [LiveKit Cloud](https://cloud.livekit.io) dashboard by navigating to **Settings** and selecting the **Project** tab.

For pricing information for any of the following plans, see the [pricing guide](https://livekit.io/pricing).

### Build plan

Projects on the `Build` (free) plan have the following limits:

- 100 concurrent participants.
- 2 concurrent [egress requests](https://docs.livekit.io/home/egress/overview.md) at a time.
- 2 concurrent [ingress requests](https://docs.livekit.io/home/ingress/overview.md) at a time.

When these limits are reached, new connections of the same type fail.

### Ship plan

Projects on the `Ship` plan have the following limits:

- 1,000 concurrent participants.
- 100 concurrent egress requests.
- 100 concurrent ingress requests.

When these limits are reached, new connections of the same type fail.

### Scale plan

Projects on the `Scale` plan have the following default limits:

- 5,000 concurrent participants.
- 100 concurrent egress requests.
- 100 concurrent ingress requests.

When these limits are reached, new connections of the same type fail.

> 💡 **Tip**
> 
> Your project admin can request an increase for specific limits in your [project settings](https://cloud.livekit.io/projects/p_/settings/project).

### Custom plan

LiveKit can work with you to ensure your project has the capacity it needs. [Contact the sales team](https://livekit.io/contact-sales?plan=Enterprise) with your project details.

## Egress time limits

Egress has time limits, depending on the output type:

| Egress output | Time limit |
| File output (MP4, OGG, WebM) | 3 hours |
| HLS segments | 12 hours |
| HLS/RTMP streaming | 12 hours |

When these time limits are reached, any in-progress egress automatically ends with the status `LIMIT_REACHED`.

You can listen for this status change using the `egress_ended` [webhook](https://docs.livekit.io/home/server/webhooks.md).

## Media subscription limits

Each participant may subscribe to a limited number of media tracks. Currently, the limits are as follows:

- Up to 100 video tracks.
- Up to 100 audio tracks.

For high volume video use cases, consider using pagination and [selective subscriptions](https://docs.livekit.io/home/client/receive.md#selective-subscription) to keep the number of subscriptions within these limits.

## API request rate limits

All projects have a 1000 requests per minute rate limit on API requests. The limit only applies to [Server API](https://docs.livekit.io/reference/server/server-apis.md) requests (for example, `RoomService` or `EgressService` API requests) and doesn't apply to SDK methods like joining a room or sending data packets.

LiveKit doesn't anticipate any project exceeding this rate limit. However, you can reach out to [support](mailto:support@livekit.io) to request an increase. Include the **Project URL** in your email. You can find your project URL in the LiveKit Cloud dashboard in your [Project Settings](https://cloud.livekit.io/projects/p_/settings/project) page.

---

This document was rendered at 2025-08-13T22:17:04.748Z.
For the latest version of this document, see [https://docs.livekit.io/home/cloud/quotas-and-limits.md](https://docs.livekit.io/home/cloud/quotas-and-limits.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).