LiveKit docs › LiveKit Cloud › Billing

---

# LiveKit Cloud billing

> Guide to LiveKit Cloud invoices and billing cycles.

## Pricing

Refer to the following page for current pricing information:

- **[LiveKit Cloud pricing](https://livekit.io/pricing)**: Current pricing, limits, and quotas for LiveKit Cloud plans.

- **[LiveKit Inference pricing](https://livekit.io/pricing/inference)**: Current pricing for LiveKit Inference models.

## Resource metering

All LiveKit Cloud pricing plans include usage-based pricing, metered by resource consumption. The following sections provide more information about how each specific type of resource is metered. For information on quotas and limits, see the [Quotas and limits](https://docs.livekit.io/home/cloud/quotas-and-limits.md) guide.

> ℹ️ **Rounding up**
> 
> Each invididual resource usage is rounded up to the minimum increment prior to summation. For example, a connection lasting 10 seconds is billed as 1 connection minute, and one lasting 70 seconds is billed as 2 connection minutes.

### Realtime media and data

LiveKit Cloud transport services, including [WebRTC media](https://docs.livekit.io/home/client/tracks.md), [SIP](https://docs.livekit.io/sip.md), and [Stream import](https://docs.livekit.io/home/ingress/overview.md), and [Recording and export](https://docs.livekit.io/home/egress/overview.md), are metered on a combination of **time** and **data transfer**. The following table shows the units and minimum increments for each resource.

| Resource type | Unit | Minimum increment |
| Time | Minute | 1 minute |
| Data transfer | GB | 0.01 GB |

### Agent deployment

Agents deployed to LiveKit Cloud are metered by the **agent session minute**, in increments of 1 minute. This reflects the amount of time the agent is actively connected to a WebRTC or SIP-based session.

### LiveKit Inference

LiveKit Inference usage is metered by **tokens**, **time**, or **characters**, depending on the specific resource, according to the following table.

| Model type | Unit | Minimum increment |
| STT | Seconds (connection time) | 1 second |
| LLM | Tokens (input and output) | 1 token |
| TTS | Characters (text) | 1 character |

## Invoices

LiveKit Cloud invoices are issued at the end of each month. The invoice total is based on resource consumption and the project's selected plan. No invoice is issued for projects with no amount due.

### Downloading invoices

Past monthly invoices are available on the project's [billing page](https://cloud.livekit.io/projects/p_/billing) for project admins. Click the **View Invoices** link in the **Statements** section to download the invoice.

---


For the latest version of this document, see [https://docs.livekit.io/home/cloud/billing.md](https://docs.livekit.io/home/cloud/billing.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).