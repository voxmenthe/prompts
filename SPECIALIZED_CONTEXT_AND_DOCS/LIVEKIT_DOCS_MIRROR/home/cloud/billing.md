LiveKit Docs › Cloud › Billing

---

# Billing

> Learn how LiveKit Cloud billing works.

## Overview

Refer to our latest [blog post](https://blog.livekit.io/towards-a-future-aligned-pricing-model/) and [pricing page](https://livekit.io/pricing) for information about our current pricing.

## Billing

### How we meter

We meter all projects and bill for resources consumed. This table shows the resources we meter and the increments we bill in:

| Resource | Unit | Minimum increment |
| Outbound transfer | GB | 0.01 GB |
| Realtime connection | minute | 1 minute |
| SIP connection | minute | 1 minute |
| Egress Transcode | minute | 1 minute |
| Ingress Transcode | minute | 1 minute |

### Billing cycle

LiveKit Cloud bills monthly. At the end of each month, we calculate the total resources consumed by your project and bill you for the resources consumed.

## Invoices

### Downloading invoices

Paying projects can download previous months' invoices as PDFs on the project's [billing page](https://cloud.livekit.io/projects/p_/billing) (accessible only to project admins) and clicking the "PDF" link in the "Statements" section.

### Customizing invoices

By default, the invoice only lists your project name. Some customers require more information on the invoice, such as a billing address or VAT number. You can add this information to your invoice by clicking the `...` menu to the right of the PDF link, then clicking `Customize “Invoice to:” field`.

This field is a plain text field that accepts any text. Newlines will be preserved on the invoice PDF. For example, you could include your business name and address like so, and the invoice PDF will have line breaks in the same places:

```
Acme Inc.
404 Nowhere Ln.
New York, NY 10001

```

After saving your “Invoice to:” field, you can click the `PDF` link to re-download the invoice PDF and it will include the new information.

---


For the latest version of this document, see [https://docs.livekit.io/home/cloud/billing.md](https://docs.livekit.io/home/cloud/billing.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).