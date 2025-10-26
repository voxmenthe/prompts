LiveKit docs › Provider-specific guides › Plivo

---

# Create and configure a Plivo SIP trunk

> Step-by-step instructions for creating inbound and outbound SIP trunks using Plivo.

Connect [Plivo's](https://plivo.com) SIP trunking with LiveKit for inbound and outbound calls.

## Prerequisites

The following are required to complete the steps in this guide:

- [Plivo account](https://console.plivo.com/)
- [LiveKit Cloud project](https://cloud.livekit.io/projects/p_/settings/project)

## Inbound calling

To accept inbound calls with Plivo and LiveKit, complete the steps in the following sections.

### Create a SIP trunk

Create an inbound trunk in Plivo, setting your LiveKit SIP endpoint as the primary URI.

1. Sign in to the [Plivo Console](https://console.plivo.com/).
2. Navigate to **Zentrunk** → [**Inbound Trunks**](https://console.plivo.com/zentrunk/inbound-trunks/).
3. Select **Create New Inbound Trunk** and provide a descriptive name for your trunk.
4. For **Primary URI**, select **Add New URI** and enter your LiveKit [SIP endpoint](https://docs.livekit.io/sip/quickstarts/configuring-sip-trunk.md#sip-endpoint). Include `;transport=tcp` in the URI. For example, `vjnxecm0tjk.sip.livekit.cloud;transport=tcp`.

If you're signed in to LiveKit Cloud, your SIP endpoint is automatically included in the following example:

```shell
%{regionalEndpointSubdomain}%.sip.livekit.cloud;transport=tcp

```

> ℹ️ **Secure trunking**
> 
> If you're setting up [secure trunking](https://docs.livekit.io/sip/secure-trunking.md), use `;transport=tls` instead of `;transport=tcp`.
5. Select **Create Trunk**.

### Connect your phone number

Connect your Plivo phone number to the inbound trunk.

1. Navigate to **Phone Numbers** → [**Your Numbers**](https://console.plivo.com/active-phone-numbers/).
2. Select the phone number to connect to the trunk.
3. In the **Number Configuration** section → **Application Type**, select **Zentrunk**.
4. For **Trunk**, select the trunk you created in the previous step.
5. Select **Update**.

### Configure LiveKit to accept calls

Set up an [inbound trunk](https://docs.livekit.io/sip/trunk-inbound.md) and [dispatch rule](https://docs.livekit.io/sip/dispatch-rule.md) in LiveKit to accepts calls to your Plivo phone number.

### Test incoming calls

Start your LiveKit agent and call your Plivo phone number. Your agent should answer the call. If you don't have an agent, see the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md) to create one.

### Troubleshooting

For help troubleshooting inbound calls, check the following logs:

- First check the [Plivo logs](https://console.plivo.com/zentrunk/logs/calls/).
- Then check the [call logs](https://cloud.livekit.io/projects/p_/telephony) in your LiveKit Cloud dashboard.

## Outbound calling

To make outbound calls with LiveKit and Plivo and LiveKit, complete the steps in the following sections.

### Create an outbound trunk in Plivo

Set up an outbound trunk with username and password authentication in Plivo.

1. Sign in to the [Plivo Console](https://console.plivo.com/).
2. Navigate to **Zentrunk** → [**Outbound Trunks**](https://console.plivo.com/zentrunk/outbound-trunks/).
3. Select **Create New Outbound Trunk** and provide a descriptive name for your trunk.
4. In the **Trunk Authentication** section → **Credentials List**, select **Add New Credentials List**.
5. Add a username and strong password for outbound call authentication. Make sure these values match the username and password you use for your LiveKit outbound trunk.
6. For **Secure Trunking**, select **Enabled** (recommended).

> 💡 **Secure trunking**
> 
> If you enable secure trunking in Plivo, you must also enable secure trunking in LiveKit. To learn more, see [Secure trunking](https://docs.livekit.io/sip/secure-trunking.md).
7. Select **Create Trunk** to complete your outbound trunk configuration.

Copy the **Termination SIP Domain** for the next step.

### Configure LiveKit to make outbound calls

Create an [outbound trunk](https://docs.livekit.io/sip/trunk-outbound.md) in LiveKit using the **Termination SIP Domain**, and username and password from the previous section.

### Place an outbound call

Test your configuration by placing an outbound call with LiveKit using the `CreateSIPParticipant` API. To learn more, see [Creating a SIP participant](https://docs.livekit.io/sip/outbound-calls.md#creating-a-sip-participant).

### Troubleshooting

If the call fails to connect, check the following common issues:

- Verify your SIP URI. It must include `;transport=tcp`.
- Verify your Plivo phone number is associated with the correct trunk.

For outbound calls, check the following logs:

- First check the [call logs](https://cloud.livekit.io/projects/p_/telephony) in your LiveKit Cloud dashboard.
- Then check the [Plivo logs](https://console.plivo.com/zentrunk/logs/calls/).

For error codes, see the [Plivo hangup codes](https://www.plivo.com/docs/voice/troubleshooting/hangup-causes) reference.

## Regional restrictions

If your calls are made from a Plivo India phone number, or you're dialing numbers in India, you must enable [region pinning](https://docs.livekit.io/sip/cloud.md#region-pinning) for your LiveKit project. This restricts calls to India to comply with local telephony regulations. Your calls will fail to connect if region pinning is not enabled.

For other countries, select the region closest to the location of your call traffic for optimal performance.

## Next steps

The following guides provide next steps for building your LiveKit telephony app.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: A quickstart guide to build a voice AI agent to answer incoming calls.

- **[Agents telephony integration](https://docs.livekit.io/agents/start/telephony.md)**: Learn how to receive and make calls with a voice AI agent

- **[Call forwarding using SIP REFER](https://docs.livekit.io/sip/transfer-cold.md)**: How to forward calls to another number or SIP endpoint with SIP REFER.

- **[Agent-assisted warm transfer](https://docs.livekit.io/sip/transfer-warm.md)**: A comprehensive guide to transferring calls using an AI agent to provide context.

- **[Secure trunking for SIP calls](https://docs.livekit.io/sip/secure-trunking.md)**: How to enable secure trunking for LiveKit SIP.

- **[Region pinning for SIP](https://docs.livekit.io/sip/cloud.md)**: Use region pinning to restrict calls to a specific region.

---


For the latest version of this document, see [https://docs.livekit.io/sip/quickstarts/configuring-plivo-trunk.md](https://docs.livekit.io/sip/quickstarts/configuring-plivo-trunk.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).