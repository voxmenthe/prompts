LiveKit docs › Accepting calls › Workflow

---

# Accepting inbound calls

> Workflow and configuration guide for accepting inbound calls.

## Inbound call workflow

When an inbound call is received, LiveKit SIP receives a text-based INVITE request. This can come from either your SIP trunking provider or through a LiveKit phone number. For third-party SIP providers, the SIP service first verifies authorization to use the trunk. This can vary based on the LiveKit trunk configuration. If you're using LiveKit Phone Numbers, no inbound trunk configuration or verification is required.

The SIP service then looks for a matching dispatch rule. If there's a matching dispatch rule, a SIP participant is created for the caller and put into a LiveKit room. Depending on the dispatch rule, other participants (for example, a voice agent or other users) might join the room.

The following diagram shows the inbound call workflow.

![Inbound SIP workflow](/images/sip/inbound-sip-workflow.svg)

1. User dials the SIP trunking provider phone number or a LiveKit Phone Number.
2. LiveKit SIP receives the INVITE request:

- For third-party SIP providers: Authenticates trunk credentials and checks if the call is allowed based on the inbound trunk configuration.
- For LiveKit Phone Numbers: Skip to the next step.
3. LiveKit SIP finds a matching dispatch rule.
4. LiveKit server creates a SIP participant for the caller and places them in a LiveKit room (per the dispatch rule).
5. User hears dial tone until LiveKit SIP responds to the call:

- If the dispatch rule has a pin, prompts the user with "Please enter room pin and press hash to confirm." If the pin is incorrect, the call is disconnected with a tone. If the pin is correct, the user is prompted to enter the room.
- User continues to hear a dial tone until another participant publishes tracks to the room.

## Setup for accepting calls

LiveKit Phone Numbers provide a simple setup process that only requires purchasing a phone number and creating a dispatch rule.

1. **Purchase a LiveKit Phone Number**

Purchase a phone number through [LiveKit Phone Numbers](https://docs.livekit.io/sip/cloud/phone-numbers.md).
2. **Create a dispatch rule**

Create a [dispatch rule](https://docs.livekit.io/sip/dispatch-rule.md). The dispatch rules dictate how SIP participants and LiveKit rooms are created for incoming calls. The rules can include whether a caller needs to enter a pin code to join a room and any custom metadata or attributes to be added to SIP participants.

### Using a third-party SIP provider

Third-party SIP providers require both an inbound trunk and a dispatch rule for proper authentication and call routing. To set up a third-party SIP provider, see the [SIP trunk setup](https://docs.livekit.io/sip/quickstarts/configuring-sip-trunk.md)  guide.

## Next steps

See the following guide to create an AI agent to receive inbound calls.

- **[Voice AI telephony guide](https://docs.livekit.io/agents/start/telephony.md)**: Create an AI agent to receive inbound calls.

---


For the latest version of this document, see [https://docs.livekit.io/sip/accepting-calls.md](https://docs.livekit.io/sip/accepting-calls.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).