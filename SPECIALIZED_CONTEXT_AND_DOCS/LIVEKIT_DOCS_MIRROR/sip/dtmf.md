LiveKit docs › Features › DTMF

---

# Handling DTMF

> Sending and receiving DTMF tones.

## Overview

LiveKit's Telephony stack fully supports Dual-tone Multi-Frequency (DTMF) tones, enabling integration with legacy Interactive Voice Response (IVR) systems. It also enables agents to receive DTMF tones from telephone users.

## Agents framework support

If you're building telephony apps with the LiveKit Agents framework, there are additional features that provide support for DTMF:

- The `ivr_detection` option for [`AgentSession`](https://docs.livekit.io/agents/build/sessions.md#session-options). When set to `True`, this automatically makes use of built-in tools to detect IVR systems and relay DTMF tones from the user back to the telephony provider.

To enable IVR detection, set `ivr_detection=True` in the `AgentSession` constructor:

```python
session = AgentSession(
  ivr_detection=True,
  # ... stt, llm, vad, turn_detection, etc.
)

```
- A prebuilt task for collecting DTMF inputs. It can be used to collect any number of digits from a caller, including, for example, a phone number or credit card number. The task supports both DTMF tones and spoken digits. To learn more, see [GetDtmfTask](https://docs.livekit.io/agents/build/tasks.md#getdtmftask).

## Sending DTMF using the API

To send DTMF tones, use the `publishDtmf` API on the `localParticipant`.

Any participant in the room can use the `publishDtmf` API to transmit DTMF tones to the room. SIP participants in the room receive the tones and relay them to the telephone user.

The `publishDtmf` API requires two parameters:

- `code`: DTMF code
- `digit`: DTMF digit

The following examples publishes the DTMF tones `1`, `2`, `3`, and `#` in sequence.

**Node.js**:

```typescript
// publishes 123# in DTMF
await localParticipant.publishDtmf(1, '1');
await localParticipant.publishDtmf(2, '2');
await localParticipant.publishDtmf(3, '3');
await localParticipant.publishDtmf(11, '#');

```

---

**Python**:

```python
# publishes 123# in DTMF
await local_participant.publish_dtmf(code=1, digit='1')
await local_participant.publish_dtmf(code=2, digit='2')
await local_participant.publish_dtmf(code=3, digit='3')
await local_participant.publish_dtmf(code=11, digit='#')

```

---

**Go**:

```go
import (
  "github.com/livekit/protocol/livekit"
)

// publishes 123# in DTMF
localParticipant.PublishDataPacket(&livekit.SipDTMF{
  Code: 1,
  Digit: "1",
})
localParticipant.PublishDataPacket(&livekit.SipDTMF{
  Code: 2,
  Digit: "2",
})
localParticipant.PublishDataPacket(&livekit.SipDTMF{
  Code: 3,
  Digit: "3",
})
localParticipant.PublishDataPacket(&livekit.SipDTMF{
  Code: 11,
  Digit: "#",
})

```

> ℹ️ **Info**
> 
> Sending DTMF tones requires both a numeric code and a string representation to ensure compatibility with various SIP implementations.
> 
> Special characters like `*` and `#` are mapped to their respective numeric codes. See [RFC 4733](https://datatracker.ietf.org/doc/html/rfc4733#section-3.2) for details.

## Receiving DTMF by listening to events

When SIP receives DTMF tones, they are relayed to the room as events that participants can listen for.

**Node.js**:

```typescript
room.on(RoomEvent.DtmfReceived, (code, digit, participant) => {
  console.log('DTMF received from participant', participant.identity, code, digit);
});

```

---

**Python**:

```python
@room.on("sip_dtmf_received")
def dtmf_received(dtmf: rtc.SipDTMF):
    logging.info(f"DTMF received from {dtmf.participant.identity}: {dtmf.code} / {dtmf.digit}")

```

---

**Go**:

```go
import (
  "fmt"

  "github.com/livekit/protocol/livekit"
  lksdk "github.com/livekit/server-sdk-go/v2"
)

func DTMFCallbackExample() {
  // Create a new callback handler
	cb := lksdk.NewRoomCallback()

	// Handle data packets received from other participants
	cb.OnDataPacket = func(data lksdk.DataPacket, params lksdk.DataReceiveParams) {
		// handle DTMF
		switch val := data.(type) {
		case *livekit.SipDTMF:
			fmt.Printf("Received DTMF from %s: %s (%d)\n", params.SenderIdentity, val.Digit, val.Code)
		}
	}

  room := lksdk.NewRoom(cb)
  ...
}

```

---


For the latest version of this document, see [https://docs.livekit.io/sip/dtmf.md](https://docs.livekit.io/sip/dtmf.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).