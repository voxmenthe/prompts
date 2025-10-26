LiveKit docs › Server APIs › Participant management

---

# Managing participants

> List, remove, and mute from your backend server.

## Initialize RoomServiceClient

Participant management is done through the room service. Create a `RoomServiceClient`:

**Go**:

```go
import (
  lksdk "github.com/livekit/server-sdk-go"
  livekit "github.com/livekit/protocol/livekit"
)

// ...

host := "https://my.livekit.host"
roomClient := lksdk.NewRoomServiceClient(host, "api-key", "secret-key")

```

---

**Python**:

```shell
uv add livekit-api

```

```python
from livekit.api import LiveKitAPI

# Will read LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET from environment variables
async with api.LiveKitAPI() as lkapi:
  # ... use your client with `lkapi.room` ...

```

---

**Node.js**:

```js
import { Room, RoomServiceClient } from 'livekit-server-sdk';

const livekitHost = 'https://my.livekit.host';
const roomService = new RoomServiceClient(livekitHost, 'api-key', 'secret-key');

```

Use the `RoomServiceClient` to manage participants in a room with the APIs in the following sections. To learn more about grants and the required privileges for each API, see [Authentication](https://docs.livekit.io/home/get-started/authentication.md).

## List participants

You can list all the participants in a room using the `ListParticipants` API.

### Required privileges

You must have the `roomList` grant to list participants.

### Examples

**Go**:

```go
res, err := roomClient.ListParticipants(context.Background(), &livekit.ListParticipantsRequest{
  Room: roomName,
})

```

---

**Python**:

```python
from livekit.api import ListParticipantsRequest

res = await lkapi.room.list_participants(ListParticipantsRequest(
  room=room_name
))

```

---

**Node.js**:

```js
const res = await roomService.listParticipants(roomName);

```

---

**LiveKit CLI**:

```shell
lk room participants list <ROOM_NAME>

```

## Get participant details

Get detailed information about a participant in a room using the `GetParticipant` API.

### Required privileges

You must have the [`roomAdmin`](https://docs.livekit.io/home/get-started/authentication.md#video-grant) grant to get detailed participant information.

### Parameters

| Name | Type | Required | Description |
| `room` | string | ✓ | Room participant is currently in. |
| `identity` | string | ✓ | Identity of the participant to get. |

### Examples

**Go**:

```go
res, err := roomClient.GetParticipant(context.Background(), &livekit.RoomParticipantIdentity{
  Room:     roomName,
  Identity: identity,
})

```

---

**Python**:

```python
from livekit.api import RoomParticipantIdentity

res = await lkapi.room.get_participant(RoomParticipantIdentity(
  room=room_name,
  identity=identity,
))

```

---

**Node.js**:

```js
const res = await roomService.getParticipant(roomName, identity);

```

---

**LiveKit CLI**:

```shell
lk room participants get --room <ROOM_NAME> <PARTICIPANT_ID>

```

## Update participant

You can modify a participant's permissions and metadata using the `UpdateParticipant` API.

### Required privileges

You must have the `roomAdmin` grant to update a participant.

### Parameters

At least one of `permission` or `metadata` must be set, along with the required `room` and `identity` fields.

| Name | Type | Required | Description |
| `room` | string | ✓ | Room participant is currently in. |
| `identity` | string | ✓ | Identity of the participant to update. |
| `permission` | [ParticipantPermission](https://docs.livekit.io/reference/server/server-apis.md#participantpermission) |  | Permissions to update for the participant. Required if `metadata` is _not_ set. |
| `metadata` | string |  | Metadata to update for the participant. Required if `permission` is _not_ set. |
| `name` | string |  | Display name to update for the participant. |
| `attributes` | map[string]string |  | Attributes to update for the participant. |

### Updating participant permissions

You can update a participant's permissions using the `Permission` field in the `UpdateParticipantRequest`. When there's a change in permissions, connected clients are notified through a `ParticipantPermissionChanged` event.

This is useful, for example, to promote an audience member to a speaker role within a room by granting them the `CanPublish` privilege.

> ℹ️ **Revoking permissions unpublishes tracks**
> 
> When you revoke the `CanPublish` permission from a participant, all tracks they've published are automatically unpublished.

**Go**:

```go
// Promotes an audience member to a speaker
res, err := c.UpdateParticipant(context.Background(), &livekit.UpdateParticipantRequest{
  Room: roomName,
  Identity: identity,
  Permission: &livekit.ParticipantPermission{
    CanSubscribe: true,
    CanPublish: true,
    CanPublishData: true,
  },
})

// ...and later revokes their publishing permissions as speaker
res, err := c.UpdateParticipant(context.Background(), &livekit.UpdateParticipantRequest{
  Room: roomName,
  Identity: identity,
  Permission: &livekit.ParticipantPermission{
    CanSubscribe: true,
    CanPublish: false,
    CanPublishData: true,
  },
})

```

---

**Python**:

```python
from livekit.api import UpdateParticipantRequest, ParticipantPermission

# Promotes an audience member to a speaker
await lkapi.room.update_participant(UpdateParticipantRequest(
  room=room_name,
  identity=identity,
  permission=ParticipantPermission(
    can_subscribe=True,
    can_publish=True,
    can_publish_data=True,
  ),
))

# ...and later move them back to audience
await lkapi.room.update_participant(UpdateParticipantRequest(
  room=room_name,
  identity=identity,
  permission=ParticipantPermission(
    can_subscribe=True,
    can_publish=False,
    can_publish_data=True,
  ),
))

```

---

**Node.js**:

```js
// Promotes an audience member to a speaker
await roomService.updateParticipant(roomName, identity, undefined, {
  canPublish: true,
  canSubscribe: true,
  canPublishData: true,
});

// ...and later move them back to audience
await roomService.updateParticipant(roomName, identity, undefined, {
  canPublish: false,
  canSubscribe: true,
  canPublishData: true,
});

```

---

**LiveKit CLI**:

```shell
lk room participants update \
  --permissions '{"can_publish":true,"can_subscribe":true,"can_publish_data":true}' \
  --room <ROOM_NAME> \
  <PARTICIPANT_ID>

```

### Updating participant metadata

You can modify a participant's metadata using the `Metadata` field in the `UpdateParticipantRequest`. When metadata is changed, connected clients receive a `ParticipantMetadataChanged` event.

**Go**:

```go
data, err := json.Marshal(values)
_, err = c.UpdateParticipant(context.Background(), &livekit.UpdateParticipantRequest{
  Room: roomName,
  Identity: identity,
  Metadata: string(data),
})

```

---

**Python**:

```python
from livekit.api import UpdateParticipantRequest

await lkapi.room.update_participant(UpdateParticipantRequest(
  room=room_name,
  identity=identity,
  metadata=json.dumps({"some": "values"}),
))

```

---

**Node.js**:

```js
const data = JSON.stringify({
  some: 'values',
});

await roomService.updateParticipant(roomName, identity, data);

```

---

**LiveKit CLI**:

```shell
lk room participants update \
  --metadata '{"some":"values"}' \
  --room <ROOM_NAME> \
  <PARTICIPANT_ID>

```

## Move participant

> ℹ️ **LiveKit Cloud feature**
> 
> This feature is only available in LiveKit Cloud.

You can move a participant from one room to a different room using the `MoveParticipant` API. Moving a participant removes them from the source room and adds them to the destination room. For example, this API can be used to move a participant from a call room to another room in an [agent-assisted call transfer](https://docs.livekit.io/sip/transfer-warm.md) workflow.

### Required privileges

You must have the `roomAdmin` grant to move a participant.

### Parameters

| Name | Type | Required | Description |
| `room` | string | ✓ | Room participant is currently in. |
| `identity` | string | ✓ | Identity of the participant to move. |
| `destination_room` | string | ✓ | Room to move participant into. |

### Examples

**Go**:

```go
res, err := roomClient.MoveParticipant(context.Background(), &livekit.MoveParticipantRequest{
  Room: roomName,
  Identity: identity,
  DestinationRoom: destinationRoom,
})

```

---

**Python**:

```python
from livekit.api import MoveParticipantRequest

await lkapi.room.move_participant(MoveParticipantRequest(
  room="<CURRENT_ROOM_NAME>",
  identity="<PARTICIPANT_ID>",
  destination_room="<NEW_ROOM_NAME>",
))

```

---

**Node.js**:

```js
await roomService.moveParticipant(roomName, identity, destinationRoom);

```

---

**LiveKit CLI**:

```shell
lk room participants move --room <CURRENT_ROOM_NAME> \
  --identity <PARTICIPANT_ID> \
  --destination-room <NEW_ROOM_NAME>

```

## Forward participant

> ℹ️ **LiveKit Cloud feature**
> 
> This feature is only available in LiveKit Cloud.

You can forward a participant to one or more rooms using the `ForwardParticipant` API. Forwarding allows you to share a participant's tracks with other rooms. For example, if you have a single ingress feed that you want simultaneously share to multiple rooms.

A forwarded participant's tracks are shared to destination rooms until the participant leaves the room or is removed from a destination room using `RemoveParticipant`.

### Required privileges

You must have the `roomAdmin` and `destinationRoom` grants to forward a participant to the room specified for the `destinationRoom` in the grant.

### Parameters

| Name | Type | Required | Description |
| `room` | string | ✓ | Room participant is currently in. |
| `identity` | string | ✓ | Identity of the participant to forward. |
| `destination_room` | string | ✓ | Room to forward participant's tracks to. |

### Examples

**Go**:

```go
res, err := roomClient.ForwardParticipant(context.Background(), &livekit.ForwardParticipantRequest{
  Room: roomName,
  Identity: identity,
  DestinationRoom: destinationRoom,
})

```

---

**Python**:

```python
from livekit.api import ForwardParticipantRequest

await lkapi.room.forward_participant(ForwardParticipantRequest(
  room="<CURRENT_ROOM_NAME>",
  identity="<PARTICIPANT_ID>",
  destination_room="<NEW_ROOM_NAME>",
))

```

---

**Node.js**:

```js
await roomService.fowardParticipant(roomName, identity, destinationRoom);

```

---

**LiveKit CLI**:

```shell
lk room participants forward --room <CURRENT_ROOM_NAME> \
  --identity <PARTICIPANT_ID> \
  --destination-room <NEW_ROOM_NAME>

```

## Remove participant

The `RemoveParticipant` API forcibly disconnects the participant from the room. However, this action doesn't invalidate the participant's token.

To prevent the participant from rejoining the same room, consider the following measures:

- Generate access tokens with a short TTL (Time-To-Live).
- Refrain from providing a new token to the same participant via your application's backend.

### Required privileges

You must have the `roomAdmin` grant to remove a participant.

### Parameters

| Name | Type | Required | Description |
| `room` | string | ✓ | Room participant is currently in. |
| `identity` | string | ✓ | Identity of the participant to remove. |

### Examples

**Go**:

```go
res, err := roomClient.RemoveParticipant(context.Background(), &livekit.RoomParticipantIdentity{
  Room:     roomName,
  Identity: identity,
})

```

---

**Python**:

```python
from livekit.api import RoomParticipantIdentity

await lkapi.room.remove_participant(RoomParticipantIdentity(
  room=room_name,
  identity=identity,
))

```

---

**Node.js**:

```js
await roomService.removeParticipant(roomName, identity);

```

---

**LiveKit CLI**:

```shell
lk room participants remove <PARTICIPANT_ID>

```

## Mute or unmute participant

To mute or unmute a specific participant track, you must first get the `TrackSid` using the `GetParticipant` [API](#getparticipant). You can then call the `MutePublishedTrack` API with the track SID.

### Required privileges

You must have the `roomAdmin` grant to mute or unmute a participant's published track.

### Parameters

| Name | Type | Required | Description |
| `room` | string | ✓ | Room participant is currently in. |
| `identity` | string | ✓ | Identity of the participant to mute. |
| `track_sid` | string | ✓ | SID of the track to mute. |
| `muted` | bool | ✓ | Whether to mute the track:- `true` to mute
- `false` to unmute |

### Examples

**Go**:

```go
res, err := roomClient.MutePublishedTrack(context.Background(), &livekit.MuteRoomTrackRequest{
  Room:     roomName,
  Identity: identity,
  TrackSid: "track_sid",
  Muted:    true,
})

```

---

**Python**:

```python
from livekit.api import MuteRoomTrackRequest

await lkapi.room.mute_published_track(MuteRoomTrackRequest(
  room=room_name,
  identity=identity,
  track_sid="track_sid",
  muted=True,
))

```

---

**Node.js**:

```js
await roomService.mutePublishedTrack(roomName, identity, 'track_sid', true);

```

---

**LiveKit CLI**:

```shell
lk room mute-track \
  --room <ROOM_NAME> \
  --identity <PARTICIPANT_ID> \
  <TRACK_SID>

```

You can also unmute the track by setting `muted` to `false`.

> ℹ️ **Note**
> 
> Being remotely unmuted can catch users by surprise, so it's turned off by default.
> 
> To allow remote unmute, select the **Admins can remotely unmute tracks** option in your [project settings](https://cloud.livekit.io/projects/p_/settings/project).
> 
> If you're self-hosting, configure `room.enable_remote_unmute: true` in your config YAML.

---


For the latest version of this document, see [https://docs.livekit.io/home/server/managing-participants.md](https://docs.livekit.io/home/server/managing-participants.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).