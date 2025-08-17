LiveKit Docs › Server APIs › Participant management

---

# Managing participants

> List, remove, and mute from your backend server.

## Initialize RoomServiceClient

Participant management is done with a RoomServiceClient, created like so:

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
pip install livekit-api

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

## List Participants

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

## Get details on a Participant

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
lk room participants get --room <ROOM_NAME> <ID>

```

## Updating permissions

You can modify a participant's permissions on-the-fly using `UpdateParticipant`. When there's a change in permissions, connected clients will be notified through the `ParticipantPermissionChanged` event.

This comes in handy, for instance, when transitioning an audience member to a speaker role within a room.

Note that if you revoke the `CanPublish` permission from a participant, all tracks they've published will be automatically unpublished.

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

// ...and later move them back to audience
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
  <ID>

```

## Updating metadata

You can modify a Participant's metadata whenever necessary. Once changed, connected clients will receive a `ParticipantMetadataChanged` event.

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
  <ID>

```

## Remove a Participant

`RemoteParticipant` will forcibly disconnect the participant from the room. However, this action doesn't invalidate the participant's token.

To prevent the participant from rejoining the same room, consider the following measures:

- Generate access tokens with a short TTL (Time-To-Live).
- Refrain from providing a new token to the same participant via your application's backend.

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
lk room participants remove <ID>

```

## Mute/unmute a Participant's Track

To mute a particular Track from a Participant, first get the TrackSid from `GetParticipant` (above), then call `MutePublishedTrack`:

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
  --identity <ID> \
  <TRACK_SID>

```

You may also unmute the track by setting `muted` to `false`.

> ℹ️ **Note**
> 
> Being remotely unmuted can catch users by surprise, so it's disabled by default.
> 
> To allow remote unmute, select the `Admins can remotely unmute tracks` option in your [project settings](https://cloud.livekit.io/projects/p_/settings/project).
> 
> If you're self-hosting, configure `room.enable_remote_unmute: true` in your config YAML.

---


For the latest version of this document, see [https://docs.livekit.io/home/server/managing-participants.md](https://docs.livekit.io/home/server/managing-participants.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).