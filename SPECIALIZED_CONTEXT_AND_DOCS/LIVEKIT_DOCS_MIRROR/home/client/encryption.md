LiveKit docs › LiveKit SDKs › End-to-end encryption

---

# End-to-end encryption

> Secure your realtime media tracks with E2EE.

## Overview

LiveKit includes built-in support for end-to-end encryption (E2EE) on realtime audio and video tracks. With E2EE enabled, media data remains fully encrypted from sender to receiver, ensuring that no intermediaries (including LiveKit servers) can access or modify the content. This feature is:

- Available for both self-hosted and LiveKit Cloud customers at no additional cost.
- Ideal for regulated industries and security-critical applications.
- Designed to provide an additional layer of protection beyond standard transport encryption.

> ℹ️ **Security is our highest priority**
> 
> Learn more about [our comprehensive approach to security](https://livekit.io/security).

## How E2EE works

E2EE is enabled at the room level and automatically applied to all media tracks from all participants in that room. You must enable it within the LiveKit SDK for each participant. In many cases you can use a built-in key provider with a single shared key for the whole room. If you require unique keys for each participant, or key rotation during the lifetime of a single room, you can implement your own key provider.

## Key distribution

It is your responsibility to securely generate, store, and distribute encryption keys to your application at runtime. LiveKit does not (and cannot) store or transport encryption keys for you.

If using a shared key, you would typically generate it on your server at the same time that you create a room and distribute it securely to participants alongside their access token for the room. When using unique keys per participant, you may need a more sophisticated method for distributing keys as new participants join the room. Remember that the key is needed for both encryption and decryption, so even when using per-participant keys, you must ensure that all participants have all keys.

## Data channel encryption

Realtime data and text are encrypted using the `encryption` field for `RoomOptions` when you create a room. When the `encryption` field is set, all outgoing data messages (including text and byte streams) are end-to-end encrypted.

End-to-end encryption for data channel messages is the default. However, for backwards compatibility, the `e2ee` field is still supported. If `encryption` is not set, data channel messages are _not_ encrypted.

> ℹ️ **e2ee field is deprecated**
> 
> The `e2ee` field is deprecated and will be removed in the next major version of each client SDK. Use the `encryption` field instead.

> ❗ **Signaling messages and APIs**
> 
> Signaling messages (control messages used to coordinate a WebRTC session) and API calls are _not_ end-to-end encrypted—they're encrypted in transit using TLS, but the LiveKit server can still read them.

## Implementation guide

The following implementation examples have been updated to use the `encryption` field. To learn more, see [Data channel encryption](#data-channel).

These examples show how to use the built-in key provider with a shared key. If you need to use a custom key provider, see the [Using a custom key provider](#custom-key-provider) section.

**JavaScript**:

```typescript
// 1. Initialize the external key provider
const keyProvider = new ExternalE2EEKeyProvider();

// 2. Configure room options
const roomOptions: RoomOptions = {
  encryption: {
    keyProvider: keyProvider,
    // Required for web implementations
    worker: new Worker(new URL('livekit-client/e2ee-worker', import.meta.url)),
  },
};

// 3. Create and configure the room
const room = new Room(roomOptions);

// 4. Set your externally distributed encryption key
await keyProvider.setKey(yourSecureKey);

// 5. Enable E2EE for all local tracks
await room.setE2EEEnabled(true);

// 6. Connect to the room
await room.connect(url, token);

```

---

**iOS**:

```swift
// 1. Initialize the key provider with options
let keyProvider = BaseKeyProvider(isSharedKey: true, sharedKey: "yourSecureKey")

// 2. Configure room options with E2EE
let roomOptions = RoomOptions(encryptionOptions: E2EEOptions(keyProvider: keyProvider))

// 3. Create the room
let room = Room(roomOptions: roomOptions)

// 4. Connect to the room
try await room.connect(url: url, token: token)

```

---

**Android**:

```kotlin
// 1. Initialize the key provider
val keyProvider = BaseKeyProvider()

// 2. Configure room options
val roomOptions = RoomOptions(
    encryptionOptions = E2EEOptions(
        keyProvider = keyProvider
    )
)
// 3. Create and configure the room
val room = LiveKit.create(context, options = roomOptions)

// 4. Set your externally distributed encryption key
keyProvider.setSharedKey(yourSecureKey)

// 5. Connect to the room
room.connect(url, token)

```

---

**Flutter**:

```dart
// 1. Initialize the key provider
final keyProvider = await BaseKeyProvider.create();

// 2. Configure room options
final roomOptions = RoomOptions(
  encryption: E2EEOptions(
    keyProvider: keyProvider,
  ),
);

// 3. Create and configure the room
final room = Room(options: roomOptions);

// 4. Set your externally distributed encryption key
await keyProvider.setSharedKey(yourSecureKey);

// 5. Connect to the room
await room.connect(url, token);

```

---

**React Native**:

```jsx
// 1. Use the hook to create an RNE2EEManager 
//    with your externally distributed shared key
// (Note: if you need a custom key provider, then you'll need 
//        to create the key provider and `RNE2EEManager` directly)
const { e2eeManager } = useRNE2EEManager({
  sharedKey: yourSecureKey,
  dataChannelEncryption: true,
});

// 2. Provide the e2eeManager in your room options
const roomOptions = {
  encryption: {
    e2eeManager,
  },
};

// 3. Pass the room options when creating your room
<LiveKitRoom
  serverUrl={url}
  token={token}
  connect={true}
  options={roomOptions}
  audio={true}
  video={true}
>
</LiveKitRoom>

```

---

**Python**:

```python
# 1. Initialize key provider options with a shared key
e2ee_options = rtc.E2EEOptions()
e2ee_options.key_provider_options.shared_key = YOUR_SHARED_KEY

# 2. Configure room options with E2EE
room_options = RoomOptions(
    auto_subscribe=True,
    e2ee=e2ee_options
)

# 3. Create and connect to the room
room = Room()
await room.connect(url, token, options=room_options)

```

---

**Node.js**:

```javascript
// 1. Initialize the key provider with options
const keyProviderOptions = {
  sharedKey: yourSecureKey, // Your externally distributed encryption key
};

// 2. Configure E2EE options
const e2eeOptions = {
  keyProviderOptions,
};

// 3. Create and configure the room
const room = new Room();

// 4. Connect to the room with E2EE enabled
await room.connect(url, token, {
    e2ee: e2eeOptions,
  }
);

```

### Examples

The following examples include full implementations of E2EE.

- **[Meet example app](https://github.com/livekit-examples/meet)**: E2EE in a production-grade JavaScript app using the `ExternalE2EEKeyProvider`.

- **[Python example app](https://github.com/livekit/python-sdks/blob/main/examples/e2ee.py)**: A simple example app using E2EE with a shared key.

- **[Android example app](https://github.com/livekit/client-sdk-android/tree/main/sample-app)**: An example implementation of E2EE using the built-in key provider.

- **[Multi-platform Flutter example](https://github.com/livekit/client-sdk-flutter/tree/main/example)**: A complete multi-platform example implementation with E2EE support using a shared key.

- **[React Native example](https://github.com/livekit/client-sdk-react-native/tree/main/example)**: A complete example app demonstrating how to use the `useRNE2EEManager` hook and a shared key.

## Using a custom key provider

If your application requires key rotation during the lifetime of a single room or unique keys per participant (such as when implementing the [MEGOLM](https://gitlab.matrix.org/matrix-org/olm/blob/master/docs/megolm.html) or [MLS](https://messaginglayersecurity.rocks/mls-architecture/draft-ietf-mls-architecture.html) protocol), you'll need to implement your own key provider.  The full details of that are beyond the scope of this guide, but a brief outline for the JS SDK is provided below (the process is similar in the other SDKs as well):

1. Extend the `BaseKeyProvider` class.
2. Call `onSetEncryptionKey` with each key/identity pair
3. Set appropriate ratcheting options (`ratchetSalt`, `ratchetWindowSize`, `failureTolerance`, `keyringSize`).
4. Implement the `onKeyRatcheted` method to handle key updates.
5. Call `ratchetKey()` when key rotation is needed.
6. Pass your custom key provider in the room options, in place of the built-in key provider.

---


For the latest version of this document, see [https://docs.livekit.io/home/client/encryption.md](https://docs.livekit.io/home/client/encryption.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).