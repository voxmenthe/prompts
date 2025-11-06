LiveKit docs › LiveKit SDKs › Realtime text & data › Remote method calls

---

# Remote method calls

> Use remote procedure calls (RPCs) to execute custom methods on other participants in the room and await a response.

## Overview

An RPC method can be used to request data, coordinate app-specific state, and more. When used to [forward tool calls](https://docs.livekit.io/agents/build/tools.md#forwarding) from an AI agent, your LLM can directly access data or manipulate the UI in your app's frontend.

Your app can pre-register any number of RPC methods before joining a LiveKit room so they are available to call as soon as another participant joins. Participants can remotely call RPC methods on other participants in the same room.

## Method registration

First register the method on the room with `room.registerRpcMethod` and provide the method's name and a handler function. Any number of methods can be registered on a room.

**JavaScript**:

```typescript
room.registerRpcMethod(
  'greet',
  async (data: RpcInvocationData) => {
    console.log(`Received greeting from ${data.callerIdentity}: ${data.payload}`);
    return `Hello, ${data.callerIdentity}!`;
  }
);

```

---

**Python**:

Pre-registration is not available in all SDKs. Use `local_participant.register_rpc_method` to register an RPC method on the local participant instead.

```python
@room.local_participant.register_rpc_method("greet")
async def handle_greet(data: RpcInvocationData):
    print(f"Received greeting from {data.caller_identity}: {data.payload}")
    return f"Hello, {data.caller_identity}!"

```

---

**Node.js**:

```typescript
room.registerRpcMethod(
  'greet',
  async (data: RpcInvocationData) => {
    console.log(`Received greeting from ${data.callerIdentity}: ${data.payload}`);
    return `Hello, ${data.callerIdentity}!`;
  }
);

```

---

**Rust**:

Pre-registration is not available in all SDKs. Use `local_participant.register_rpc_method` to register an RPC method on the local participant instead.

```rust
room.local_participant().register_rpc_method(
    "greet".to_string(),
    |data| {
        Box::pin(async move {
            println!(
                "Received greeting from {}: {}",
                data.caller_identity,
                data.payload
            );
            return Ok("Hello, ".to_string() + &data.caller_identity);
        })
    },
);

```

---

**Android**:

```kotlin
room.registerRpcMethod(
    "greet"
) { data ->
    println("Received greeting from ${data.callerIdentity}: ${data.payload}")
    "Hello, ${data.callerIdentity}!"
}

```

---

**Swift**:

```swift
room.registerRpcMethod("greet") { data in
    print("Received greeting from \(data.callerIdentity): \(data.payload)")
    return "Hello, \(data.callerIdentity)!"
}

```

---

**Go**:

```go
greetHandler := func(data livekit.RpcInvocationData) (string, error) {
  fmt.Printf("Received greeting from %s: %s\n", data.CallerIdentity, data.Payload)
  return "Hello, " + data.CallerIdentity + "!", nil
}
room.RegisterRpcMethod("greet", greetHandler)

```

## Calling a method

Use `localParticipant.performRpc` to call the registered RPC method on a remote participant by providing the destination participant's identity, method name, and payload. This is an asynchronous operation that returns a string, and might raise an error.

> ℹ️ **Hidden participants**
> 
> [Hidden participants](https://docs.livekit.io/home/get-started/api-primitives.md#hidden-participants) cannot call RPC methods. Any RPC attempt by a hidden participant fails with an error.

**JavaScript**:

```typescript
try {
  const response = await localParticipant.performRpc({
    destinationIdentity: 'recipient-identity',
    method: 'greet',
    payload: 'Hello from RPC!',
  });
  console.log('RPC response:', response);
} catch (error) {
  console.error('RPC call failed:', error);
}

```

---

**Python**:

```python
try:
  response = await room.local_participant.perform_rpc(
    destination_identity='recipient-identity',
    method='greet',
    payload='Hello from RPC!'
  )
  print(f"RPC response: {response}")
except Exception as e:
  print(f"RPC call failed: {e}")

```

---

**Node.js**:

```typescript
try {
  const response = await localParticipant.performRpc({
    destinationIdentity: 'recipient-identity',
    method: 'greet',
    payload: 'Hello from RPC!',
  });
  console.log('RPC response:', response);
} catch (error) {
  console.error('RPC call failed:', error);
}

```

---

**Rust**:

```rust
match room
    .local_participant()
    .perform_rpc(PerformRpcParams {
        destination_identity: "recipient-identity".to_string(),
        method: "greet".to_string(),
        payload: "Hello from RPC!".to_string(),
        ..Default::default()
    })
    .await
{
    Ok(response) => {
        println!("RPC response: {}", response);
    }
    Err(e) => log::error!("RPC call failed: {:?}", e),
}

```

---

**Android**:

```kotlin
try {
    val response = localParticipant.performRpc(
        destinationIdentity = "recipient-identity",
        method = "greet",
        payload = "Hello from RPC!"
    ).await()
    println("RPC response: $response")
} catch (e: RpcError) {
    println("RPC call failed: $e")
}

```

---

**Swift**:

```swift
do {
    let response = try await localParticipant.performRpc(
      destinationIdentity: "recipient-identity",
      method: "greet",
      payload: "Hello from RPC!"
    )
    print("RPC response: \(response)")
} catch let error as RpcError {
    print("RPC call failed: \(error)")
}

```

---

**Go**:

```go
res, err := room.LocalParticipant.PerformRpc(livekit.PerformRpcParams{
  DestinationIdentity: "recipient-identity",
  Method: "greet",
  Payload: "Hello from RPC!",
})
if err != nil {
  fmt.Printf("RPC call failed: %v\n", err)
}
fmt.Printf("RPC response: %s\n", res)

```

## Method names

Method names can be any string, up to 64 bytes long (UTF-8).

## Payload format

RPC requests and responses both support a string payload, with a maximum size of 15KiB (UTF-8). You may use any format that makes sense, such as JSON or base64-encoded data.

## Response timeout

`performRpc` uses a timeout to hang up automatically if the response takes too long. The default timeout is 10 seconds, but you are free to change it as needed in your `performRpc` call. In general, you should set a timeout that is as short as possible while still satisfying your use case.

The timeout you set is used for the entire duration of the request, including network latency. This means the timeout the handler is provided will be shorter than the overall timeout.

## Errors

`performRpc` will return certain built-in errors (detailed below), or your own custom errors generated in your remote method handler.

To return a custom error to the caller, handlers should throw an error of the type `RpcError` with the following properties:

- `code`: A number that indicates the type of error. Codes 1001-1999 are reserved for LiveKit internal errors.
- `message`: A string that provides a readable description of the error.
- `data`: An optional string that provides even more context about the error, with the same format and limitations as request/response payloads.

Any other error thrown in a handler will be caught and the caller will receive a generic `1500 Application Error`.

#### Built-in error types

| Code | Name | Description |
| 1400 | UNSUPPORTED_METHOD | Method not supported at destination |
| 1401 | RECIPIENT_NOT_FOUND | Recipient not found |
| 1402 | REQUEST_PAYLOAD_TOO_LARGE | Request payload too large |
| 1403 | UNSUPPORTED_SERVER | RPC not supported by server |
| 1404 | UNSUPPORTED_VERSION | Unsupported RPC version |
| 1500 | APPLICATION_ERROR | Application error in method handler |
| 1501 | CONNECTION_TIMEOUT | Connection timeout |
| 1502 | RESPONSE_TIMEOUT | Response timeout |
| 1503 | RECIPIENT_DISCONNECTED | Recipient disconnected |
| 1504 | RESPONSE_PAYLOAD_TOO_LARGE | Response payload too large |
| 1505 | SEND_FAILED | Failed to send |

## Examples

The following SDKs have full RPC examples.

- **[RPC in Go](https://github.com/livekit/server-sdk-go/blob/main/examples/rpc/main.go)**: Example showing how to register and call RPC methods in Go.

- **[RPC in JavaScript](https://github.com/livekit/client-sdk-js/tree/main/examples/rpc)**: Example showing how to register and call RPC methods in JavaScript.

- **[RPC in Flutter](https://github.com/livekit-examples/flutter-examples/blob/main/packages/rpc-demo/lib/main.dart)**: Example showing how to register and call RPC methods in Flutter.

- **[RPC in Python](https://github.com/livekit/python-sdks/blob/main/examples/rpc.py)**: Example showing how to register and call RPC methods in Python.

- **[RPC in Rust](https://github.com/livekit/rust-sdks/tree/main/examples/rpc)**: Example showing how to register and call RPC methods in Rust.

- **[RPC in Node.js](https://github.com/livekit/node-sdks/tree/main/examples/rpc)**: Example showing how to register and call RPC methods in Node.js.

---


For the latest version of this document, see [https://docs.livekit.io/home/client/data/rpc.md](https://docs.livekit.io/home/client/data/rpc.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).