# `pipeUIMessageStreamToResponse`

The `pipeUIMessageStreamToResponse` function pipes streaming data to a Node.js ServerResponse object (see [Streaming Data](../../ai-sdk-ui/streaming-data.md)).

## Import

```
import { pipeUIMessageStreamToResponse } from "ai"
```

## Example

```tsx
pipeUIMessageStreamToResponse({
  response: serverResponse,
  status: 200,
  statusText: 'OK',
  headers: {
    'Custom-Header': 'value',
  },
  stream: myUIMessageStream,
  consumeSseStream: ({ stream }) => {
    // Optional: consume the SSE stream independently
    console.log('Consuming SSE stream:', stream);
  },
});
```

## API Signature

### Parameters

### response:

ServerResponse

The Node.js ServerResponse object to pipe the data to.

### stream:

ReadableStream<UIMessageChunk>

The UI message stream to pipe to the response.

### status:

number

The status code for the response.

### statusText:

string

The status text for the response.

### headers:

Headers | Record<string, string>

Additional headers for the response.

### consumeSseStream:

({ stream }: { stream: ReadableStream }) => void

Optional function to consume the SSE stream independently. The stream is teed and this function receives a copy.
