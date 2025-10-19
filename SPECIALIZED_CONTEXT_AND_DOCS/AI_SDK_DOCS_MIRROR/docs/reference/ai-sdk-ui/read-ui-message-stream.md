# readUIMessageStream

Transforms a stream of `UIMessageChunk`s into an `AsyncIterableStream` of `UIMessage`s.

UI message streams are useful outside of Chat use cases, e.g. for terminal UIs, custom stream consumption on the client, or RSC (React Server Components).

## Import

```tsx
import { readUIMessageStream } from 'ai';
```

## API Signature

### Parameters

### message?:

UIMessage

The last assistant message to use as a starting point when the conversation is resumed. Otherwise undefined.

### stream:

ReadableStream<UIMessageChunk>

The stream of UIMessageChunk objects to read.

### onError?:

(error: unknown) => void

A function that is called when an error occurs during stream processing.

### terminateOnError?:

boolean

Whether to terminate the stream if an error occurs. Defaults to false.

### Returns

An `AsyncIterableStream` of `UIMessage`s. Each stream part represents a different state of the same message as it is being completed.

For comprehensive examples and use cases, see [Reading UI Message Streams](../../ai-sdk-ui/reading-ui-message-streams.md).
