# `createStreamableValue`

AI SDK RSC is currently experimental. We recommend using [AI SDK
UI](../../ai-sdk-ui/overview.md) for production. For guidance on migrating from
RSC to UI, see our [migration guide](../../ai-sdk-rsc/migrating-to-ui.md).

Create a stream that sends values from the server to the client. The value can be any serializable data.

## Import

```
import { createStreamableValue } from "@ai-sdk/rsc"
```

## API Signature

### Parameters

### value:

any

Any data that RSC supports. Example, JSON.

### Returns

### value:

streamable

This creates a special value that can be returned from Actions to the client. It holds the data inside and can be updated via the update method.
