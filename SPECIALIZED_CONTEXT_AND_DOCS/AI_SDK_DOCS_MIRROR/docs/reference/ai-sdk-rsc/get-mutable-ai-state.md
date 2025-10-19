# `getMutableAIState`

AI SDK RSC is currently experimental. We recommend using [AI SDK
UI](../../ai-sdk-ui/overview.md) for production. For guidance on migrating from
RSC to UI, see our [migration guide](../../ai-sdk-rsc/migrating-to-ui.md).

Get a mutable copy of the AI state. You can use this to update the state in the server.

## Import

```
import { getMutableAIState } from "@ai-sdk/rsc"
```

## API Signature

### Parameters

### key?:

string

Returns the value of the specified key in the AI state, if it's an object.

### Returns

The mutable AI state.

### Methods

### update:

(newState: any) => void

Updates the AI state with the new state.

### done:

(newState: any) => void

Updates the AI state with the new state, marks it as finalized and closes the stream.

## Examples

[Learn to persist and restore states AI and UI states in Next.js](/examples/next-app/state-management/save-and-restore-states)
