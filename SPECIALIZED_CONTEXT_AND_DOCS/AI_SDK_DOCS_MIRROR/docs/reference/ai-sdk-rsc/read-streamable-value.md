# `readStreamableValue`

AI SDK RSC is currently experimental. We recommend using [AI SDK
UI](../../ai-sdk-ui/overview.md) for production. For guidance on migrating from
RSC to UI, see our [migration guide](../../ai-sdk-rsc/migrating-to-ui.md).

It is a function that helps you read the streamable value from the client that was originally created using [`createStreamableValue`](create-streamable-value.md) on the server.

## Import

```
import { readStreamableValue } from "@ai-sdk/rsc"
```

## Example

```ts
async function generate() {
  'use server';
  const streamable = createStreamableValue();

  streamable.update(1);
  streamable.update(2);
  streamable.done(3);

  return streamable.value;
}
```
```tsx
import { readStreamableValue } from '@ai-sdk/rsc';

export default function Page() {
  const [generation, setGeneration] = useState('');

  return (
    <div>
      <button
        onClick={async () => {
          const stream = await generate();

          for await (const delta of readStreamableValue(stream)) {
            setGeneration(generation => generation + delta);
          }
        }}
      >
        Generate
      </button>
    </div>
  );
}
```

## API Signature

### Parameters

### stream:

StreamableValue

The streamable value to read from.

### Returns

It returns an async iterator that contains the values emitted by the streamable value.
