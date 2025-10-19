# Streaming Values

AI SDK RSC is currently experimental. We recommend using [AI SDK
UI](../ai-sdk-ui/overview.md) for production. For guidance on migrating from
RSC to UI, see our [migration guide](migrating-to-ui.md).

The RSC API provides several utility functions to allow you to stream values from the server to the client. This is useful when you need more granular control over what you are streaming and how you are streaming it.

These utilities can also be paired with [AI SDK Core](../ai-sdk-core-folder-description.md)
functions like [`streamText`](../reference/ai-sdk-core/stream-text.md) and
[`streamObject`](../reference/ai-sdk-core/stream-object.md) to easily stream
LLM generations from the server to the client.

There are two functions provided by the RSC API that allow you to create streamable values:

- [`createStreamableValue`](../reference/ai-sdk-rsc/create-streamable-value.md) - creates a streamable (serializable) value, with full control over how you create, update, and close the stream.
- [`createStreamableUI`](../reference/ai-sdk-rsc/create-streamable-ui.md) - creates a streamable React component, with full control over how you create, update, and close the stream.

## `createStreamableValue`

The RSC API allows you to stream serializable Javascript values from the server to the client using [`createStreamableValue`](../reference/ai-sdk-rsc/create-streamable-value.md), such as strings, numbers, objects, and arrays.

This is useful when you want to stream:

- Text generations from the language model in real-time.
- Buffer values of image and audio generations from multi-modal models.
- Progress updates from multi-step agent runs.

## Creating a Streamable Value

You can import `createStreamableValue` from `@ai-sdk/rsc` and use it to create a streamable value.

```tsx
'use server';

import { createStreamableValue } from '@ai-sdk/rsc';

export const runThread = async () => {
  const streamableStatus = createStreamableValue('thread.init');

  setTimeout(() => {
    streamableStatus.update('thread.run.create');
    streamableStatus.update('thread.run.update');
    streamableStatus.update('thread.run.end');
    streamableStatus.done('thread.end');
  }, 1000);

  return {
    status: streamableStatus.value,
  };
};
```

## Reading a Streamable Value

You can read streamable values on the client using `readStreamableValue`. It returns an async iterator that yields the value of the streamable as it is updated:

```tsx
import { readStreamableValue } from '@ai-sdk/rsc';
import { runThread } from '@/actions';

export default function Page() {
  return (
    <button
      onClick={async () => {
        const { status } = await runThread();

        for await (const value of readStreamableValue(status)) {
          console.log(value);
        }
      }}
    >
      Ask
    </button>
  );
}
```

Learn how to stream a text generation (with `streamText`) using the Next.js App Router and `createStreamableValue` in this [example](/examples/next-app/basics/streaming-text-generation).

## `createStreamableUI`

`createStreamableUI` creates a stream that holds a React component. Unlike AI SDK Core APIs, this function does not call a large language model. Instead, it provides a primitive that can be used to have granular control over streaming a React component.

## Using `createStreamableUI`

Let's look at how you can use the `createStreamableUI` function with a Server Action.

```tsx
'use server';

import { createStreamableUI } from '@ai-sdk/rsc';

export async function getWeather() {
  const weatherUI = createStreamableUI();

  weatherUI.update(<div style={{ color: 'gray' }}>Loading...</div>);

  setTimeout(() => {
    weatherUI.done(<div>It&apos;s a sunny day!</div>);
  }, 1000);

  return weatherUI.value;
}
```

First, you create a streamable UI with an empty state and then update it with a loading message. After 1 second, you mark the stream as done passing in the actual weather information as its final value. The `.value` property contains the actual UI that can be sent to the client.

## Reading a Streamable UI

On the client side, you can call the `getWeather` Server Action and render the returned UI like any other React component.

```tsx
'use client';

import { useState } from 'react';
import { readStreamableValue } from '@ai-sdk/rsc';
import { getWeather } from '@/actions';

export default function Page() {
  const [weather, setWeather] = useState<React.ReactNode | null>(null);

  return (
    <div>
      <button
        onClick={async () => {
          const weatherUI = await getWeather();
          setWeather(weatherUI);
        }}
      >
        What&apos;s the weather?
      </button>

      {weather}
    </div>
  );
}
```

When the button is clicked, the `getWeather` function is called, and the returned UI is set to the `weather` state and rendered on the page. Users will see the loading message first and then the actual weather information after 1 second.

Learn more about handling multiple streams in a single request in the [Multiple Streamables](../advanced/multiple-streamables.md) guide.

Learn more about handling state for more complex use cases with  [AI/UI State](generative-ui-state.md) .
