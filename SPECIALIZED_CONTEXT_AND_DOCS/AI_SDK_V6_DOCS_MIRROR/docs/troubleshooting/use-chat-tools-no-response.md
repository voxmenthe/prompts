# `useChat` No Response

## Issue

I am using [`useChat`](../reference/ai-sdk-ui/use-chat.md).
When I log the incoming messages on the server, I can see the tool call and the tool result, but the model does not respond with anything.

## Solution

To resolve this issue, convert the incoming messages to the `ModelMessage` format using the [`convertToModelMessages`](../reference/ai-sdk-ui/convert-to-model-messages.md) function.

```tsx
import { openai } from '@ai-sdk/openai';
import { convertToModelMessages, streamText } from 'ai';

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: openai('gpt-4o'),
    messages: convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse();
}
```
