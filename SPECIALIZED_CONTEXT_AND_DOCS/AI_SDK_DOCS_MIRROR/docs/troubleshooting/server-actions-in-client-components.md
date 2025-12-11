# Server Actions in Client Components

You may use Server Actions in client components, but sometimes you may encounter the following issues.

## Issue

It is not allowed to define inline `"use server"` annotated Server Actions in Client Components.

## Solution

To use Server Actions in a Client Component, you can either:

- Export them from a separate file with `"use server"` at the top.
- Pass them down through props from a Server Component.
- Implement a combination of [`createAI`](../reference/ai-sdk-rsc/create-ai.md) and [`useActions`](../reference/ai-sdk-rsc/use-actions.md) hooks to access them.

Learn more about [Server Actions and Mutations](https://nextjs.org/docs/app/api-reference/functions/server-actions#with-client-components).

```ts
'use server';

import { generateText } from 'ai';

export async function getAnswer(question: string) {
  'use server';

  const { text } = await generateText({
    model: "anthropic/claude-sonnet-4.5",
    prompt: question,
  });

  return { answer: text };
}
```
