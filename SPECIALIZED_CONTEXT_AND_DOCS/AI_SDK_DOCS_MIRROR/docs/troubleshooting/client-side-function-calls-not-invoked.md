# Client-Side Function Calls Not Invoked

## Issue

I upgraded the AI SDK to v3.0.20 or newer. I am using [`OpenAIStream`](../reference/stream-helpers/openai-stream.md). Client-side function calls are no longer invoked.

## Solution

You will need to add a stub for `experimental_onFunctionCall` to [`OpenAIStream`](../reference/stream-helpers/openai-stream.md) to enable the correct forwarding of the function calls to the client.

```tsx
const stream = OpenAIStream(response, {
  async experimental_onFunctionCall() {
    return;
  },
});
```
