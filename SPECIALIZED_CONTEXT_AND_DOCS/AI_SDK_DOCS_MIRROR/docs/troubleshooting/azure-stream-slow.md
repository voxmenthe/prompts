# Azure OpenAI Slow To Stream

## Issue

When using OpenAI hosted on Azure, streaming is slow and in big chunks.

## Cause

This is a Microsoft Azure issue. Some users have reported the following solutions:

- **Update Content Filtering Settings**:
  Inside [Azure AI Studio](https://ai.azure.com/), within "Shared resources" > "Content filters", create a new
  content filter and set the "Streaming mode (Preview)" under "Output filter" from "Default"
  to "Asynchronous Filter".

## Solution

You can use the [`smoothStream` transformation](../ai-sdk-core/generating-text.md#smoothing-streams) to stream each word individually.

```tsx
import { smoothStream, streamText } from 'ai';

const result = streamText({
  model,
  prompt,
  experimental_transform: smoothStream(),
});
```
