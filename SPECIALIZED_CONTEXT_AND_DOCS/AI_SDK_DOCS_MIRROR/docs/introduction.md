# AI SDK

The AI SDK is the TypeScript toolkit designed to help developers build AI-powered applications and agents with React, Next.js, Vue, Svelte, Node.js, and more.

## Why use the AI SDK?

Integrating large language models (LLMs) into applications is complicated and heavily dependent on the specific model provider you use.

The AI SDK standardizes integrating artificial intelligence (AI) models across [supported providers](foundations/providers-and-models.md). This enables developers to focus on building great AI applications, not waste time on technical details.

For example, here’s how you can generate text with various models using the AI SDK:

import { generateText } from "ai"

import { xai } from "@ai-sdk/xai"

const { text } = await generateText({

model: xai("grok-4"),

prompt: "What is love?"

})

Love is a universal emotion that is characterized by feelings of affection, attachment, and warmth towards someone or something. It is a complex and multifaceted experience that can take many different forms, including romantic love, familial love, platonic love, and self-love.

The AI SDK has two main libraries:

- **[AI SDK Core](ai-sdk-core-folder-description.md):** A unified API for generating text, structured objects, tool calls, and building agents with LLMs.
- **[AI SDK UI](ai-sdk-ui-folder-description.md):** A set of framework-agnostic hooks for quickly building chat and generative user interface.

## Model Providers

The AI SDK supports [multiple model providers](/providers).

- [xAI Grok Image Input Image Generation Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/xai)
- [OpenAI Image Input Image Generation Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/openai)
- [Azure Image Input Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/azure)
- [Anthropic Image Input Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/anthropic)
- [Amazon Bedrock Image Input Image Generation Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/amazon-bedrock)
- [Groq Image Input Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/groq)
- [Fal AI Image Generation](/providers/ai-sdk-providers/fal)
- [DeepInfra Image Input Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/deepinfra)
- [Google Generative AI Image Input Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/google-generative-ai)
- [Google Vertex AI Image Input Image Generation Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/google-vertex)
- [Mistral Image Input Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/mistral)
- [Together.ai Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/togetherai)
- [Cohere Tool Usage Tool Streaming](/providers/ai-sdk-providers/cohere)
- [Fireworks Image Generation Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/fireworks)
- [DeepSeek Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/deepseek)
- [Cerebras Object Generation Tool Usage Tool Streaming](/providers/ai-sdk-providers/cerebras)
- [Perplexity](/providers/ai-sdk-providers/perplexity)
- [Luma AI Image Generation](/providers/ai-sdk-providers/luma)
- [Baseten Object Generation Tool Usage](/providers/ai-sdk-providers/baseten)

## Templates

We've built some [templates](https://vercel.com/templates?type=ai) that include AI SDK integrations for different use cases, providers, and frameworks. You can use these templates to get started with your AI-powered application.

### Starter Kits

- [Chatbot Starter Template Uses the AI SDK and Next.js. Features persistence, multi-modal chat, and more.](https://vercel.com/templates/next.js/nextjs-ai-chatbot)
- [Internal Knowledge Base (RAG) Uses AI SDK Language Model Middleware for RAG and enforcing guardrails.](https://vercel.com/templates/next.js/ai-sdk-internal-knowledge-base)
- [Multi-Modal Chat Uses Next.js and AI SDK useChat hook for multi-modal message chat interface.](https://vercel.com/templates/next.js/multi-modal-chatbot)
- [Semantic Image Search An AI semantic image search app template built with Next.js, AI SDK, and Postgres.](https://vercel.com/templates/next.js/semantic-image-search)
- [Natural Language PostgreSQL Query PostgreSQL using natural language with AI SDK and GPT-4o.](https://vercel.com/templates/next.js/natural-language-postgres)

### Feature Exploration

- [Feature Flags Example AI SDK with Next.js, Feature Flags, and Edge Config for dynamic model switching.](https://vercel.com/templates/next.js/ai-sdk-feature-flags-edge-config)
- [Chatbot with Telemetry AI SDK chatbot with OpenTelemetry support.](https://vercel.com/templates/next.js/ai-chatbot-telemetry)
- [Structured Object Streaming Uses AI SDK useObject hook to stream structured object generation.](https://vercel.com/templates/next.js/use-object)
- [Multi-Step Tools Uses AI SDK streamText function to handle multiple tool steps automatically.](https://vercel.com/templates/next.js/ai-sdk-roundtrips)

### Frameworks

- [Next.js OpenAI Starter Uses OpenAI GPT-4, AI SDK, and Next.js.](https://github.com/vercel/ai/tree/main/examples/next-openai)
- [Nuxt OpenAI Starter Uses OpenAI GPT-4, AI SDK, and Nuxt.js.](https://github.com/vercel/ai/tree/main/examples/nuxt-openai)
- [SvelteKit OpenAI Starter Uses OpenAI GPT-4, AI SDK, and SvelteKit.](https://github.com/vercel/ai/tree/main/examples/sveltekit-openai)
- [Solid OpenAI Starter Uses OpenAI GPT-4, AI SDK, and Solid.](https://github.com/vercel/ai/tree/main/examples/solidstart-openai)

### Generative UI

- [Gemini Chatbot Uses Google Gemini, AI SDK, and Next.js.](https://vercel.com/templates/next.js/gemini-ai-chatbot)
- [Generative UI with RSC (experimental) Uses Next.js, AI SDK, and streamUI to create generative UIs with React Server Components.](https://vercel.com/templates/next.js/rsc-genui)

### Security

- [Bot Protection Uses Kasada, OpenAI GPT-4, AI SDK, and Next.js.](https://vercel.com/templates/next.js/advanced-ai-bot-protection)
- [Rate Limiting Uses Vercel KV, OpenAI GPT-4, AI SDK, and Next.js.](https://github.com/vercel/ai/tree/main/examples/next-openai-upstash-rate-limits)

## Join our Community

If you have questions about anything related to the AI SDK, you're always welcome to ask our community on [the Vercel Community](https://community.vercel.com/c/ai-sdk/62).

## `llms.txt` (for Cursor, Windsurf, Copilot, Claude etc.)

You can access the entire AI SDK documentation in Markdown format at [ai-sdk.dev/llms.txt](/llms.txt). This can be used to ask any LLM (assuming it has a big enough context window) questions about the AI SDK based on the most up-to-date documentation.

### Example Usage

For instance, to prompt an LLM with questions about the AI SDK:

1. Copy the documentation contents from [ai-sdk.dev/llms.txt](/llms.txt)
2. Use the following prompt format:

```prompt
Documentation:
{paste documentation here}
---
Based on the above documentation, answer the following:
{your question}
```
