# Navigating the Library

The AI SDK is a powerful toolkit for building AI applications. This page will help you pick the right tools for your requirements.

Let’s start with a quick overview of the AI SDK, which is comprised of three parts:

- **[AI SDK Core](../ai-sdk-core/overview.md):** A unified, provider agnostic API for generating text, structured objects, and tool calls with LLMs.
- **[AI SDK UI](../ai-sdk-ui/overview.md):** A set of framework-agnostic hooks for building chat and generative user interfaces.
- [AI SDK RSC](../ai-sdk-rsc/overview.md): Stream generative user interfaces with React Server Components (RSC). Development is currently experimental and we recommend using [AI SDK UI](../ai-sdk-ui/overview.md).

## Choosing the Right Tool for Your Environment

When deciding which part of the AI SDK to use, your first consideration should be the environment and existing stack you are working with. Different components of the SDK are tailored to specific frameworks and environments.

| Library | Purpose | Environment Compatibility |
| --- | --- | --- |
| [AI SDK Core](../ai-sdk-core/overview.md) | Call any LLM with unified API (e.g. [generateText](../reference/ai-sdk-core/generate-text.md) and [generateObject](../reference/ai-sdk-core/generate-object.md)) | Any JS environment (e.g. Node.js, Deno, Browser) |
| [AI SDK UI](../ai-sdk-ui/overview.md) | Build streaming chat and generative UIs (e.g. [useChat](../reference/ai-sdk-ui/use-chat.md)) | React & Next.js, Vue & Nuxt, Svelte & SvelteKit |
| [AI SDK RSC](../ai-sdk-rsc/overview.md) | Stream generative UIs from Server to Client (e.g. [streamUI](../reference/ai-sdk-rsc/stream-ui.md)). Development is currently experimental and we recommend using [AI SDK UI](../ai-sdk-ui/overview.md). | Any framework that supports React Server Components (e.g. Next.js) |

## Environment Compatibility

These tools have been designed to work seamlessly with each other and it's likely that you will be using them together. Let's look at how you could decide which libraries to use based on your application environment, existing stack, and requirements.

The following table outlines AI SDK compatibility based on environment:

| Environment | [AI SDK Core](../ai-sdk-core/overview.md) | [AI SDK UI](../ai-sdk-ui/overview.md) | [AI SDK RSC](../ai-sdk-rsc/overview.md) |
| --- | --- | --- | --- |
| None / Node.js / Deno |  |  |  |
| Vue / Nuxt |  |  |  |
| Svelte / SvelteKit |  |  |  |
| Next.js Pages Router |  |  |  |
| Next.js App Router |  |  |  |

## When to use AI SDK UI

AI SDK UI provides a set of framework-agnostic hooks for quickly building **production-ready AI-native applications**. It offers:

- Full support for streaming chat and client-side generative UI
- Utilities for handling common AI interaction patterns (i.e. chat, completion, assistant)
- Production-tested reliability and performance
- Compatibility across popular frameworks

## AI SDK UI Framework Compatibility

AI SDK UI supports the following frameworks: [React](https://react.dev/), [Svelte](https://svelte.dev/), and [Vue.js](https://vuejs.org/). Here is a comparison of the supported functions across these frameworks:

| Function | React | Svelte | Vue.js |
| --- | --- | --- | --- |
| [useChat](../reference/ai-sdk-ui/use-chat.md) |  |  |  |
| [useChat](../reference/ai-sdk-ui/use-chat.md) tool calling |  |  |  |
| [useCompletion](../reference/ai-sdk-ui/use-completion.md) |  |  |  |
| [useObject](../reference/ai-sdk-ui/use-object.md) |  |  |  |

[Contributions](https://github.com/vercel/ai/blob/main/CONTRIBUTING.md) are
welcome to implement missing features for non-React frameworks.

## When to use AI SDK RSC

AI SDK RSC is currently experimental. We recommend using [AI SDK
UI](../ai-sdk-ui/overview.md) for production. For guidance on migrating from
RSC to UI, see our [migration guide](../ai-sdk-rsc/migrating-to-ui.md).

[React Server Components](https://nextjs.org/docs/app/building-your-application/rendering/server-components)
(RSCs) provide a new approach to building React applications that allow components
to render on the server, fetch data directly, and stream the results to the client,
reducing bundle size and improving performance. They also introduce a new way to
call server-side functions from anywhere in your application called [Server Actions](https://nextjs.org/docs/app/building-your-application/data-fetching/server-actions-and-mutations).

AI SDK RSC provides a number of utilities that allow you to stream values and UI directly from the server to the client. However, **it's important to be aware of current limitations**:

- **Cancellation**: currently, it is not possible to abort a stream using Server Actions. This will be improved in future releases of React and Next.js.
- **Increased Data Transfer**: using [`createStreamableUI`](../reference/ai-sdk-rsc/create-streamable-ui.md) can lead to quadratic data transfer (quadratic to the length of generated text). You can avoid this using  [`createStreamableValue`](../reference/ai-sdk-rsc/create-streamable-value.md)  instead, and rendering the component client-side.
- **Re-mounting Issue During Streaming**: when using `createStreamableUI`, components re-mount on `.done()`, causing [flickering](https://github.com/vercel/ai/issues/2232).

Given these limitations, **we recommend using [AI SDK UI](../ai-sdk-ui/overview.md) for production applications**.
