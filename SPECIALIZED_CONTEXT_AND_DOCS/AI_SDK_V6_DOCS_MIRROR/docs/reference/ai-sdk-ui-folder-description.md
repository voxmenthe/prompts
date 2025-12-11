# AI SDK UI

[AI SDK UI](../ai-sdk-ui-folder-description.md) is designed to help you build interactive chat, completion, and assistant applications with ease.
It is framework-agnostic toolkit, streamlining the integration of advanced AI functionalities into your applications.

AI SDK UI contains the following hooks:

- [useChat Use a hook to interact with language models in a chat interface.](ai-sdk-ui/use-chat.md)
- [useCompletion Use a hook to interact with language models in a completion interface.](ai-sdk-ui/use-completion.md)
- [useObject Use a hook for consuming a streamed JSON objects.](ai-sdk-ui/use-object.md)
- [convertToModelMessages Convert useChat messages to ModelMessages for AI functions.](ai-sdk-ui/convert-to-model-messages.md)
- [pruneMessages Prunes model messages from a list of model messages.](ai-sdk-ui/prune-messages.md)
- [createUIMessageStream Create a UI message stream to stream additional data to the client.](ai-sdk-ui/create-ui-message-stream.md)
- [createUIMessageStreamResponse Create a response object to stream UI messages to the client.](ai-sdk-ui/create-ui-message-stream-response.md)
- [pipeUIMessageStreamToResponse Pipe a UI message stream to a Node.js ServerResponse object.](ai-sdk-ui/pipe-ui-message-stream-to-response.md)
- [readUIMessageStream Transform a stream of UIMessageChunk objects into an AsyncIterableStream of UIMessage objects.](ai-sdk-ui/read-ui-message-stream.md)

## UI Framework Support

AI SDK UI supports the following frameworks: [React](https://react.dev/), [Svelte](https://svelte.dev/), [Vue.js](https://vuejs.org/),
[Angular](https://angular.dev/), and [SolidJS](https://www.solidjs.com/).

Here is a comparison of the supported functions across these frameworks:

|  | [useChat](ai-sdk-ui/use-chat.md) | [useCompletion](ai-sdk-ui/use-completion.md) | [useObject](ai-sdk-ui/use-object.md) |
| --- | --- | --- | --- |
| React `@ai-sdk/react` |  |  |  |
| Vue.js `@ai-sdk/vue` |  |  |  |
| Svelte `@ai-sdk/svelte` | Chat | Completion | StructuredObject |
| Angular `@ai-sdk/angular` | Chat | Completion | StructuredObject |
| [SolidJS](https://github.com/kodehort/ai-sdk-solid) (community) |  |  |  |

[Contributions](https://github.com/vercel/ai/blob/main/CONTRIBUTING.md) are
welcome to implement missing features for non-React frameworks.
