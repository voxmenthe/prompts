# Migrate AI SDK 3.4 to 4.0

Check out the [AI SDK 4.0 release blog
post](https://vercel.com/blog/ai-sdk-4-0) for more information about the
release.

## Recommended Migration Process

1. Backup your project. If you use a versioning control system, make sure all previous versions are committed.
2. [Migrate to AI SDK 3.4](../troubleshooting/migration-guide/migration-guide-3-4.md).
3. Upgrade to AI SDK 4.0.
4. Automatically migrate your code using [codemods](#codemods).
   > If you don't want to use codemods, we recommend resolving all deprecation warnings before upgrading to AI SDK 4.0.
5. Follow the breaking changes guide below.
6. Verify your project is working as expected.
7. Commit your changes.

## AI SDK 4.0 package versions

You need to update the following packages to the following versions in your `package.json` file(s):

- `ai` package: `4.0.*`
- `ai-sdk@provider-utils` package: `2.0.*`
- `ai-sdk/*` packages: `1.0.*` (other `@ai-sdk` packages)

## Codemods

The AI SDK provides Codemod transformations to help upgrade your codebase when a
feature is deprecated, removed, or otherwise changed.

Codemods are transformations that run on your codebase programmatically. They
allow you to easily apply many changes without having to manually go through
every file.

Codemods are intended as a tool to help you with the upgrade process. They may
not cover all of the changes you need to make. You may need to make additional
changes manually.

You can run all codemods provided as part of the 4.0 upgrade process by running
the following command from the root of your project:

```sh
npx @ai-sdk/codemod upgrade
```

To run only the v4 codemods:

```sh
npx @ai-sdk/codemod v4
```

Individual codemods can be run by specifying the name of the codemod:

```sh
npx @ai-sdk/codemod <codemod-name> <path>
```

For example, to run a specific v4 codemod:

```sh
npx @ai-sdk/codemod v4/replace-baseurl src/
```

See also the [table of codemods](#codemod-table). In addition, the latest set of
codemods can be found in the
[`@ai-sdk/codemod`](https://github.com/vercel/ai/tree/main/packages/codemod/src/codemods)
repository.

## Provider Changes

### Removed `baseUrl` option

The `baseUrl` option has been removed from all providers. Please use the `baseURL` option instead.

```ts
const perplexity = createOpenAI({
  // ...
  baseUrl: 'https://api.perplexity.ai/',
});
```
```ts
const perplexity = createOpenAI({
  // ...
  baseURL: 'https://api.perplexity.ai/',
});
```

### Anthropic Provider

#### Removed `Anthropic` facade

The `Anthropic` facade has been removed from the Anthropic provider.
Please use the `anthropic` object or the `createAnthropic` function instead.

```ts
const anthropic = new Anthropic({
  // ...
});
```
```ts
const anthropic = createAnthropic({
  // ...
});
```

#### Removed `topK` setting

There is no codemod available for this change. Please review and update your
code manually.

The model specific `topK` setting has been removed from the Anthropic provider.
You can use the standard `topK` setting instead.

```ts
const result = await generateText({
  model: anthropic('claude-3-5-sonnet-latest', {
    topK: 0.5,
  }),
});
```
```ts
const result = await generateText({
  model: anthropic('claude-3-5-sonnet-latest'),
  topK: 0.5,
});
```

### Google Generative AI Provider

#### Removed `Google` facade

The `Google` facade has been removed from the Google Generative AI provider.
Please use the `google` object or the `createGoogleGenerativeAI` function instead.

```ts
const google = new Google({
  // ...
});
```
```ts
const google = createGoogleGenerativeAI({
  // ...
});
```

#### Removed `topK` setting

There is no codemod available for this change. Please review and update your
code manually.

The model-specific `topK` setting has been removed from the Google Generative AI provider.
You can use the standard `topK` setting instead.

```ts
const result = await generateText({
  model: google('gemini-1.5-flash', {
    topK: 0.5,
  }),
});
```
```ts
const result = await generateText({
  model: google('gemini-1.5-flash'),
  topK: 0.5,
});
```

### Google Vertex Provider

#### Removed `topK` setting

There is no codemod available for this change. Please review and update your
code manually.

The model-specific `topK` setting has been removed from the Google Vertex provider.
You can use the standard `topK` setting instead.

```ts
const result = await generateText({
  model: vertex('gemini-1.5-flash', {
    topK: 0.5,
  }),
});
```
```ts
const result = await generateText({
  model: vertex('gemini-1.5-flash'),
  topK: 0.5,
});
```

### Mistral Provider

#### Removed `Mistral` facade

The `Mistral` facade has been removed from the Mistral provider.
Please use the `mistral` object or the `createMistral` function instead.

```ts
const mistral = new Mistral({
  // ...
});
```
```ts
const mistral = createMistral({
  // ...
});
```

### OpenAI Provider

#### Removed `OpenAI` facade

The `OpenAI` facade has been removed from the OpenAI provider.
Please use the `openai` object or the `createOpenAI` function instead.

```ts
const openai = new OpenAI({
  // ...
});
```
```ts
const openai = createOpenAI({
  // ...
});
```

### LangChain Adapter

#### Removed `toAIStream`

The `toAIStream` function has been removed from the LangChain adapter.
Please use the `toDataStream` function instead.

```ts
LangChainAdapter.toAIStream(stream);
```
```ts
LangChainAdapter.toDataStream(stream);
```

## AI SDK Core Changes

### `streamText` returns immediately

Instead of returning a Promise, the `streamText` function now returns immediately.
It is not necessary to await the result of `streamText`.

```ts
const result = await streamText({
  // ...
});
```
```ts
const result = streamText({
  // ...
});
```

### `streamObject` returns immediately

Instead of returning a Promise, the `streamObject` function now returns immediately.
It is not necessary to await the result of `streamObject`.

```ts
const result = await streamObject({
  // ...
});
```
```ts
const result = streamObject({
  // ...
});
```

### Remove roundtrips

The `maxToolRoundtrips` and `maxAutomaticRoundtrips` options have been removed from the `generateText` and `streamText` functions.
Please use the `maxSteps` option instead.

The `roundtrips` property has been removed from the `GenerateTextResult` type.
Please use the `steps` property instead.

```ts
const { text, roundtrips } = await generateText({
  maxToolRoundtrips: 1, // or maxAutomaticRoundtrips
  // ...
});
```
```ts
const { text, steps } = await generateText({
  maxSteps: 2,
  // ...
});
```

### Removed `nanoid` export

The `nanoid` export has been removed. Please use [`generateId`](../reference/ai-sdk-core/generate-id.md) instead.

```ts
import { nanoid } from 'ai';
```
```ts
import { generateId } from 'ai';
```

### Increased default size of generated IDs

There is no codemod available for this change. Please review and update your
code manually.

The [`generateId`](../reference/ai-sdk-core/generate-id.md) function now
generates 16-character IDs. The previous default was 7 characters.

This might e.g. require updating your database schema if you limit the length of
IDs.

```ts
import { generateId } from 'ai';

const id = generateId(); // now 16 characters
```

### Removed `ExperimentalMessage` types

The following types have been removed:

- `ExperimentalMessage` (use `ModelMessage` instead)
- `ExperimentalUserMessage` (use `CoreUserMessage` instead)
- `ExperimentalAssistantMessage` (use `CoreAssistantMessage` instead)
- `ExperimentalToolMessage` (use `CoreToolMessage` instead)

```ts
import {
  ExperimentalMessage,
  ExperimentalUserMessage,
  ExperimentalAssistantMessage,
  ExperimentalToolMessage,
} from 'ai';
```
```ts
import {
  ModelMessage,
  CoreUserMessage,
  CoreAssistantMessage,
  CoreToolMessage,
} from 'ai';
```

### Removed `ExperimentalTool` type

The `ExperimentalTool` type has been removed. Please use the `CoreTool` type instead.

```ts
import { ExperimentalTool } from 'ai';
```
```ts
import { CoreTool } from 'ai';
```

### Removed experimental AI function exports

The following exports have been removed:

- `experimental_generateText` (use `generateText` instead)
- `experimental_streamText` (use `streamText` instead)
- `experimental_generateObject` (use `generateObject` instead)
- `experimental_streamObject` (use `streamObject` instead)

```ts
import {
  experimental_generateText,
  experimental_streamText,
  experimental_generateObject,
  experimental_streamObject,
} from 'ai';
```
```ts
import { generateText, streamText, generateObject, streamObject } from 'ai';
```

### Removed AI-stream related methods from `streamText`

The following methods have been removed from the `streamText` result:

- `toAIStream`
- `pipeAIStreamToResponse`
- `toAIStreamResponse`

Use the `toDataStream`, `pipeDataStreamToResponse`, and `toDataStreamResponse` functions instead.

```ts
const result = await streamText({
  // ...
});

result.toAIStream();
result.pipeAIStreamToResponse(response);
result.toAIStreamResponse();
```
```ts
const result = streamText({
  // ...
});

result.toDataStream();
result.pipeDataStreamToResponse(response);
result.toUIMessageStreamResponse();
```

### Renamed "formatStreamPart" to "formatDataStreamPart"

The `formatStreamPart` function has been renamed to `formatDataStreamPart`.

```ts
formatStreamPart('text', 'Hello, world!');
```
```ts
formatDataStreamPart('text', 'Hello, world!');
```

### Renamed "parseStreamPart" to "parseDataStreamPart"

The `parseStreamPart` function has been renamed to `parseDataStreamPart`.

```ts
const part = parseStreamPart(line);
```
```ts
const part = parseDataStreamPart(line);
```

### Renamed `TokenUsage`, `CompletionTokenUsage` and `EmbeddingTokenUsage` types

The `TokenUsage`, `CompletionTokenUsage` and `EmbeddingTokenUsage` types have
been renamed to `LanguageModelUsage` (for the first two) and
`EmbeddingModelUsage` (for the last).

```ts
import { TokenUsage, CompletionTokenUsage, EmbeddingTokenUsage } from 'ai';
```
```ts
import { LanguageModelUsage, EmbeddingModelUsage } from 'ai';
```

### Removed deprecated telemetry data

There is no codemod available for this change. Please review and update your
code manually.

The following telemetry data values have been removed:

- `ai.finishReason` (now in `ai.response.finishReason`)
- `ai.result.object` (now in `ai.response.object`)
- `ai.result.text` (now in `ai.response.text`)
- `ai.result.toolCalls` (now in `ai.response.toolCalls`)
- `ai.stream.msToFirstChunk` (now in `ai.response.msToFirstChunk`)

This change will apply to observability providers and any scripts or automation that you use for processing telemetry data.

### Provider Registry

#### Removed experimental_Provider, experimental_ProviderRegistry, and experimental_ModelRegistry

The `experimental_Provider` interface, `experimental_ProviderRegistry` interface, and `experimental_ModelRegistry` interface have been removed.
Please use the `Provider` interface instead.

```ts
import { experimental_Provider, experimental_ProviderRegistry } from 'ai';
```
```ts
import { Provider } from 'ai';
```

The model registry is not available any more. Please [register
providers](../reference/ai-sdk-core/provider-registry.md#setup) instead.

#### Removed `experimental_​createModelRegistry` function

The `experimental_createModelRegistry` function has been removed.
Please use the `experimental_createProviderRegistry` function instead.

```ts
import { experimental_createModelRegistry } from 'ai';
```
```ts
import { experimental_createProviderRegistry } from 'ai';
```

The model registry is not available any more. Please [register
providers](../reference/ai-sdk-core/provider-registry.md#setup) instead.

### Removed `rawResponse` from results

There is no codemod available for this change. Please review and update your
code manually.

The `rawResponse` property has been removed from the `generateText`, `streamText`, `generateObject`, and `streamObject` results.
You can use the `response` property instead.

```ts
const { text, rawResponse } = await generateText({
  // ...
});
```
```ts
const { text, response } = await generateText({
  // ...
});
```

### Removed `init` option from `pipeDataStreamToResponse` and `toDataStreamResponse`

There is no codemod available for this change. Please review and update your
code manually.

The `init` option has been removed from the `pipeDataStreamToResponse` and `toDataStreamResponse` functions.
You can set the values from `init` directly into the `options` object.

```ts
const result = await streamText({
  // ...
});

result.toUIMessageStreamResponse(response, {
  init: {
    headers: {
      'X-Custom-Header': 'value',
    },
  },
  // ...
});
```
```ts
const result = streamText({
  // ...
});

result.toUIMessageStreamResponse(response, {
  headers: {
    'X-Custom-Header': 'value',
  },
  // ...
});
```

### Removed `responseMessages` from `generateText` and `streamText`

There is no codemod available for this change. Please review and update your
code manually.

The `responseMessages` property has been removed from the `generateText` and `streamText` results.
This includes the `onFinish` callback.
Please use the `response.messages` property instead.

```ts
const { text, responseMessages } = await generateText({
  // ...
});
```
```ts
const { text, response } = await generateText({
  // ...
});

const responseMessages = response.messages;
```

### Removed `experimental_​continuationSteps` option

The `experimental_continuationSteps` option has been removed from the `generateText` function.
Please use the `experimental_continueSteps` option instead.

```ts
const result = await generateText({
  experimental_continuationSteps: true,
  // ...
});
```
```ts
const result = await generateText({
  experimental_continueSteps: true,
  // ...
});
```

### Removed `LanguageModelResponseMetadataWithHeaders` type

The `LanguageModelResponseMetadataWithHeaders` type has been removed.
Please use the `LanguageModelResponseMetadata` type instead.

```ts
import { LanguageModelResponseMetadataWithHeaders } from 'ai';
```
```ts
import { LanguageModelResponseMetadata } from 'ai';
```

#### Changed `streamText` warnings result to Promise

There is no codemod available for this change. Please review and update your
code manually.

The `warnings` property of the `StreamTextResult` type is now a Promise.

```ts
const result = await streamText({
  // ...
});

const warnings = result.warnings;
```
```ts
const result = streamText({
  // ...
});

const warnings = await result.warnings;
```

#### Changed `streamObject` warnings result to Promise

There is no codemod available for this change. Please review and update your
code manually.

The `warnings` property of the `StreamObjectResult` type is now a Promise.

```ts
const result = await streamObject({
  // ...
});

const warnings = result.warnings;
```
```ts
const result = streamObject({
  // ...
});

const warnings = await result.warnings;
```

#### Renamed `simulateReadableStream` `values` to `chunks`

There is no codemod available for this change. Please review and update your
code manually.

The `simulateReadableStream` function from `ai/test` has been renamed to `chunks`.

```ts
import { simulateReadableStream } from 'ai/test';

const stream = simulateReadableStream({
  values: [1, 2, 3],
  chunkDelayInMs: 100,
});
```
```ts
import { simulateReadableStream } from 'ai/test';

const stream = simulateReadableStream({
  chunks: [1, 2, 3],
  chunkDelayInMs: 100,
});
```

## AI SDK RSC Changes

There are no codemods available for the changes in this section. Please review
and update your code manually.

### Removed `render` function

The AI SDK RSC 3.0 `render` function has been removed.
Please use the `streamUI` function instead or [switch to AI SDK UI](../ai-sdk-rsc/migrating-to-ui.md).

```ts
import { render } from '@ai-sdk/rsc';
```
```ts
import { streamUI } from '@ai-sdk/rsc';
```

## AI SDK UI Changes

### Removed Svelte, Vue, and SolidJS exports

This codemod only operates on `.ts` and `.tsx` files. If you have code in
files with other suffixes, please review and update your code manually.

The `ai` package no longer exports Svelte, Vue, and SolidJS UI integrations.
You need to install the `@ai-sdk/svelte`, `@ai-sdk/vue`, and `@ai-sdk/solid` packages directly.

```ts
import { useChat } from 'ai/svelte';
```
```ts
import { useChat } from '@ai-sdk/svelte';
```

### Removed `experimental_StreamData`

The `experimental_StreamData` export has been removed.
Please use the `StreamData` export instead.

```ts
import { experimental_StreamData } from 'ai';
```
```ts
import { StreamData } from 'ai';
```

### `useChat` hook

There are no codemods available for the changes in this section. Please review
and update your code manually.

#### Removed `streamMode` setting

The `streamMode` options has been removed from the `useChat` hook.
Please use the `streamProtocol` parameter instead.

```ts
const { messages } = useChat({
  streamMode: 'text',
  // ...
});
```
```ts
const { messages } = useChat({
  streamProtocol: 'text',
  // ...
});
```

#### Replaced roundtrip setting with `maxSteps`

The following options have been removed from the `useChat` hook:

- `experimental_maxAutomaticRoundtrips`
- `maxAutomaticRoundtrips`
- `maxToolRoundtrips`

Please use the [`maxSteps`](../ai-sdk-core/tools-and-tool-calling.md#multi-step-calls) option instead.
The value of `maxSteps` is equal to roundtrips + 1.

```ts
const { messages } = useChat({
  experimental_maxAutomaticRoundtrips: 2,
  // or maxAutomaticRoundtrips
  // or maxToolRoundtrips
  // ...
});
```
```ts
const { messages } = useChat({
  maxSteps: 3, // 2 roundtrips + 1
  // ...
});
```

#### Removed `options` setting

The `options` parameter in the `useChat` hook has been removed.
Please use the `headers` and `body` parameters instead.

```ts
const { messages } = useChat({
  options: {
    headers: {
      'X-Custom-Header': 'value',
    },
  },
  // ...
});
```
```ts
const { messages } = useChat({
  headers: {
    'X-Custom-Header': 'value',
  },
  // ...
});
```

#### Removed `experimental_addToolResult` method

The `experimental_addToolResult` method has been removed from the `useChat` hook.
Please use the `addToolResult` method instead.

```ts
const { messages, experimental_addToolResult } = useChat({
  // ...
});
```
```ts
const { messages, addToolResult } = useChat({
  // ...
});
```

#### Changed default value of `keepLastMessageOnError` to true and deprecated the option

The `keepLastMessageOnError` option has been changed to default to `true`.
The option will be removed in the next major release.

```ts
const { messages } = useChat({
  keepLastMessageOnError: true,
  // ...
});
```
```ts
const { messages } = useChat({
  // ...
});
```

### `useCompletion` hook

There are no codemods available for the changes in this section. Please review
and update your code manually.

#### Removed `streamMode` setting

The `streamMode` options has been removed from the `useCompletion` hook.
Please use the `streamProtocol` parameter instead.

```ts
const { text } = useCompletion({
  streamMode: 'text',
  // ...
});
```
```ts
const { text } = useCompletion({
  streamProtocol: 'text',
  // ...
});
```

### `useAssistant` hook

#### Removed `experimental_useAssistant` export

The `experimental_useAssistant` export has been removed from the `useAssistant` hook.
Please use the `useAssistant` hook directly instead.

```ts
import { experimental_useAssistant } from '@ai-sdk/react';
```
```ts
import { useAssistant } from '@ai-sdk/react';
```

#### Removed `threadId` and `messageId` from `AssistantResponse`

There is no codemod available for this change. Please review and update your
code manually.

The `threadId` and `messageId` parameters have been removed from the `AssistantResponse` function.
Please use the `threadId` and `messageId` variables from the outer scope instead.

```ts
return AssistantResponse(
  { threadId: myThreadId, messageId: myMessageId },
  async ({ forwardStream, sendDataMessage, threadId, messageId }) => {
    // use threadId and messageId here
  },
);
```
```ts
return AssistantResponse(
  { threadId: myThreadId, messageId: myMessageId },
  async ({ forwardStream, sendDataMessage }) => {
    // use myThreadId and myMessageId here
  },
);
```

#### Removed `experimental_​AssistantResponse` export

There is no codemod available for this change. Please review and update your
code manually.

The `experimental_AssistantResponse` export has been removed.
Please use the `AssistantResponse` function directly instead.

```ts
import { experimental_AssistantResponse } from 'ai';
```
```ts
import { AssistantResponse } from 'ai';
```

### `experimental_useObject` hook

There are no codemods available for the changes in this section. Please review
and update your code manually.

The `setInput` helper has been removed from the `experimental_useObject` hook.
Please use the `submit` helper instead.

```ts
const { object, setInput } = useObject({
  // ...
});
```
```ts
const { object, submit } = useObject({
  // ...
});
```

## AI SDK Errors

### Removed `isXXXError` static methods

The `isXXXError` static methods have been removed from AI SDK errors.
Please use the `isInstance` method of the corresponding error class instead.

```ts
import { APICallError } from 'ai';

APICallError.isAPICallError(error);
```
```ts
import { APICallError } from 'ai';

APICallError.isInstance(error);
```

### Removed `toJSON` method

There is no codemod available for this change. Please review and update your
code manually.

The `toJSON` method has been removed from AI SDK errors.

## AI SDK 2.x Legacy Changes

There are no codemods available for the changes in this section. Please review
and update your code manually.

### Removed 2.x legacy providers

Legacy providers from AI SDK 2.x have been removed. Please use the new [AI SDK provider architecture](../foundations/providers-and-models.md) instead.

#### Removed 2.x legacy function and tool calling

The legacy `function_call` and `tools` options have been removed from `useChat` and `Message`.
The `name` property from the `Message` type has been removed.
Please use the [AI SDK Core tool calling](../ai-sdk-core/tools-and-tool-calling.md) instead.

### Removed 2.x prompt helpers

Prompt helpers for constructing message prompts are no longer needed with the AI SDK provider architecture and have been removed.

### Removed 2.x `AIStream`

The `AIStream` function and related exports have been removed.
Please use the [`streamText`](../reference/ai-sdk-core/stream-text.md) function and its `toDataStream()` method instead.

### Removed 2.x `StreamingTextResponse`

The `StreamingTextResponse` function has been removed.
Please use the [`streamText`](../reference/ai-sdk-core/stream-text.md) function and its `toDataStreamResponse()` method instead.

### Removed 2.x `streamToResponse`

The `streamToResponse` function has been removed.
Please use the [`streamText`](../reference/ai-sdk-core/stream-text.md) function and its `pipeDataStreamToResponse()` method instead.

### Removed 2.x RSC `Tokens` streaming

The legacy `Tokens` RSC streaming from 2.x has been removed.
`Tokens` were implemented prior to AI SDK RSC and are no longer needed.

## Codemod Table

The following table lists codemod availability for the AI SDK 4.0 upgrade
process. Note the codemod `upgrade` command will run all of them for you. This
list is provided to give visibility into which migrations have some automation.
It can also be helpful to find the codemod names if you'd like to run a subset
of codemods. For more, see the [Codemods](#codemods) section.

| Change | Codemod |
| --- | --- |
| **Provider Changes** |  |
| Removed baseUrl option | `v4/replace-baseurl` |
| **Anthropic Provider** |  |
| Removed Anthropic facade | `v4/remove-anthropic-facade` |
| Removed topK setting | *N/A* |
| **Google Generative AI Provider** |  |
| Removed Google facade | `v4/remove-google-facade` |
| Removed topK setting | *N/A* |
| **Google Vertex Provider** |  |
| Removed topK setting | *N/A* |
| **Mistral Provider** |  |
| Removed Mistral facade | `v4/remove-mistral-facade` |
| **OpenAI Provider** |  |
| Removed OpenAI facade | `v4/remove-openai-facade` |
| **LangChain Adapter** |  |
| Removed toAIStream | `v4/replace-langchain-toaistream` |
| **AI SDK Core Changes** |  |
| streamText returns immediately | `v4/remove-await-streamtext` |
| streamObject returns immediately | `v4/remove-await-streamobject` |
| Remove roundtrips | `v4/replace-roundtrips-with-maxsteps` |
| Removed nanoid export | `v4/replace-nanoid` |
| Increased default size of generated IDs | *N/A* |
| Removed ExperimentalMessage types | `v4/remove-experimental-message-types` |
| Removed ExperimentalTool type | `v4/remove-experimental-tool` |
| Removed experimental AI function exports | `v4/remove-experimental-ai-fn-exports` |
| Removed AI-stream related methods from streamText | `v4/remove-ai-stream-methods-from-stream-text-result` |
| Renamed "formatStreamPart" to "formatDataStreamPart" | `v4/rename-format-stream-part` |
| Renamed "parseStreamPart" to "parseDataStreamPart" | `v4/rename-parse-stream-part` |
| Renamed TokenUsage, CompletionTokenUsage and EmbeddingTokenUsage types | `v4/replace-token-usage-types` |
| Removed deprecated telemetry data | *N/A* |
| **Provider Registry** |  |
| → Removed experimental_Provider, experimental_ProviderRegistry, and experimental_ModelRegistry | `v4/remove-deprecated-provider-registry-exports` |
| → Removed experimental_createModelRegistry function | *N/A* |
| Removed rawResponse from results | *N/A* |
| Removed init option from pipeDataStreamToResponse and toDataStreamResponse | *N/A* |
| Removed responseMessages from generateText and streamText | *N/A* |
| Removed experimental_continuationSteps option | `v4/replace-continuation-steps` |
| Removed LanguageModelResponseMetadataWithHeaders type | `v4/remove-metadata-with-headers` |
| Changed streamText warnings result to Promise | *N/A* |
| Changed streamObject warnings result to Promise | *N/A* |
| Renamed simulateReadableStream values to chunks | *N/A* |
| **AI SDK RSC Changes** |  |
| Removed render function | *N/A* |
| **AI SDK UI Changes** |  |
| Removed Svelte, Vue, and SolidJS exports | `v4/rewrite-framework-imports` |
| Removed experimental_StreamData | `v4/remove-experimental-streamdata` |
| **useChat hook** |  |
| Removed streamMode setting | *N/A* |
| Replaced roundtrip setting with maxSteps | `v4/replace-roundtrips-with-maxsteps` |
| Removed options setting | *N/A* |
| Removed experimental_addToolResult method | *N/A* |
| Changed default value of keepLastMessageOnError to true and deprecated the option | *N/A* |
| **useCompletion hook** |  |
| Removed streamMode setting | *N/A* |
| **useAssistant hook** |  |
| Removed experimental_useAssistant export | `v4/remove-experimental-useassistant` |
| Removed threadId and messageId from AssistantResponse | *N/A* |
| Removed experimental_AssistantResponse export | *N/A* |
| **experimental_useObject hook** |  |
| Removed setInput helper | *N/A* |
| **AI SDK Errors** |  |
| Removed isXXXError static methods | `v4/remove-isxxxerror` |
| Removed toJSON method | *N/A* |
| **AI SDK 2.x Legacy Changes** |  |
| Removed 2.x legacy providers | *N/A* |
| Removed 2.x legacy function and tool calling | *N/A* |
| Removed 2.x prompt helpers | *N/A* |
| Removed 2.x AIStream | *N/A* |
| Removed 2.x StreamingTextResponse | *N/A* |
| Removed 2.x streamToResponse | *N/A* |
| Removed 2.x RSC Tokens streaming | *N/A* |
