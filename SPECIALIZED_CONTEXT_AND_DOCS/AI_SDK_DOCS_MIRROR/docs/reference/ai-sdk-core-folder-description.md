# AI SDK Core

[AI SDK Core](../ai-sdk-core-folder-description.md) is a set of functions that allow you to interact with language models and other AI models.
These functions are designed to be easy-to-use and flexible, allowing you to generate text, structured data,
and embeddings from language models and other AI models.

AI SDK Core contains the following main functions:

- [generateText() Generate text and call tools from a language model.](ai-sdk-core/generate-text.md)
- [streamText() Stream text and call tools from a language model.](ai-sdk-core/stream-text.md)
- [generateObject() Generate structured data from a language model.](ai-sdk-core/generate-object.md)
- [streamObject() Stream structured data from a language model.](ai-sdk-core/stream-object.md)
- [embed() Generate an embedding for a single value using an embedding model.](ai-sdk-core/embed.md)
- [embedMany() Generate embeddings for several values using an embedding model (batch embedding).](ai-sdk-core/embed-many.md)
- [experimental_generateImage() Generate images based on a given prompt using an image model.](ai-sdk-core/generate-image.md)
- [experimental_transcribe() Generate a transcript from an audio file.](ai-sdk-core/transcribe.md)
- [experimental_generateSpeech() Generate speech audio from text.](ai-sdk-core/generate-speech.md)

It also contains the following helper functions:

- [tool() Type inference helper function for tools.](ai-sdk-core/tool.md)
- [experimental_createMCPClient() Creates a client for connecting to MCP servers.](ai-sdk-core/create-mcp-client.md)
- [jsonSchema() Creates AI SDK compatible JSON schema objects.](ai-sdk-core/json-schema.md)
- [zodSchema() Creates AI SDK compatible Zod schema objects.](ai-sdk-core/zod-schema.md)
- [createProviderRegistry() Creates a registry for using models from multiple providers.](ai-sdk-core/provider-registry.md)
- [cosineSimilarity() Calculates the cosine similarity between two vectors, e.g. embeddings.](ai-sdk-core/cosine-similarity.md)
- [simulateReadableStream() Creates a ReadableStream that emits values with configurable delays.](ai-sdk-core/simulate-readable-stream.md)
- [wrapLanguageModel() Wraps a language model with middleware.](ai-sdk-core/wrap-language-model.md)
- [extractReasoningMiddleware() Extracts reasoning from the generated text and exposes it as a `reasoning` property on the result.](ai-sdk-core/extract-reasoning-middleware.md)
- [simulateStreamingMiddleware() Simulates streaming behavior with responses from non-streaming language models.](ai-sdk-core/simulate-streaming-middleware.md)
- [defaultSettingsMiddleware() Applies default settings to a language model.](ai-sdk-core/default-settings-middleware.md)
- [smoothStream() Smooths text streaming output.](ai-sdk-core/smooth-stream.md)
- [generateId() Helper function for generating unique IDs](ai-sdk-core/generate-id.md)
- [createIdGenerator() Creates an ID generator](ai-sdk-core/create-id-generator.md)
