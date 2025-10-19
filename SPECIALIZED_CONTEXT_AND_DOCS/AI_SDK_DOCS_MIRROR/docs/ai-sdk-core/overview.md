# AI SDK Core

Large Language Models (LLMs) are advanced programs that can understand, create, and engage with human language on a large scale.
They are trained on vast amounts of written material to recognize patterns in language and predict what might come next in a given piece of text.

AI SDK Core **simplifies working with LLMs by offering a standardized way of integrating them into your app** - so you can focus on building great AI applications for your users, not waste time on technical details.

For example, here’s how you can generate text with various models using the AI SDK:

import { generateText } from "ai"

import { xai } from "@ai-sdk/xai"

const { text } = await generateText({

model: xai("grok-4"),

prompt: "What is love?"

})

Love is a universal emotion that is characterized by feelings of affection, attachment, and warmth towards someone or something. It is a complex and multifaceted experience that can take many different forms, including romantic love, familial love, platonic love, and self-love.

## AI SDK Core Functions

AI SDK Core has various functions designed for [text generation](generating-text.md), [structured data generation](generating-structured-data.md), and [tool usage](tools-and-tool-calling.md).
These functions take a standardized approach to setting up [prompts](prompts.md) and [settings](settings.md), making it easier to work with different models.

- [`generateText`](generating-text.md): Generates text and [tool calls](tools-and-tool-calling.md).
  This function is ideal for non-interactive use cases such as automation tasks where you need to write text (e.g. drafting email or summarizing web pages) and for agents that use tools.
- [`streamText`](generating-text.md): Stream text and tool calls.
  You can use the `streamText` function for interactive use cases such as [chat bots](../ai-sdk-ui/chatbot.md) and [content streaming](../ai-sdk-ui/completion.md).
- [`generateObject`](generating-structured-data.md): Generates a typed, structured object that matches a [Zod](https://zod.dev/) schema.
  You can use this function to force the language model to return structured data, e.g. for information extraction, synthetic data generation, or classification tasks.
- [`streamObject`](generating-structured-data.md): Stream a structured object that matches a Zod schema.
  You can use this function to [stream generated UIs](../ai-sdk-ui/object-generation.md).

## API Reference

Please check out the [AI SDK Core API Reference](../reference/ai-sdk-core-folder-description.md) for more details on each function.
