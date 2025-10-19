# Tools

While [large language models (LLMs)](overview.md#large-language-models) have incredible generation capabilities,
they struggle with discrete tasks (e.g. mathematics) and interacting with the outside world (e.g. getting the weather).

Tools are actions that an LLM can invoke.
The results of these actions can be reported back to the LLM to be considered in the next response.

For example, when you ask an LLM for the "weather in London", and there is a weather tool available, it could call a tool
with London as the argument. The tool would then fetch the weather data and return it to the LLM. The LLM can then use this
information in its response.

## What is a tool?

A tool is an object that can be called by the model to perform a specific task.
You can use tools with [`generateText`](../reference/ai-sdk-core/generate-text.md)
and [`streamText`](../reference/ai-sdk-core/stream-text.md) by passing one or more tools to the `tools` parameter.

A tool consists of three properties:

- **`description`**: An optional description of the tool that can influence when the tool is picked.
- **`inputSchema`**: A [Zod schema](tools.md#schema-specification-and-validation-with-zod) or a [JSON schema](../reference/ai-sdk-core/json-schema.md) that defines the input required for the tool to run. The schema is consumed by the LLM, and also used to validate the LLM tool calls.
- **`execute`**: An optional async function that is called with the arguments from the tool call.

`streamUI` uses UI generator tools with a `generate` function that can return
React components.

If the LLM decides to use a tool, it will generate a tool call.
Tools with an `execute` function are run automatically when these calls are generated.
The output of the tool calls are returned using tool result objects.

You can automatically pass tool results back to the LLM
using [multi-step calls](../ai-sdk-core/tools-and-tool-calling.md#multi-step-calls) with `streamText` and `generateText`.

## Schemas

Schemas are used to define the parameters for tools and to validate the [tool calls](../ai-sdk-core/tools-and-tool-calling.md).

The AI SDK supports both raw JSON schemas (using the [`jsonSchema` function](../reference/ai-sdk-core/json-schema.md))
and [Zod](https://zod.dev/) schemas (either directly or using the [`zodSchema` function](../reference/ai-sdk-core/zod-schema.md)).

[Zod](https://zod.dev/) is a popular TypeScript schema validation library.
You can install it with:

```
pnpm add zod
```

You can then specify a Zod schema, for example:

```ts
import z from 'zod';

const recipeSchema = z.object({
  recipe: z.object({
    name: z.string(),
    ingredients: z.array(
      z.object({
        name: z.string(),
        amount: z.string(),
      }),
    ),
    steps: z.array(z.string()),
  }),
});
```

You can also use schemas for structured output generation with
[`generateObject`](../reference/ai-sdk-core/generate-object.md) and
[`streamObject`](../reference/ai-sdk-core/stream-object.md).

## Toolkits

When you work with tools, you typically need a mix of application specific tools and general purpose tools.
There are several providers that offer pre-built tools as **toolkits** that you can use out of the box:

- **[agentic](https://docs.agentic.so/marketplace/ts-sdks/ai-sdk)** - A collection of 20+ tools. Most tools connect to access external APIs such as [Exa](https://exa.ai/) or [E2B](https://e2b.dev/).
- **[browserbase](https://docs.browserbase.com/integrations/vercel/introduction#vercel-ai-integration)** - Browser tool that runs a headless browser
- **[browserless](https://docs.browserless.io/ai-integrations/vercel-ai-sdk)** - Browser automation service with AI integration - self hosted or cloud based
- **[Stripe agent tools](https://docs.stripe.com/agents?framework=vercel)** - Tools for interacting with Stripe.
- **[StackOne ToolSet](https://docs.stackone.com/agents/typescript/frameworks/vercel-ai-sdk)** - Agentic integrations for hundreds of [enterprise SaaS](https://www.stackone.com/integrations)
- **[Toolhouse](https://docs.toolhouse.ai/toolhouse/toolhouse-sdk/using-vercel-ai)** - AI function-calling in 3 lines of code for over 25 different actions.
- **[Agent Tools](https://ai-sdk-agents.vercel.app/?item=introduction)** - A collection of tools for agents.
- **[AI Tool Maker](https://github.com/nihaocami/ai-tool-maker)** - A CLI utility to generate AI SDK tools from OpenAPI specs.
- **[Composio](https://docs.composio.dev/providers/vercel)** - Composio provides 250+ tools like GitHub, Gmail, Salesforce and [more](https://composio.dev/tools).
- **[Interlify](https://www.interlify.com/docs/integrate-with-vercel-ai)** - Convert APIs into tools so that AI can connect to your backend in minutes.
- **[JigsawStack](http://www.jigsawstack.com/docs/integration/vercel)** - JigsawStack provides over 30+ small custom fine tuned models available for specific uses.
- **[AI Tools Registry](https://ai-tools-registry.vercel.app)** - A Shadcn compatible tool definitions and components registry for the AI SDK.
- **[DeepAgent](https://deepagent.amardeep.space/docs/vercel-ai-sdk)** - A powerful suite of 50+ AI tools and integrations, seamlessly connecting with APIs like Tavily, E2B, Airtable and [more](https://deepagent.amardeep.space/docs) to build enterprise-ready AI agents.
- **[Smithery](https://smithery.ai/docs/integrations/vercel_ai_sdk)** - Smithery provides an open marketplace of 6K+ MCPs, including [Browserbase](https://browserbase.com/) and [Exa](https://exa.ai/).

Do you have open source tools or tool libraries that are compatible with the
AI SDK? Please [file a pull request](https://github.com/vercel/ai/pulls) to
add them to this list.

## Learn more

The AI SDK Core [Tool Calling](../ai-sdk-core/tools-and-tool-calling.md)
and [Agents](agents.md) documentation has more information about tools and tool calling.
