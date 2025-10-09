LiveKit Docs › Models › Large language models (LLM) › Plugins › DeepSeek

---

# DeepSeek LLM plugin guide

> How to use DeepSeek models with LiveKit Agents.

Available in:
- [x] Node.js
- [x] Python

## Overview

This plugin allows you to use the [DeepSeek API](https://platform.deepseek.com/) as an LLM provider for your voice agents.

> 💡 **LiveKit Inference**
> 
> DeepSeek models are also available in various providers in LiveKit Inference, with billing and integration handled automatically. See [the docs](https://docs.livekit.io/agents/models/llm.md) for more information.

## Usage

Use the OpenAI plugin's `with_deepseek` method to set the default agent session LLM to DeepSeek:

**Python**:

```shell
pip install "livekit-agents[openai]~=1.2"

```

---

**Node.js**:

```shell
pnpm add @livekit/agents-plugin-openai@1.x

```

Set the following environment variable in your `.env` file:

```shell
DEEPSEEK_API_KEY=<your-deepseek-api-key>

```

**Python**:

```python
from livekit.plugins import openai

session = AgentSession(
    llm=openai.LLM.with_deepseek(
        model="deepseek-chat", # this is DeepSeek-V3
    ),
)

```

---

**Node.js**:

```typescript
import * as openai from '@livekit/agents-plugin-openai';

const session = new voice.AgentSession({
   llm: openai.LLM.withDeepSeek({
    model: "deepseek-chat",  // this is DeepSeek-V3
   })
});

```

## Parameters

This section describes some of the available parameters. For a complete reference of all available parameters, see the plugin reference links in the [Additional resources](#additional-resources) section.

- **`model`** _(str | DeepSeekChatModels)_ (optional) - Default: `deepseek-chat`: DeepSeek model to use. See [models and pricing](https://api-docs.deepseek.com/quick_start/pricing) for a complete list.

- **`temperature`** _(float)_ (optional) - Default: `1.0`: Controls the randomness of the model's output. Higher values, for example 0.8, make the output more random, while lower values, for example 0.2, make it more focused and deterministic.

Valid values are between `0` and `2`.

- **`parallel_tool_calls`** _(bool)_ (optional): Controls whether the model can make multiple tool calls in parallel. When enabled, the model can make multiple tool calls simultaneously, which can improve performance for complex tasks.

- **`tool_choice`** _(ToolChoice | Literal['auto', 'required', 'none'])_ (optional) - Default: `auto`: Controls how the model uses tools. Set to 'auto' to let the model decide, 'required' to force tool usage, or 'none' to disable tool usage.

## Additional resources

The following links provide more information about the DeepSeek LLM integration.

- **[DeepSeek docs](https://platform.deepseek.com/docs)**: DeepSeek API documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and DeepSeek.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/llm/plugins/deepseek.md](https://docs.livekit.io/agents/models/llm/plugins/deepseek.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).