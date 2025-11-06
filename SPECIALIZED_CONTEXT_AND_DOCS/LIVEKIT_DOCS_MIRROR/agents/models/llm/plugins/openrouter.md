LiveKit docs › Models › Large language models (LLM) › Plugins › OpenRouter

---

# OpenRouter LLM plugin guide

> How to use OpenRouter with LiveKit Agents to access 500+ AI models.

Available in:
- [ ] Node.js
- [x] Python

## Overview

This plugin allows you to use [OpenRouter](https://openrouter.ai/) as an LLM provider for your voice agents. OpenRouter provides access to hundreds of models from multiple providers through a unified API, with automatic fallback support and intelligent routing.

## Usage

Install the OpenAI plugin to add OpenRouter support:

```shell
uv add "livekit-agents[openai]~=1.2"

```

### Authentication

The OpenRouter plugin requires an [OpenRouter API key](https://openrouter.ai/settings/keys).

Set `OPENROUTER_API_KEY` in your `.env` file.

Create an OpenRouter LLM using the `with_openrouter` method:

```python
from livekit.plugins import openai

session = AgentSession(
    llm=openai.LLM.with_openrouter(model="anthropic/claude-sonnet-4.5"),
    # ... tts, stt, vad, turn_detection, etc.
)

```

### Parameters

This section describes some of the available parameters. See the usage examples below and the plugin reference links in the [Additional resources](#additional-resources) section for a complete list of all available parameters.

- **`model`** _(string)_ (optional) - Default: `openrouter/auto`: Model to use. Can be "openrouter/auto" to let OpenRouter choose, or specify a specific model like "anthropic/claude-sonnet-4.5". For a list of available models, see [OpenRouter models](https://openrouter.ai/models).

- **`site_url`** _(string)_ (optional): Your site URL for analytics and ranking on OpenRouter. This is sent as the `HTTP-Referer` header.

- **`app_name`** _(string)_ (optional): Your app name for analytics on OpenRouter. This is sent as the `X-Title` header.

- **`fallback_models`** _(list[string])_ (optional): List of fallback models to use if the primary model is unavailable. Example: `fallback_models=["anthropic/claude-sonnet-4", "openai/gpt-5-mini"]`.

- **`provider`** _(dict)_ (optional): Provider routing preferences for fine-grained control over model selection. Can include:

- `order`: List of preferred providers in order
- `allow_fallbacks`: Whether to allow fallback to other providers
- `require_parameters`: Whether to require specific parameters
- `data_collection`: Data collection preference, either "allow" or "deny"
- `only`: List of providers to exclusively use
- `ignore`: List of providers to exclude
- `quantizations`: List of accepted quantization levels
- `sort`: Sort providers by "price", "throughput", or "latency"
- `max_price`: Maximum price per token
Refer to the [OpenRouter documentation](https://openrouter.ai/docs/features/provider-routing) for more information.

- **`plugins`** _(list[OpenRouterWebPlugin])_ (optional): List of OpenRouter plugins to enable. Currently supports web search plugin with configuration for max results and search prompts.

### Examples

The following examples demonstrate usage of various OpenRouter parameters.

Configure multiple fallback models to use if the primary model is unavailable:

```python
from livekit.plugins import openai

llm = openai.LLM.with_openrouter(
    model="openai/gpt-4o",
    fallback_models=[
        "anthropic/claude-sonnet-4",
        "openai/gpt-5-mini",
    ],
)

```

Control which providers are used for model inference:

```python
from livekit.plugins import openai

llm = openai.LLM.with_openrouter(
    model="deepseek/deepseek-chat-v3.1",
    provider={
        "order": ["novita/fp8", "gmicloud/fp8", "google-vertex"],
        "allow_fallbacks": True,
        "sort": "latency",
    },
)

```

Enable OpenRouter's web search capabilities:

```python
from livekit.plugins import openai

llm = openai.LLM.with_openrouter(
    model="google/gemini-2.5-flash-preview-09-2025",
    plugins=[
        openai.OpenRouterWebPlugin(
            max_results=5,
            search_prompt="Search for relevant information",
        )
    ],
)

```

Include site and app information for OpenRouter analytics:

```python
from livekit.plugins import openai

llm = openai.LLM.with_openrouter(
    model="openrouter/auto",
    site_url="https://myapp.com",
    app_name="My Voice Agent",
)

```

## Additional resources

The following links provide more information about the OpenRouter integration.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai)**: View the source or contribute to the LiveKit OpenAI LLM plugin.

- **[OpenRouter docs](https://openrouter.ai/docs)**: OpenRouter API documentation and model list.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and OpenRouter.

---


For the latest version of this document, see [https://docs.livekit.io/agents/models/llm/plugins/openrouter.md](https://docs.livekit.io/agents/models/llm/plugins/openrouter.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).