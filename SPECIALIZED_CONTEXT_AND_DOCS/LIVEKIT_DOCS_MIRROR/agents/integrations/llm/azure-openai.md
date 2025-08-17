LiveKit Docs › Integration guides › Large language models (LLM) › Azure OpenAI

---

# Azure OpenAI LLM integration guide

> How to use the Azure OpenAI LLM plugin for LiveKit Agents.

## Overview

[Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) provides access to OpenAI's powerful language models like `gpt-4o` and `o1` through Azure's managed service. With LiveKit's Azure OpenAI integration and the Agents framework, you can build sophisticated voice AI applications using their industry-leading models.

> ℹ️ **Note**
> 
> Using the OpenAI platform instead of Azure? See our [OpenAI LLM integration guide](https://docs.livekit.io/agents/integrations/llm/openai.md).

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```bash
pip install "livekit-agents[openai]~=1.2"

```

### Authentication

The Azure OpenAI plugin requires either an [Azure OpenAI API key](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource) or a Microsoft Entra ID token.

Set the following environment variables in your `.env` file:

- `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_ENTRA_TOKEN`
- `AZURE_OPENAI_ENDPOINT`
- `OPENAI_API_VERSION`

### Usage

Use Azure OpenAI within an `AgentSession` or as a standalone LLM service. For example, you can use this LLM in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import openai

session = AgentSession(
    llm=openai.LLM.with_azure(
        azure_deployment="<model-deployment>",
        azure_endpoint="https://<endpoint>.openai.azure.com/", # or AZURE_OPENAI_ENDPOINT
        api_key="<api-key>", # or AZURE_OPENAI_API_KEY
        api_version="2024-10-01-preview", # or OPENAI_API_VERSION
    ),
    # ... tts, stt, vad, turn_detection, etc.
)

```

### Parameters

This section describes the Azure-specific parameters. For a complete list of all available parameters, see the [plugin documentation](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.LLM.with_azure).

- **`azure_deployment`** _(string)_: Name of your model deployment.

- **`entra_token`** _(string)_ (optional): Microsoft Entra ID authentication token. Required if not using API key authentication. To learn more see Azure's [Authentication](https://learn.microsoft.com/en-us/azure/ai-services/openai/realtime-audio-reference#authentication) documentation.

## Additional resources

The following links provide more information about the Azure OpenAI LLM plugin.

- **[Python package](https://pypi.org/project/livekit-plugins-openai/)**: The `livekit-plugins-openai` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/openai/index.html.md#livekit.plugins.openai.LLM.with_azure)**: Reference for the Azure OpenAI LLM plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai)**: View the source or contribute to the LiveKit OpenAI LLM plugin.

- **[Azure OpenAI docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/)**: Azure OpenAI service documentation.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Azure OpenAI.

- **[Azure ecosystem overview](https://docs.livekit.io/agents/integrations/azure.md)**: Overview of the entire Azure AI ecosystem and LiveKit Agents integration.

---


For the latest version of this document, see [https://docs.livekit.io/agents/integrations/llm/azure-openai.md](https://docs.livekit.io/agents/integrations/llm/azure-openai.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).