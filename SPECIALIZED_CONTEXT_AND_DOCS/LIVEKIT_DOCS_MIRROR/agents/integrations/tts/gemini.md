LiveKit Docs › Integration guides › Text-to-speech (TTS) › Gemini

---

# Gemini TTS integration guide

> How to use the Gemini TTS plugin for LiveKit Agents.

## Overview

[Gemini TTS](https://ai.google.dev/gemini-api/docs/speech-generation) generates speech from text with customizable style, tone, accent, and pace through natural language prompts. It’s designed for scenarios that require precise recitation and nuanced audio output. With LiveKit's Gemini TTS integration and the Agents framework, you can build voice AI applications with accurate and nuanced speech.

## Quick reference

This section includes a basic usage example and some reference material. For links to more detailed documentation, see [Additional resources](#additional-resources).

### Installation

Install the plugin from PyPI:

```bash
pip install "livekit-agents[google]~=1.2"

```

### Authentication

Credentials must be provided by one of the following methods:

- To use VertexAI: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the [service account key](https://cloud.google.com/iam/docs/keys-create-delete) file.
- To use Gemini API: Set the `api_key` argument or the `GOOGLE_API_KEY` environment variable.

### Usage

Use a Gemini TTS in an `AgentSession` or as a standalone speech generator. For example, you can use this TTS in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md).

```python
from livekit.plugins import google

session = AgentSession(
  tts = google.beta.GeminiTTS(
   model="gemini-2.5-flash-preview-tts",
   voice_name="Zephyr",
   instructions="Speak in a friendly and engaging tone.",
  ),
  # ... llm, stt, etc.
)

```

### Parameters

This section describes some of the available parameters. See the [plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/google/index.html.md#livekit.plugins.google.TTS) for a complete list of all available parameters.

- **`model`** _(string)_ (optional) - Default: `gemini-2.5-flash-preview-tts`: The model to use for speech generation. For a list of models, see [Supported models](https://ai.google.dev/gemini-api/docs/speech-generation#supported-models).

- **`voice_name`** _(string)_ (optional) - Default: `Kore`: Voice name. For supported voices, see [Voice options](https://ai.google.dev/gemini-api/docs/speech-generation#voices).

- **`voice_name`** _(string)_ (optional): Name of the voice to use for speech. For a full list of voices, see [Supported voices and languages](https://cloud.google.com/text-to-speech/docs/voices).

- **`instructions`** _(string)_ (optional): Prompt to control the style, tone, accent, and pace. To learn more, see [Controlling speech style with prompts](https://ai.google.dev/gemini-api/docs/speech-generation#controllable).

## Additional resources

The following resources provide more information about using Gemini TTS with LiveKit Agents.

- **[Python package](https://pypi.org/project/livekit-plugins-google/)**: The `livekit-plugins-google` package on PyPI.

- **[Plugin reference](https://docs.livekit.io/reference/python/v1/livekit/plugins/google/beta/index.html.md#livekit.plugins.google.beta.TTS)**: Reference for the Gemini TTS plugin.

- **[GitHub repo](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-google)**: View the source or contribute to the LiveKit Google plugin.

- **[Gemini TTS docs](https://ai.google.dev/gemini-api/docs/speech-generation)**: Gemini Developer API docs for TTS.

- **[Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)**: Get started with LiveKit Agents and Gemini TTS.

- **[Google ecosystem guide](https://docs.livekit.io/agents/integrations/google.md)**: Overview of the entire Google AI and LiveKit Agents integration.

---


For the latest version of this document, see [https://docs.livekit.io/agents/integrations/tts/gemini.md](https://docs.livekit.io/agents/integrations/tts/gemini.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).